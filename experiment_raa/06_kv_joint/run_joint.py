"""串行采集真实 Qwen 请求，冻结正式复用模型和两套成本表，再运行四组联合评价。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE.parent / "02_kv_split_among_gpu_cpu"),
                str(HERE.parent / "03_kv_restore_cost")]

from run_cost import digest, request, save
from tier_baseline import metrics, monitor, settled

from memmgr.inference.models import local_snapshot
from memmgr.short_term.config import local_path, transfer_config

GPU_BYTES = 32 * 2**20
CPU_BYTES = 56 * 2**20
MAX_MODEL_LEN = 1088
OUTPUT_TOKENS = 8
MARGIN_MS = 0.5
MODES = (("lru", "always"), ("lru", "gate"), ("learned", "always"), ("learned", "gate"))
REQUIRED_METRICS = {"vllm:request_success_total", "vllm:prefix_cache_queries_total",
                    "vllm:prefix_cache_hits_total", "vllm:request_prefill_kv_computed_tokens_sum",
                    "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
                    "vllm:num_preemptions_total"}


def make_plan(tokenizer, stage, rounds, seed):
    """用 tokenizer 构造 stage 独立首块；rounds 为轮数，seed 固定 nonce 及组内顺序。"""
    rng = random.Random(seed)
    corpus = tokenizer.encode("The memory system stores keys and values for attention. " * 120,
                              add_special_tokens=False)
    plan = []

    def prompt(length):
        """生成 length 个 token，并将阶段与随机 nonce 编入首块。"""
        nonce = tokenizer.encode(f"{stage} {rng.randbytes(32).hex()}\n", add_special_tokens=False)
        return (nonce + corpus)[:length]

    def append(case, number, phase, gap, tokens):
        """记录 case、轮次 number、phase、干扰间隔 gap 与输入 tokens。"""
        plan.append(dict(index=len(plan), case=case, round=number, phase=phase, gap=gap,
                         prompt_token_ids=tokens, prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))

    if stage == "warmup":
        append("warmup", 0, "prime", 0, prompt(1025))
        return plan

    for number in range(rounds):
        cases = [(length, gap) for length in (65, 513) for gap in (1, 2)]
        rng.shuffle(cases)
        for length, gap in cases:
            case = f"{stage}-{number}-l{length}-g{gap}"
            target = prompt(length + 64)
            for _ in range(4):
                append(case, number, "prime", gap, target[:length])
            for _ in range(gap):
                append(case, number, "filler", gap, prompt(1025))
            append(case, number, "revisit", gap, target)
    return plan


def command(args, directory, policy, restore):
    """用 args 的项目模型和固定容量启动正式 policy/restore，审计写入 directory。"""
    options = dict(tier="cpu", policy=policy, batch_invariant=True, block_tokens=64,
                   gpu_bytes=GPU_BYTES, cpu_bytes=CPU_BYTES, restore=restore,
                   margin_ms=MARGIN_MS, audit_path=str(directory / "audit.jsonl"))
    if policy == "learned":
        options["model_dir"] = str(args.output / "model")
    if restore == "gate":
        options["cost_profile"] = str(args.output / "calibration" / f"{policy}.json")
    server = dict(args.server, max_model_len=MAX_MODEL_LEN, max_num_seqs=1, kv_cache=options)
    transfer = transfer_config(server, ROOT)
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model),
            "--served-model-name", "cost-model", "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", server["dtype"], "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", "1",
            "--block-size", "64", "--kv-cache-memory-bytes", str(GPU_BYTES),
            "--gpu-memory-utilization", "0.8", "--enforce-eager", "--enable-prefix-caching",
            "--generation-config", "vllm", "--no-enable-log-requests", "--kv-transfer-config", json.dumps(transfer)]


def ready(client, process, port, log):
    """等待 port 监听及 client 的健康检查；process 退出时指出 log。"""
    deadline = time.monotonic() + 600
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited ({process.returncode}): {log}")
        with socket.socket() as probe:
            listening = probe.connect_ex(("127.0.0.1", port)) == 0
        if listening:
            client.get("/health", timeout=10).raise_for_status()
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"vLLM startup exceeded 600 seconds: {log}")
        time.sleep(0.5)


def capture(client, plan, directory, seed, sampler, prefix, audit):
    """用 client 按 seed/prefix 回放 plan，并用 audit 关联真实请求 ID 和 Prom。"""
    raw, before = metrics(client)
    if missing := REQUIRED_METRICS - before.keys():
        raise RuntimeError(f"Missing Prometheus metrics: {sorted(missing)}")
    (directory / "initial.prom").write_text(raw, encoding="utf-8")
    with (directory / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for row in plan:
            if sampler.done():
                sampler.result()
                raise RuntimeError("Resource sampler stopped unexpectedly")
            index = row["index"]
            tokens = row["prompt_token_ids"]
            request_id = f"{prefix}-{index:04}"
            result = request(client, tokens, OUTPUT_TOKENS, seed, request_id)
            raw, after, delta = settled(client, before, len(tokens))
            begins = [event for line in audit if (event := json.loads(line))["kind"] == "begin"]
            if (len(begins) != 1 or begins[0]["prompt_sha256"] != row["prompt_sha256"]
                    or begins[0]["prompt_tokens"] != len(tokens)):
                raise RuntimeError(f"Unexpected connector request: {request_id}")
            server_ttft = delta["vllm:time_to_first_token_seconds_sum"]
            if (server_ttft <= 0 or delta["vllm:time_to_first_token_seconds_count"] != 1
                    or delta["vllm:num_preemptions_total"] != 0):
                raise RuntimeError(f"Invalid per-request counters: {request_id}")
            if any(not item.startswith("token_id:") or not item[9:].isdigit() for item in result["generated_tokens"]):
                raise RuntimeError(f"vLLM did not return token IDs: {request_id}")
            (directory / f"{index:04}.prom").write_text(raw, encoding="utf-8")
            record = {key: value for key, value in row.items() if key != "prompt_token_ids"}
            record.update(request_id=begins[0]["request_id"], client_request_id=request_id,
                          prompt_tokens=len(tokens), server_ttft_seconds=server_ttft, metrics=delta, **result)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            before = after
    (directory / "final.prom").write_text(raw, encoding="utf-8")


def run(args, directory, policy, restore, plan, warmup):
    """在 directory 启动独立 policy/restore 进程，先执行 warmup，再回放 plan。"""
    directory.mkdir(parents=True)
    (directory / "warmup").mkdir()
    argv = command(args, directory, policy, restore)
    save(directory / "server.json", dict(argv=argv, env=args.env, policy=policy, restore=restore))
    print(f"开始 {directory}", flush=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    log = directory / "server.log"
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=os.environ | args.env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        stop = threading.Event()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                sampler = pool.submit(monitor, process.pid, stop, directory / "resources.jsonl")
                try:
                    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120,
                                      trust_env=False) as client:
                        ready(client, process, args.port, log)
                        with (directory / "audit.jsonl").open(encoding="utf-8") as audit:
                            capture(client, warmup, directory / "warmup", args.seed, sampler, "warm", audit)
                            capture(client, plan, directory, args.seed, sampler, "joint", audit)
                finally:
                    stop.set()
                sampler.result()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if (directory / "audit.jsonl.1").exists() or not (directory / "audit.jsonl").stat().st_size:
        raise RuntimeError(f"Incomplete connector audit: {directory}")
    print(f"完成 {directory}", flush=True)


def formal(output, module, *argv):
    """在项目环境执行正式 module 命令，追加 argv 及退出信息到 output 的命令日志。"""
    cmd = [sys.executable, "-m", f"memmgr.short_term.{module}", *map(str, argv)]
    with (output / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(cmd) + "\n")
    with (output / f"{module}.log").open("a", encoding="utf-8") as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    """固定实验环境，先采集和训练，再按策略独立校准和评价。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".agent/kv-joint/output")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--through", choices=("smoke", "collection", "calibration", "eval"), default="eval")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    if importlib.metadata.version("vllm") != "0.29.0":
        raise ValueError("Require vLLM 0.29.0")
    args.output = local_path(args.output, ROOT)
    args.server = json.loads((ROOT / "config/server.json").read_text(encoding="utf-8"))
    args.model = local_snapshot(args.server["model"], args.server["revision"])
    model = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if args.server["dtype"] != "float16" or model["hidden_size"] % model["num_attention_heads"]:
        raise ValueError("Unsupported model KV dtype or head dimensions")
    kv_bytes_per_block = (model["num_hidden_layers"] * 2 * model["num_key_value_heads"]
                          * (model["hidden_size"] // model["num_attention_heads"]) * 2 * 64)
    args.env = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
                    VLLM_BATCH_INVARIANT="1", VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
                    VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), HERE.parent / "02_kv_split_among_gpu_cpu" / "tier_baseline.py",
               HERE.parent / "03_kv_restore_cost" / "run_cost.py", *sorted((ROOT / "memmgr/short_term").glob("*.py"))]
    save(args.output / "environment.json", dict(arguments=vars(args), command=[sys.executable, *sys.argv],
         kv_bytes_per_block=kv_bytes_per_block, model_config_sha256=digest(args.model / "config.json"),
         packages={name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "scikit-learn")},
         source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources},
         gpu=subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    plans = {stage: make_plan(tokenizer, stage, rounds, args.seed + offset)
             for offset, (stage, rounds) in enumerate((("warmup", 1), ("smoke", 1), ("collect", 4),
                                                       ("calibration", 4), ("eval", 3)))}
    plan_dir = args.output / "plans"
    plan_dir.mkdir()
    prior = set()
    for stage, plan in plans.items():
        first_blocks = {tuple(row["prompt_token_ids"][:64]) for row in plan}
        cases = {(row["case"], tuple(row["prompt_token_ids"][:64])) for row in plan if row["phase"] != "filler"}
        if prior & first_blocks or len(first_blocks) != len(cases) + sum(row["phase"] == "filler" for row in plan):
            raise ValueError(f"Shared prefix across independent cases or stages: {stage}")
        prior.update(first_blocks)
        save(plan_dir / f"{stage}.json", plan)
    if args.through == "smoke":
        run(args, args.output / "smoke" / "lru-always", "lru", "always", plans["smoke"], plans["warmup"])
        return
    collection = args.output / "collect" / "lru-always"
    run(args, collection, "lru", "always", plans["collect"], plans["warmup"])
    formal(args.output, "train", "--audit", collection / "audit.jsonl", "--output", args.output / "model")
    if args.through == "collection":
        return
    calibration = args.output / "calibration"
    for policy in ("lru", "learned"):
        always, never = (calibration / f"{policy}-{mode}" for mode in ("always", "never"))
        for mode, directory in (("always", always), ("never", never)):
            run(args, directory, policy, mode, plans["calibration"], plans["warmup"])
        formal(args.output, "cost", "--pair", always, never, "--output", calibration / f"{policy}.json")
    if args.through == "calibration":
        return
    for policy, mode in MODES:
        run(args, args.output / "eval" / f"{policy}-{mode}", policy, mode, plans["eval"], plans["warmup"])


if __name__ == "__main__":
    main()
