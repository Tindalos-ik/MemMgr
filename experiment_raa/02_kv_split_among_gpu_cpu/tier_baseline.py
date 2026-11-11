"""在独占 vLLM 上回放超出 GPU 容量的请求，测量 CPU KV 回迁与重算成本。"""

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
import psutil
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer


def save(path, value):
    """将 value 以 UTF-8 JSON 写入 path。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metrics(client):
    """读取 client 服务的原始 Prometheus 文本和按名称求和的计数。"""
    response = client.get("/metrics")
    response.raise_for_status()
    counters = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name.endswith(("_total", "_sum", "_count")):
                counters[sample.name] = counters.get(sample.name, 0) + sample.value
    return response.text, counters


def settled(client, previous, prompt_tokens):
    """等待 client 的请求计数及输入用量相对 previous 增加，核验 prompt_tokens。"""
    deadline = time.monotonic() + 25
    while True:
        raw, current = metrics(client)
        completed = current["vllm:request_success_total"] - previous["vllm:request_success_total"]
        if completed == 1:
            # 再跨过一次统计周期，收齐异步 store 完成记录。
            time.sleep(0.6)
            raw, current = metrics(client)
            delta = {key: current[key] - previous.get(key, 0) for key in current}
            if delta["vllm:request_success_total"] != 1:
                raise RuntimeError("Exclusive service received an unexpected request")
            if delta["vllm:prefix_cache_queries_total"] != prompt_tokens:
                raise RuntimeError(f"Invalid metric boundary: {delta}")
            if any(value < 0 for value in delta.values()):
                raise RuntimeError("Service counters decreased")
            return raw, current, delta
        if completed > 1 or time.monotonic() > deadline:
            raise RuntimeError("Request metrics failed to publish at an exclusive boundary")
        time.sleep(0.1)


def trace(tokenizer, repeats, seed):
    """用 tokenizer、repeats 和 seed 生成三种长度及复访距离的固定请求序列。"""
    rng = random.Random(seed)
    corpus = tokenizer.encode("The memory system stores keys and values for attention. " * 4096,
                              add_special_tokens=False)
    cases = [(length, gap) for length in (256, 1024, 3072) for gap in (0, 2, 6)]
    plan = []
    for repeat in range(repeats):
        rng.shuffle(cases)
        for length, gap in cases:
            case = f"r{repeat}-l{length}-g{gap}"
            prompts = []
            for index, size in enumerate([length] + [1024] * gap):
                prefix = tokenizer.encode(rng.randbytes(32).hex() + "\n", add_special_tokens=False)
                tokens = (prefix + corpus)[:size]
                if len(tokens) != size:
                    raise ValueError("Prompt construction failed")
                prompts.append(tokens)
            for phase, tokens in [("cold", prompts[0])] + [("filler", p) for p in prompts[1:]] + [("revisit", prompts[0])]:
                plan.append(dict(case=case, repeat=repeat, length=length, gap=gap,
                                 phase=phase, prompt_token_ids=tokens))
    first_blocks = [tuple(row["prompt_token_ids"][:64]) for row in plan if row["phase"] != "revisit"]
    if len(set(first_blocks)) != len(first_blocks):
        raise ValueError("Distinct requests share a first cache block")
    return plan


def monitor(pid, stop, path):
    """采样 pid 进程树 RSS、系统内存和整卡 GPU 指标，直到 stop 置位，写入 path。"""
    with path.open("w", encoding="utf-8") as stream:
        while not stop.is_set():
            parent = psutil.Process(pid)
            processes = [parent] + parent.children(recursive=True)
            rss = 0
            for process in processes:
                if process.is_running():
                    rss += process.memory_info().rss
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"], text=True).strip()
            stream.write(json.dumps(dict(time=time.time(), process_tree_rss=rss,
                                         host_available=psutil.virtual_memory().available, gpu=gpu)) + "\n")
            stream.flush()
            stop.wait(1)


def experiment(args, mode, epoch, plan, request):
    """用 args 启动 mode 独占服务，回放 epoch 的 plan，使用 request 记录真实流式响应。"""
    output = args.output / f"e{epoch}-{mode}"
    output.mkdir()
    cpu_bytes = {"gpu": 0, "cpu512": 512 * 2**20, "cpu64": 64 * 2**20}[mode]
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
            "--served-model-name", "tier-model", "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "float16", "--max-model-len", "3584", "--max-num-seqs", "1",
            "--block-size", "64", "--num-gpu-blocks-override", "64",
            "--kv-cache-memory-bytes", str(112 * 2**20), "--gpu-memory-utilization", "0.65", "--enforce-eager",
            "--enable-prefix-caching", "--generation-config", "vllm", "--no-enable-log-requests"]
    if cpu_bytes:
        argv += ["--kv-transfer-config", json.dumps(dict(kv_connector="OffloadingConnector", kv_role="kv_both",
                     kv_connector_extra_config=dict(cpu_bytes_to_use=cpu_bytes, block_size=64,
                                                     eviction_policy="lru")))]
    save(output / "server.json", dict(argv=argv, cpu_bytes=cpu_bytes, gpu_blocks=64,
                                       block_tokens=64, kv_bytes_per_token=28672))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stop = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        sampler = pool.submit(monitor, process.pid, stop, output / "resources.jsonl")
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120, trust_env=False) as client:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM exited: {output / 'server.log'}")
                    try:
                        response = client.get("/health", timeout=2)
                    except httpx.ConnectError:
                        response = None
                    if response is not None:
                        response.raise_for_status()
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("vLLM startup exceeded 600 seconds")
                    time.sleep(1)
                models = client.get("/v1/models")
                models.raise_for_status()
                save(output / "models.json", models.json())
                # 预热覆盖 prefill、decode 和实际回迁的三种长度。
                warmups = []
                for length in (256, 1024, 3072):
                    warm = [length] + [16] * (length - 1)
                    result = request(client, "tier-model", warm, args.output_tokens, args.seed)
                    warmups.append(dict(phase="initial", length=length, **result))
                for index in range(6):
                    result = request(client, "tier-model", [1000 + index] + [16] * 1023,
                                     args.output_tokens, args.seed)
                    warmups.append(dict(phase="pressure", length=1024, **result))
                for length in (256, 1024, 3072):
                    result = request(client, "tier-model", [length] + [16] * (length - 1),
                                     args.output_tokens, args.seed)
                    warmups.append(dict(phase="revisit", length=length, **result))
                save(output / "warmup.json", warmups)
                time.sleep(1.5)
                raw, before = metrics(client)
                (output / "initial.prom").write_text(raw, encoding="utf-8")
                required = {"vllm:request_success_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"}
                if cpu_bytes:
                    # load 的 label 在首次真实回迁后创建，预热阶段只要求 store 指标。
                    required |= {f"vllm:kv_offload_store_{quantity}_total" for quantity in ("bytes", "time")}
                if required - before.keys():
                    raise RuntimeError(f"Missing required metrics: {required - before.keys()}")
                with (output / "requests.jsonl").open("w", encoding="utf-8") as stream:
                    for index, row in enumerate(plan):
                        if sampler.done():
                            sampler.result()
                            raise RuntimeError("Resource sampler exited unexpectedly")
                        tokens = row["prompt_token_ids"]
                        result = request(client, "tier-model", tokens, args.output_tokens, args.seed)
                        raw, after, delta = settled(client, before, len(tokens))
                        if mode == "cpu512" and row["phase"] == "revisit" and row["gap"] == 6:
                            if delta.get("vllm:kv_offload_load_bytes_total", 0) <= 0:
                                raise RuntimeError("Expected CPU promotion did not occur")
                        (output / f"{index:04}.prom").write_text(raw, encoding="utf-8")
                        record = dict(index=index, **{k: v for k, v in row.items() if k != "prompt_token_ids"},
                                      prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                                      metrics=delta, **result)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        before = after
                        if row["phase"] == "revisit":
                            print(json.dumps(dict(epoch=epoch, mode=mode, case=row["case"],
                                                  ttft=result["ttft_seconds"], gpu_hits=delta["vllm:prefix_cache_hits_total"],
                                                  load_bytes=delta.get("vllm:kv_offload_load_bytes_total"))), flush=True)
                (output / "final.prom").write_text(raw, encoding="utf-8")
        finally:
            stop.set()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            sampler.result()
            pool.shutdown()


def main():
    """读取命令行参数，在独立输出目录执行正反顺序的三组真实分层实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--client-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.output_tokens <= 128:
        raise ValueError("Invalid experiment size")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
                      VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
                      VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC", HF_HOME=str(args.runtime / "hf"),
                      VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"),
                      TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"))
    sys.path.insert(0, str(args.client_dir))
    from prefix_baseline import request
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    save(args.output / "environment.json", dict(arguments=vars(args), python=sys.version,
         source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), args.client_dir / "prefix_baseline.py")},
         cache_layout="LBHNC", stats_interval_seconds=0.5,
         packages={name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "httpx", "prometheus-client", "psutil")},
         model_config=json.loads((Path(args.model) / "config.json").read_text()),
         gpu=subprocess.check_output(["nvidia-smi", "-q"], text=True)))
    for epoch, modes in enumerate((("gpu", "cpu512", "cpu64"), ("cpu64", "cpu512", "gpu"))):
        plan = trace(tokenizer, args.repeats, args.seed + epoch)
        if args.pilot:
            plan = [row for row in plan if row["repeat"] == 0 and row["length"] == 1024 and row["gap"] == 6]
            modes = ("gpu", "cpu512")
        save(args.output / f"trace-{epoch}.json", plan)
        for mode in modes:
            experiment(args, mode, epoch, plan, request)
        if args.pilot:
            break


if __name__ == "__main__":
    main()
