"""每次运行一个持锁的真实服务，采集独立 KV 校准或固定未见回放。"""

import argparse
from collections import Counter
from importlib import import_module
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
from types import SimpleNamespace

import httpx
import psutil
from transformers import AutoTokenizer

from memmgr.short_term.config import local_path, transfer_config
from memmgr.short_term.retention import RetentionProfile, filesystem_device, fit, retention_context

from .historical import BLOCK_BYTES, PHASE, POLICIES, digest, read, rows, save, sha
from .review import process_io, prometheus, verify_metrics
from .ssd_evidence import replay as replay_ssd, verify_serial


ssd = import_module("experiment_raa.07_kv_joint_ssd.run_ssd_joint")
joint = ssd.joint
ROOT = Path(__file__).resolve().parents[2]
BLOCK = 64
CPU_BYTES = 28 * 2**20
MAX_DEPTH = CPU_BYTES // BLOCK_BYTES
GBDT = {"model.json": "4bdd0df8cb08f250a89c297b3f8469349a817799006a1bcf3d60217549e019e6",
        "model.joblib": "5c97e4045088bd5e47046f3282b31a2ee6bd10ce942e178bf75993a7efe60373"}
ENV = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", PYTHONHASHSEED="0",
           VLLM_BATCH_INVARIANT="1", VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
           VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC")


def verify_model(directory):
    """核对 directory 中冻结的 GBDT 文件、特征元数据与准入阈值。"""
    if any(sha(directory / name) != expected for name, expected in GBDT.items()):
        raise ValueError("Expected experiment 06's original frozen GBDT")


def freeze(output, source):
    """在新 output 冻结输入；source 提供 06 的冻结模型、训练及历史校准计划。"""
    output.mkdir(parents=True)
    server = read(ROOT / "config/server.json")
    model = joint.local_snapshot(server["model"], server["revision"])
    model_dir = source / "model"
    verify_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    body = tokenizer.encode(" A cache entry can be shared by requests with the same ordered token prefix." * 256,
                            add_special_tokens=False)
    randoms = {phase: random.Random(seed) for phase, seed in (("calibration", 13071), ("eval", 13072))}

    def prompt(phase, length):
        """生成 phase 独立的 length 个 token，首块包括独立随机内容。"""
        prefix = tokenizer.encode(f"{phase} {randoms[phase].randbytes(24).hex()}\n", add_special_tokens=False)
        tokens = (prefix + body)[:length]
        if len(tokens) != length or len(prefix) >= BLOCK:
            raise ValueError("Invalid tokenized experiment prompt")
        return tokens

    calibration, evaluation = [], []
    for depth in (1, 4, 8, MAX_DEPTH):
        for repeat in range(4):
            tokens = prompt("calibration", min(depth * BLOCK + 65, joint.MAX_MODEL_LEN - joint.OUTPUT_TOKENS))
            calibration.append(dict(case=f"b{depth:02}-{repeat}", blocks=depth, tokens=tokens,
                                    filler=prompt("calibration", MAX_DEPTH * BLOCK + 1)))
    randoms["calibration"].shuffle(calibration)
    targets = [prompt("eval", length) for length in (129, 321, 577, 1025) for _ in range(3)]
    randoms["eval"].shuffle(targets)
    for i, target in enumerate(targets):
        evaluation.extend(dict(case=f"target-{i:02}", phase="prime", tokens=target) for _ in range(4))
        for j in range(1 + (i % 3 == 2)):
            evaluation.append(dict(case=f"filler-{i:02}-{j}", phase="filler", tokens=prompt("eval", 1025)))
    order = list(range(len(targets)))
    for pass_id in range(2):
        randoms["eval"].shuffle(order)
        evaluation.extend(dict(case=f"target-{i:02}", phase=f"revisit-{pass_id}", tokens=targets[i]) for i in order)
    warmup = {stage: prompt(stage, 1025) for stage in randoms}
    cal_prefixes = {digest(tokens[:BLOCK]) for tokens in [warmup["calibration"], *(
        item[name] for item in calibration for name in ("tokens", "filler"))]}
    eval_prefixes = {digest(row["tokens"][:BLOCK]) for row in evaluation} | {digest(warmup["eval"][:BLOCK])}
    historic_prefixes = {digest(row["prompt_token_ids"][:BLOCK]) for stage in ("collect", "calibration", "eval", "warmup")
                         for row in read(source / f"plans/{stage}.json")}
    if (len(cal_prefixes) != 1 + 2 * len(calibration) or len(eval_prefixes) != len(targets) + 17
            or cal_prefixes & eval_prefixes or (cal_prefixes | eval_prefixes) & historic_prefixes):
        raise ValueError("Experiment prefix domains are not independent")
    save(output / "plan.json", dict(calibration=calibration, evaluation=evaluation, warmup=warmup,
                                    calibration_prefixes=sorted(cal_prefixes), evaluation_prefixes=sorted(eval_prefixes)))
    save(output / "experiment.json", dict(server=server, model=str(model), model_dir=str(model_dir), env=ENV,
                                          model_sha256=GBDT["model.joblib"], gpu_bytes=joint.GPU_BYTES,
                                          cpu_bytes=CPU_BYTES, seed=13073, plan_sha256=sha(output / "plan.json")))


def command(config, directory, port, policy, tier, profile):
    """为 config 准备真实服务 argv；port、policy、tier、profile 只接入这一组。"""
    server = config["server"]
    args = SimpleNamespace(model=config["model"], server=server, port=port)
    argv = joint.command(args, directory, "lru", "always")
    options = dict(tier=tier, policy=policy, batch_invariant=True, gpu_bytes=config["gpu_bytes"],
                   cpu_bytes=config["cpu_bytes"], audit_path=str(directory / "audit.jsonl"))
    if policy != "lru":
        options["model_dir"] = config["model_dir"]
    if policy == "time_value":
        options["retention_profile"] = str(profile)
    if tier == "ssd":
        path = directory / "ssd"
        path.mkdir()
        options.update(ssd_root=str(path), ssd_bytes=ssd.SSD_BYTES)
        argv[-2:-2] = ["--shutdown-timeout", "20"]
    transfer = transfer_config(dict(server, max_model_len=joint.MAX_MODEL_LEN, max_num_seqs=1, kv_cache=options), ROOT)
    argv[-1] = json.dumps(transfer)
    return argv


def available_events(audit):
    """从 audit 读取可用完整文本行，保留正在写入的行。"""
    while True:
        offset = audit.tell()
        line = audit.readline()
        if not line.endswith("\n"):
            audit.seek(offset)
            return
        yield json.loads(line)


def transfer_totals(events):
    """从 events 得到 CPU 提交与完成，以及 SSD 完成的读写块和用时。"""
    counts = Counter()
    for event in events:
        kind = event["kind"]
        if kind in ("store", "stored", "loaded"):
            counts[kind] += len(event["keys"])
        elif kind == "fs_complete":
            if not event["success"]:
                raise ValueError("Failed audited SSD transfer")
            direction = "ssd_" + event["direction"]
            counts[direction] += len(event["keys"])
            counts[direction + "_seconds"] += event["transfer_seconds"]
    return counts


def generate(client, tokens, seed, tag):
    """发送 tokens 并从 client 首个真实 token ID 事件计时，seed 和 tag 跨组固定。"""
    payload = dict(model="cost-model", prompt=tokens, stream=True, temperature=0, seed=seed,
                   request_id=tag, max_tokens=joint.OUTPUT_TOKENS, ignore_eos=True,
                   logprobs=1, return_tokens_as_token_ids=True, stream_options={"include_usage": True})
    generated, first, usage, stopped = [], None, None, False
    started = time.perf_counter()
    with client.stream("POST", "/v1/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                stopped = True
                break
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item["error"])
            usage = item.get("usage") or usage
            for choice in item["choices"]:
                if choice["logprobs"] and (tokens_out := choice["logprobs"]["tokens"]):
                    if first is None:
                        first = time.perf_counter() - started
                    generated.extend(tokens_out)
    if (not stopped or first is None or usage is None or len(generated) != joint.OUTPUT_TOKENS
            or (usage["prompt_tokens"], usage["completion_tokens"]) != (len(tokens), len(generated))):
        raise ValueError(f"Incomplete generated token stream: {tag}")
    return dict(client_ttft_seconds=first, generated_tokens=generated, usage=usage)


def capture(client, audit, previous, directory, index, config, tokens, tier):
    """采集 tokens 的真实输出与数据路径，逐请求核对原生计数和 audit。"""
    tag = f"tv13-{index:04}"
    row = generate(client, tokens, config["seed"], tag)
    events = []
    deadline = time.monotonic() + 25
    jobs = set()
    while True:
        for event in available_events(audit):
            events.append(event)
            if event["kind"] == "fs_submit":
                jobs.add(event["job"])
            elif event["kind"] == "fs_complete":
                jobs.remove(event["job"])
        raw, counters = joint.metrics(client)
        delta = {key: counters[key] - previous.get(key, 0) for key in counters}
        counts = transfer_totals(events)
        external = delta.get("vllm:external_prefix_cache_hits_total", 0)
        if (delta["vllm:request_success_total"] == 1 and counts["store"] == counts["stored"]
                and delta.get("vllm:kv_offload_store_bytes_total", 0) == counts["stored"] * BLOCK_BYTES
                and delta.get("vllm:kv_offload_load_bytes_total", 0) == counts["loaded"] * BLOCK_BYTES
                and external == counts["loaded"] * BLOCK and any(event["kind"] == "store" for event in events)
                and (tier == "cpu" or (not jobs and any(event["kind"] == "ssd_request_closed" for event in events)
                    and delta.get("vllm:kv_offload_tiering_read_bytes_total", 0) == counts["ssd_load"] * BLOCK_BYTES
                    and delta.get("vllm:kv_offload_tiering_write_bytes_total", 0) == counts["ssd_store"] * BLOCK_BYTES))):
            break
        if delta["vllm:request_success_total"] > 1 or time.monotonic() >= deadline:
            raise RuntimeError(f"Request metrics did not drain: {tag}, {counts}, {list(delta.items())}")
        time.sleep(0.05)
    begin, = (event for event in events if event["kind"] == "begin")
    gpu = int(delta["vllm:prefix_cache_hits_total"])
    computed = int(delta["vllm:request_prefill_kv_computed_tokens_sum"])
    ttft = delta["vllm:time_to_first_token_seconds_sum"]
    if (not begin["request_id"].startswith(f"cmpl-{tag}-0-") or begin["prompt_sha256"] != digest(tokens)
            or any(event["request_id"] != begin["request_id"] for event in events)
            or gpu + external + computed != begin["prompt_tokens"] or computed <= 0
            or delta["vllm:prefix_cache_queries_total"] != len(tokens)
            or delta["vllm:time_to_first_token_seconds_count"] != 1 or not math.isfinite(ttft) or ttft <= 0
            or delta["vllm:num_preemptions_total"] or not row["generated_tokens"]
            or not all(token.startswith("token_id:") and token[9:].isdigit() for token in row["generated_tokens"])):
        raise ValueError(f"Invalid request boundaries, cache or output IDs: {tag}")
    (directory / f"{index:04}.prom").write_text(raw, encoding="utf-8")
    row.update(index=index, request_id=begin["request_id"], prompt_tokens=len(tokens), prompt_sha256=digest(tokens),
               ttft_seconds=ttft, gpu_tokens=gpu, cpu_tokens=int(external), computed_tokens=computed,
               cpu_write_blocks=counts["stored"], cpu_read_blocks=counts["loaded"],
               cpu_write_seconds=delta.get("vllm:kv_offload_store_time_total", 0),
               cpu_read_seconds=delta.get("vllm:kv_offload_load_time_total", 0),
               ssd_write_blocks=counts["ssd_store"], ssd_read_blocks=counts["ssd_load"],
               ssd_write_seconds=counts["ssd_store_seconds"], ssd_read_seconds=counts["ssd_load_seconds"])
    return counters, row


def inherited_lock(path):
    """找出 path 的同一 Linux flock 独占描述符，保证服务进程能够继承。"""
    identity = path.stat()
    for entry in Path("/proc/self/fd").iterdir():
        try:
            opened = entry.stat()
        except FileNotFoundError:
            # 枚举目录时使用的临时描述符可能已经关闭。
            continue
        if (opened.st_dev, opened.st_ino) != (identity.st_dev, identity.st_ino):
            continue
        info = (Path("/proc/self/fdinfo") / entry.name).read_text()
        if any(line.startswith("lock:") and "FLOCK" in line.split() and "WRITE" in line.split()
               for line in info.splitlines()):
            return int(entry.name)
    raise RuntimeError(f"Start each GPU service with flock -x {path}")


def service_lock(process, fd):
    """验证 process 继承 fd 对应的同一个打开文件，并保留 Linux 证据。"""
    local = Path(f"/proc/self/fd/{fd}").stat()
    remote = Path(f"/proc/{process.pid}/fd/{fd}").stat()
    info = Path(f"/proc/{process.pid}/fdinfo/{fd}").read_text()
    if ((local.st_dev, local.st_ino) != (remote.st_dev, remote.st_ino)
            or not any(line.startswith("lock:") and {"FLOCK", "WRITE"} <= set(line.split())
                       for line in info.splitlines())):
        raise RuntimeError("vLLM did not inherit the shared exclusive GPU flock")
    return dict(pid=process.pid, fd=fd, device=remote.st_dev, inode=remote.st_ino, fdinfo=info)


def collect(args, config, plan, directory, profile):
    """启动一个服务，执行 args 指定的路径或策略，并在返回前等待进程停止。"""
    lock_fd = inherited_lock(ROOT / ".agent/kv-experiment-gpu.lock")
    ssd.require_idle_gpu()
    directory.mkdir(parents=True)
    policy = "lru" if args.stage == "calibration" else args.arm
    argv = command(config, directory, args.port, policy, args.tier, profile)
    settings = config["env"] | {"VLLM_SERVER_DEV_MODE": "1" if args.stage == "calibration" else "0"}
    env = os.environ | settings
    save(directory / "server.json", dict(argv=argv, env=settings, tier=args.tier, arm=args.arm,
                                         stage=args.stage, plan_sha256=config["plan_sha256"]))
    log_path = directory / "server.log"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with log_path.open("x") as log:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock_fd,))
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120, trust_env=False) as client:
                joint.ready(client, process, args.port, log_path)
                save(directory / "gpu-lock.json", service_lock(process, lock_fd))
                with ((directory / "audit.jsonl").open() as audit,
                      (directory / "requests.jsonl").open("x") as records,
                      (directory / "resources.jsonl").open("x") as resources):
                    context, = (event["context"] for event in available_events(audit) if event["kind"] == "context")
                    device = dict(mount=filesystem_device(directory / "ssd"), capacity=ssd.SSD_BYTES,
                                  write="eager") if args.tier == "ssd" else None
                    current = retention_context(context, args.tier, device, env)
                    if profile is not None:
                        RetentionProfile(profile, current)
                    raw, previous = joint.metrics(client)
                    (directory / "initial.prom").write_text(raw, encoding="utf-8")
                    index = 0

                    def send(tokens, **fields):
                        """提交 tokens 并将 fields 和原始观测写入逐请求日志。"""
                        nonlocal index, previous
                        previous, row = capture(client, audit, previous, directory, index, config, tokens, args.tier)
                        row.update(fields)
                        records.write(json.dumps(row, allow_nan=False) + "\n")
                        records.flush()
                        index += 1
                        return row

                    send(plan["warmup"][args.stage], phase="warmup")
                    parent = psutil.Process(process.pid)
                    ssd.snapshot(parent, resources, "formal_start")
                    if args.stage == "calibration":
                        for case in plan["calibration"]:
                            measure_probe(client, send, case, args.arm)
                    else:
                        for item in plan["evaluation"]:
                            send(item["tokens"], case=item["case"], phase=item["phase"])
                    ssd.snapshot(parent, resources, "formal_end")
                    client.get("/health").raise_for_status()
                    if process.poll() is not None or (directory / "audit.jsonl.1").exists():
                        raise RuntimeError(f"Service exited early or audit rolled over: {directory}")
        finally:
            if process.poll() is None:
                if args.tier == "ssd":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if args.tier == "ssd":
        shutdown = rows(directory / "audit.jsonl")[-1]
        if shutdown["kind"] != "ssd_shutdown" or shutdown["success"] is not True:
            raise RuntimeError(f"SSD engine did not drain before shutdown: {directory}")
    save(directory / "complete.json", dict(context=current, requests=index, arm=args.arm, stage=args.stage,
                                          evidence={name: sha(directory / name) for name in ("server.json", "audit.jsonl",
                                                  "requests.jsonl", "resources.jsonl", "gpu-lock.json")}))


def measure_probe(client, send, case, arm):
    """让 case 的同一输入走 arm 指定的连续层级，按发出的实际请求核对命中数。"""
    blocks, tokens = case["blocks"], case["tokens"]
    prime = None
    if arm != "recompute":
        prime = send(tokens[:blocks * BLOCK + 1], phase="prime", case=case["case"])
        if prime["cpu_write_blocks"] != blocks or prime["gpu_tokens"] or prime["cpu_tokens"]:
            raise ValueError(f"Calibration prime did not write expected new CPU blocks: {prime}")
        if arm == "ssd":
            send(case["filler"], phase="filler", case=case["case"])
        reset = client.post("/reset_prefix_cache", params={"reset_external": "false"})
        reset.raise_for_status()
        if reset.json() != {"success": True}:
            raise RuntimeError(f"GPU-only cache reset failed: {reset.text}")
    row = send(tokens, phase=PHASE, case=case["case"], path=arm, blocks=blocks,
               write_seconds=prime["cpu_write_seconds"] if prime else None,
               prime_ssd_write_seconds=prime["ssd_write_seconds"] if prime else None)
    if (row["gpu_tokens"] or row["cpu_tokens"] != (0 if arm == "recompute" else BLOCK * blocks)
            or row["ssd_read_blocks"] != (blocks if arm == "ssd" else 0)):
        raise ValueError(f"Calibration probe used an unexpected path: {row}")
    print(f"{arm} {case['case']} {row['ttft_seconds'] * 1000:.2f} ms", flush=True)


def calibrate(output, tier, plan):
    """用 output 中同一 tier 的独立已完成校准组估计 profile；plan 提供隔离证明。"""
    arms = ("recompute", "cpu", "ssd") if tier == "ssd" else ("recompute", "cpu")
    context = None
    samples = []
    evidence = {}
    sources = {"calibration_plan": digest(plan["calibration"])}
    expected = {case["case"]: digest(case["tokens"]) for case in plan["calibration"]}
    for arm in arms:
        directory = output / f"calibration-{tier}" / arm
        marker = read(directory / "complete.json")
        current = marker["context"]
        if marker["stage"] != "calibration" or marker["arm"] != arm or (context is not None and context != current):
            raise ValueError("Calibration arms differ in phase or device/runtime")
        context = current
        for name, expected_sha in marker["evidence"].items():
            if sha(directory / name) != expected_sha:
                raise ValueError(f"Calibration evidence changed: {directory / name}")
            sources[str((directory / name).relative_to(output))] = expected_sha
        records = rows(directory / "requests.jsonl")
        if tier == "ssd":
            transfer = replay_ssd(directory, records)
            before = prometheus(directory / "initial.prom")
            for row in records:
                after = prometheus(directory / f"{row['index']:04}.prom")
                verify_metrics(row, before, after)
                before = after
            io = process_io(directory / "resources.jsonl")
            probe_blocks = sum(row["blocks"] for row in records if row["phase"] == PHASE)
            expected_reads = probe_blocks if arm == "ssd" else 0
            if (transfer["full_ssd_chain_blocks"] != expected_reads
                    or io["read_bytes"] < expected_reads * BLOCK_BYTES
                    or io["write_bytes"] < sum(row["ssd_write_blocks"] for row in records[1:]) * BLOCK_BYTES):
                raise ValueError(f"Calibration did not execute the requested native route: {arm}")
            evidence[arm] = dict(chain=transfer, process_io=io, complete_sha256=sha(directory / "complete.json"))
        observed = [row for row in records if row["phase"] == PHASE]
        if len(observed) != len(expected) or {row["case"]: row["prompt_sha256"] for row in observed} != expected:
            raise ValueError("Calibration probes do not match the independent frozen plan")
        for row in observed:
            row["ssd_write_seconds"] = row["prime_ssd_write_seconds"]
        samples.extend(observed)
    if tier == "ssd":
        verify_serial([item["chain"] for item in evidence.values()])
    profile = fit(samples, context, plan["calibration_prefixes"], sources)
    path = output / f"profile-{tier}.json"
    save(path, profile)
    RetentionProfile(path, context)
    if tier == "ssd":
        save(output / "review-ssd-calibration.json", dict(profile_sha256=sha(path), probes=len(expected), arms=evidence))


def main():
    """读取实验命令行；每次服务调用通过 WSL flock 持有共享锁。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "calibration", "fit", "eval"))
    parser.add_argument("output")
    parser.add_argument("--source", default=".agent/kv-joint/output")
    parser.add_argument("--tier", choices=("cpu", "ssd"), default="cpu")
    parser.add_argument("--arm")
    parser.add_argument("--port", type=int, default=18113)
    args = parser.parse_args()
    output = local_path(args.output, ROOT)
    if args.stage == "prepare":
        freeze(output, local_path(args.source, ROOT))
        return
    plan, config = read(output / "plan.json"), read(output / "experiment.json")
    verify_model(Path(config["model_dir"]))
    if config["plan_sha256"] != sha(output / "plan.json") or config["model_sha256"] != GBDT["model.joblib"]:
        raise ValueError("Frozen plan or model identity changed")
    if args.stage == "fit":
        calibrate(output, args.tier, plan)
        return
    arms = POLICIES if args.stage == "eval" else (("recompute", "cpu", "ssd") if args.tier == "ssd" else ("recompute", "cpu"))
    if args.arm not in arms or not 1 <= args.port <= 65535:
        parser.error(f"--arm must be one of {arms}, port 1–65535")
    profile = output / f"profile-{args.tier}.json" if args.stage == "eval" else None
    if profile is not None:
        known = set(read(profile)["calibration_prefix_sha256"])
        if known & set(plan["evaluation_prefixes"]):
            raise ValueError("Evaluation includes calibration first blocks")
    collect(args, config, plan, output / f"{args.stage}-{args.tier}" / args.arm, profile)


if __name__ == "__main__":
    main()
