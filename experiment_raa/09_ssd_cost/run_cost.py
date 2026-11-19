"""从实验 07 的真实 SSD 命中出发，串行回放独立的 SSD 成本实验。"""

import argparse
from collections import Counter
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
import psutil

from .plan import BLOCK, events, load, make_plan, token_hash

joint = importlib.import_module("experiment_raa.06_kv_joint.run_joint")
ssd_joint = importlib.import_module("experiment_raa.07_kv_joint_ssd.run_ssd_joint")
ROOT = Path(__file__).resolve().parents[2]
MODULE = "experiment_raa.09_ssd_cost.connector"
KV_BYTES = 1_835_008
MODES = ("restore", "recompute")


def command(args, directory, mode, baseline, profile):
    """保留 baseline 中的模型、容量和运行选项，只替换本组路径、端口及 connector。"""
    argv = list(baseline["argv"])
    argv[0] = sys.executable
    argv[argv.index("--port") + 1] = str(args.port)
    transfer = json.loads(argv[-1])
    transfer.update(kv_connector="SSDCostConnector", kv_connector_module_path=MODULE)
    extra = transfer["kv_connector_extra_config"]
    extra.update(spec_name="SSDCostSpec", spec_module_path=MODULE, ssd_cost_mode=mode)
    if profile is not None:
        extra["ssd_cost_profile"] = profile
    ssd = directory / "ssd"
    ssd.mkdir(mode=0o700)
    extra["policy_config"].update(ssd_root=str(ssd), audit_path=str(directory / "audit.jsonl"))
    extra["secondary_tiers"][0]["root_dir"] = str(ssd / "invariant-vllm-0.29.0")
    argv[-1] = json.dumps(transfer)
    return argv


def request_events(stream, request_id, process):
    """从 stream 等待 request_id 的三级写入和读取全部完成；检查 process 存活。"""
    deadline = time.monotonic() + 30
    found, jobs, writes = [], {}, Counter()
    engine_id, closed = None, False
    while time.monotonic() < deadline and process.poll() is None:
        for line in stream:
            e = json.loads(line)
            req = e.get("request_id", "")
            if not req.startswith(f"cmpl-{request_id}-0-"):
                raise RuntimeError(f"Unexpected audit event while draining {request_id}: {e}")
            if engine_id is None:
                engine_id = req
            if req != engine_id:
                raise RuntimeError("More than one engine request ID")
            found.append(e)
            kind = e["kind"]
            if kind == "ssd_request_closed":
                closed = True
            elif kind == "fs_submit":
                if e["job"] in jobs:
                    raise RuntimeError("Duplicate filesystem job")
                jobs[e["job"]] = (e["direction"], e["keys"])
            elif kind == "fs_complete":
                if not e["success"] or jobs.pop(e["job"]) != (e["direction"], e["keys"]):
                    raise RuntimeError("Invalid filesystem completion")
            elif kind in {"store", "stored"}:
                for key in e["keys"]:
                    writes[key] += 1 if kind == "store" else -1
                    if writes[key] < 0:
                        raise RuntimeError("Unmatched GPU store completion")
        if closed and not jobs and not any(writes.values()):
            return found
        time.sleep(0.01)
    raise RuntimeError(f"Timed out draining {request_id}: {closed}, {jobs.keys()}, {dict(writes)}")


def io_sample(engine, block_stats):
    """读取 engine 的 Linux 进程 I/O、创建时间和 block_stats 的共享设备计数。"""
    return dict(pid=engine.pid, created=engine.create_time(), process=engine.io_counters()._asdict(),
                device=[int(value) for value in block_stats.read_text().split()], time_ns=time.monotonic_ns())


def verify_probe(row, audit, before, after, shape):
    """用 audit、Linux I/O 边界和预期 shape 核对 row 的命中、重算与真实迁移链。"""
    decision, = (e for e in audit if e["kind"] == "ssd_cost_decision")
    gpu, cpu, ssd = shape
    if (decision["gpu"], decision["layers"]) != (gpu * BLOCK, ["cpu"] * cpu + ["ssd"] * ssd):
        raise ValueError(f"Wrong probe geometry: expected {shape}, observed {decision}")
    target = decision["ssd_keys"] if decision["admit"] else []
    cpu_keys = decision["cpu_keys"]
    promotions = [e["key"] for e in audit if e["kind"] == "promotion" and e["accepted"]]
    loads = [e for e in audit if e["kind"] == "fs_complete" and e["direction"] == "load"]
    submissions = {e["job"]: e for e in audit if e["kind"] == "fs_submit" and e["direction"] == "load"}
    restored = [key for e in loads for key in e["keys"]]
    gpu_loaded = [key for e in audit if e["kind"] == "loaded" for key in e["keys"]]
    # 完成回调按调度器内部集合迭代，核对每块恰好出现一次。
    if (Counter(promotions) != Counter(target) or Counter(restored) != Counter(target)
            or Counter(gpu_loaded) != Counter(cpu_keys + target)
            or set(submissions) != {e["job"] for e in loads}
            or any(not e["success"] or e["keys"] != submissions[e["job"]]["keys"]
                   or e["time_ns"] <= submissions[e["job"]]["time_ns"] or e["transfer_seconds"] <= 0 for e in loads)):
        raise ValueError(f"Wrong physical read/transfer chain: {decision=}, {promotions=}, {restored=}, {gpu_loaded=}")
    metrics = row["metrics"]
    reused = len(gpu_loaded) * BLOCK
    ssd_seconds = sum(e["transfer_seconds"] for e in loads)
    dma_seconds = metrics.get("vllm:kv_offload_load_time_total", 0)
    if (metrics.get("vllm:external_prefix_cache_hits_total", 0) != reused
            or metrics["vllm:prefix_cache_hits_total"] != gpu * BLOCK
            or metrics["vllm:request_prefill_kv_computed_tokens_sum"] != row["prompt_tokens"] - gpu * BLOCK - reused
            or metrics.get("vllm:kv_offload_load_bytes_total", 0) != len(gpu_loaded) * KV_BYTES
            or metrics.get("vllm:kv_offload_tiering_read_bytes_total", 0) != len(restored) * KV_BYTES
            or not math.isclose(metrics.get("vllm:kv_offload_tiering_read_time_total", 0), ssd_seconds, abs_tol=1e-9)
            or ((dma_seconds > 0) != bool(gpu_loaded))
            or metrics["vllm:time_to_first_token_seconds_count"] != 1
            or metrics["vllm:time_to_first_token_seconds_sum"] <= 0
            or metrics["vllm:num_preemptions_total"]):
        raise ValueError(f"Wrong native counters: {row['client_request_id']}")
    if (before["pid"], before["created"]) != (after["pid"], after["created"]):
        raise ValueError("Engine changed during probe")
    read_bytes = after["process"]["read_bytes"] - before["process"]["read_bytes"]
    write_bytes = after["process"]["write_bytes"] - before["process"]["write_bytes"]
    if read_bytes < len(restored) * KV_BYTES or write_bytes < 0:
        raise ValueError(f"Bad physical process I/O delta: {read_bytes}, {write_bytes}")
    return dict(decision=decision, ssd_logical_read_bytes=len(restored) * KV_BYTES,
                engine_read_bytes=read_bytes, engine_write_bytes=write_bytes,
                guest_read_bytes=(after["device"][2] - before["device"][2]) * 512,
                guest_write_bytes=(after["device"][6] - before["device"][6]) * 512,
                ssd_io_seconds=ssd_seconds,
                ssd_submit_to_observed_seconds=sum((e["time_ns"] - submissions[e["job"]]["time_ns"]) / 1e9 for e in loads),
                cpu_gpu_seconds=dma_seconds,
                computed_tokens=int(metrics["vllm:request_prefill_kv_computed_tokens_sum"]))


def probe_baseline(client, index, cpu_bytes, ssd_bytes):
    """等 client 的 Prometheus 快照覆盖之前 index 条请求和已证实的两级读取字节。"""
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        raw, counters = joint.metrics(client)
        observed = (counters["vllm:request_success_total"],
                    counters.get("vllm:kv_offload_load_bytes_total", 0),
                    counters.get("vllm:kv_offload_tiering_read_bytes_total", 0))
        expected = (index, cpu_bytes, ssd_bytes)
        if observed == expected:
            return raw, counters
        if any(actual > wanted for actual, wanted in zip(observed, expected, strict=True)):
            raise ValueError(f"Baseline passed its request boundary: {observed}, expected {expected}")
        time.sleep(0.05)
    raise TimeoutError(f"Undrained Prometheus baseline for request {index}: {observed}, expected {expected}")


def requests(plan):
    """按 plan 的固定顺序迭代请求，标出各案例的最后一个探测请求。"""
    yield "warmup", plan["warmup"], False, None
    for case in plan["cases"]:
        for index, tokens in enumerate(case["inputs"]):
            yield case["id"], tokens, index == len(case["inputs"]) - 1, case["shape"]


def capture(client, process, directory, plan, seed, block_stats):
    """回放 plan，按 seed 采样 client 的真实流、进程 I/O 和原生指标，保存到 directory。"""
    with ((directory / "audit.jsonl").open(encoding="utf-8") as audit,
          (directory / "requests.jsonl").open("x", encoding="utf-8") as stream):
        begin = [json.loads(line) for line in audit]
        engine_event, = (e for e in begin if e["kind"] == "ssd_cost_process")
        init, = (e for e in begin if e["kind"] == "fs_init")
        if not init["direct_io"] or init["existing_bytes"] or init["block_bytes"] != KV_BYTES:
            raise ValueError(f"Unexpected SSD file layout: {init}")
        engine = psutil.Process(engine_event["pid"])
        if engine not in psutil.Process(process.pid).children(recursive=True):
            raise ValueError("Audit PID does not belong to the isolated service")
        cpu_bytes = ssd_bytes = 0
        for index, (case, tokens, probe, shape) in enumerate(requests(plan)):
            request_id = f"ssd09-{'probe' if probe else 'setup'}-{index:04}"
            if probe:
                raw, previous = probe_baseline(client, index, cpu_bytes, ssd_bytes)
                (directory / f"{index:04}-before.prom").write_text(raw, encoding="utf-8")
                before = io_sample(engine, block_stats)
            row = joint.request(client, tokens, joint.OUTPUT_TOKENS, seed, request_id)
            observed = request_events(audit, request_id, process)
            cpu_bytes += sum(len(e["keys"]) * KV_BYTES for e in observed if e["kind"] == "loaded")
            ssd_bytes += sum(len(e["keys"]) * KV_BYTES for e in observed
                             if e["kind"] == "fs_complete" and e["direction"] == "load")
            request_begin, = (e for e in observed if e["kind"] == "begin")
            digest = token_hash(tokens)
            if (request_begin["prompt_sha256"], request_begin["prompt_tokens"]) != (digest, len(tokens)):
                raise ValueError("Engine received different input tokens")
            if any(not key.startswith("token_id:") or not key[9:].isdigit() for key in row["generated_tokens"]):
                raise ValueError("Server did not return output token IDs")
            row.update(index=index, case=case, probe=probe, prompt_tokens=len(tokens), prompt_sha256=digest,
                       client_request_id=request_id, engine_request_id=request_begin["request_id"])
            if probe:
                after = io_sample(engine, block_stats)
                raw, _, counters = joint.settled(client, previous, len(tokens))
                (directory / f"{index:04}-after.prom").write_text(raw, encoding="utf-8")
                row.update(metrics=counters, io_before=before, io_after=after)
                row.update(verify_probe(row, observed, before, after, shape))
                print(f"{directory.name} {case}: {row['decision']['geometry']} {row['decision']['reason']}", flush=True)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()


def run(args, baseline, plan, mode, block_stats, profile):
    """按 baseline 在 args 的独立目录启动 mode 服务，运行 plan 并在退出时释放 GPU。"""
    ssd_joint.require_idle_gpu()
    directory = args.output / mode
    directory.mkdir()
    argv = command(args, directory, mode, baseline, profile)
    env = baseline["env"] | {"PYTHONHASHSEED": "0"}
    joint.save(directory / "server.json", dict(argv=argv, env=env))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    log = directory / "server.log"
    print(f"启动 {directory}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=os.environ | env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120, trust_env=False) as client:
                joint.ready(client, process, args.port, log)
                capture(client, process, directory, plan, args.seed, block_stats)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def verify_pairs(output, modes, plan):
    """核对 output 的 modes 使用同一 plan、输出 token 一致且初始层级候选相同。"""
    rows = {mode: list(events(output / mode / "requests.jsonl")) for mode in modes}
    paired = []
    for index, ((case, tokens, probe, _), *items) in enumerate(zip(requests(plan), *rows.values(), strict=True)):
        reference = items[0]
        if (reference["index"], reference["case"], reference["probe"], reference["prompt_sha256"]) != (
            index, case, probe, token_hash(tokens)
        ):
            raise ValueError(f"Request {index} differs from its plan")
        for row in items[1:]:
            for field in ("index", "case", "probe", "prompt_sha256", "generated_tokens"):
                if row[field] != reference[field]:
                    raise ValueError(f"Paired request {row['index']} differs in {field}")
            if row["probe"]:
                for field in ("gpu", "cpu_keys", "ssd_keys", "geometry"):
                    if row["decision"][field] != reference["decision"][field]:
                        raise ValueError(f"Paired residency differs in {field}: {row['case']}")
        if reference["probe"]:
            paired.append(dict(zip(modes, items, strict=True)))
    if len(paired) != len(plan["cases"]):
        raise ValueError("Missing probe pairs")
    return paired


def main():
    """读取正式基线和命令行，在调用者持有的 GPU 锁内运行一个独立阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "calibration", "heldout"))
    parser.add_argument("--source", default=".agent/kv-joint-ssd/output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--port", type=int, default=18089)
    args = parser.parse_args()
    args.source = joint.local_path(args.source, ROOT)
    args.output = joint.local_path(args.output, ROOT)
    frozen = load(args.source / "environment.json")
    packages = {name: importlib.metadata.version(name) for name in frozen["packages"]}
    if packages != frozen["packages"] or joint.digest(Path(frozen["arguments"]["model"]) / "config.json") != frozen["model_config_sha256"]:
        raise ValueError("Software or model differs from experiment 07")
    args.seed = frozen["arguments"]["seed"]
    baseline = load(args.source / "epoch-0/ssd-lru/server.json")
    profile = load(joint.local_path(args.profile, ROOT)) if args.profile else None
    if (profile is not None) != (args.stage == "heldout"):
        parser.error("--profile is required only for heldout")
    plan = make_plan(args, frozen)
    args.output.mkdir(parents=True)
    mount = subprocess.check_output(["findmnt", "-n", "-o", "MAJ:MIN,SOURCE,FSTYPE", "-T", args.output], text=True).split()
    block_stats = Path(f"/sys/dev/block/{mount[0]}/stat")
    if not block_stats.is_file():
        raise ValueError(f"No Linux block device for the experimental output: {mount}")
    sources = {path.name: joint.digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    if profile is not None:
        if (profile["schema"], profile["mount"], profile["connector_sha256"]) != (1, mount, sources["connector.py"]):
            raise ValueError("SSD mount or experimental connector differs from calibration")
        prefix_hashes = {token_hash(case["inputs"][-1][:BLOCK]) for case in plan["cases"]}
        if prefix_hashes.intersection(profile["calibration_prefix_sha256"]):
            raise ValueError("Held-out input has a calibration prefix")
    modes = MODES if profile is None else ("recompute", "gate", "restore")
    joint.save(args.output / "environment.json", dict(stage=args.stage, modes=modes, mount=mount,
               source_environment_sha256=joint.digest(args.source / "environment.json"), packages=packages,
               source_sha256=sources,
               source_review_sha256=joint.digest(args.source / "review.json"), block_bytes=KV_BYTES))
    joint.save(args.output / "plan.json", plan)
    if profile is not None:
        joint.save(args.output / "profile.json", profile)
    for mode in modes:
        run(args, baseline, plan, mode, block_stats, profile if mode == "gate" else None)
    verify_pairs(args.output, modes, plan)
    ssd_joint.mark_complete(args.output, modes=modes, probes=len(plan["cases"]))


if __name__ == "__main__":
    main()
