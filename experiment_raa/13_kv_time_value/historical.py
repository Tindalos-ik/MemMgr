"""用 06 独立校准原始记录与 08 压力审计回放冻结 CPU 策略。"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from memmgr.short_term.connector import PolicyController
from memmgr.short_term.retention import RetentionProfile, fit, retention_context


BLOCK_BYTES = 1_835_008
PHASE = "retention-calibration"
POLICIES = ("admission", "learned", "time_value")


def read(path):
    """读取 path 的 JSON 对象。"""
    return json.loads(path.read_text())


def rows(path):
    """读取 path 的完整 JSONL，发现截断立即报错。"""
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def sha(path):
    """计算工件 path 的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tokens):
    """计算 token ID 列表 tokens 的 SHA-256。"""
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def save(path, value):
    """在新建 path 中保存 JSON value。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pilot(source, trace):
    """从 source 的 06 校准配对读取 TTFT/DMA，并与 trace 的 08 评价首块检查隔离。"""
    plan_file = source / "plans/calibration.json"
    prefixes = {digest(row["prompt_token_ids"][:64]) for row in read(plan_file)}
    trace_prefixes = {digest(row["prompt_token_ids"][:64]) for row in read(trace)}
    if prefixes & trace_prefixes:
        raise ValueError("Historical evaluation overlaps calibration")
    sources = {str(plan_file): sha(plan_file)}
    measurements = []
    context = None
    for mode, path in (("always", "cpu"), ("never", "recompute")):
        directory = source / f"calibration/lru-{mode}"
        requests, audit = (directory / name for name in ("requests.jsonl", "audit.jsonl"))
        sources.update({str(file): sha(file) for file in (requests, audit)})
        events = rows(audit)
        current, = (event["context"] for event in events if event["kind"] == "context")
        if context is not None and context != current:
            raise ValueError("Historical device or runtime changed across calibration arms")
        context = current
        lookups = {event["request_id"]: event for event in events if event["kind"] == "lookup" and event["external"]}
        primes = {}
        for row in rows(requests):
            metric = row["metrics"]
            case = row["case"]
            if row["phase"] == "prime" and metric.get("vllm:kv_offload_store_bytes_total", 0):
                primes.setdefault(case, metric)
            if row["phase"] != "revisit" or row["request_id"] not in lookups:
                continue
            lookup = lookups[row["request_id"]]
            loaded = lookup["external"] if path == "cpu" else 0
            blocks = lookup["external"] // 64
            if (lookup["gpu"] or lookup["admitted"] != loaded
                    or metric.get("vllm:kv_offload_load_bytes_total", 0) != loaded // 64 * BLOCK_BYTES
                    or metric["vllm:request_prefill_kv_computed_tokens_sum"] != row["prompt_tokens"] - loaded
                    or primes[case]["vllm:kv_offload_store_bytes_total"] != blocks * BLOCK_BYTES):
                raise ValueError(f"Invalid historical calibration geometry: {case}")
            measurements.append(dict(phase=PHASE, path=path, case=case, blocks=blocks,
                                     prompt_tokens=row["prompt_tokens"], prompt_sha256=row["prompt_sha256"],
                                     generated_tokens=row["generated_tokens"], ttft_seconds=row["server_ttft_seconds"],
                                     write_seconds=primes[case]["vllm:kv_offload_store_time_total"]))
    return fit(measurements, retention_context(context, "cpu", None), prefixes, sources)


def replay(trace, model, output, profile, policy):
    """用 trace 的键链驱动实际 CPU 管理器，对 policy 统计外层冷 GPU 命中反事实。"""
    init, = (event for event in trace if event["kind"] == "init")
    identity = init["identity"]
    manager = CPUOffloadingManager(init["capacity"], cache_policy="ManagedLRU",
                                   cache_policy_module_path="memmgr.short_term.connector")
    controller = PolicyController(manager, dict(model_revision=identity["model_revision"], dtype=identity["dtype"],
                                                 max_model_len=1088, policy_config=dict(policy=policy, model_dir=model,
                                                 audit_path=output / f"{policy}.jsonl")))
    if controller.predictor.sha256 != init["model_sha256"]:
        raise ValueError("Offline replay predictor differs from historical audit")
    if policy == "time_value":
        controller.retention = RetentionProfile(output / "pilot.json", profile["context"])
    observations = []
    for event in trace:
        if event["kind"] != "begin":
            continue
        request_id = event["request_id"]
        keys = [bytes.fromhex(value) for value in event["keys"]]
        context = ReqContext(request_id)
        admitted = controller.observe(request_id, keys)
        if [key.hex() for key in admitted] != event["admitted"]:
            raise ValueError("Frozen policy differs from historical admission")
        hits = 0
        for key in keys[:(event["prompt_tokens"] - 1) // 64]:
            if key not in manager._policy.blocks:
                break
            hits += 1
        selected = controller.select(keys, context)
        result = manager.prepare_store(selected, context)
        manager.complete_store(result.keys_to_store, context, True)
        controller.protected = set()
        del controller.active[request_id]
        observations.append(dict(written=len(result.keys_to_store), evicted=len(result.evicted_keys),
                                 cpu_hit_tokens=hits * 64, computed_tokens=event["prompt_tokens"] - hits * 64))
    manager.shutdown()
    controller.handler.close()
    sums = defaultdict(int)
    for row in observations:
        for key, value in row.items():
            sums[key] += value
    return dict(totals=dict(sums), requests=observations)


def main():
    """使用命令行中的真实历史文件，原样保留输出目录中的新审计和反事实统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path(".agent/kv-joint/output"))
    parser.add_argument("--trace", type=Path, default=Path(".agent/kv-capacity/pressure-14"))
    args = parser.parse_args()
    trace_file = args.trace / "epoch-0/cpu-learned/audit.jsonl"
    plan_file = args.trace / "plans/pressure.json"
    profile = pilot(args.source, plan_file)
    trace = rows(trace_file)
    if [row["prompt_sha256"] for row in trace if row["kind"] == "begin"][1:] != [
            row["prompt_sha256"] for row in read(plan_file)]:
        raise ValueError("Historical trace does not match its published plan")
    args.output.mkdir(parents=True)
    save(args.output / "pilot.json", profile)
    policies = {policy: replay(trace, args.source / "model", args.output, profile, policy) for policy in POLICIES}
    save(args.output / "review.json", dict(audit_sha256=sha(trace_file), plan_sha256=sha(plan_file),
                                          calibration_cpu_blocks=profile["context"]["cpu_blocks"], policies=policies))
    print({policy: result["totals"] for policy, result in policies.items()})


if __name__ == "__main__":
    main()
