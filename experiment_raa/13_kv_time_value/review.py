"""复核实验 13 的输入、逐 token 输出、实际迁移与配对 TTFT。"""

import argparse
from collections import Counter, defaultdict
import itertools
import math
from pathlib import Path
from statistics import median

from prometheus_client.parser import text_string_to_metric_families

from memmgr.short_term.retention import RetentionProfile

from .historical import BLOCK_BYTES, POLICIES, digest, read, rows, save, sha


TIMES = ("ttft_seconds", "client_ttft_seconds")
COUNTERS = ("gpu_tokens", "cpu_tokens", "computed_tokens", "cpu_write_blocks", "cpu_read_blocks",
            "ssd_write_blocks", "ssd_read_blocks", "cpu_write_seconds", "cpu_read_seconds",
            "ssd_write_seconds", "ssd_read_seconds")
PROM_FIELDS = {
    "gpu_tokens": ("prefix_cache_hits_total", 1),
    "cpu_tokens": ("external_prefix_cache_hits_total", 1),
    "computed_tokens": ("request_prefill_kv_computed_tokens_sum", 1),
    "ttft_seconds": ("time_to_first_token_seconds_sum", 1),
    "cpu_write_blocks": ("kv_offload_store_bytes_total", BLOCK_BYTES),
    "cpu_read_blocks": ("kv_offload_load_bytes_total", BLOCK_BYTES),
    "ssd_write_blocks": ("kv_offload_tiering_write_bytes_total", BLOCK_BYTES),
    "ssd_read_blocks": ("kv_offload_tiering_read_bytes_total", BLOCK_BYTES),
    "cpu_write_seconds": ("kv_offload_store_time_total", 1),
    "cpu_read_seconds": ("kv_offload_load_time_total", 1),
    "ssd_write_seconds": ("kv_offload_tiering_write_time_total", 1),
    "ssd_read_seconds": ("kv_offload_tiering_read_time_total", 1),
}


def prometheus(path):
    """解析 path 的原始 Prometheus 文本并对相同名称的引擎计数求和。"""
    counters = Counter()
    for family in text_string_to_metric_families(path.read_text()):
        for sample in family.samples:
            if sample.name.endswith(("_total", "_sum", "_count")):
                counters[sample.name] += sample.value
    return counters


def verify_metrics(row, before, after):
    """用 before、after 的原始指标差值验证 row 的路径和服务端时延。"""
    delta = {name: value - before[name] for name, value in after.items()}
    if (delta["vllm:request_success_total"] != 1 or delta["vllm:num_preemptions_total"] != 0
            or delta["vllm:time_to_first_token_seconds_count"] != 1
            or delta["vllm:prefix_cache_queries_total"] != row["prompt_tokens"]):
        raise ValueError(f"Raw metrics disagree with request boundary {row['index']}")
    for field, (metric, unit) in PROM_FIELDS.items():
        observed = delta.get(f"vllm:{metric}", 0)
        if not math.isclose(row[field] * unit, observed, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Raw {metric} disagrees with request {row['index']}: {observed}")


def distribution(seconds):
    """以毫秒报告 seconds 的中位数与最近秩 P95。"""
    ordered = sorted(seconds)
    return dict(median_ms=1000 * median(ordered), p95_ms=1000 * ordered[math.ceil(0.95 * len(ordered)) - 1])


def process_io(path):
    """按 path 前后进程快照求累计 Linux I/O 字节；要求没有进程替换。"""
    start, end = rows(path)
    if (start["phase"], end["phase"]) != ("formal_start", "formal_end"):
        raise ValueError(f"Bad I/O boundary: {path}")
    before, after = ({(entry["pid"], entry["created_at"]): entry["io"] for entry in event["processes"]}
                     for event in (start, end))
    if before.keys() != after.keys():
        raise ValueError("Service process tree changed during measurement")
    return {name: sum(after[key][name] - before[key][name] for key in before)
            for name in ("read_bytes", "write_bytes")}


def review_retention(events, totals, tier):
    """从 events 验证真实 store 是否使用计价结果，并在 totals 累计变化和成本。"""
    decision = None
    for event in events:
        if event["kind"] == "retain_value":
            if decision is not None:
                raise ValueError("Unmatched retention decision")
            decision = event
        elif event["kind"] == "store":
            if decision is None:
                if event["keys"]:
                    raise ValueError("Time-value write bypassed retention decision")
                continue
            incoming, victims = (decision[field] for field in ("incoming", "victims"))
            written, planned = (decision[field] for field in ("written", "planned_evictions"))
            if event["keys"] != [key[0] for key in incoming[:written]]:
                raise ValueError("Physical write disagrees with retention decision")
            if event["evicted"] != [key[0] for key in victims[:planned]]:
                totals["eviction_plan_differed"] += 1
                if tier == "cpu":
                    raise ValueError("CPU actual victims differ from priced victims")
            totals["decisions"] += 1
            totals["blocked_or_shortened"] += written < len(incoming)
            totals["planned_evictions"] += planned
            totals["incoming_ssd_alternative"] += sum(row[2] == "ssd" for row in incoming)
            totals["incoming_recompute_alternative"] += sum(row[2] == "recompute" for row in incoming)
            totals["evicted_ssd_alternative"] += sum(row[2] == "ssd" for row in victims[:planned])
            totals["evicted_recompute_alternative"] += sum(row[2] == "recompute" for row in victims[:planned])
            for key in ("benefit_seconds", "write_seconds", "displaced_seconds", "net_seconds"):
                totals[key] += decision[key]
            decision = None
    if decision is not None:
        raise ValueError("No physical store for time-value decision")


def group(path, config, expected, profile, tier, policy):
    """核对 path 的策略审计与 expected，返回配对用原始记录和真实计量统计。"""
    marker = read(path / "complete.json")
    if marker["stage"] != "eval" or marker["arm"] != policy:
        raise ValueError(f"Not an evaluation group: {path}")
    if profile is not None:
        RetentionProfile(profile, marker["context"])
    elif marker["context"]["tier"] != tier:
        raise ValueError("Unexpected evaluation tier")
    for name, value in marker["evidence"].items():
        if sha(path / name) != value:
            raise ValueError(f"Evaluation evidence changed: {path / name}")
    audit = rows(path / "audit.jsonl")
    startup, = [row for row in audit if row["kind"] == "init"]
    if startup["policy"] != policy or startup["model_sha256"] != config["model_sha256"]:
        raise ValueError("Evaluation policy or frozen predictor differs")
    if policy == "time_value":
        approved, = (row["sha256"] for row in audit if row["kind"] == "retention_profile")
        if approved != sha(profile):
            raise ValueError("Service used a different retention profile")
    by_request = defaultdict(list)
    for event in audit:
        if "request_id" in event:
            by_request[event["request_id"]].append(event)
    results = rows(path / "requests.jsonl")
    if len(results) != marker["requests"] or len(by_request) != len(results):
        raise ValueError("Incomplete request audits")
    retention, evictions = Counter(), 0
    before = prometheus(path / "initial.prom")
    for index, (row, plan) in enumerate(zip(results, expected, strict=True)):
        events = by_request[row["request_id"]]
        begin, = [event for event in events if event["kind"] == "begin"]
        if (row["index"] != index or row["prompt_sha256"] != digest(plan["tokens"])
                or begin["prompt_sha256"] != row["prompt_sha256"]
                or row["phase"] != plan["phase"] or row.get("case") != plan.get("case")):
            raise ValueError(f"Group {policy} differs from plan at request {index}")
        after = prometheus(path / f"{index:04}.prom")
        verify_metrics(row, before, after)
        before = after
        row["admitted"] = begin["admitted"]
        if index:
            evictions += sum(len(event["evicted"]) for event in events if event["kind"] in ("store", "promotion"))
            if policy == "time_value":
                review_retention(events, retention, tier)
    warmup_ids = results[0]["generated_tokens"]
    results = results[1:]
    metrics = {key: sum(row[key] for row in results) for key in COUNTERS}
    for name in ("cpu_write", "cpu_read", "ssd_write", "ssd_read"):
        metrics[f"{name}_bytes"] = metrics[f"{name}_blocks"] * BLOCK_BYTES
    stats = dict(requests=len(results), totals=metrics, evicted_blocks=evictions, process_io=process_io(path / "resources.jsonl"),
                 warmup_ids=warmup_ids,
                 retention=dict(retention), timings={key: distribution(row[key] for row in results) for key in TIMES})
    return results, stats, marker["context"]


def review(output, tier):
    """复核 output 的 tier 三组，按相同请求位置计算处理策略减对照策略。"""
    plan, config = (read(output / name) for name in ("plan.json", "experiment.json"))
    if sha(output / "plan.json") != config["plan_sha256"]:
        raise ValueError("Frozen plan digest differs")
    expected = [dict(tokens=plan["warmup"]["eval"], phase="warmup"), *plan["evaluation"]]
    profile = output / f"profile-{tier}.json"
    groups, stats, contexts = {}, {}, []
    for policy in POLICIES:
        groups[policy], stats[policy], context = group(output / f"eval-{tier}" / policy, config, expected, profile, tier, policy)
        contexts.append(context)
    if any(context != contexts[0] for context in contexts[1:]):
        raise ValueError("Incomparable device or runtime across policies")
    if any(stats[policy]["warmup_ids"] != stats[POLICIES[0]]["warmup_ids"] for policy in POLICIES[1:]):
        raise ValueError("Warmup token IDs differ across the policies")
    comparisons = {}
    for control, treatment in itertools.combinations(POLICIES, 2):
        pairs = list(zip(groups[control], groups[treatment], strict=True))
        for before, after in pairs:
            for key in ("prompt_sha256", "generated_tokens", "admitted"):
                if before[key] != after[key]:
                    raise ValueError(f"Paired {key} differs for {treatment}/{control}: {before['index']}")
        windows = ("all", "revisits")
        summary = {}
        for window in windows:
            common = pairs if window == "all" else [pair for pair in pairs if pair[0]["phase"].startswith("revisit-")]
            summary[window] = dict(requests=len(common), **{
                key: distribution(after[key] - before[key] for before, after in common) for key in TIMES},
                different_paths=sum(any(after[key] != before[key] for key in COUNTERS if not key.endswith("_seconds"))
                                    for before, after in common),
                totals={key: sum(after[key] - before[key] for before, after in common) for key in COUNTERS})
        comparisons[f"{treatment}-{control}"] = summary
    return dict(tier=tier, plan_sha256=config["plan_sha256"], model_sha256=config["model_sha256"],
                output_ids_sha256=digest([row["generated_tokens"] for row in groups[POLICIES[0]]]),
                calibration_sha256=sha(profile), anchors=read(profile)["anchors"], strategies=stats, paired=comparisons)


def main():
    """接收实验输出目录和层级，复核后独立写出机器可读结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tier", choices=("cpu", "ssd"), default="cpu")
    args = parser.parse_args()
    result = review(args.output, args.tier)
    save(args.output / f"review-{args.tier}.json", result)
    print({key: dict(requests=value["all"]["requests"], ttft=value["all"]["ttft_seconds"],
                     paths=value["all"]["different_paths"]) for key, value in result["paired"].items()})


if __name__ == "__main__":
    main()
