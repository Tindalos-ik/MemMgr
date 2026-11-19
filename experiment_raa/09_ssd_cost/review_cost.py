"""离线复核 SSD 配对原始证据，冻结成本门槛并检查未见首块回放。"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from prometheus_client.parser import text_string_to_metric_families

from .plan import BLOCK, events, load, token_hash
from .run_cost import KV_BYTES, joint, verify_pairs, verify_probe

MARGIN_MS = 0.5
ACCEPT_MIN = 3
METRICS = {
    "client_ttft_ms": "ttft_seconds",
    "ssd_io_ms": "ssd_io_seconds",
    "ssd_submit_to_observed_ms": "ssd_submit_to_observed_seconds",
    "cpu_gpu_ms": "cpu_gpu_seconds",
    "logical_ssd_read_bytes": "ssd_logical_read_bytes",
    "engine_read_bytes": "engine_read_bytes",
    "engine_write_bytes": "engine_write_bytes",
    "guest_read_bytes": "guest_read_bytes",
    "guest_write_bytes": "guest_write_bytes",
    "computed_tokens": "computed_tokens",
}


def prom(path):
    """从 path 的原始 Prometheus 文本按名称累加计数、样本数与和。"""
    counters = Counter()
    for family in text_string_to_metric_families(path.read_text(encoding="utf-8")):
        for sample in family.samples:
            if sample.name.endswith(("_total", "_sum", "_count")):
                counters[sample.name] += sample.value
    return counters


def replay(root):
    """从 root 的完成标记、输入计划、Prom、审计与 I/O 原始快照重建全部配对。"""
    plan = load(root / "plan.json")
    environment = load(root / "environment.json")
    complete = load(root / "complete.json")
    modes = environment["modes"]
    if modes != complete["modes"] or complete["probes"] != len(plan["cases"]):
        raise ValueError("Run completion disagrees with the plan and modes")
    pairs = verify_pairs(root, modes, plan)
    cases = {case["id"]: case for case in plan["cases"]}
    profile = load(root / "profile.json") if "gate" in modes else None
    context = None
    for mode in modes:
        directory = root / mode
        by_request = defaultdict(list)
        persisted, evicted, at_decision = set(), set(), {}
        for event in events(directory / "audit.jsonl"):
            kind = event["kind"]
            if kind == "context":
                if context is not None and context != event["context"]:
                    raise ValueError("vLLM context differs across arms")
                context = event["context"]
            if kind in {"promotion", "store"}:
                evicted.update(persisted.intersection(event["evicted"]))
            if kind == "fs_complete" and event["success"] and event["direction"] == "store":
                persisted.update(event["keys"])
            if kind == "ssd_cost_decision":
                at_decision[event["request_id"]] = set(event["ssd_keys"]) <= evicted
            if req := event.get("request_id"):
                by_request[req].append(event)
        for pair in pairs:
            row = pair[mode]
            index, req = row["index"], row["engine_request_id"]
            case = cases[row["case"]]
            if token_hash(case["inputs"][-1]) != row["prompt_sha256"] or not at_decision[req]:
                raise ValueError(f"Missing input or prior SSD persistence/CPU eviction proof: {req}")
            before, after = (prom(directory / f"{index:04}-{bound}.prom") for bound in ("before", "after"))
            counters = {key: value - before[key] for key, value in after.items()}
            if counters != row["metrics"]:
                raise ValueError(f"Raw Prometheus snapshot differs: {req}")
            observed = verify_probe(row, by_request[req], row["io_before"], row["io_after"], case["shape"])
            if any(row[key] != value for key, value in observed.items()):
                raise ValueError(f"Raw I/O or connector audit differs: {req}")
            decision = observed["decision"]
            if mode == "gate":
                rule = profile["rows"].get(decision["geometry"])
                expected = (rule["admit"], "calibrated") if rule else (True, "outside-profile")
            else:
                expected = (mode == "restore", mode)
            if (decision["admit"], decision["reason"]) != expected:
                raise ValueError(f"Wrong runtime decision: {req}")
    return plan, environment, pairs, context


def measures(row):
    """将 row 的真实指标转为毫秒、字节和 token。"""
    values = {key: row[field] * (1000 if key.endswith("_ms") else 1) for key, field in METRICS.items()}
    values["server_ttft_ms"] = row["metrics"]["vllm:time_to_first_token_seconds_sum"] * 1000
    values["cpu_gpu_bytes"] = row["metrics"].get("vllm:kv_offload_load_bytes_total", 0)
    return values


def summary(pairs):
    """按实际几何聚合 pairs，保留各臂中位数、配对差和实际 I/O 合计。"""
    by_geometry = defaultdict(list)
    for pair in pairs:
        by_geometry[pair["restore"]["decision"]["geometry"]].append(pair)
    result = {}
    for geometry, cases in sorted(by_geometry.items(), key=lambda item: tuple(map(int, item[0].split(":")))):
        modes = list(cases[0])
        arms = {mode: [measures(pair[mode]) for pair in cases] for mode in modes}
        medians = {mode: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
                   for mode, rows in arms.items()}
        diffs = {}
        for mode, reference in (("restore", "recompute"), ("gate", "restore"), ("gate", "recompute")):
            if mode not in arms:
                continue
            metrics = {}
            for key in ("server_ttft_ms", "client_ttft_ms"):
                values = [row[key] - base[key] for row, base in zip(arms[mode], arms[reference], strict=True)]
                metrics[key] = dict(median=statistics.median(values), min=min(values), max=max(values),
                                    faster=sum(value < 0 for value in values))
            diffs[f"{mode}_minus_{reference}"] = metrics
        result[geometry] = dict(n=len(cases), case_ids=[pair["restore"]["case"] for pair in cases],
                                median=medians, paired=diffs,
                                decisions={mode: dict(Counter(pair[mode]["decision"]["reason"]
                                                              for pair in cases)) for mode in modes},
                                total={mode: {key: sum(row[key] for row in rows)
                                              for key in ("engine_read_bytes", "engine_write_bytes",
                                                          "guest_read_bytes", "guest_write_bytes")}
                                       for mode, rows in arms.items()})
    return result


def freeze(root, plan, environment, context, report):
    """用 root 的校准配对、plan、environment、context 和 report 冻结精确几何门槛。"""
    if plan["stage"] != "calibration" or set(environment["modes"]) != {"restore", "recompute"}:
        raise ValueError("Only calibration may produce a cost profile")
    rows = {}
    for geometry, row in report.items():
        delta = row["paired"]["restore_minus_recompute"]["server_ttft_ms"]
        if row["n"] < ACCEPT_MIN:
            raise ValueError(f"Insufficient calibration pairs for {geometry}")
        rows[geometry] = dict(n=row["n"], restore_minus_recompute_ms=delta["median"],
                              admit=delta["median"] < -MARGIN_MS)
    return dict(schema=1, margin_ms=MARGIN_MS, context=context, mount=environment["mount"],
                connector_sha256=environment["source_sha256"]["connector.py"],
                calibration_plan_sha256=joint.digest(root / "plan.json"),
                calibration_prefix_sha256=sorted(token_hash(case["inputs"][-1][:BLOCK]) for case in plan["cases"]),
                rows=rows)


def compare_gate(pairs):
    """以 pairs 的真实三臂对照量化门槛对回迁和重算基线的 TTFT 变化。"""
    result = {}
    for scope, group in (("all", pairs), ("calibrated", [p for p in pairs if p["gate"]["decision"]["reason"] == "calibrated"])):
        rows = {}
        for reference in ("restore", "recompute"):
            deltas = defaultdict(list)
            for pair in group:
                gate, base = measures(pair["gate"]), measures(pair[reference])
                for key in ("server_ttft_ms", "client_ttft_ms"):
                    deltas[key].append(gate[key] - base[key])
            rows[f"gate_minus_{reference}"] = {key: dict(median=statistics.median(values),
                                                         faster=sum(value < 0 for value in values))
                                                 for key, values in deltas.items()}
        correct = sum(
            pair["gate"]["decision"]["admit"] == (
                measures(pair["restore"])["server_ttft_ms"] - measures(pair["recompute"])["server_ttft_ms"] < -MARGIN_MS
            )
            for pair in group
        )
        result[scope] = dict(n=len(group), admitted=sum(p["gate"]["decision"]["admit"] for p in group),
                             agrees_with_heldout_pair=correct, paired=rows)
    return result


def main():
    """读取已结束的实验目录，输出统计并在校准阶段冻结决策表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    plan, environment, pairs, context = replay(args.root)
    rows = summary(pairs)
    report = dict(stage=plan["stage"], n=len(pairs), block_bytes=KV_BYTES, context=context, rows=rows)
    if plan["stage"] == "calibration":
        joint.save(args.root / "profile.json", freeze(args.root, plan, environment, context, rows))
    if plan["stage"] == "heldout":
        profile = load(args.root / "profile.json")
        prefixes = {token_hash(case["inputs"][-1][:BLOCK]) for case in plan["cases"]}
        if prefixes.intersection(profile["calibration_prefix_sha256"]):
            raise ValueError("Held-out probe has a calibration prefix")
        report["gate"] = compare_gate(pairs)
    joint.save(args.root / "review.json", report)


if __name__ == "__main__":
    main()
