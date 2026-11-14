"""复核回迁准入日志、原始指标、校准隔离和配对性能，拒绝不完整实验。"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from prometheus_client.parser import text_string_to_metric_families

from cost_policy import CostPolicy, geometry
from run_cost import digest, save


def read_rows(path):
    """从 path 读取完整且非空的 JSONL 文件。"""
    raw = path.read_text(encoding="utf-8")
    if not raw or not raw.endswith("\n"):
        raise ValueError(f"Incomplete JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def counters(path):
    """读取 path 的原始 Prometheus 快照，按名称累加 counter 和 histogram sum/count。"""
    result = {}
    for family in text_string_to_metric_families(path.read_text(encoding="utf-8")):
        for sample in family.samples:
            if sample.name.endswith(("_total", "_sum", "_count")):
                result[sample.name] = result.get(sample.name, 0) + sample.value
    return result


def load_stage(root, stage, modes):
    """复核 root 下 stage 的 modes，返回真实请求行；核验逐请求准入、字节量和计算 token。"""
    environment = json.loads((root / "environment.json").read_text())
    policy = CostPolicy(root / "calibration.json", environment["arguments"]["margin_ms"]) if "gate" in modes else None
    rows = []
    for epoch in (0, 1):
        plan = json.loads((root / f"{stage}-trace-{epoch}.json").read_text())
        for mode in modes:
            directory = root / f"{stage}-e{epoch}-{mode}"
            server = json.loads((directory / "server.json").read_text())
            if server["context"] != environment["context"]:
                raise ValueError("Runtime context changed")
            actual = read_rows(directory / "requests.jsonl")
            if len(actual) != len(plan):
                raise ValueError(f"Missing requests: {directory}")
            decisions = defaultdict(list)
            for decision in read_rows(directory / "decisions.jsonl"):
                decisions[decision["request_id"]].append(decision)
            previous = counters(directory / "initial.prom")
            for index, (row, expected) in enumerate(zip(actual, plan)):
                tokens = expected["prompt_token_ids"]
                if (row["index"] != index or row["request_id"] != f"cmpl-cost-{index}-0"
                        or row["prompt_sha256"] != hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                        or any(row[key] != value for key, value in expected.items() if key != "prompt_token_ids")):
                    raise ValueError("Replay identity mismatch")
                matched = [key for key in decisions if re.fullmatch(re.escape(row["request_id"]) + r"-[0-9a-f]{8}", key)]
                if len(matched) != 1:
                    raise ValueError("Missing or ambiguous engine request ID")
                audit = decisions.pop(matched[0])
                if len(audit) != 1:
                    raise ValueError("Unexpected repeated scheduling decision")
                audit = audit[0]
                current = counters(directory / f"{index:04}.prom")
                delta = {key: value - previous.get(key, 0) for key, value in current.items()}
                if delta != row["metrics"] or any(value < 0 for value in delta.values()):
                    raise ValueError("Saved counters differ from raw snapshots")
                previous = current
                gpu = delta["vllm:prefix_cache_hits_total"]
                external = delta.get("vllm:external_prefix_cache_hits_total", 0)
                computed = delta["vllm:request_prefill_kv_computed_tokens_sum"]
                loaded = delta.get("vllm:kv_offload_load_bytes_total", 0)
                if (delta["vllm:request_success_total"] != 1
                        or delta["vllm:prefix_cache_queries_total"] != len(tokens)
                        or delta["vllm:num_preemptions_total"] != 0
                        or row["usage"]["prompt_tokens"] != len(tokens)
                        or row["usage"]["completion_tokens"] != environment["context"]["output_tokens"]
                        or row["finish_reason"] != "length"
                        or len(row["generated_tokens"]) != environment["context"]["output_tokens"]):
                    raise ValueError("Invalid request boundary, output or preemption")
                geometry(len(tokens), int(gpu), audit["cpu_tokens"])
                if (gpu + external + computed != len(tokens)
                        or loaded != external * 28672
                        or audit["prompt_tokens"] != len(tokens) or audit["gpu_tokens"] != gpu
                        or audit["missing_tokens"] != len(tokens) - gpu
                        or audit["admitted_tokens"] != external or audit["mode"] != mode
                        or audit["remaining_tokens"] != computed):
                    raise ValueError("Admission, transfer and compute accounting disagree")
                if row["phase"] in ("cold", "filler") and (gpu or audit["cpu_tokens"]):
                    raise ValueError("Unique first block unexpectedly reused")
                if row["phase"] == "revisit":
                    if audit["cpu_tokens"] <= 0:
                        raise ValueError("Expected recoverable CPU prefix is absent")
                    if row["gap"] == 2 and gpu != 0:
                        raise ValueError("Pressure did not evict GPU prefix")
                    if row["gap"] == 0 and gpu <= 0:
                        raise ValueError("Expected partial GPU hit is absent")
                if mode == "always" and external != audit["cpu_tokens"]:
                    raise ValueError("Always mode rejected candidates")
                if mode == "never" and (external or loaded):
                    raise ValueError("Never mode performed a load")
                if mode == "gate" and audit["cpu_tokens"]:
                    estimate = policy.decide(len(tokens), int(gpu), audit["cpu_tokens"])
                    admitted = audit["cpu_tokens"] if estimate["admit"] else 0
                    if (estimate != audit["estimate"] or audit["profile_sha256"] != policy.sha256
                            or external != admitted):
                        raise ValueError("Frozen policy and actual decision disagree")
                rows.append(dict(**row, stage=stage, epoch=epoch, mode=mode, audit=audit,
                                 gpu_tokens=gpu, cpu_tokens=audit["cpu_tokens"], load_bytes=loaded,
                                 server_ttft_seconds=delta["vllm:time_to_first_token_seconds_sum"]))
            if any(not key.startswith("cmpl-warm-") for key in decisions):
                raise ValueError("Unaccounted requests in scheduler log")
            if previous != counters(directory / "final.prom"):
                raise ValueError("Final snapshot mismatch")
            read_rows(directory / "resources.jsonl")
    return rows


def pair_rows(rows, reference):
    """用 rows 的 epoch/index 将每条记录与 reference 模式的相同输入配对，并核验 GPU/CPU 候选相同。"""
    lookup = {(row["epoch"], row["index"]): row for row in rows if row["mode"] == reference}
    pairs = []
    for row in rows:
        other = lookup[(row["epoch"], row["index"])]
        if any(row[key] != other[key] for key in ("prompt_sha256", "gpu_tokens", "cpu_tokens")):
            raise ValueError("Paired strategies encountered different cache candidates")
        pairs.append((row, other))
    return pairs


def output_mismatches(rows):
    """对 rows 中同输入的策略响应及同用例 cold/revisit 响应检查实际输出 token，返回差异位置。"""
    differences, seen = [], {}
    for row in rows:
        key = (row["epoch"], row["prompt_sha256"])
        previous = seen.setdefault(key, row)
        if row["generated_tokens"] != previous["generated_tokens"] or row["answer"] != previous["answer"]:
            positions = [i for i, (left, right) in enumerate(zip(previous["generated_tokens"], row["generated_tokens"])) if left != right]
            differences.append(dict(stage=row["stage"], epoch=row["epoch"], mode=row["mode"],
                                    index=row["index"], reference_mode=previous["mode"],
                                    reference_index=previous["index"], differing_positions=positions))
    return differences


def calibrate(root):
    """从 root 的独立 always/never 回放生成成本表，使用服务端 TTFT 保留传输、调度和剩余计算的整体成本。"""
    rows = load_stage(root, "calibrate", ("always", "never"))
    groups = defaultdict(list)
    for row, other in pair_rows(rows, "never"):
        if row["mode"] == "always" and row["cpu_tokens"]:
            key = geometry(row["usage"]["prompt_tokens"], int(row["gpu_tokens"]), row["cpu_tokens"])
            groups[key].append((row, other))
    entries = {}
    for key, pairs in groups.items():
        prompt, gpu, cpu = map(int, key.split(":"))
        entries[key] = dict(prompt_tokens=prompt, gpu_tokens=gpu, cpu_tokens=cpu, n=len(pairs),
                            restore_seconds=statistics.median(row["server_ttft_seconds"] for row, _ in pairs),
                            recompute_seconds=statistics.median(row["server_ttft_seconds"] for _, row in pairs),
                            samples=[dict(epoch=left["epoch"], index=left["index"],
                                          restore_seconds=left["server_ttft_seconds"],
                                          recompute_seconds=right["server_ttft_seconds"]) for left, right in pairs])
    environment = json.loads((root / "environment.json").read_text())
    save(root / "calibration.json", dict(schema=1, vllm="0.29.0", context=environment["context"],
         measure="server_ttft_seconds", entries=entries))
    CostPolicy(root / "calibration.json", environment["arguments"]["margin_ms"])


def review(root):
    """复核 root 的独立校准及正式实验，保存配对统计与输出差异；完整性或输出一致性失败时报错。"""
    calibration = load_stage(root, "calibrate", ("always", "never"))
    evaluation = load_stage(root, "evaluate", ("always", "never", "gate"))
    calibrated_blocks = {tuple(row["prompt_token_ids"][:64]) for epoch in (0, 1)
                         for row in json.loads((root / f"calibrate-trace-{epoch}.json").read_text())}
    evaluated_blocks = {tuple(row["prompt_token_ids"][:64]) for epoch in (0, 1)
                        for row in json.loads((root / f"evaluate-trace-{epoch}.json").read_text())}
    if calibrated_blocks & evaluated_blocks:
        raise ValueError("Calibration and evaluation share a first cache block")
    environment = json.loads((root / "environment.json").read_text())
    policy = CostPolicy(root / "calibration.json", environment["arguments"]["margin_ms"])
    # 从原始校准记录复算冻结成本，拒绝评价后手工调整的表。
    grouped = defaultdict(list)
    for row, other in pair_rows(calibration, "never"):
        if row["mode"] == "always" and row["cpu_tokens"]:
            grouped[geometry(row["usage"]["prompt_tokens"], int(row["gpu_tokens"]), row["cpu_tokens"])].append((row, other))
    if grouped.keys() != policy.entries.keys():
        raise ValueError("Calibration coverage changed")
    for key, pairs in grouped.items():
        entry = policy.entries[key]
        if (entry["n"] != len(pairs)
                or entry["restore_seconds"] != statistics.median(row["server_ttft_seconds"] for row, _ in pairs)
                or entry["recompute_seconds"] != statistics.median(row["server_ttft_seconds"] for _, row in pairs)):
            raise ValueError("Calibration costs differ from original observations")
    groups = defaultdict(list)
    for row, baseline in pair_rows(evaluation, "always"):
        if row["phase"] == "revisit":
            groups[(row["stored_tokens"], row["length"], row["gap"], row["mode"])].append((row, baseline))
    summaries = []
    for (stored, length, gap, mode), pairs in sorted(groups.items()):
        paired = [(base["ttft_seconds"] - row["ttft_seconds"]) * 1000 for row, base in pairs]
        summaries.append(dict(stored_tokens=stored, input_tokens=length, gap=gap, mode=mode, n=len(pairs),
                              ttft_ms=statistics.median(row["ttft_seconds"] for row, _ in pairs) * 1000,
                              paired_saved_ms=statistics.median(paired), paired_range_ms=[min(paired), max(paired)],
                              paired_wins=sum(value > 0 for value in paired),
                              epoch_paired_saved_ms={str(epoch): statistics.median(
                                  (base["ttft_seconds"] - row["ttft_seconds"]) * 1000 for row, base in pairs if row["epoch"] == epoch) for epoch in (0, 1)},
                              gpu_tokens=statistics.median(row["gpu_tokens"] for row, _ in pairs),
                              cpu_tokens=statistics.median(row["cpu_tokens"] for row, _ in pairs),
                              admitted_tokens=statistics.median(row["audit"]["admitted_tokens"] for row, _ in pairs),
                              load_MiB=statistics.median(row["load_bytes"] for row, _ in pairs) / 2**20))
    differences = output_mismatches(calibration) + output_mismatches(evaluation)
    gated = [row for row in evaluation if row["mode"] == "gate" and row["cpu_tokens"]]
    if not any(row["audit"]["admitted_tokens"] == 0 for row in gated) or not any(row["audit"]["admitted_tokens"] > 0 for row in gated):
        raise ValueError("Evaluation did not exercise both admission branches")
    save(root / "summary.json", dict(groups=summaries, output_mismatches=differences,
         calibration_sha256=policy.sha256, calibration_requests=len(calibration), evaluation_requests=len(evaluation),
         admitted=sum(row["audit"]["admitted_tokens"] > 0 for row in gated),
         rejected=sum(row["audit"]["admitted_tokens"] == 0 for row in gated)))
    (root / "validated-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in evaluation), encoding="utf-8")
    print(json.dumps(dict(groups=summaries, output_mismatches=differences), ensure_ascii=False, indent=2))
    if differences:
        raise RuntimeError("Generated tokens differ; inspect summary.json and per-request logprobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    review(args.root.resolve())
