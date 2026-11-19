"""复核前缀保留、CPU 占用、实际回迁与实验 08 的逐 token 对照。"""

import argparse
from collections import Counter
import json
from pathlib import Path

from .run_prefix_capacity import GROUPS, INPUTS, ROOT, frozen_plans, review08, ssd


joint, require = review08.joint, review08.require


def consecutive(keys, occupied):
    """返回 keys 从开头起连续出现在 occupied 中的块数。"""
    return next((index for index, key in enumerate(keys) if key not in occupied), len(keys))


def prefix_audit(directory):
    """按 directory 的写入、提升与淘汰预约复算占用和每条请求的连续保留长度。"""
    path = directory / "audit.jsonl"
    occupied, requests = set(), {}
    slots, slot_by_key = {}, {}
    peak = transfer_slots_checked = 0
    for event in joint.read_rows(path):
        kind = event["kind"]
        if kind == "begin":
            keys = event["keys"]
            requests[event["request_id"]] = dict(keys=keys, max_after_store_blocks=0, written_blocks=0)
        elif kind in ("store", "promotion"):
            req = requests[event["request_id"]]
            keys = req["keys"]
            before = consecutive(keys, occupied)
            evicted = event["evicted"]
            incoming = event["keys"] if kind == "store" else [event["key"]]
            for key in evicted:
                occupied.remove(key)
                if key in slot_by_key:
                    del slots[slot_by_key.pop(key)]
            occupied.update(incoming)
            new_slots = event.get("slots")
            if kind == "promotion" and "slot" in event:
                new_slots = [event["slot"]]
            if new_slots is not None:
                for key, slot in zip(incoming, new_slots, strict=True):
                    require(type(slot) is int and 0 <= slot < 8 and slot not in slots, path, "CPU 槽仍在占用")
                    slots[slot], slot_by_key[key] = key, slot
            peak = max(peak, len(occupied))
            require(peak <= 8, path, "CPU 占用超过 8 槽")
            if kind == "store":
                length = consecutive(keys, occupied)
                require(length >= before and set(incoming) <= set(keys[:length]), path,
                        "写入破坏驻留命中或留下前缀空洞")
                req["max_after_store_blocks"] = max(req["max_after_store_blocks"], length)
                req["written_blocks"] += len(incoming)
        elif kind in ("fs_submit", "fs_complete") and "slots" in event:
            for key, slot in zip(event["keys"], event["slots"], strict=True):
                require(slots.get(slot) == key, path, "盘层传输前后 CPU 槽身份改变")
                transfer_slots_checked += 1
    warmup, *formal = [dict(complete_blocks=len(req["keys"]), retained_blocks=req["max_after_store_blocks"],
                           written_blocks=req["written_blocks"]) for req in requests.values()]
    require(warmup == dict(complete_blocks=16, retained_blocks=8, written_blocks=8)
            and len(formal) == 78 and peak == 8, path, "超长预热保留长度或请求数不符")
    return dict(warmup=warmup, peak_blocks=peak, transfer_slots_checked=transfer_slots_checked,
                formal_retained_blocks_histogram=dict(sorted(Counter(row["retained_blocks"] for row in formal).items())),
                formal=formal)


def historical_results(root, env, plan, warmup):
    """从 env 绑定的实验 08 复核报告和原始请求取失败基线及成功组 token。"""
    historical = Path(env["historical"])
    report_path = historical / "review.json"
    require(review08.digest(historical / "environment.json") == env["historical_environment_sha256"]
            and review08.digest(report_path) == env["historical_review_sha256"], root, "历史基线摘要改变")
    old = joint.read_json(report_path)
    require(old["status"] == "failed_groups" and old["cpu_mib"] == 14 and old["workload"] == "frozen-78"
            and old["environment_sha256"] == env["historical_environment_sha256"]
            and all(old["artifacts_sha256"][name] == env["artifacts_sha256"][name] for name in INPUTS),
            report_path, "实验 08 基线或输入不符")
    failures, controls = {}, {}
    for epoch, data in old["epochs"].items():
        for group in GROUPS:
            failed = data["excluded"][group]
            require(failed["engine_errors"] == ["CPU KV allocation failed"] and failed["actual_cpu_blocks"] == 8
                    and failed["completed_request_records"] == dict(warmup=0, formal=0)
                    and failed["last_begin"]["prompt_sha256"] == warmup[0]["prompt_sha256"],
                    report_path, "历史 LRU 失败位置不符")
            failures[f"{epoch}/{group}"] = dict(
                error=failed["engine_errors"][0], cpu_blocks=failed["actual_cpu_blocks"],
                warmup=failed["last_begin"], completed=failed["completed_request_records"],
                evidence_sha256=failed["evidence"]["sha256"])
        for group, stats in data["groups"].items():
            controls[f"{epoch}/{group}"] = {key: stats[key] for key in (
                "cpu_peak_blocks", "computed_tokens", "hit_external_tokens", "ssd_recovered_tokens",
                "ttft", "output_tokens_sha256")}
    baseline = historical / "epoch-0/cpu-admission"
    records = joint.read_rows(baseline / "warmup/requests.jsonl") + joint.read_rows(baseline / "requests.jsonl")
    require([row["prompt_sha256"] for row in records] == [row["prompt_sha256"] for row in warmup + plan],
            baseline, "历史原始请求与计划次序不符")
    tokens = [row["generated_tokens"] for row in records]
    require(all(group["output_tokens_sha256"] == joint.digest(tokens) for group in controls.values()),
            baseline, "历史输出摘要与实际 token 不一致")
    return tokens, dict(root=str(historical), environment_sha256=env["historical_environment_sha256"],
                        review_sha256=env["historical_review_sha256"], lru_failures=failures, controls=controls)


def review(root):
    """从 root 读取两组独立真机原始工件，核对历史输出并计算 SSD 减 CPU 的配对指标。"""
    path = root / "environment.json"
    env = joint.read_json(path)
    frozen_plans(root, env["artifacts_sha256"])
    plan, warmup = (joint.read_plan(root, name) for name in ("eval", "warmup"))
    reference_tokens, history = historical_results(root, env, plan, warmup)
    reference_argv = joint.read_json(Path(env["historical"]) / "epoch-0/cpu-lru/server.json")["argv"]
    summaries, rows_by_group, contexts = {}, {}, []
    sources = {Path(name).name: digest for name, digest in env["source_sha256"].items()
               if Path(name).parent == Path("memmgr/short_term")}
    for group in GROUPS:
        directory = root / "epoch-0" / group
        rows, tokens, stats, context = review08.review_group(directory, 0, plan, warmup, env, reference_argv, None)
        require(context["source"] == sources and stats["cpu_capacity_blocks"] == 8
                and stats["cpu_peak_blocks"] <= 8, directory, "软件摘要或 CPU 上限不符")
        require(tokens == reference_tokens, directory, "逐请求输出 token 与实验 08 不同")
        contexts.append(context)
        prefix = prefix_audit(directory)
        for planned, row, bounds in zip(plan, rows, prefix.pop("formal"), strict=True):
            bounds.update(index=planned["index"], case=planned["case"], gpu_hit_tokens=row["gpu"],
                          external_restored_tokens=row["cpu"], ssd_restored_tokens=64 * row["ssd_chain_blocks"],
                          computed_tokens=row["computed"])
            row.update(bounds)
        summaries[group] = dict(stats, prefix=prefix, per_request=rows)
        rows_by_group[group] = rows
    require(contexts[0] == contexts[1]
            and len({group["audit_session"] for group in summaries.values()}) == len(GROUPS),
            root, "运行上下文不一致或审计会话复用")
    return dict(status="complete", requests_per_group=len(plan), warmup_requests_per_group=len(warmup),
                environment_sha256=review08.digest(path), plans_sha256=env["artifacts_sha256"],
                output_tokens_sha256=joint.digest(reference_tokens), history=history, groups=summaries,
                paired=review08.comparisons(rows_by_group))


def main():
    """从命令行读取实验 11 原始工件目录，输出完整 JSON 复核。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(review(ssd.local_path(args.root, ROOT)), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
