"""从真实 KV 槽、作业、Prom 和进程 I/O 复核实验 12 的配对结果。"""

import argparse
from collections import Counter
from importlib import import_module
import json
import math
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

from memmgr.short_term.reuse import Predictor

run = import_module("experiment_raa.12_ssd_write.run_ssd_write")
capacity = import_module("experiment_raa.08_kv_capacity.review_capacity")
ssd, joint, require = capacity.ssd, capacity.joint, capacity.require


def replay(path, warm, records, env, mode):
    """按 path 重放 warm 和 records 的真实 CPU 槽、SSD 作业与回读链，env 和 mode 固定布局。"""
    require(not path.with_name(path.name + ".1").exists(), path, "审计日志已轮转")
    events = joint.read_rows(path)
    init, fs, context = events[:3]
    stride = env["kv_bytes_per_block"]
    slots_total = env["cpu_bytes"] // stride
    require(init["kind"] == "init" and init["capacity"] == slots_total and init["policy"] == "learned"
            and fs["kind"] == "fs_init" and fs["ssd_write"] == mode and fs["direct_io"] is True
            and fs["block_bytes"] == stride and fs["capacity"] == env["ssd_bytes"] and fs["existing_bytes"] == 0
            and context["kind"] == "context", path, "布局、策略或 O_DIRECT 初始化不符")
    ctx = context["context"]
    sources = {Path(name).name: digest for name, digest in env["source_sha256"].items()
               if Path(name).parent.as_posix() == "memmgr/short_term"}
    require(ctx["cpu_blocks"] == slots_total and ctx["gpu_blocks"] == env["gpu_bytes"] // stride
            and ctx["gpu_bytes"] == env["gpu_bytes"] and ctx["batch_invariant"] is True
            and ctx["gpu"].split(",", 1)[0] in env["gpu"]
            and ctx["source"] == sources and ctx["identity"] == init["identity"]
            and ctx["model_sha256"] == init["model_sha256"],
            path, "运行上下文与启动清单不符")
    warm_ids = [row["request_id"] for row in warm]
    formal_ids = [row["request_id"] for row in records]
    planned = warm_ids + formal_ids
    measured = set(formal_ids)
    require(len(set(planned)) == len(planned), path, "请求 ID 重复")
    tracked, resident, reserved, occupied = {}, {}, {}, {}
    jobs, seen_jobs, pins, disk, evicted_once = {}, set(), set(), set(), set()
    deferred, wait = None, None
    stats = Counter()
    before = start = end = None
    previous = context["time_ns"]
    for event in events[3:]:
        kind, ts = event["kind"], event["time_ns"]
        require(event["session"] == init["session"] and ts >= previous, path, "审计会话或单调时钟不符")
        rid = event.get("request_id")
        formal = rid in measured
        if kind == "ssd_shutdown":
            require(event is events[-1] and event["success"] is True, path, "未正常关闭盘层")
            continue
        if kind == "begin":
            require(len(tracked) < len(planned) and rid == planned[len(tracked)], path, "请求顺序不符")
            require(len(set(event["keys"])) == len(event["keys"]) == event["prompt_tokens"] // 64,
                    path, "完整前缀块数与输入不符")
            if formal and start is None:
                require(not (jobs or pins or reserved or deferred or wait), path, "预热迁移未排空")
                before, start = previous, ts
                stats.update(cpu_start_blocks=len(occupied), cpu_peak_blocks=len(occupied), ssd_start_files=len(disk))
            tracked[rid] = dict(begin=event, stored=0, loaded=0, loaded_keys=set(), chain_keys=set())
            previous = ts
            continue
        previous = ts
        require(rid in tracked and (formal or start is None), path, "事件跨越预热边界或缺少请求")
        req = tracked[rid]
        prefix = req["begin"]["keys"]
        keys = event.get("keys", [])
        keyset = set(keys)
        require(len(keyset) == len(keys), path, "作业键重复")
        if formal:
            end = ts

        if kind == "retain":
            require("retain" not in req, path, "保留判断重复")
            req["retain"] = event
        elif kind == "lookup":
            require("lookup" not in req and event["admitted"] == event["external"] and event["estimate"] is None,
                    path, "查找或 always 回迁决定不符")
            first = event["gpu"] // 64
            count = event["external"] // 64
            expected = prefix[first:first + count]
            require(event["external"] % 64 == event["gpu"] % 64 == 0 and len(expected) == count
                    and all(key in resident for key in expected), path, "查找使用了尚不可用的 CPU 槽")
            req.update(lookup=event, candidate_keys=set(expected))
        elif kind in ("store", "promotion"):
            victims = set(event["evicted"])
            require(len(victims) == len(event["evicted"]) and victims <= resident.keys()
                    and victims <= disk and not victims & pins, path, "CPU 淘汰早于写盘完成或淘汰了在途槽")
            if deferred is not None:
                require(wait is None and deferred[0] == rid and deferred[1] == ("gpu_store" if kind == "store" else kind)
                        and deferred[2] <= victims, path, "延写作业未对应到同次实际 CPU 淘汰")
                deferred = None
            evicted_once.update(victims)
            for key in victims:
                del occupied[resident.pop(key)[0]]
            if kind == "promotion":
                key = event["key"]
                require(event["accepted"] is True and key in prefix and key in disk and key in evicted_once,
                        path, "提升缺少此前完成的盘写或 CPU 淘汰")
                keys, slots = [key], [event["slot"]]
            else:
                slots = event["slots"]
                require(keyset <= set(req["begin"]["admitted"])
                        and (not keys or req["retain"]["accepted"]), path, "CPU 写超出冻结 GBDT 准入")
            for key, slot in zip(keys, slots, strict=True):
                require(slot not in occupied and 0 <= slot < slots_total and key not in resident and key not in reserved,
                        path, "CPU 槽重复分配或超出容量")
                occupied[slot] = key
                reserved[key] = (slot, rid, kind)
            if formal:
                stats["cpu_peak_blocks"] = max(stats["cpu_peak_blocks"], len(occupied))
                stats["cpu_evictions"] += len(victims)
        elif kind == "stored":
            for key in keys:
                slot, writer, cause = reserved.pop(key)
                require((writer, cause) == (rid, "store"), path, "CPU 写完成与预约不符")
                resident[key] = (slot, None)
            req["stored"] += len(keys)
        elif kind == "loaded":
            require(keyset <= req["candidate_keys"] - req["loaded_keys"] and keyset <= resident.keys(),
                    path, "GPU 回读缺少已完成 CPU 源块")
            req["loaded_keys"].update(keys)
            req["loaded"] += len(keys)
            req["chain_keys"].update(key for key in keys if resident[key][1] == rid)
        elif kind == "ssd_request_closed":
            require(not req.get("closed"), path, "请求关闭重复")
            req["closed"] = True
        elif kind == "ssd_slot_wait":
            seconds = event["seconds"]
            require(event["reason"] in ("gpu_store", "promotion") and 0 < event["slots"] <= slots_total
                    and seconds > 0 and math.isfinite(seconds), path, "槽等待指标无效")
            if mode == "on_evict":
                require(wait is not None and wait[:3] == (rid, event["reason"], event["slots"]) and seconds >= wait[3],
                        path, "槽等待未涵盖完整的延写作业时间")
                wait = None
            if formal:
                stats["slot_wait_seconds"] += seconds
                stats["slot_wait_max_seconds"] = max(stats["slot_wait_max_seconds"], seconds)
                stats["slot_wait_count"] += 1
        elif kind in ("fs_submit", "fs_complete"):
            job, direction, reason, slots = (event[name] for name in ("job", "direction", "reason", "slots"))
            signature = (rid, direction, reason, keys, slots)
            require(keys and len(keys) == len(slots) == len(set(slots)), path, "盘层作业槽几何不符")
            if direction == "store":
                require((mode == "eager" and reason == mode and keyset <= set(prefix))
                        or (mode == "on_evict" and reason in ("gpu_store", "promotion")), path, "写入触发条件不符")
                require(all(key in resident and resident[key][0] == slot and occupied[slot] == key
                            for key, slot in zip(keys, slots, strict=True)), path, "写入期间源槽已被覆盖")
            else:
                require(direction == "load" and reason == "promotion" and keyset <= disk & evicted_once
                        and all(reserved[key] == (slot, rid, "promotion") and occupied[slot] == key
                                for key, slot in zip(keys, slots, strict=True)), path, "读回缺少成功写盘及目标槽")
            if kind == "fs_submit":
                require(job not in seen_jobs, path, "盘作业 ID 重用")
                seen_jobs.add(job)
                jobs[job] = signature
                if direction == "store":
                    require(not (keyset & (pins | disk)), path, "同块重复写盘")
                    pins.update(keys)
                    require((len(disk) + len(pins)) * stride <= env["ssd_bytes"], path, "SSD 配额超限")
                    if formal:
                        stats["ssd_write_pinned_peak"] = max(stats["ssd_write_pinned_peak"], len(pins))
                continue
            transfer_seconds = event["transfer_seconds"]
            require(jobs.pop(job) == signature and event["success"] is True and 0 <= transfer_seconds < math.inf,
                    path, "盘作业未成功完成或块槽发生变化")
            if direction == "store":
                pins.difference_update(keys)
                disk.update(keys)
                if mode == "on_evict":
                    require(deferred is None and wait is None, path, "交接完成前启动了新一次延写")
                    deferred = (rid, reason, keyset)
                    wait = (rid, reason, len(keys), transfer_seconds)
                    if formal and reason == "promotion":
                        stats["promotion_triggered_write_blocks"] += len(keys)
            else:
                for key in keys:
                    resident[key] = (reserved.pop(key)[0], rid)
            if formal:
                stats[f"ssd_{direction}_blocks"] += len(keys)
        else:
            raise ValueError(f"{path}: 未知事件 {kind}")

    require(events[-1]["kind"] == "ssd_shutdown" and list(tracked) == planned and end is not None
            and not (jobs or pins or reserved or deferred or wait), path, "作业或请求未排空")
    for rid, req in tracked.items():
        require(req["closed"] and req["loaded_keys"] == req["candidate_keys"], path, f"请求没有完整结束: {rid}")
    stats["ssd_end_files"] = len(disk)
    stats["cpu_end_blocks"] = len(occupied)
    stats["audit_start_ns"] = init["time_ns"]
    stats["audit_shutdown_ns"] = events[-1]["time_ns"]
    return tracked, stats, ctx, (before, start, end)


def metrics(path):
    """从 path 取盘层按作业计数、查询命中和分配阻塞秒数。"""
    names = {"write_bytes", "read_bytes", "chunk_queries", "chunk_hits"}
    wanted = {f"vllm:kv_offload_tiering_{name}_total": name for name in names}
    values = dict.fromkeys(names, 0.0)
    for family in text_string_to_metric_families(path.read_text(encoding="utf-8")):
        for sample in family.samples:
            if sample.name in wanted and sample.labels.get("tier", "").startswith("1:"):
                values[wanted[sample.name]] += sample.value
            elif sample.name == "vllm:kv_offload_ssd_slot_wait_seconds_total":
                values["slot_wait_seconds"] = sample.value
    require("slot_wait_seconds" in values, path, "缺少实际源槽等待指标")
    return values


def group(root, workload, epoch, mode, env, predictor, plan, warm):
    """核对 root 下 workload/epoch/mode 的启动设置和真实原始指标，predictor 和两段计划固定输入。"""
    directory = root / workload / f"epoch-{epoch}" / mode
    marker = dict(status="complete", workload=workload, epoch=epoch, mode=mode,
                  requests=len(plan), warmup_requests=len(warm))
    require(joint.read_json(directory / "complete.json") == marker, directory, "组未完成")
    server = joint.read_json(directory / "server.json")
    transfer = json.loads(server["argv"][-1])
    extra = transfer["kv_connector_extra_config"]
    options = extra["policy_config"]
    cache = extra["secondary_tiers"][0]["root_dir"]
    require(server["env"] == env["env"] and extra["spec_name"] == "ManagedTieringSpec"
            and options["tier"] == "ssd" and options["policy"] == "learned" and options["restore"] == "always"
            and options["ssd_write"] == mode and Path(options["model_dir"]) == root / "model"
            and Path(options["ssd_root"]) == directory / "ssd" and Path(cache).parent == directory / "ssd"
            and all(options[name] == env[name] for name in ("cpu_bytes", "gpu_bytes", "ssd_bytes")),
            directory, "启动配置与对照设计不符")
    audit = directory / "audit.jsonl"
    warmdir = directory / "warmup"
    warm_records = joint.read_rows(warmdir / "requests.jsonl")
    records = joint.read_rows(directory / "requests.jsonl")
    tracked, counts, ctx, window = replay(audit, warm_records, records, env, mode)
    require(ctx["model_sha256"] == predictor.sha256, directory, "推理实际使用的 GBDT 不符")
    admission = capacity.check_admission(audit, tracked, records, predictor)
    stride = env["kv_bytes_per_block"]
    warm_rows, _, warm_end = joint.review_prom(warmdir, warm_records, warm, tracked, stride, 64, "warm")
    rows, initial, _ = joint.review_prom(directory, records, plan, tracked, stride, 64, "joint")
    require(warm_end == initial and all(len(row["tokens"]) == 8 for row in warm_rows + rows), directory, "预热边界不符")
    for row, request in zip(rows, records, strict=True):
        state = tracked[request["request_id"]]
        row["chain_positions"] = [i for i, key in enumerate(state["begin"]["keys"]) if key in state["chain_keys"]]
    io = ssd.process_io(directory / "resources.jsonl", window)
    disk = joint.read_json(directory / "disk.json")
    write, read = (counts[f"ssd_{way}_blocks"] * stride for way in ("store", "load"))
    first, last = (metrics(directory / f"{name}.prom") for name in ("initial", "final"))
    delta = {key: last[key] - first[key] for key in first}
    require(io["process_write_bytes"] >= write and io["process_read_bytes"] >= read and disk["root"] == cache
            and disk["bytes"] == disk["files"] * stride and delta["chunk_queries"] >= delta["chunk_hits"] >= 0
            and disk["files"] == counts["ssd_end_files"] == counts["ssd_start_files"] + counts["ssd_store_blocks"]
            and delta["write_bytes"] == write and delta["read_bytes"] == read
            and math.isclose(delta["slot_wait_seconds"], counts["slot_wait_seconds"], abs_tol=1e-9),
            directory, "盘容量、实际 I/O、逻辑 I/O 或槽等待的计账不符")
    tokens = [row["tokens"] for row in warm_rows + rows]
    chain_blocks = sum(len(row["chain_positions"]) for row in rows)
    stats = dict({key: counts[key] for key in ("cpu_start_blocks", "cpu_peak_blocks", "cpu_end_blocks", "cpu_evictions",
                 "audit_start_ns", "audit_shutdown_ns",
                 "ssd_write_pinned_peak", "ssd_store_blocks", "ssd_load_blocks", "ssd_start_files", "ssd_end_files",
                 "promotion_triggered_write_blocks", "slot_wait_count", "slot_wait_seconds", "slot_wait_max_seconds")},
                 **io, **admission, disk_bytes=disk["bytes"], logical_write_bytes=write, logical_read_bytes=read,
                 ssd_lookup_misses=int(delta["chunk_queries"] - delta["chunk_hits"]), ssd_gpu_chain_blocks=chain_blocks,
                 ssd_gpu_chain_requests=sum(bool(row["chain_positions"]) for row in rows),
                 read_without_gpu_chain_blocks=counts["ssd_load_blocks"] - chain_blocks,
                 computed_tokens=sum(row["computed"] for row in rows), output_sha256=joint.digest(tokens),
                 ttft={field: joint.latencies(row[field] for row in rows) for field in ("ttft", "server_ttft")},
                 evidence=capacity.evidence(directory))
    return stats, tokens, rows, ctx


def compare(root, env, workload, predictor, warm):
    """配对 root 中 workload 的正反两轮，每个组由 env、predictor、warm 约束。"""
    plan = joint.read_json(root / ("plans/eval.json" if workload == "frozen-78" else "plans/pressure.json"))
    results, output_reference, context_reference = {}, None, None
    for epoch in range(2):
        groups = {}
        for mode in run.MODES:
            stats, tokens, rows, ctx = group(root, workload, epoch, mode, env, predictor, plan, warm)
            if output_reference is None:
                output_reference, context_reference = tokens, ctx
            require(tokens == output_reference and ctx == context_reference, root, "逐 token 输出或运行上下文不一致")
            groups[mode] = (stats, rows)
        eager, after = (groups[mode][1] for mode in run.MODES)
        pairs = list(zip(eager, after, strict=True))
        lost_chains = lost_hits = 0
        for a, b in pairs:
            positions = set(a["chain_positions"])
            lost_chains += len(positions - set(b["chain_positions"]))
            lost_hits += sum(index >= (b["gpu"] + b["cpu"]) // 64 for index in positions)
        keys = ("computed_tokens", "logical_write_bytes", "process_write_bytes", "ssd_lookup_misses", "ssd_gpu_chain_blocks")
        delta = {key: groups["on_evict"][0][key] - groups["eager"][0][key] for key in keys}
        results[f"epoch-{epoch}"] = dict(groups={mode: groups[mode][0] for mode in run.MODES},
            on_evict_minus_eager=delta, eager_chain_not_repeated_blocks=lost_chains,
            eager_chain_missed_in_any_tier_blocks=lost_hits,
            extra_computed_tokens=sum(max(0, b["computed"] - a["computed"]) for a, b in pairs),
            paired_ttft=joint.paired(pairs))
    return results


def main():
    """复核命令行指定的实验根目录并将完整结果写到标准输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output.resolve()
    env = joint.read_json(root / "environment.json")
    require(joint.read_json(root / "complete.json") == dict(status="complete", groups=env["groups"])
            and env["groups"] == [list(item) for item in run.GROUPS], root, "组不全或顺序不符")
    model = ssd.frozen_model(root, {name: env["artifacts_sha256"][name] for name in ssd.FROZEN})
    require(capacity.digest(root / "plans/pressure.json") == env["artifacts_sha256"]["plans/pressure.json"],
            root, "热点计划摘要不符")
    identity = dict(model_revision=env["server"]["revision"], dtype="float16", block_tokens=64,
                    features="block-chain-v1", vllm="0.29.0")
    predictor = Predictor(root / "model", identity)
    require(predictor.sha256 == model, root, "GBDT 摘要不符")
    warm = joint.read_json(root / "plans/warmup.json")
    results = {workload: compare(root, env, workload, predictor, warm) for workload in run.WORKLOADS}
    ordered = [results[workload][f"epoch-{epoch}"]["groups"][mode] for workload, epoch, mode in run.GROUPS]
    require(all(a["audit_shutdown_ns"] < b["audit_start_ns"] for a, b in zip(ordered, ordered[1:])),
            root, "服务未按正反序逐组关闭再启动")
    print(json.dumps(dict(status="complete", environment_sha256=capacity.digest(root / "environment.json"),
                          output=str(root), workloads=results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
