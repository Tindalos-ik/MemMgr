"""离线复核实验 07 的逐请求指标、三级块迁移和配对时延。"""

import argparse
from collections import Counter
from importlib import import_module
import hashlib
import json
import math
from pathlib import Path
import statistics


joint = import_module("experiment_raa.06_kv_joint.review_joint")
require = joint.require
GROUPS = ("cpu-lru", "cpu-learned", "ssd-lru", "ssd-learned")
EPOCHS = ("epoch-0", "epoch-1")
IO_FIELDS = ("read_bytes", "write_bytes")
MONOTONIC_IO = (*IO_FIELDS, "read_count", "write_count", "read_chars", "write_chars")
FROZEN = ("plans/eval.json", "plans/warmup.json", "model/model.json", "model/model.joblib")


def audit_run(path, warmup_records, records, tier, policy, block_bytes):
    """重放 path 的块事件；两段请求划分预热和正式窗口，tier、policy、block_bytes 来自运行配置。"""
    require(not path.with_name(path.name + ".1").exists(), path, "审计已轮转")
    events = joint.read_rows(path)
    init = events[0]
    require(init["kind"] == "init" and init["policy"] == policy, path, "审计初始策略不符")
    capacity = init["capacity"]
    block_tokens = init["identity"]["block_tokens"]
    require(type(capacity) is int and capacity > 0 and block_tokens == 64
            and type(block_bytes) is int and block_bytes > 0, path, "CPU 容量或块规格无效")
    warmup = [row["request_id"] for row in warmup_records]
    planned = warmup + [row["request_id"] for row in records]
    measured = set(planned[len(warmup):])
    require(len(set(planned)) == len(planned), path, "请求 ID 重复")

    tracked, resident, reserved, disk = {}, {}, {}, set()
    jobs, seen_jobs, write_pins = {}, set(), Counter()
    formal = Counter()
    runtime = fs = None
    start = end = None
    warm_end = None
    previous = init["time_ns"]

    for event in events[1:]:
        kind, timestamp = event["kind"], event["time_ns"]
        location = f"{path}/{kind}@{timestamp}"
        require(event["session"] == init["session"] and timestamp >= previous,
                location, "会话改变或时间倒序")
        if start is None and kind == "begin" and event["request_id"] in measured:
            warm_end = previous
        previous = timestamp
        if kind == "context":
            require(runtime is None and not tracked, location, "重复或延迟的运行上下文")
            runtime = event["context"]
            require(runtime["identity"] == init["identity"] and runtime["policy"] == policy
                    and runtime["cpu_blocks"] == capacity and runtime["model_sha256"] == init["model_sha256"],
                    location, "运行上下文与初始策略或布局不符")
            continue
        if kind == "fs_init":
            require(tier == "ssd" and fs is None and not tracked and event["direct_io"] is True
                    and event["existing_bytes"] == 0 and type(event["block_bytes"]) is int
                    and 0 < event["block_bytes"] <= event["capacity"], location, "SSD 须使用空缓存和 O_DIRECT")
            fs = event
            continue

        rid = event["request_id"]
        in_window = rid in measured
        require(runtime is not None and (fs is not None) == (tier == "ssd"), location, "缺少运行或文件层初始化")
        if kind == "begin":
            require(len(tracked) < len(planned) and rid == planned[len(tracked)], location, "请求顺序与采集不符")
            keys = event["keys"]
            require(len(set(keys)) == len(keys) == event["prompt_tokens"] // block_tokens
                    and event["admitted"] == keys[:len(event["admitted"])]
                    and (policy != "lru" or event["admitted"] == keys), location, "前缀块或准入范围无效")
            if in_window and start is None:
                require(not reserved and not jobs and not write_pins, location, "正式开始前未排空预热迁移")
                start = dict(cpu_blocks=len(resident), ssd_files=len(disk), time_ns=timestamp)
                formal["cpu_peak_blocks"] = len(resident)
            tracked[rid] = dict(begin=event, selected=set(event["admitted"]), stored=0, loaded=0,
                                loaded_keys=set(), read_blocks=0, chain_blocks=0)
            continue

        require(rid in tracked and (in_window or start is None), location, "未开始的请求或预热事件越过正式边界")
        req = tracked[rid]
        prefix = req["begin"]["keys"]
        keys = event.get("keys", [])
        keyset = set(keys)
        require(len(keyset) == len(keys) and keyset.issubset(prefix), location, "事件块重复或不属于请求前缀")

        if kind == "lookup":
            gpu, external = event["gpu"], event["external"]
            require("lookup" not in req and event["prompt"] == req["begin"]["prompt_tokens"]
                    and 0 <= gpu <= event["prompt"] and 0 <= external <= event["prompt"] - gpu
                    and gpu % block_tokens == external % block_tokens == 0
                    and event["admitted"] == external and event["estimate"] is None,
                    location, "查找几何或 always 回迁决定不符")
            candidate_keys = prefix[gpu // block_tokens:(gpu + external) // block_tokens]
            expected = []
            for key in prefix[gpu // block_tokens:]:
                if key not in resident:
                    break
                expected.append(key)
            require(candidate_keys == expected, location, "候选与 CPU 上连续可用的前缀不符")
            req.update(lookup=event, candidate_keys=set(candidate_keys))
        elif kind == "retain":
            require(policy == "learned" and "retain" not in req and type(event["accepted"]) is bool,
                    location, "保留决定重复或策略不符")
            req["retain"] = event
            if not event["accepted"]:
                req["selected"].clear()
        elif kind in ("store", "promotion"):
            victims = set(event["evicted"])
            require(len(victims) == len(event["evicted"]) and victims <= resident.keys()
                    and victims.isdisjoint(write_pins), location, "淘汰块不在 CPU 上或仍有 SSD 写入")
            for key in victims:
                resident.pop(key)
            if kind == "promotion":
                key = event["key"]
                require(tier == "ssd" and event["accepted"] is True and key in prefix and key in disk,
                        location, "提升缺少请求前缀或此前完成的 SSD 写入")
                keyset = {key}
            else:
                require(keyset <= req["selected"] and (policy != "learned" or "retain" in req),
                        location, "CPU 写入超出策略准入范围")
            require(keyset.isdisjoint(resident.keys() | reserved.keys()) and keyset.isdisjoint(victims),
                    location, "CPU 预约与占用冲突")
            reserved.update(dict.fromkeys(keyset, (rid, kind)))
            occupied = len(resident) + len(reserved)
            require(occupied <= capacity, location, "CPU 容量超限")
            if in_window:
                formal["cpu_evicted_blocks"] += len(victims)
                formal["cpu_peak_blocks"] = max(formal["cpu_peak_blocks"], occupied)
        elif kind == "stored":
            for key in keys:
                require(reserved.pop(key) == (rid, "store"), location, "CPU 写完成与预约不符")
                resident[key] = None
            req["stored"] += len(keys)
        elif kind == "loaded":
            require(keyset <= req["candidate_keys"] - req["loaded_keys"] and keyset <= resident.keys(),
                    location, "GPU 回迁未准入、重复或 CPU 源块缺失")
            req["loaded_keys"].update(keys)
            req["loaded"] += len(keys)
            req["chain_blocks"] += sum(resident[key] == rid for key in keys)
        elif kind in ("fs_submit", "fs_complete"):
            job, direction = event["job"], event["direction"]
            require(tier == "ssd" and keys and direction in ("store", "load"), location, "文件传输事件无效")
            signature = (rid, direction, keys)
            if kind == "fs_submit":
                require(job not in seen_jobs, location, "SSD 作业 ID 重复")
                seen_jobs.add(job)
                jobs[job] = signature
                if direction == "store":
                    require(not req.get("ssd_closed"), location, "SSD 请求关闭后有新写入")
                    # 原生级联在 CPU 写完成的审计行之前提交；完成时 CPU 块必须已经可读。
                    require(all(key in resident or reserved.get(key) == (rid, "store") for key in keys),
                            location, "SSD 写入缺少 CPU 源块")
                    write_pins.update(keys)
                else:
                    require(all(reserved.get(key) == (rid, "promotion") for key in keys),
                            location, "SSD 读取缺少已淘汰块的 CPU 提升预约")
                continue
            require(job in jobs and jobs.pop(job) == signature and event["success"] is True
                    and math.isfinite(event["transfer_seconds"]) and event["transfer_seconds"] >= 0,
                    location, "SSD 作业失败或完成记录与提交不符")
            if direction == "store":
                require(keyset <= resident.keys(), location, "SSD 写完成时 CPU 源块缺失")
                disk.update(keys)
                require(len(disk) * fs["block_bytes"] <= fs["capacity"], location, "SSD 容量超限")
                for key in keys:
                    write_pins[key] -= 1
                    if not write_pins[key]:
                        del write_pins[key]
            else:
                for key in keys:
                    require(reserved.pop(key) == (rid, "promotion"), location, "SSD 读完成与提升预约不符")
                    resident[key] = rid
                req["read_blocks"] += len(keys)
            if in_window:
                formal[f"ssd_{direction}_blocks"] += len(keys)
        elif kind == "ssd_request_closed":
            require(tier == "ssd" and not req.get("ssd_closed"), location, "SSD 请求关闭事件重复或出现在 CPU 组")
            req["ssd_closed"] = True
        else:
            raise ValueError(f"{location}: 未知审计事件")
        if in_window:
            end = timestamp

    require(list(tracked) == planned and start is not None and end is not None
            and not jobs and not write_pins and not reserved, path, "请求不完整或迁移未排空")
    for rid, req in tracked.items():
        require("lookup" in req and req["loaded_keys"] == req["candidate_keys"]
                and req["chain_blocks"] <= req["read_blocks"]
                and (tier != "ssd" or req.get("ssd_closed")), f"{path}/{rid}", "查找、读取或回迁未闭合")
    chains = sum(tracked[rid]["chain_blocks"] for rid in measured)
    ssd_block_bytes = fs["block_bytes"] if fs else 0
    accounting = dict(cpu_capacity_blocks=capacity, cpu_capacity_bytes=capacity * block_bytes,
                      cpu_start_blocks=start["cpu_blocks"], cpu_end_blocks=len(resident),
                      cpu_peak_blocks=formal["cpu_peak_blocks"], cpu_evicted_blocks=formal["cpu_evicted_blocks"],
                      ssd_capacity_bytes=fs["capacity"] if fs else 0, ssd_block_bytes=ssd_block_bytes,
                      ssd_start_files=start["ssd_files"], ssd_end_files=len(disk),
                      ssd_store_blocks=formal["ssd_store_blocks"], ssd_load_blocks=formal["ssd_load_blocks"],
                      ssd_logical_write_bytes=formal["ssd_store_blocks"] * ssd_block_bytes,
                      ssd_logical_read_bytes=formal["ssd_load_blocks"] * ssd_block_bytes,
                      ssd_to_gpu_chain_blocks=chains,
                      ssd_reads_without_same_request_gpu_load=formal["ssd_load_blocks"] - chains)
    accounting["audit_session"] = init["session"]
    return tracked, accounting, runtime, (warm_end, start["time_ns"], end)


def process_io(path, audit_window):
    """从 path 中同一组进程的正式首尾快照取 I/O 差；audit_window 含预热末尾及正式首尾。"""
    samples = joint.read_rows(path)
    phases = [row["phase"] for row in samples]
    require(len(samples) >= 3 and phases[:2] == ["warmup_start", "formal_start"]
            and phases[-1] == "formal_end" and all(phase == "formal_sample" for phase in phases[2:-1]),
            path, "进程 I/O 快照边界缺失或顺序错误")
    times = [row["time_ns"] for row in samples]
    require(all(type(t) is int and t > 0 for t in times)
            and all(a < b for a, b in zip(times, times[1:]))
            and times[0] <= audit_window[0] <= times[1] <= audit_window[1] <= audit_window[2] <= times[-1],
            path, "进程 I/O 窗口未覆盖正式审计")
    initial = previous = None
    previous_birth = None
    parent_pid = samples[0]["processes"][0]["pid"]
    clock_shifts = []
    for sample in samples:
        processes = sample["processes"]
        values = {row["pid"]: tuple(row["io"][key] for key in MONOTONIC_IO) for row in processes}
        births = {row["pid"]: row["created_at"] for row in processes}
        require(len(values) >= 2 and len(values) == len(processes) and processes[0]["pid"] == parent_pid
                and all(type(pid) is int and pid > 0 and math.isfinite(created) and created > 0
                        for pid, created in births.items())
                and all(type(n) is int and n >= 0 for pair in values.values() for n in pair),
                path, "进程身份重复或 I/O 计数无效")
        if previous is not None:
            require(values.keys() == previous.keys()
                    and all(all(a <= b for a, b in zip(previous[key], values[key])) for key in values),
                    path, "采样进程改变或内核 I/O 计数回退")
            if births != previous_birth:
                # psutil 每次枚举子进程时重算 btime + starttime；父进程创建时间从同一对象缓存。
                offsets = {births[pid] - previous_birth[pid] for pid in births if pid != parent_pid}
                require(len(values) >= 3 and births[parent_pid] == previous_birth[parent_pid]
                        and len(offsets) == 1, path, "子进程创建时间发生不一致的变化")
                shift = offsets.pop()
                require(shift != 0 and shift == round(shift), path, "子进程创建时间变化并非共同的整秒漂移")
                clock_shifts.append(int(shift))
        if sample["phase"] == "formal_start":
            initial = values
        previous, previous_birth = values, births
    deltas = {f"process_{field}": sum(values[key][i] - initial[key][i] for key in values)
              for i, field in enumerate(IO_FIELDS)}
    return dict(deltas, process_creation_clock_shifts_seconds=clock_shifts)


def review_group(directory, epoch, plan, warmup, block_bytes, model_sha256):
    """核对 directory 的第 epoch 轮分组；两段计划、块大小和冻结模型摘要限定可比数据。"""
    tier, policy = directory.name.split("-")
    marker = directory / "complete.json"
    completed = joint.read_json(marker)
    require(completed == dict(status="complete", epoch=epoch, group=directory.name,
                              requests=len(plan), warmup_requests=len(warmup)),
            marker, "组完成标记与目录或计划不符")
    server_path = directory / "server.json"
    server = joint.read_json(server_path)
    argv = server["argv"]
    transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
    extra = transfer["kv_connector_extra_config"]
    options = extra["policy_config"]
    require(transfer["kv_connector"] == "ManagedConnector"
            and options["tier"] == server["tier"] == tier and options["policy"] == server["policy"] == policy
            and options["restore"] == server["restore"] == "always" and not options["cost_profile"]
            and options["batch_invariant"] is True and argv[argv.index("--max-num-seqs") + 1] == "1"
            and extra["spec_name"] == ("ManagedTieringSpec" if tier == "ssd" else "ManagedCPUSpec"),
            server_path, "服务与分组策略不符")
    require(Path(options["audit_path"]).name == "audit.jsonl", server_path, "服务未启用本组审计")
    records = joint.read_rows(directory / "requests.jsonl")
    warm_dir = directory / "warmup"
    warmup_records = joint.read_rows(warm_dir / "requests.jsonl")
    tracked, stats, context, audit_window = audit_run(directory / "audit.jsonl", warmup_records, records,
                                                     tier, policy, block_bytes)
    block_tokens = context["identity"]["block_tokens"]
    require(options["cpu_bytes"] == extra["cpu_bytes_to_use"] == server["cpu_bytes"]
            and options["cpu_bytes"] // block_bytes == stats["cpu_capacity_blocks"]
            and options["block_tokens"] == extra["block_size"] == block_tokens
            and extra["model_revision"] == context["identity"]["model_revision"]
            and extra["dtype"] == context["identity"]["dtype"]
            and extra["gpu_bytes"] == options["gpu_bytes"] == context["gpu_bytes"] == server["gpu_bytes"]
            and context["batch_invariant"] is True, server_path, "配置与审计块布局不符")
    if policy == "learned":
        require(Path(options["model_dir"]).name == "model", server_path, "服务未使用冻结模型目录")
        require(context["model_sha256"] == model_sha256, server_path, "冻结模型摘要不符")
    else:
        require(context["model_sha256"] is None, server_path, "LRU 组加载了 learned 模型")
    if tier == "ssd":
        require(options["ssd_bytes"] == stats["ssd_capacity_bytes"] and stats["ssd_block_bytes"] >= block_bytes,
                server_path, "SSD 容量或文件块大小与审计不符")
    warm_rows, _, warm_end = joint.review_prom(warm_dir, warmup_records, warmup, tracked,
                                                block_bytes, block_tokens, "warm")
    rows, initial, _ = joint.review_prom(directory, records, plan, tracked, block_bytes, block_tokens, "joint")
    require(warm_end == initial, directory, "预热结束与正式开始的 Prometheus 快照不一致")
    require(all(len(row["tokens"]) == 8 for row in warm_rows + rows), directory, "输出长度与实验 06 不符")
    for row, record in zip(rows, records):
        row["ssd_chain_blocks"] = tracked[record["request_id"]]["chain_blocks"]

    io = process_io(directory / "resources.jsonl", audit_window)
    new_file_bytes = (stats["ssd_end_files"] - stats["ssd_start_files"]) * stats["ssd_block_bytes"]
    require(io["process_read_bytes"] >= stats["ssd_logical_read_bytes"]
            and io["process_write_bytes"] >= new_file_bytes,
            directory, "进程实际 I/O 未覆盖 O_DIRECT 读取或新增文件")
    disk_path = directory / "disk.json"
    disk = joint.read_json(disk_path)
    require(disk["files"] == stats["ssd_end_files"]
            and disk["bytes"] == stats["ssd_end_files"] * stats["ssd_block_bytes"]
            and disk["root"] == server["ssd_root"] and ((disk["root"] is None) == (tier == "cpu")),
            disk_path, "盘面文件数或文件字节与成功写入审计不符")
    if tier == "ssd":
        require(disk["root"] == extra["secondary_tiers"][0]["root_dir"] and Path(disk["root"]).is_absolute(),
                disk_path, "盘面路径与文件层配置不符")
    stats.update(io, requests=len(rows), disk_root=disk["root"], ssd_end_bytes=disk["bytes"],
                 ssd_new_file_bytes=new_file_bytes,
                 ssd_start_bytes=stats["ssd_start_files"] * stats["ssd_block_bytes"],
                 gpu_capacity_blocks=context["gpu_blocks"], gpu_capacity_bytes=context["gpu_bytes"],
                 ssd_recovery=("not_applicable" if tier == "cpu"
                               else "confirmed" if stats["ssd_to_gpu_chain_blocks"] else "zero"),
                 ssd_recovered_tokens=stats["ssd_to_gpu_chain_blocks"] * block_tokens,
                 ssd_recovery_requests=sum(row["ssd_chain_blocks"] > 0 for row in rows),
                 hit_gpu_tokens=sum(row["gpu"] for row in rows),
                 hit_external_tokens=sum(row["cpu"] for row in rows),
                 cpu_only_recovered_tokens=sum(row["cpu"] for row in rows) - stats["ssd_to_gpu_chain_blocks"] * block_tokens,
                 computed_tokens=sum(row["computed"] for row in rows),
                 gpu_to_cpu_bytes=sum(row["store_bytes"] for row in rows),
                 cpu_to_gpu_bytes=sum(row["load_bytes"] for row in rows),
                 ttft=joint.latencies(row["ttft"] for row in rows),
                 server_ttft=joint.latencies(row["server_ttft"] for row in rows))
    return rows, warm_rows, stats, context


def paired(pairs):
    """按 pairs 中同计划位置的 (对照, 处理) 求 TTFT 差和重算 token；差为处理减对照。"""
    result = joint.paired(pairs)
    control = sum(left["computed"] for left, _ in pairs)
    treatment = sum(right["computed"] for _, right in pairs)
    result["computed_tokens"] = dict(control=control, treatment=treatment, delta=treatment - control,
                                     paired_median_delta=statistics.median(
                                         right["computed"] - left["computed"] for left, right in pairs) if pairs else None)
    return result


def frozen_model(root, expected):
    """核对 root 中四个冻结输入文件的字节摘要 expected，返回 learned 模型摘要。"""
    require(set(expected) == set(FROZEN), root, "冻结清单不完整")
    for name in FROZEN:
        path = root / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == expected[name], path, "与采集清单的文件摘要不符")
    model_sha256 = expected["model/model.joblib"]
    require(joint.read_json(root / "model/model.json")["model_sha256"] == model_sha256,
            root, "冻结模型与元数据的摘要不符")
    return model_sha256


def review(root):
    """复核 root 下两个独立服务轮次；缺正式完成标记时给出未测成状态。"""
    markers = {epoch: [name for name in GROUPS if (root / epoch / name / "complete.json").is_file()]
               for epoch in EPOCHS}
    complete_path = root / "complete.json"
    complete = complete_path.is_file()
    report = dict(status="complete" if complete else "not_measured", epochs={})
    if not complete and all(len(names) < len(GROUPS) for names in markers.values()):
        report["epochs"] = {epoch: dict(status="not_measured", missing_groups=[g for g in GROUPS if g not in names])
                            for epoch, names in markers.items()}
        return report
    plan = joint.read_plan(root, "eval")
    warmup = joint.read_plan(root, "warmup")
    require(len(plan) == 78 and len(warmup) == 1, root, "冻结的 06 评价或预热计划长度不符")
    environment = joint.read_json(root / "environment.json")
    block_bytes = environment["kv_bytes_per_block"]
    model_sha256 = frozen_model(root, environment["artifacts_sha256"])
    order = None
    if complete:
        marker = joint.read_json(complete_path)
        order = marker["epochs"]
        require(marker["status"] == "complete" and marker["requests_per_group"] == len(plan)
                and marker["warmup_requests_per_group"] == len(warmup)
                and order == environment["epochs"] and len(order) == len(EPOCHS)
                and sorted(order[0]) == sorted(GROUPS) and order[1] == order[0][::-1]
                and all(len(names) == len(GROUPS) for names in markers.values()),
                complete_path, "总完成标记与请求数量、分组或正反顺序不符")
    else:
        report["status"] = "partial"
    baseline = context_baseline = ssd_baseline = None
    sessions, disk_roots = set(), set()
    for index, (epoch, names) in enumerate(markers.items()):
        if len(names) < len(GROUPS):
            report["epochs"][epoch] = dict(status="not_measured", missing_groups=[g for g in GROUPS if g not in names])
            continue
        rows_by_group, summaries = {}, {}
        for name in order[index] if order else names:
            directory = root / epoch / name
            rows, warm_rows, stats, context = review_group(directory, index, plan, warmup, block_bytes, model_sha256)
            common = {key: value for key, value in context.items() if key not in ("policy", "model_sha256")}
            tokens = [row["tokens"] for row in warm_rows + rows]
            if baseline is None:
                baseline, context_baseline = tokens, common
            require(common == context_baseline, directory, "轮次或组之间的推理环境或缓存几何不符")
            for i, (expected, actual) in enumerate(zip(baseline, tokens)):
                require(actual == expected, directory, f"预热/正式序列第 {i} 条输出 token ID 不一致")
            session = stats.pop("audit_session")
            require(session not in sessions, directory, "各组须使用独立审计会话")
            sessions.add(session)
            if stats["disk_root"] is not None:
                require(stats["disk_root"] not in disk_roots, directory, "各 SSD 组须使用独立缓存目录")
                disk_roots.add(stats["disk_root"])
                ssd_layout = (stats["ssd_block_bytes"], stats["ssd_capacity_bytes"])
                ssd_baseline = ssd_layout if ssd_baseline is None else ssd_baseline
                require(ssd_layout == ssd_baseline, directory, "SSD 组的文件块或容量不同")
            rows_by_group[name], summaries[name] = rows, stats
        comparisons = {}
        for control, treatment in (("cpu-lru", "ssd-lru"), ("cpu-learned", "ssd-learned"),
                                   ("cpu-lru", "cpu-learned"), ("ssd-lru", "ssd-learned")):
            pairs = list(zip(rows_by_group[control], rows_by_group[treatment]))
            comparison = dict(control=control, treatment=treatment, all=paired(pairs))
            if treatment.startswith("ssd-") and control.startswith("cpu-"):
                comparison["ssd_recovery_requests"] = paired([pair for pair in pairs if pair[1]["ssd_chain_blocks"]])
            comparisons[f"{treatment}_vs_{control}"] = comparison
        report["epochs"][epoch] = dict(status="complete", execution_order=order[index] if order else None,
                                        input_and_output_tokens_verified=len(plan), groups=summaries, paired=comparisons)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="实验 07 采集输出目录")
    args = parser.parse_args()
    result = review(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "complete" else 2)
