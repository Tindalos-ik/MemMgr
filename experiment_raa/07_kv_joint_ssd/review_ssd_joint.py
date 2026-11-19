"""离线复核实验 07 的逐请求指标、三级块迁移和配对时延。"""

from collections import Counter
from importlib import import_module
import math


joint = import_module("experiment_raa.06_kv_joint.review_joint")
require = joint.require


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
    previous = init["time_ns"]

    for event in events[1:]:
        kind, timestamp = event["kind"], event["time_ns"]
        location = f"{path}/{kind}@{timestamp}"
        require(event["session"] == init["session"] and timestamp >= previous,
                location, "会话改变或时间倒序")
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
        else:
            raise ValueError(f"{location}: 未知审计事件")
        if in_window:
            end = timestamp

    require(list(tracked) == planned and start is not None and end is not None
            and not jobs and not write_pins and not reserved, path, "请求不完整或迁移未排空")
    for rid, req in tracked.items():
        require("lookup" in req and req["loaded_keys"] == req["candidate_keys"]
                and req["chain_blocks"] <= req["read_blocks"], f"{path}/{rid}", "查找、读取或回迁未闭合")
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
    return tracked, accounting, runtime, (start["time_ns"], end)
