"""按实验 13 的真实事件重放 eager SSD 副本发布、CPU 槽和分支成本。"""

from collections import Counter
import math

from .historical import BLOCK_BYTES, read, rows, sha


def close(expected, observed):
    """要求用于复核的 expected 和 observed 秒数在数值容差内一致。"""
    if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Measured cost differs: {expected} != {observed}")


def verify_value(event, profile, parents, depths, disk, resident, pending, stats):
    """按 event 时点的 profile、键链、已发布/在途副本和 CPU 槽复算净收益，更新 stats。"""
    incoming, victims = event["incoming"], event["victims"]
    for field, candidates in (("incoming", incoming), ("victim", victims)):
        for key, probability, alternative, gain, *write in candidates:
            root = parents[key]
            reachable = key in disk
            while reachable and root is not None:
                reachable = root in resident or root in disk
                root = parents[root]
            expected = "ssd" if reachable else "recompute"
            if alternative != expected or not 0 <= probability <= 1 or (key in resident) != (field == "victim"):
                raise ValueError(f"Invalid priced {field} {key}: {alternative}, expected {expected}")
            close(probability * profile.gain(depths[key], reachable), gain)
            if field == "incoming":
                seconds, = write
                close(profile.cpu_write + (0 if key in disk or key in pending else profile.ssd_write), seconds)
            stats[f"priced_{field}_{alternative}"] += 1
    count = event["written"]
    free = event["free"]
    needed = max(0, count - free)
    if event["proposed"] != len(incoming) or event["planned_evictions"] != needed or not 0 <= needed <= len(victims):
        raise ValueError("Retention decision has an invalid prefix length or slot count")
    benefit = sum(row[3] for row in incoming[:count])
    writes = sum(row[4] for row in incoming[:count])
    displaced = sum(row[3] for row in victims[:needed])
    net = benefit - writes - displaced
    for name, expected in (("benefit_seconds", benefit), ("write_seconds", writes),
                           ("displaced_seconds", displaced), ("net_seconds", net)):
        close(expected, event[name])
    for length in range(min(len(incoming), len(victims) + free) + 1):
        candidate = (sum(row[3] for row in incoming[:length]) - sum(row[4] for row in incoming[:length])
                     - sum(row[3] for row in victims[:max(0, length - free)]))
        if candidate - net > 1e-12:
            raise ValueError("The actual retention plan left a better-priced prefix unused")
    stats["victim_ssd_selected"] += sum(row[2] == "ssd" for row in victims[:needed])
    stats["incoming_ssd_selected"] += sum(row[2] == "ssd" for row in incoming[:count])
    return [row[0] for row in incoming[:count]], [row[0] for row in victims[:needed]]


def verify_serial(groups):
    """核对各 groups 的 Linux 锁为同一 inode，服务时间窗按录制顺序不重叠。"""
    identity = groups[0]["lock_identity"]
    last = 0
    for group in groups:
        start, stop = group["service_window_ns"]
        if group["lock_identity"] != identity or not last < start < stop:
            raise ValueError("SSD group service windows overlap or use different GPU lock files")
        last = stop


def replay(directory, records, profile=None):
    """对 directory 的 records 重放原生 SSD 作业，返回逐请求读回链和真实分支证据。"""
    marker = read(directory / "complete.json")
    lock_path = directory / "gpu-lock.json"
    if sha(lock_path) != marker["evidence"]["gpu-lock.json"]:
        raise ValueError(f"SSD group lacks verified GPU lock evidence: {directory}")
    lock = read(lock_path)
    if not any(line.startswith("lock:") and {"FLOCK", "WRITE"} <= set(line.split())
               for line in lock["fdinfo"].splitlines()):
        raise ValueError(f"SSD process did not hold the exclusive GPU lock: {directory}")
    audit = rows(directory / "audit.jsonl")
    begin, storage, context = audit[:3]
    shutdown = audit[-1]
    if ((begin["kind"], storage["kind"], context["kind"], shutdown["kind"]) !=
            ("init", "fs_init", "context", "ssd_shutdown") or shutdown["success"] is not True
            or storage["direct_io"] is not True or storage["ssd_write"] != "eager"
            or storage["existing_bytes"] != 0 or storage["block_bytes"] != BLOCK_BYTES
            or storage["capacity"] != marker["context"]["device"]["capacity"]):
        raise ValueError(f"Unexpected O_DIRECT configuration or shutdown: {directory}")
    resident, reserved, slots = {}, {}, {}
    disk, evicted, jobs = {}, set(), {}
    parents, depths = {}, {}
    stats, requests = Counter(), {}
    witnesses = {}
    active = None
    previous = begin["time_ns"]
    for event in audit[1:-1]:
        ts, kind = event["time_ns"], event["kind"]
        if ts < previous or event["session"] != begin["session"]:
            raise ValueError(f"Nonmonotonic or mixed SSD sessions: {directory}")
        previous = ts
        if kind in ("fs_init", "context", "retention_profile"):
            continue
        request_id = event["request_id"]
        if kind == "begin":
            if active is not None and (not active["closed"] or reserved or jobs or active["pricing"] is not None):
                raise ValueError(f"Previous request still owns unfinished SSD work: {request_id}")
            if request_id in requests:
                raise ValueError("Duplicate request in SSD audit")
            active = dict(keys=event["keys"], gpu_store=set(), gpu_load=set(), drive_read=Counter(),
                          drive_write=set(), closed=False, pricing=None, lookups=0, priced_ssd=0)
            requests[request_id] = active
            for depth, key in enumerate(event["keys"], 1):
                parent = event["keys"][depth - 2] if depth > 1 else None
                if parents.setdefault(key, parent) != parent or depths.setdefault(key, depth) != depth:
                    raise ValueError("Chained key changed its parent or depth")
            continue
        if active is not requests.get(request_id):
            raise ValueError(f"Cross-request SSD activity: {request_id}")
        keys = event.get("keys", [])
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate keys in {kind}")
        if kind in ("store", "promotion"):
            if kind == "store" and profile is not None:
                pricing = active["pricing"]
                if pricing is None and keys:
                    raise ValueError("GPU write bypassed time_value")
                if pricing is not None and pricing != (keys, event["evicted"]):
                    raise ValueError(f"SSD actual CPU allocation differs from pricing: {request_id}")
                active["pricing"] = None
            for key in event["evicted"]:
                if key not in disk:
                    raise ValueError(f"CPU victim had no published disk replica: {key}")
                evicted.add(key)
                del slots[resident.pop(key)]
            if kind == "promotion":
                key = event["key"]
                if not event["accepted"] or key not in disk or key not in evicted:
                    raise ValueError(f"Promotion without a prior published write and eviction: {key}")
                keys, target = [key], [event["slot"]]
            else:
                target = event["slots"]
            for key, slot in zip(keys, target, strict=True):
                if (key not in active["keys"] or key in resident or key in reserved or slot in slots
                        or not 0 <= slot < begin["capacity"]):
                    raise ValueError(f"Invalid native slot allocation {key} -> {slot}")
                reserved[key] = (slot, kind)
                slots[slot] = key
        elif kind == "stored":
            for key in keys:
                slot, source = reserved.pop(key)
                if source != "store":
                    raise ValueError(f"GPU DMA completed a non-store slot: {key}")
                resident[key] = slot
            active["gpu_store"].update(keys)
        elif kind in ("fs_submit", "fs_complete"):
            direction, reason, target = (event[name] for name in ("direction", "reason", "slots"))
            signature = (request_id, direction, reason, keys, target)
            if direction == "store" and reason == "eager":
                if not keys or any(key in disk or resident[key] != slot for key, slot in zip(keys, target, strict=True)):
                    raise ValueError("Eager write did not have a unique live CPU source")
            elif direction == "load" and reason == "promotion":
                if not keys or any(key not in disk or reserved[key] != (slot, "promotion")
                                   for key, slot in zip(keys, target, strict=True)):
                    raise ValueError("SSD load lacked a published file or allocated promotion slot")
            else:
                raise ValueError(f"Unexpected SSD path: {direction}, {reason}")
            token = event["job"]
            if kind == "fs_submit":
                if token in jobs:
                    raise ValueError(f"Duplicate SSD submission {token}")
                jobs[token] = signature
            else:
                if jobs.pop(token) != signature or event["success"] is not True or not 0 < event["transfer_seconds"] < math.inf:
                    raise ValueError(f"SSD job failed or completion mismatched: {token}")
                if direction == "store":
                    disk.update(dict.fromkeys(keys, event))
                    active["drive_write"].update(keys)
                else:
                    for key in keys:
                        resident[key] = reserved.pop(key)[0]
                    active["drive_read"].update(keys)
        elif kind == "lookup":
            first, tail = (event[name] for name in ("gpu", "external"))
            match = active["keys"][first // 64:(first + tail) // 64]
            if (event["estimate"] is not None or event["admitted"] != tail or len(match) * 64 != tail
                    or any(key not in resident for key in match)):
                raise ValueError("Lookup lacked a ready continuous CPU prefix after promotion")
            active["lookups"] += 1
            active["matched"] = set(match)
        elif kind == "loaded":
            if set(keys) & active["gpu_load"] or any(key not in resident for key in keys):
                raise ValueError("GPU load lacked a completed CPU source")
            active["gpu_load"].update(keys)
        elif kind == "retain_value":
            if profile is None or active["pricing"] is not None:
                raise ValueError("Unexpected or ambiguous time_value decision")
            counts = Counter()
            pending = {key for job in jobs.values() if job[1] == "store" for key in job[3]}
            active["pricing"] = verify_value(event, profile, parents, depths, disk, resident, pending, counts)
            active["priced_ssd"] += counts["priced_incoming_ssd"] + counts["priced_victim_ssd"]
            if request_id != next(iter(requests)):
                stats.update(counts)
                for field, name, selected in (("incoming", "incoming", active["pricing"][0]),
                                              ("victims", "victims", active["pricing"][1]),
                                              ("incoming", "incoming_selected", active["pricing"][0]),
                                              ("victims", "victims_selected", active["pricing"][1])):
                    priced = next((row for row in event[field] if row[2] == "ssd"
                                   and (field == name or row[0] in selected)), None)
                    if name not in witnesses and priced is not None:
                        key, probability, _, gain, *write = priced
                        published = disk[key]
                        witnesses[name] = dict(key=key, request_id=request_id, decision_ns=ts,
                                               published_ns=published["time_ns"], publish_job=published["job"],
                                               publish_request_id=published["request_id"], depth=depths[key],
                                               probability=probability, value_seconds=gain, selected=key in selected,
                                               **({"write_seconds": write[0]} if write else {}))
        elif kind == "ssd_request_closed":
            active["closed"] = True
        elif kind not in ("retain", "ssd_slot_wait"):
            raise ValueError(f"Unexpected event in SSD audit: {kind}")
    if reserved or jobs or list(requests) != [row["request_id"] for row in records]:
        raise ValueError("SSD audit did not drain or differs from captured request sequence")
    for row, request in zip(records, requests.values(), strict=True):
        if (not request["closed"] or request["lookups"] != 1 or request["pricing"] is not None
                or request["gpu_load"] != request["matched"]):
            raise ValueError(f"Incomplete CPU→GPU chain for {row['request_id']}")
        reads = request["drive_read"]
        physical = reads.total()
        for field, value in (("cpu_write_blocks", len(request["gpu_store"])), ("cpu_read_blocks", len(request["gpu_load"])),
                             ("ssd_read_blocks", physical), ("ssd_write_blocks", len(request["drive_write"]))):
            if row[field] != value:
                raise ValueError(f"SSD audit and transfer counters differ: {row['request_id']}, {field}")
        if row["index"]:
            forwarded = request["gpu_load"] & reads.keys()
            stats["full_ssd_chain_requests"] += bool(forwarded)
            stats["full_ssd_chain_blocks"] += len(forwarded)
            stats["physical_ssd_read_blocks"] += physical
            stats["repeated_ssd_read_blocks"] += physical - len(reads)
            stats["ssd_read_not_forwarded_blocks"] += len(reads) - len(forwarded)
            stats["published_replica_pricing_requests"] += bool(request["priced_ssd"])
    paths = list((directory / "ssd").rglob("*.bin"))
    if len(paths) != len(disk) or any(path.stat().st_size != BLOCK_BYTES for path in paths):
        raise ValueError(f"O_DIRECT file layout differs from successfully published replicas: {directory}")
    stats["published_files"] = len(disk)
    return dict(stats) | dict(first_replica_pricing=witnesses, lock_identity=[lock["device"], lock["inode"]],
                              service_window_ns=[begin["time_ns"], shutdown["time_ns"]])
