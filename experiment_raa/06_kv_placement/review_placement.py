"""从真实原始指标、前缀访问与异步块事件复核同容量三级放置实验。"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from memmgr.short_term.placement import PlacementCosts
from memmgr.short_term.reuse import History, Predictor
from run_placement import calibrate, counters, digest, save


def rows(path):
    """读取 path 的完整 JSONL，拒绝空记录或缺失结尾换行。"""
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def audit(directory, costs, model, formal_ids):
    """重建 directory 的真实块生命周期，核对 costs、model 与 formal_ids 对应的传输。"""
    events = rows(directory / "audit.jsonl")
    cpu, ssd, jobs, features = {}, set(), {}, {}
    written, removed, promoted, chained = set(), set(), set(), set()
    history, decisions, policy_times = History(), Counter(), {}
    formal = Counter()
    snapshots, capacity, block_bytes = 0, None, None
    reserved, frozen_targets = {}, {}
    max_cpu, max_ssd = 0, 0
    current = None
    for event in events:
        kind = event["kind"]
        if kind == "init":
            cpu_capacity = event["capacity"]
            placement = event["policy"] == "placement"
        elif kind == "fs_init":
            if not event["direct_io"] or event["existing_bytes"]:
                raise ValueError("Require a fresh real O_DIRECT cache")
            capacity, block_bytes = event["capacity"], event["block_bytes"]
        elif kind == "begin":
            current = event["request_id"]
            history.observe(event["keys"], 64, set(cpu) | ssd)
            actual = [history.features(key) for key in event["keys"]]
            if history.clock != event["clock"] or actual != event["features"]:
                raise ValueError("Online prefix history differs from reconstruction")
            features[current] = dict(zip(event["keys"], actual))
            if current in formal_ids:
                policy_times[current] = event["policy_seconds"]
        elif kind == "placement_context" and costs is not None:
            if event["context"] != costs.profile["context"]:
                raise ValueError("Evaluation and calibration runtime identities differ")
        elif kind == "placement":
            keys = list(event["targets"])
            expected = model.score([features[current][key] for key in keys])
            if any(abs(event["probabilities"][key] - probability) > 1e-12 for key, probability in zip(keys, expected)):
                raise ValueError("Placement probability differs from the frozen predictor")
            expected_gap = model.delay([features[current][key] for key in keys])
            if any(abs(event["reuse_intervals"][key] - gap) > 1e-12 for key, gap in zip(keys, expected_gap)):
                raise ValueError("Predicted reuse interval differs from the frozen regressor")
            missing = event["missing"]
            probability = [event["probabilities"][key] for key in missing]
            interval = sum(event["reuse_intervals"][key] for key in missing) / len(missing) if missing else 0.0
            if abs(interval - event["incoming_interval"]) > 1e-12:
                raise ValueError("Residence interval does not match the predicted prefix interval")
            for tier, raw in event["raw_seconds"].items():
                computed = costs.gain(probability, event["positions"], tier,
                                      event["cpu_writes"] if tier == "cpu" else event["ssd_staging_writes"],
                                      event["ssd_writes"] if tier == "ssd" else 0)
                if abs(raw - computed) > 1e-12:
                    raise ValueError("Placement benefit differs from calibrated path costs")
            candidates = event["candidates_seconds"]
            if "cpu" in candidates and abs(candidates["cpu"] - event["raw_seconds"]["cpu"] + event["cpu_loss"]) > 1e-12:
                raise ValueError("CPU replacement loss was not applied")
            if "ssd" in candidates and abs(candidates["ssd"] - event["raw_seconds"]["ssd"] + event["staging_loss"] + event["ssd_loss"]) > 1e-12:
                raise ValueError("SSD replacement or staging loss was not applied")
            selected = max(candidates, key=candidates.get)
            if candidates[selected] <= .0005:
                selected = "drop"
            if selected != event["target"] or set(event["targets"].values()) - {"cpu", "ssd", "drop"}:
                raise ValueError("Placement did not choose its maximum measured net benefit")
            if event["request_id"] in formal_ids:
                decisions[event["target"]] += 1
                policy_times[current] += event["policy_seconds"]
            frozen_targets[history.clock] = event["targets"]
        elif kind == "store":
            for key in event["evicted"]:
                if cpu.pop(key) != 0:
                    raise ValueError("Evicted a pinned CPU block")
                removed.add(key)
            for key in event["keys"]:
                if key in cpu:
                    raise ValueError("Duplicate CPU slot allocation")
                cpu[key] = -1
            if event["request_id"] in formal_ids:
                formal["cpu_write_bytes"] += len(event["keys"]) * block_bytes
        elif kind == "stored":
            for key in event["keys"]:
                if cpu[key] == -1:
                    cpu[key] = 0
        elif kind == "promotion":
            if not event["accepted"] or event["key"] in cpu:
                raise ValueError("Invalid native SSD promotion allocation")
            for key in event["evicted"]:
                if cpu.pop(key) != 0:
                    raise ValueError("Promotion evicted a pinned CPU block")
                removed.add(key)
            cpu[event["key"]] = -1
        elif kind == "fs_evict":
            protected = {key for job in jobs.values() for key in job["keys"]}
            for key in event["keys"]:
                if key in protected:
                    raise ValueError("Removed a file participating in actual I/O")
                ssd.remove(key)
        elif kind == "fs_submit":
            if event["job"] in jobs:
                raise ValueError("Duplicate in-flight SSD job")
            jobs[event["job"]] = event
            if event["direction"] == "store":
                for key in event["keys"]:
                    if cpu[key] == -1:
                        cpu[key] = 0
                    cpu[key] += 1
            else:
                if not set(event["keys"]) <= ssd:
                    raise ValueError("SSD load lacks a completed real file")
        elif kind == "fs_reserve":
            reserved[event["job"]] = event["keys"]
        elif kind == "fs_complete":
            submitted = jobs.pop(event["job"])
            if (not event["success"] or event["direction"] != submitted["direction"]
                    or event["keys"] != submitted["keys"]):
                raise ValueError("SSD completion differs from its actual submission")
            if event["direction"] == "store":
                ssd.update(event["keys"])
                written.update(event["keys"])
                for key in event["keys"]:
                    cpu[key] -= 1
                    if cpu[key] < 0:
                        raise ValueError("Unbalanced SSD source reference")
            else:
                for key in event["keys"]:
                    if cpu[key] != -1:
                        raise ValueError("SSD destination is not a pending primary write")
                    cpu[key] = 0
                    if key in written and key in removed:
                        promoted.add((event["request_id"], key))
            if event["request_id"] in formal_ids:
                formal["ssd_" + ("read_bytes" if event["direction"] == "load" else "write_bytes")] += len(event["keys"]) * block_bytes
                if event["direction"] == "store":
                    formal["ssd_new_file_bytes"] += len(reserved.pop(event["job"])) * block_bytes
        elif kind == "load":
            for key in event["keys"]:
                if cpu[key] < 0:
                    raise ValueError("GPU attempted to read an unfinished primary write")
                cpu[key] += 1
        elif kind == "loaded":
            for key in event["keys"]:
                cpu[key] -= 1
                if cpu[key] < 0:
                    raise ValueError("Unbalanced GPU source reference")
                if (event["request_id"], key) in promoted:
                    chained.add((event["request_id"], key))
            if event["request_id"] in formal_ids:
                formal["cpu_read_bytes"] += len(event["keys"]) * block_bytes
        elif kind == "cpu_release":
            for key in event["keys"]:
                if cpu.pop(key) != 0:
                    raise ValueError("Released a pinned or incomplete primary block")
                removed.add(key)
        elif kind == "residency":
            if set(event["cpu"]) != set(cpu) or set(event["ssd"]) != ssd or any(cpu.values()):
                raise ValueError("Audited state differs from the native idle residency")
            if event["allocated"] != len(ssd) * block_bytes or event["allocated"] > capacity:
                raise ValueError("SSD residency exceeds or differs from the physical quota")
            if len(cpu) + event["cpu_free"] != cpu_capacity:
                raise ValueError("Native CPU block-pool accounting differs")
            if placement and event["clock"] in frozen_targets:
                for key, target in frozen_targets[event["clock"]].items():
                    if (target == "cpu" and key not in cpu or target == "ssd" and (key not in ssd or key in cpu)
                            or target == "drop" and (key in cpu or key in ssd)):
                        raise ValueError("Frozen placement differs from the actual idle tier")
                if len(cpu) > cpu_capacity - costs.max_blocks:
                    raise ValueError("Placement consumed its reserved CPU staging space")
            snapshots += 1
            max_cpu, max_ssd = max(max_cpu, len(cpu)), max(max_ssd, len(ssd))
        if capacity is not None and len(cpu) > cpu_capacity:
            raise ValueError("Actual CPU allocation exceeded its capacity")
    if jobs or any(cpu.values()) or not snapshots:
        raise ValueError("Incomplete real transfers or missing idle residency evidence")
    sorted_times = sorted(policy_times.values())
    return dict(actions=dict(decisions), complete_chain_blocks=len(chained), snapshots=snapshots,
                final_cpu_blocks=len(cpu), final_ssd_blocks=len(ssd),
                max_idle_cpu_blocks=max_cpu, max_idle_ssd_blocks=max_ssd,
                policy_p95_ms=sorted_times[math.ceil(len(sorted_times) * .95) - 1] * 1000,
                **{name: formal[name] for name in ("cpu_read_bytes", "cpu_write_bytes", "ssd_read_bytes", "ssd_write_bytes", "ssd_new_file_bytes")})


def review(root):
    """从 root 的原始记录复核输入、容量、成本、传输和输出，写入可重复复算的统计。"""
    root = Path(root)
    profile = json.loads((root / "costs.json").read_text())
    recalculated = calibrate(root / "collect", root / "costs-recomputed.json")
    (root / "costs-recomputed.json").unlink()
    if recalculated != profile:
        raise ValueError("Frozen costs differ from raw path measurements")
    metadata = json.loads((root / "model/model.json").read_text())
    predictor = Predictor(root / "model", metadata["identity"])
    costs = PlacementCosts(root / "costs.json", profile["context"])
    training_ids = {row["query_id"] for row in json.loads((root / "collect-trace.json").read_text())}
    warmup = json.loads((root / "collect/warmup.json").read_text())
    training_ids.add(warmup["query_id"])
    environment = json.loads((root / "environment.json").read_text())
    if any(digest(ROOT / name) != value for name, value in environment["source"].items()):
        raise ValueError("Current implementation differs from the recorded experiment")
    groups, differences, conditions = [], [], []
    for epoch in range(2):
        plan = json.loads((root / f"evaluate-trace-{epoch}.json").read_text())
        if training_ids & {row["query_id"] for row in plan}:
            raise ValueError("Training and independent evaluation questions overlap")
        reference = rows(root / f"e{epoch}-lru/requests.jsonl")
        for mode in ("lru", "learned", "placement"):
            directory = root / f"e{epoch}-{mode}"
            records = rows(directory / "requests.jsonl")
            actual_warmup = json.loads((directory / "warmup.json").read_text())
            if (actual_warmup["query_id"] != warmup["query_id"]
                    or actual_warmup["prompt_sha256"] != warmup["prompt_sha256"]):
                raise ValueError("Strategies did not use the same training-only warmup")
            server = json.loads((directory / "server.json").read_text())
            kv = server["kv_cache"]
            conditions.append((kv["gpu_bytes"], kv["cpu_bytes"], kv["ssd_bytes"], kv["batch_invariant"]))
            if len(records) != len(plan) or len(reference) != len(plan):
                raise ValueError("Incomplete evaluation requests")
            before = counters((directory / "initial.prom").read_text())
            paired, times, totals = [], [], Counter()
            for index, (record, original, expected) in enumerate(zip(records, reference, plan)):
                after = counters((directory / f"{index:04}.prom").read_text())
                delta = {key: after[key] - before.get(key, 0) for key in after}
                if (delta != record["metrics"] or delta["vllm:request_success_total"] != 1
                        or delta["vllm:num_preemptions_total"] != 0 or min(delta.values()) < 0
                        or record["index"] != index
                        or record["prompt_sha256"] != hashlib.sha256(json.dumps(expected["prompt_token_ids"]).encode()).hexdigest()):
                    raise ValueError("Raw counters or frozen input differ")
                gpu = delta["vllm:prefix_cache_hits_total"]
                external = delta.get("vllm:external_prefix_cache_hits_total", 0)
                computed = delta["vllm:request_prefill_kv_computed_tokens_sum"]
                if gpu + external + computed != len(expected["prompt_token_ids"]):
                    raise ValueError("Prompt tokens do not balance across GPU, offload and compute")
                if record["generated_tokens"] != original["generated_tokens"] or record["answer"] != original["answer"]:
                    differences.append(dict(epoch=epoch, mode=mode, index=index))
                paired.append((original["ttft_seconds"] - record["ttft_seconds"]) * 1000)
                times.append(record["ttft_seconds"] * 1000)
                totals.update(computed_tokens=computed, recovered_tokens=external,
                              cpu_write_bytes=delta.get("vllm:kv_offload_store_bytes_total", 0),
                              cpu_read_bytes=delta.get("vllm:kv_offload_load_bytes_total", 0),
                              ssd_write_bytes=delta.get("vllm:kv_offload_tiering_write_bytes_total", 0),
                              ssd_read_bytes=delta.get("vllm:kv_offload_tiering_read_bytes_total", 0))
                before = after
            checked = audit(directory, costs, predictor, {row["request_id"] for row in records})
            if any(checked[name] != totals[name] for name in ("cpu_read_bytes", "cpu_write_bytes", "ssd_read_bytes", "ssd_write_bytes")):
                raise ValueError("Completed real block transfers differ from the published byte counters")
            phases = []
            for phase in ("stable", "scan", "shift", "long_gap"):
                indices = [index for index, row in enumerate(plan) if row["phase"] == phase]
                phases.append(dict(phase=phase, requests=len(indices),
                                   ttft_median_ms=statistics.median(times[index] for index in indices),
                                   paired_saved_median_ms=statistics.median(paired[index] for index in indices)))
            groups.append(dict(epoch=epoch, mode=mode, requests=len(records), phases=phases,
                               ttft_median_ms=statistics.median(times), ttft_p95_ms=sorted(times)[math.ceil(len(times) * .95) - 1],
                               paired_saved_median_ms=statistics.median(paired), paired_wins=sum(value > 0 for value in paired),
                               computed_tokens=totals["computed_tokens"], recovered_tokens=totals["recovered_tokens"], **checked))
    if len(set(conditions)) != 1:
        raise ValueError("Comparison did not use identical physical tier capacities and numeric modes")
    result = dict(groups=groups, output_mismatches=differences, capacities=conditions[0],
                  model_sha256=predictor.sha256, costs_sha256=costs.sha256)
    save(root / "summary.json", result)
    if differences:
        raise ValueError("Generation differs across the real policies")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    review(parser.parse_args().root)
