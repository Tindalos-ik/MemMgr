"""从原始请求、传输生命周期和指标快照复核 GBDT 策略、时间隔离与配对性能。"""

import argparse
from collections import defaultdict, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np
from prometheus_client.parser import text_string_to_metric_families

from reuse_features import History, Predictor, digest, read_rows, save
from train_reuse import dataset, fit


def counters(path):
    """读取 path 原始快照，按名称汇总 counter 和 histogram sum/count。"""
    result = {}
    for family in text_string_to_metric_families(Path(path).read_text()):
        for sample in family.samples:
            if sample.name.endswith(("_total", "_sum", "_count")):
                result[sample.name] = result.get(sample.name, 0) + sample.value
    return result


def audit_cache(directory, predictor):
    """复算 directory 的审计生命周期及模型决定；predictor 为冻结模型，LRU 时为空。"""
    events = read_rows(directory / "audit.jsonl")
    if events[0]["kind"] != "init":
        raise ValueError("Missing manager initialization")
    initial = events[0]
    capacity, block_bytes = initial["capacity"], initial["bytes_per_block"]
    if capacity != 36 or block_bytes != 64 * 28672:
        raise ValueError("Unexpected physical CPU capacity or block size")
    if initial["model_sha256"] != (predictor.sha256 if predictor else None):
        raise ValueError("Runtime model digest changed")
    history, cache, evictable, requests, lives = History(), {}, OrderedDict(), {}, []
    pending_eviction = None
    for event in events[1:]:
        kind = event["kind"]
        req_id = event.get("request_id")
        if kind == "begin":
            if req_id in requests:
                raise ValueError("Duplicate request identity")
            vector = history.observe(event["root"], event["features"][-1])
            score = predictor.score([vector])[0] if predictor else None
            if (vector != event["features"] or event["clock"] != history.clock
                    or score != event["score"] or event["history_size"] != len(history.entries)
                    or event["accepted"] != (predictor is None or score >= predictor.threshold)):
                raise ValueError("Online history, score or admission differs from reconstruction")
            resident = [key for key in event["keys"] if key in cache and cache[key]["ref"] >= 0]
            if resident != event["resident"]:
                raise ValueError("Ready CPU snapshot differs from completed stores")
            requests[req_id] = dict(begin=event, lookup=[], store_keys=[], load_keys=[],
                                    policy_seconds=event["policy_seconds"])
        elif kind == "lookup":
            state = requests[req_id]
            state["lookup"].append(event)
            for key in reversed(state["begin"]["keys"]):
                if key in evictable:
                    evictable.move_to_end(key)
        elif kind == "evict":
            if initial["mode"] != "learned" or pending_eviction is not None:
                raise ValueError("Unexpected learned eviction")
            groups = {}
            current_root = requests[req_id]["begin"]["root"]
            for rank, key in enumerate(evictable):
                block = cache[key]
                if key not in event["protected"] and block["root"] != current_root:
                    groups.setdefault(block["root"], []).append((key, block["position"], rank))
            roots = list(groups)
            vectors = [history.features(root) for root in roots]
            scores = predictor.score(vectors)
            expected_candidates = [dict(root=root, features=vector, score=score,
                                        keys=[key for key, _, _ in groups[root]])
                                   for root, vector, score in zip(roots, vectors, scores)]
            ranked = sorted(zip(roots, scores), key=lambda item: (item[1], min(k[2] for k in groups[item[0]])))
            chosen = [key for root, _ in ranked
                      for key, _, _ in sorted(groups[root], key=lambda item: -item[1])][:event["n"]]
            if expected_candidates != event["candidates"] or chosen != event["keys"] or len(chosen) != event["n"]:
                raise ValueError("Eviction differs from frozen scores, LRU ties or tail-first order")
            pending_eviction = event
            requests[req_id]["policy_seconds"] += event["policy_seconds"]
        elif kind == "store":
            state = requests[req_id]
            begin = state["begin"]
            if event["accepted"] != begin["accepted"] or any(key not in begin["keys"] for key in event["offered"]):
                raise ValueError("Store admission or offered keys changed")
            selected = event["offered"] if begin["accepted"] else []
            new = [key for key in selected if key not in cache]
            n = max(0, len(new) - (capacity - len(cache)))
            if initial["mode"] == "learned":
                expected_evicted = pending_eviction["keys"] if pending_eviction else []
                if pending_eviction and pending_eviction["request_id"] != req_id:
                    raise ValueError("Eviction and store request differ")
                pending_eviction = None
            else:
                expected_evicted = [key for key in evictable if key not in selected][:n]
            if event["keys"] != new or event["evicted"] != expected_evicted or len(expected_evicted) != n:
                raise ValueError("Native allocation or write filtering differs from policy")
            for key in event["evicted"]:
                block = cache.pop(key)
                if block["ref"] != 0 or key not in evictable:
                    raise ValueError("Attempted to evict an in-flight block")
                del evictable[key]
                block["evicted"] = True
            for key in new:
                block = dict(key=key, root=begin["root"], position=begin["keys"].index(key),
                             request_id=req_id, ref=-1, used=False, evicted=False)
                cache[key] = block
                lives.append(block)
            if len(cache) > capacity or len(cache) != event["resident_blocks"]:
                raise ValueError("CPU resident capacity exceeded")
            if new:
                for key in reversed(begin["keys"]):
                    if key in evictable:
                        evictable.move_to_end(key)
        elif kind == "stored":
            for key in event["keys"]:
                block = cache[key]
                if block["ref"] != -1 or block["request_id"] != req_id:
                    raise ValueError("Invalid store completion")
                block["ref"] = 0
                evictable[key] = None
            requests[req_id]["store_keys"].extend(event["keys"])
        elif kind == "load":
            for key in event["keys"]:
                block = cache[key]
                if block["ref"] < 0:
                    raise ValueError("Read before store completion")
                if block["ref"] == 0:
                    del evictable[key]
                block["ref"] += 1
        elif kind == "loaded":
            for key in event["keys"]:
                block = cache[key]
                if block["ref"] <= 0:
                    raise ValueError("Load completion without a matching pinned block")
                block["ref"] -= 1
                block["used"] = True
                if block["ref"] == 0:
                    evictable[key] = None
            requests[req_id]["load_keys"].extend(event["keys"])
        elif kind == "end":
            if event["resident_blocks"] != len(cache) or event["active_requests"]:
                raise ValueError("Unfinished manager state")
        else:
            raise ValueError(f"Unknown audit event: {kind}")
    if pending_eviction or any(block["ref"] != 0 for block in cache.values()):
        raise ValueError("Unfinished eviction or transfer")
    formal_lives = [block for block in lives if block["request_id"].startswith("cmpl-reuse-")]
    accounting = dict(written_bytes=len(formal_lives) * block_bytes,
                      unused_evicted_bytes=sum(block["evicted"] and not block["used"] for block in formal_lives) * block_bytes,
                      used_evicted_bytes=sum(block["evicted"] and block["used"] for block in formal_lives) * block_bytes,
                      used_resident_bytes=sum(not block["evicted"] and block["used"] for block in formal_lives) * block_bytes,
                      pending_observation_bytes=sum(not block["evicted"] and not block["used"] for block in formal_lives) * block_bytes)
    if accounting["written_bytes"] != sum(value for key, value in accounting.items() if key != "written_bytes"):
        raise ValueError("Write lifetime accounting mismatch")
    return requests, accounting


def verify_run(root, name, mode, plan, predictor):
    """复核 root/name 的 mode 回放与 plan，使用 predictor 验证模型决定和原始指标。"""
    directory = root / name
    environment = json.loads((root / "environment.json").read_text())
    server = json.loads((directory / "server.json").read_text())
    if server["context"] != environment["context"] or server["mode"] != mode:
        raise ValueError("Runtime context changed")
    audits, accounting = audit_cache(directory, predictor)
    rows = read_rows(directory / "requests.jsonl")
    if len(rows) != len(plan):
        raise ValueError("Missing replay requests")
    previous = counters(directory / "initial.prom")
    for index, (row, expected) in enumerate(zip(rows, plan)):
        tokens = expected["prompt_token_ids"]
        matches = [key for key in audits if re.fullmatch(re.escape(row["request_id"]) + r"-[0-9a-f]{8}", key)]
        if len(matches) != 1 or row["request_id"] != f"cmpl-reuse-{index}-0":
            raise ValueError("Missing or ambiguous engine request")
        audit = audits.pop(matches[0])
        begin = audit["begin"]
        if (any(row[key] != value for key, value in expected.items() if key != "prompt_token_ids")
                or row["prompt_sha256"] != hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                or begin["prompt_sha256"] != row["prompt_sha256"]
                or begin["root"] != hashlib.sha256(json.dumps(tokens[:64]).encode()).hexdigest()
                or len(begin["keys"]) != len(tokens) // 64):
            raise ValueError("Replay input identity mismatch")
        current = counters(directory / f"{index:04}.prom")
        delta = {key: value - previous.get(key, 0) for key, value in current.items()}
        if delta != row["metrics"] or any(value < 0 for value in delta.values()):
            raise ValueError("Raw counter snapshots differ from saved deltas")
        previous = current
        lookups = audit["lookup"]
        if len(lookups) != 1 or lookups[0]["cpu_tokens"] is None:
            raise ValueError("Unexpected repeated or deferred lookup")
        gpu, cpu = lookups[0]["gpu_tokens"], lookups[0]["cpu_tokens"]
        if (gpu != delta["vllm:prefix_cache_hits_total"]
                or cpu != delta.get("vllm:external_prefix_cache_hits_total", 0)
                or gpu + cpu + delta["vllm:request_prefill_kv_computed_tokens_sum"] != len(tokens)
                or len(audit["store_keys"]) * 64 * 28672 != delta.get("vllm:kv_offload_store_bytes_total", 0)
                or len(audit["load_keys"]) * 64 * 28672 != delta.get("vllm:kv_offload_load_bytes_total", 0)
                or cpu * 28672 != delta.get("vllm:kv_offload_load_bytes_total", 0)
                or delta["vllm:request_success_total"] != 1
                or delta["vllm:prefix_cache_queries_total"] != len(tokens)
                or delta["vllm:num_preemptions_total"] != 0
                or row["usage"]["prompt_tokens"] != len(tokens)
                or row["usage"]["completion_tokens"] != environment["context"]["output_tokens"]
                or row["finish_reason"] != "length"
                or len(row["generated_tokens"]) != environment["context"]["output_tokens"]
                or len(row["token_logprobs"]) != environment["context"]["output_tokens"]):
            raise ValueError(f"Transfer, compute or output accounting mismatch: {name}/{index}")
        row.update(mode=mode, run=name, gpu_tokens=gpu, cpu_tokens=cpu,
                   resident_tokens=len(begin["resident"]) * 64,
                   previously_seen=begin["features"][0] > 1, policy_seconds=audit["policy_seconds"],
                   accepted=begin["accepted"], score=begin["score"],
                   server_ttft_seconds=delta["vllm:time_to_first_token_seconds_sum"])
    if any(not key.startswith("cmpl-warm-") for key in audits):
        raise ValueError("Unaccounted engine requests")
    if previous != counters(directory / "final.prom"):
        raise ValueError("Final snapshot mismatch")
    if sum(row["metrics"].get("vllm:kv_offload_store_bytes_total", 0) for row in rows) != accounting["written_bytes"]:
        raise ValueError("Physical writes differ from lifetime totals")
    read_rows(directory / "resources.jsonl")
    return rows, accounting


def percentile(values, fraction):
    """返回非空 values 的最近秩 fraction 分位数。"""
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def summarize(rows, reference):
    """汇总 rows 的计时和命中，用 reference 按轮次/请求配对计算节省。"""
    paired = [(reference[(row["epoch"], row["index"])]["ttft_seconds"] - row["ttft_seconds"]) * 1000 for row in rows]
    hot = [row for row in rows if row["hot"] and row["previously_seen"]]
    opportunity = sum(row["usage"]["prompt_tokens"] - row["gpu_tokens"] for row in hot)
    return dict(n=len(rows), hot_revisits=len(hot),
                ttft_median_ms=statistics.median(row["ttft_seconds"] for row in rows) * 1000,
                ttft_p95_ms=percentile([row["ttft_seconds"] for row in rows], 0.95) * 1000,
                server_ttft_median_ms=statistics.median(row["server_ttft_seconds"] for row in rows) * 1000,
                request_total_seconds=sum(row["total_seconds"] for row in rows),
                paired_saved_median_ms=statistics.median(paired), paired_wins=sum(value > 0 for value in paired),
                hot_resident_ratio=sum(row["resident_tokens"] for row in hot) / sum(row["usage"]["prompt_tokens"] for row in hot),
                hot_cpu_recovery_ratio=sum(row["cpu_tokens"] for row in hot) / opportunity,
                hot_missing_gpu_tokens=opportunity,
                store_bytes=sum(row["metrics"].get("vllm:kv_offload_store_bytes_total", 0) for row in rows),
                load_bytes=sum(row["metrics"].get("vllm:kv_offload_load_bytes_total", 0) for row in rows),
                policy_p95_ms=percentile([row["policy_seconds"] for row in rows], 0.95) * 1000,
                accepted=sum(row["accepted"] for row in rows), rejected=sum(not row["accepted"] for row in rows))


def review(root):
    """完整复核 root 的真实采集、时间隔离、冻结模型、六次评价与生成一致性。"""
    root = Path(root)
    environment = json.loads((root / "environment.json").read_text())
    metadata = json.loads((root / "model.json").read_text())
    predictor = Predictor(root)
    samples, split = dataset(root / "collect" / "audit.jsonl", environment["arguments"]["horizon"])
    if (samples != read_rows(root / "samples.jsonl") or split != metadata["split"]
            or digest(root / "samples.jsonl") != metadata["samples_sha256"]
            or digest(root / "collect" / "audit.jsonl") != metadata["source_audit_sha256"]):
        raise ValueError("Training data or chronological split changed")
    rebuilt, threshold, validation = fit(samples, environment["arguments"]["seed"])
    vectors = np.asarray([row["features"] for row in samples], dtype=np.float32)
    if (threshold != metadata["threshold"] or validation != metadata["metrics"]
            or rebuilt.get_params() != metadata["model_parameters"]
            or not np.array_equal(rebuilt.predict_proba(vectors), predictor.model.predict_proba(vectors))):
        raise ValueError("Frozen model differs from reproducible training")
    collect_plan = json.loads((root / "collect-trace.json").read_text())
    collected, _ = verify_run(root, "collect", "lru", collect_plan, None)
    all_blocks = {tuple(row["prompt_token_ids"][:64]) for row in collect_plan}
    rows, accounting = [], []
    for epoch in (0, 1):
        plan = json.loads((root / f"evaluate-trace-{epoch}.json").read_text())
        blocks = {tuple(row["prompt_token_ids"][:64]) for row in plan}
        if blocks & all_blocks:
            raise ValueError("Collection and independent evaluation prefixes overlap")
        all_blocks |= blocks
        for mode in ("lru", "admission", "learned"):
            verified, writes = verify_run(root, f"e{epoch}-{mode}", mode, plan, predictor if mode != "lru" else None)
            rows.extend(dict(**row, epoch=epoch) for row in verified)
            accounting.append(dict(epoch=epoch, mode=mode, **writes))
    reference = {(row["epoch"], row["index"]): row for row in rows if row["mode"] == "lru"}
    differences = []
    for row in rows:
        baseline = reference[(row["epoch"], row["index"])]
        if row["prompt_sha256"] != baseline["prompt_sha256"]:
            raise ValueError("Paired inputs differ")
        if row["generated_tokens"] != baseline["generated_tokens"] or row["answer"] != baseline["answer"]:
            differences.append(dict(epoch=row["epoch"], mode=row["mode"], index=row["index"],
                                    positions=[i for i, (a, b) in enumerate(zip(row["generated_tokens"], baseline["generated_tokens"])) if a != b]))
    groups = []
    for mode in ("lru", "admission", "learned"):
        selected = [row for row in rows if row["mode"] == mode]
        if mode != "lru" and (all(row["accepted"] for row in selected) or not any(row["accepted"] for row in selected)):
            raise ValueError("Evaluation did not exercise both write admission branches")
        groups.append(dict(mode=mode, phase="all", **summarize(selected, reference)))
        for phase in ("stable", "scan", "shift", "long_gap"):
            groups.append(dict(mode=mode, phase=phase, **summarize([row for row in selected if row["phase"] == phase], reference)))
    for path, expected in environment["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Delivered source differs from measured source")
    save(root / "summary.json", dict(collection_requests=len(collected), evaluation_requests=len(rows),
         model_sha256=predictor.sha256, threshold=predictor.threshold, validation=validation,
         groups=groups, write_lifetimes=accounting, output_mismatches=differences,
         epochs=[dict(epoch=epoch, mode=mode, **summarize([row for row in rows if row["epoch"] == epoch and row["mode"] == mode], reference))
                 for epoch in (0, 1) for mode in ("lru", "admission", "learned")]))
    (root / "validated-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(dict(groups=[group for group in groups if group["phase"] == "all"],
                          output_mismatches=len(differences)), indent=2), flush=True)
    if differences:
        raise RuntimeError("Generated output differs; inspect summary.json and token logprobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    review(parser.parse_args().root.resolve())
