"""从原始请求、指标与块事件复核三级 KV 恢复和生成一致性。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from prometheus_client.parser import text_string_to_metric_families


def rows(path):
    """读取 path 的完整 JSONL，拒绝缺失尾换行的记录。"""
    raw = path.read_text(encoding="utf-8")
    if not raw or not raw.endswith("\n"):
        raise ValueError(f"Incomplete JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def counters(path):
    """从 path 的原始 Prometheus 文本重算累计指标。"""
    result = {}
    for family in text_string_to_metric_families(path.read_text()):
        for sample in family.samples:
            if sample.name.endswith(("_total", "_sum", "_count")):
                result[sample.name] = result.get(sample.name, 0) + sample.value
    return result


def audit(path):
    """复核 path 中写入、淘汰、提升与 GPU 完成的块身份和事件顺序。"""
    written, evicted, promoted = set(), set(), set()
    pending, chained = {}, set()
    block_bytes, loads, stores = None, 0, 0
    for row in rows(path):
        kind = row["kind"]
        if kind == "fs_init":
            if not row["direct_io"]:
                raise ValueError("Buffered I/O cannot validate SSD reads")
            block_bytes = row["block_bytes"]
        elif kind in ("cpu_store", "promotion"):
            evicted.update(row["evicted"])
        elif kind == "fs_submit":
            if row["job"] in pending:
                raise ValueError("Duplicate SSD job")
            pending[row["job"]] = row
        elif kind == "fs_complete":
            submitted = pending.pop(row["job"])
            if (not row["success"] or row["keys"] != submitted["keys"]
                    or row["direction"] != submitted["direction"]):
                raise ValueError("SSD job completion differs from submission")
            if row["direction"] == "store":
                written.update(row["keys"])
                stores += len(row["keys"])
            else:
                if not set(row["keys"]) <= written & evicted:
                    raise ValueError("SSD read lacks a completed write and prior CPU eviction")
                promoted.update((row["request_id"], key) for key in row["keys"])
                loads += len(row["keys"])
        elif kind == "gpu_loaded":
            chained.update((row["request_id"], key) for key in row["keys"]
                           if (row["request_id"], key) in promoted)
    if pending:
        raise ValueError("Unfinished SSD transfers")
    return dict(fs_store_blocks=stores, fs_load_blocks=loads, complete_chain_blocks=len(chained),
                fs_store_payload_bytes=stores * block_bytes if block_bytes else 0,
                fs_load_payload_bytes=loads * block_bytes if block_bytes else 0)


def process_io(path):
    """从 path 的健康进程固定身份采样复算内核实际 I/O 字节增量。"""
    samples = rows(path)
    first = {(row["pid"], row["created_at"]): row["io"] for row in samples[0]["processes"]}
    last = {(row["pid"], row["created_at"]): row["io"] for row in samples[-1]["processes"]}
    if first.keys() != last.keys():
        raise ValueError("Sampled process identities changed")
    result = {}
    for field in ("read_bytes", "write_bytes"):
        differences = [last[key][field] - first[key][field] for key in first]
        if any(value < 0 for value in differences):
            raise ValueError("Kernel I/O counters decreased")
        result["process_" + field] = sum(differences)
    return result


def review(root):
    """从 root 原始数据复算分组、配对时延和路径证据，生成差异时失败退出。"""
    root = Path(root)
    groups, differences, derived = [], [], []
    for epoch in range(2):
        plan = json.loads((root / f"trace-{epoch}.json").read_text())
        unique = {hashlib.sha256(json.dumps(row["prompt_token_ids"][:end]).encode()).hexdigest()
                  for row in plan for end in range(64, len(row["prompt_token_ids"]) + 1, 64)}
        if len(unique) <= 64 + 36:
            raise ValueError("Working set must exceed combined GPU and CPU capacities")
        reference = rows(root / f"e{epoch}-gpu/requests.jsonl")
        for mode in ("gpu", "cpu", "ssd"):
            directory = root / f"e{epoch}-{mode}"
            records = rows(directory / "requests.jsonl")
            if len(records) != len(plan) or len(reference) != len(plan):
                raise ValueError("Request count mismatch")
            before = counters(directory / "initial.prom")
            recovered, computed, saved, logprob_max = 0, 0, [], 0.0
            for index, (row, original, prompt) in enumerate(zip(records, reference, plan)):
                after = counters(directory / f"{index:04}.prom")
                delta = {key: after[key] - before.get(key, 0) for key in after}
                if delta != row["metrics"] or row["index"] != index:
                    raise ValueError("Raw metrics or request ordering differ")
                count = len(prompt["prompt_token_ids"])
                if (row["client_request_id"] != f"ssd-{index}"
                        or row["prompt_sha256"] != hashlib.sha256(json.dumps(prompt["prompt_token_ids"]).encode()).hexdigest()):
                    raise ValueError("Client request identity or input digest differs")
                if row["usage"]["prompt_tokens"] != count or delta["vllm:prefix_cache_queries_total"] != count:
                    raise ValueError("Prompt token count mismatch")
                if any(value < 0 for value in delta.values()) or delta["vllm:request_success_total"] != 1:
                    raise ValueError("Invalid exclusive metric boundary")
                if row["generated_tokens"] != original["generated_tokens"] or row["answer"] != original["answer"]:
                    differences.append(dict(epoch=epoch, mode=mode, index=index,
                                            expected=original["generated_tokens"], actual=row["generated_tokens"]))
                probability_delta = max(abs(a - b) for a, b in zip(row["token_logprobs"], original["token_logprobs"]))
                if not math.isfinite(probability_delta):
                    raise ValueError("Non-finite logprob difference")
                logprob_max = max(logprob_max, probability_delta)
                gpu = delta["vllm:prefix_cache_hits_total"]
                compute = delta["vllm:request_prefill_kv_computed_tokens_sum"]
                external = delta.get("vllm:external_prefix_cache_hits_total", 0)
                if gpu + compute + external != count:
                    raise ValueError("GPU hits, external hits and prefill do not conserve prompt tokens")
                recovered += external
                computed += compute
                saved.append((original["ttft_seconds"] - row["ttft_seconds"]) * 1000)
                derived.append(dict(epoch=epoch, mode=mode, index=index, input_tokens=count,
                                    gpu_tokens=gpu, external_tokens=external, computed_tokens=compute))
                before = after
            path = audit(directory / "audit.jsonl") if mode != "gpu" else {}
            io = process_io(directory / "resources.jsonl")
            if mode == "ssd" and not path["complete_chain_blocks"]:
                raise ValueError("No complete SSD -> CPU -> GPU recovery after eviction")
            if mode == "ssd" and io["process_read_bytes"] < path["fs_load_payload_bytes"]:
                raise ValueError("Kernel I/O does not cover the audited direct SSD reads")
            groups.append(dict(epoch=epoch, mode=mode, n=len(records), unique_blocks=len(unique),
                 ttft_median_ms=statistics.median(row["ttft_seconds"] * 1000 for row in records),
                 ttft_p95_ms=sorted(row["ttft_seconds"] * 1000 for row in records)[math.ceil(len(records) * .95) - 1],
                 paired_saved_median_ms=statistics.median(saved), recovered_tokens=recovered,
                 computed_tokens=computed, max_logprob_delta=logprob_max, **path, **io))
    result = dict(groups=groups, output_mismatches=differences)
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (root / "validated-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in derived))
    (root / "review-metadata.json").write_text(json.dumps(dict(
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment_sha256=hashlib.sha256((root / "environment.json").read_bytes()).hexdigest()), indent=2) + "\n")
    if differences:
        raise ValueError(f"Generation differs for {len(differences)} requests; see summary.json")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    review(parser.parse_args().root)
