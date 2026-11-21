"""从正式联合实验的原始请求、审计和 Prometheus 快照复算结果。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from prometheus_client.parser import text_string_to_metric_families

from memmgr.short_term.cost import CostPolicy, geometry
from memmgr.short_term.reuse import History, Predictor


GROUPS = ("lru-always", "lru-gate", "learned-always", "learned-gate")
METRICS = {
    "requests": "vllm:request_success_total",
    "queries": "vllm:prefix_cache_queries_total",
    "gpu": "vllm:prefix_cache_hits_total",
    "cpu": "vllm:external_prefix_cache_hits_total",
    "computed": "vllm:request_prefill_kv_computed_tokens_sum",
    "store_bytes": "vllm:kv_offload_store_bytes_total",
    "load_bytes": "vllm:kv_offload_load_bytes_total",
    "ttft": "vllm:time_to_first_token_seconds_sum",
    "ttft_count": "vllm:time_to_first_token_seconds_count",
    "preemptions": "vllm:num_preemptions_total",
}
LAZY_COUNTERS = {"store_bytes", "load_bytes"}
PLAN_FIELDS = ("index", "case", "round", "phase", "gap", "prompt_sha256")


def require(condition, context, message):
    """condition 不成立时，报告数据位置 context 和错误 message。"""
    if not condition:
        raise ValueError(f"{context}: {message}")


def read_json(path):
    """读取 path 的 UTF-8 JSON。"""
    return json.loads(path.read_bytes())


def read_rows(path):
    """读取 path 中非空且以完整换行结束的 JSONL。"""
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), path, "JSONL 为空或末行不完整")
    return [json.loads(line) for line in raw.splitlines()]


def digest(value):
    """按采集器的 JSON 编码计算 token 序列 value 的 SHA-256。"""
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def read_plan(root, name):
    """读取 root 下 name 对应的冻结输入计划，核验顺序和 token 摘要。"""
    path = root / "plans" / f"{name}.json"
    rows = read_json(path)
    require(rows, path, "输入计划为空")
    for index, row in enumerate(rows):
        ids = row["prompt_token_ids"]
        require(row["index"] == index and ids and all(type(t) is int and t >= 0 for t in ids)
                and row["prompt_sha256"] == digest(ids), path, f"输入计划第 {index} 条的索引或 token 无效")
    return rows


def counters(path):
    """从 path 原始 Prometheus 快照按名称汇总复核所需计数器和直方图。"""
    wanted = {metric: key for key, metric in METRICS.items()}
    # 传输计数器在第一次实际传输前尚未导出。
    values = dict.fromkeys(LAZY_COUNTERS, 0)
    for family in text_string_to_metric_families(path.read_text(encoding="utf-8")):
        for sample in family.samples:
            key = wanted.get(sample.name)
            if key is not None:
                require(math.isfinite(sample.value) and sample.value >= 0, path, f"指标无效：{sample.name}")
                values[key] = values.get(key, 0) + sample.value
    require(values.keys() == METRICS.keys(), path, f"缺少指标：{sorted(METRICS.keys() - values.keys())}")
    return values


def audit_run(directory, warmup_records, records, model_dir, cost_path, margin_ms, block_bytes):
    """复算 directory 审计，对照两段请求、冻结模型与成本表；block_bytes 来自运行清单。"""
    path = directory / "audit.jsonl"
    require(not path.with_name("audit.jsonl.1").exists(), path, "审计发生轮转，无法证明完整会话")
    initial, runtime, *events = read_rows(path)
    policy, restore = directory.name.split("-")
    context = runtime["context"]
    block_tokens = context["identity"]["block_tokens"]
    capacity = initial["capacity"]
    require(initial["kind"] == "init" and runtime["kind"] == "context"
            and initial["policy"] == context["policy"] == policy
            and initial["identity"] == context["identity"] and block_tokens == 64
            and capacity == context["cpu_blocks"] > 0 and type(block_bytes) is int and block_bytes > 0,
            path, "会话、策略或块布局不一致")
    predictor = Predictor(model_dir, context["identity"]) if policy == "learned" else None
    require(initial["model_sha256"] == context["model_sha256"] == (predictor.sha256 if predictor else None),
            path, "冻结模型摘要不匹配")
    cost = CostPolicy(cost_path, context, margin_ms) if restore == "gate" else None
    warmup = [row["request_id"] for row in warmup_records]
    expected = warmup + [row["request_id"] for row in records]
    measured = set(expected[len(warmup):])
    require(len(set(expected)) == len(expected), path, "请求 ID 重复")
    history = History()
    tracked = {}
    # 每次完成写入建立一个独立的驻留周期；相同块再次写入时重新计数。
    resident, pending = {}, {}
    evictions = Counter()
    previous = initial["time_ns"]
    for event in (runtime, *events):
        require(event["session"] == initial["session"] and event["time_ns"] >= previous,
                path, "混入其他会话或事件时间倒序")
        previous = event["time_ns"]
        kind = event["kind"]
        if event is runtime:
            continue
        rid = event["request_id"]
        location = f"{path}/{rid}/{kind}"
        if kind == "begin":
            require(len(tracked) < len(expected) and rid == expected[len(tracked)], location, "请求与采集顺序不一致")
            keys = event["keys"]
            require(len(keys) == event["prompt_tokens"] // block_tokens, location, "完整前缀块数量不一致")
            history.observe(keys, block_tokens, resident.keys() | pending.keys())
            features = [history.features(key) for key in keys]
            require(event["clock"] == history.clock and event["features"] == features,
                    location, "块访问时钟或模型输入特征不一致")
            admitted = keys
            if predictor:
                scores = predictor.score(features)
                limit = next((i for i, score in enumerate(scores) if score < predictor.threshold), len(keys))
                admitted = keys[:limit]
            require(event["admitted"] == admitted, location, "连续前缀准入与冻结模型阈值不一致")
            tracked[rid] = dict(begin=event, selected=set(admitted), stored=0, loaded=0, load_keys=set())
            continue
        require(rid in tracked, location, "事件早于请求开始")
        request = tracked[rid]
        if kind == "lookup":
            require("lookup" not in request and event["prompt"] == request["begin"]["prompt_tokens"],
                    location, "重复查找或输入 token 数不一致")
            gpu, external = event["gpu"], event["external"]
            shape = geometry(event["prompt"], gpu, external)
            require(gpu % block_tokens == external % block_tokens == 0, location, "命中数量未按完整块计量")
            keys = request["begin"]["keys"]
            candidates = []
            for key in keys[gpu // block_tokens:]:
                if key not in resident:
                    break
                candidates.append(key)
            require(external == len(candidates) * block_tokens, location, "CPU 候选与已驻留的连续前缀不一致")
            estimate = None
            if cost and external:
                entry = cost.entries[shape]
                savings = entry["recompute_seconds"] - entry["restore_seconds"]
                estimate = dict(admit=savings > cost.margin, expected_saved_seconds=savings,
                                restore_seconds=entry["restore_seconds"], recompute_seconds=entry["recompute_seconds"])
            admitted = external if (estimate["admit"] if estimate else restore == "always") else 0
            require(event["estimate"] == estimate and event["admitted"] == admitted,
                    location, "审计决定或估计成本与配置不一致")
            request["lookup"] = event
            request["load_keys"] = set(candidates) if admitted else set()
        elif kind == "retain":
            require(predictor and "retain" not in request, location, "额外的 learned 保留决定")
            occupied = resident.keys() | pending.keys()
            missing = [key for key in request["begin"]["admitted"] if key not in occupied]
            victims = event["victims"]
            required = max(0, len(missing) + len(occupied) - capacity)
            require(event["required"] == required == len(set(victims)) == len(victims)
                    and set(victims) <= resident.keys() - set(request["begin"]["keys"]),
                    location, "写入所需空位或候选淘汰块不一致")
            incoming, displaced = (sum(predictor.score([history.features(key) for key in keys]))
                                   for keys in (missing, victims))
            accepted = bool(request["selected"]) and (not required or incoming > displaced)
            require(event["incoming"] == incoming and event["displaced"] == displaced
                    and event["accepted"] is accepted, location, "保留决定与模型评分不一致")
            request["retain"] = event
            if not accepted:
                request["selected"].clear()
        elif kind == "store":
            keys, victims = set(event["keys"]), set(event["evicted"])
            require((not predictor or "retain" in request) and keys <= request["selected"]
                    and not keys.intersection(resident.keys() | pending.keys())
                    and victims <= resident.keys() and (not predictor or victims.isdisjoint(request["begin"]["keys"]))
                    and len(keys) == len(event["keys"]) and len(victims) == len(event["evicted"]),
                    location, "写入或淘汰转换无效")
            for key in victims:
                owner, read = resident.pop(key)
                if rid in measured:
                    evictions["evicted"] += 1
                    if not read:
                        evictions["evicted_without_read"] += 1
                        evictions["evicted_without_read_from_warmup"] += owner not in measured
            pending.update(dict.fromkeys(keys, rid))
            require(len(pending) + len(resident) <= capacity, location, "CPU 容量超限")
        elif kind == "stored":
            for key in event["keys"]:
                require(key in pending and pending[key] == rid, location, "完成写入找不到对应预约")
                del pending[key]
                resident[key] = (rid, False)
                request["stored"] += 1
        elif kind == "loaded":
            for key in event["keys"]:
                require(key in request["load_keys"] and key in resident, location, "完成回迁找不到对应准入和驻留块")
                request["load_keys"].remove(key)
                resident[key] = (resident[key][0], True)
                request["loaded"] += 1
        else:
            raise ValueError(f"{location}: 未知 CPU 审计事件")
    require(not pending and all("lookup" in req and not req["load_keys"] for req in tracked.values()),
            path, "缺失查找或未完成传输")
    require(list(tracked) == expected, path, "审计与请求的顺序或数量不一致")
    evictions["unread_at_end"] = sum(not read for _, read in resident.values())
    evictions["unread_at_end_from_warmup"] = sum(not read and rid not in measured for rid, read in resident.values())
    accounting = {f"{name}_bytes": evictions[name] * block_bytes for name in (
        "evicted", "evicted_without_read", "evicted_without_read_from_warmup", "unread_at_end", "unread_at_end_from_warmup")}
    accounting.update(block_bytes=block_bytes, cpu_capacity_blocks=capacity,
                      resident_blocks_at_end=len(resident), warmup_requests=len(warmup),
                      model_sha256=predictor.sha256 if predictor else None,
                      model_threshold=predictor.threshold if predictor else None,
                      cost_sha256=cost.sha256 if cost else None, cost_margin_ms=margin_ms if cost else None)
    return tracked, accounting, context


def review_prom(directory, records, plan, tracked, block_bytes, block_tokens, client_prefix):
    """用 directory 的快照核对 records/plan；tracked 提供审计，块大小和客户端 ID 前缀限定采样口径。"""
    require(len(records) == len(plan), directory, f"请求数 {len(records)} != 计划数 {len(plan)}")
    initial = previous = counters(directory / "initial.prom")
    reviewed = []
    for index, (record, planned) in enumerate(zip(records, plan)):
        location = f"{directory}/{index:04}"
        audited = tracked[record["request_id"]]
        begin, lookup = audited["begin"], audited["lookup"]
        length = len(planned["prompt_token_ids"])
        require(all(record[key] == planned[key] for key in PLAN_FIELDS)
                and record["client_request_id"] == f"{client_prefix}-{index:04}"
                and record["prompt_sha256"] == begin["prompt_sha256"]
                and record["prompt_tokens"] == begin["prompt_tokens"] == record["usage"]["prompt_tokens"] == length,
                location, "计划、请求与审计输入不一致")
        current = counters(directory / f"{index:04}.prom")
        delta = {key: value - previous[key] for key, value in current.items()}
        require(all(value >= 0 for value in delta.values()), location, "Prometheus 计数回退")
        saved = record["metrics"]
        require(all(delta[key] == (saved.get(metric, 0) if key in LAZY_COUNTERS else saved[metric])
                    for key, metric in METRICS.items()), location, "请求记录的指标与原始快照不一致")
        previous = current
        require(delta["requests"] == delta["ttft_count"] == 1 and delta["preemptions"] == 0
                and delta["queries"] == length and delta["gpu"] == lookup["gpu"]
                and delta["cpu"] == lookup["admitted"] == audited["loaded"] * block_tokens
                and delta["gpu"] + delta["cpu"] + delta["computed"] == length
                and delta["store_bytes"] == audited["stored"] * block_bytes
                and delta["load_bytes"] == audited["loaded"] * block_bytes,
                location, "命中、重算或迁移与原始 Prometheus 增量不一致")
        generated = record["generated_tokens"]
        require(generated and record["finish_reason"] == "length"
                and len(generated) == record["usage"]["completion_tokens"] == len(record["token_logprobs"])
                and record["usage"]["total_tokens"] == length + len(generated)
                and all(isinstance(token, str) and re.fullmatch(r"token_id:[0-9]+", token) for token in generated),
                location, "缺少完整的实际输出 token ID")
        ttft = record["ttft_seconds"]
        require(math.isfinite(ttft) and 0 < ttft <= record["total_seconds"] < math.inf
                and delta["ttft"] > 0 and record["server_ttft_seconds"] == delta["ttft"],
                location, "客户端或服务端 TTFT 无效")
        reviewed.append(dict(ttft=ttft, server_ttft=delta["ttft"], tokens=generated, phase=planned["phase"],
                             gpu=int(delta["gpu"]), cpu=int(delta["cpu"]), computed=int(delta["computed"]),
                             store_bytes=int(delta["store_bytes"]), load_bytes=int(delta["load_bytes"]),
                             geometry=(length, lookup["gpu"], lookup["external"]),
                             threshold_blocks=len(begin["admitted"]),
                             retained=audited.get("retain", {}).get("accepted")))
    require(counters(directory / "final.prom") == previous, directory, "最后一次请求后指标发生变化")
    return reviewed, initial, previous


def review_group(root, directory, plan, warmup, block_bytes):
    """将 directory 的请求与 plan、warmup 配对；root 提供策略文件，block_bytes 来自运行清单。"""
    records = read_rows(directory / "requests.jsonl")
    warmup_records = read_rows(directory / "warmup" / "requests.jsonl")
    server = read_json(directory / "server.json")
    argv = server["argv"]
    transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
    options = transfer["kv_connector_extra_config"]["policy_config"]
    policy, restore = directory.name.split("-")
    require(transfer["kv_connector"] == "ManagedConnector" and options["tier"] == "cpu"
            and options["policy"] == server["policy"] == policy
            and options["restore"] == server["restore"] == restore,
            directory, "服务配置与评价组不一致")
    tracked, accounting, context = audit_run(directory, warmup_records, records, root / "model",
                                             root / "calibration" / f"{policy}.json", options["margin_ms"], block_bytes)
    block_tokens = context["identity"]["block_tokens"]
    require(options["cpu_bytes"] // block_bytes == accounting["cpu_capacity_blocks"],
            directory, "清单中的每块迁移字节与原生 CPU 容量不一致")
    _, _, warmup_final = review_prom(directory / "warmup", warmup_records, warmup, tracked,
                                     block_bytes, block_tokens, "warm")
    reviewed, initial, _ = review_prom(directory, records, plan, tracked, block_bytes, block_tokens, "joint")
    require(warmup_final == initial, directory, "预热结束与正式请求开始的指标快照不一致")
    return reviewed, accounting, context


def check_frozen_sources(root):
    """核对 root 的冻结模型和成本表分别来自本次真实采集和匹配策略的校准日志。"""
    model = read_json(root / "model" / "model.json")
    collection = root / "collect" / "lru-always" / "audit.jsonl"
    require(model["source_audit_sha256"] == hashlib.sha256(collection.read_bytes()).hexdigest(),
            collection, "训练审计摘要与冻结模型不一致")
    for policy in ("lru", "learned"):
        calibration = root / "calibration"
        path = calibration / f"{policy}.json"
        sources = read_json(path)["sources"]
        observed = {tuple(Path(name).parts[-2:]): sha for name, sha in sources.items()}
        expected = {}
        for mode in ("always", "never"):
            for filename in ("requests.jsonl", "audit.jsonl"):
                source = (f"{policy}-{mode}", filename)
                expected[source] = hashlib.sha256(calibration.joinpath(*source).read_bytes()).hexdigest()
        require(len(sources) == len(expected) and observed == expected, path, "成本校准来源或摘要不一致")


def latencies(values):
    """将秒数样本 values 转为毫秒，计算中位数和最近秩 P95；空样本标为空。"""
    ordered = sorted(values)
    return dict(n=len(ordered), median_ms=statistics.median(ordered) * 1000 if ordered else None,
                p95_ms=ordered[math.ceil(0.95 * len(ordered)) - 1] * 1000 if ordered else None)


def paired(pairs):
    """按 pairs 的 (基线, 处理组) 请求计算配对 TTFT；差值为处理组减基线。"""
    result = {}
    for field in ("ttft", "server_ttft"):
        differences = [right[field] - left[field] for left, right in pairs]
        result[field] = dict(**latencies(differences), faster=sum(value < 0 for value in differences),
                             slower=sum(value > 0 for value in differences),
                             tied=sum(value == 0 for value in differences))
    return result


def review(root):
    """复核 root 下四组真实评价；按输入顺序配对请求，输出可复算的统计。"""
    plans = {name: read_plan(root, name) for name in ("warmup", "collect", "calibration", "eval")}
    prefix_sets = {name: {tuple(row["prompt_token_ids"][:64]) for row in rows} for name, rows in plans.items()}
    require(not prefix_sets["eval"].intersection(prefix_sets["warmup"] | prefix_sets["collect"] | prefix_sets["calibration"]),
            root, "评价计划与预热、采集或校准计划共享首块前缀")
    block_bytes = read_json(root / "environment.json")["kv_bytes_per_block"]
    check_frozen_sources(root)
    groups, summary, contexts = {}, {}, {}
    for name in GROUPS:
        rows, accounting, context = review_group(root, root / "eval" / name, plans["eval"], plans["warmup"], block_bytes)
        groups[name], contexts[name] = rows, context
        summary[name] = dict(**accounting, requests=len(rows),
                             hit_gpu_tokens=sum(row["gpu"] for row in rows),
                             hit_cpu_tokens=sum(row["cpu"] for row in rows),
                             computed_tokens=sum(row["computed"] for row in rows),
                             gpu_to_cpu_bytes=sum(row["store_bytes"] for row in rows),
                             cpu_to_gpu_bytes=sum(row["load_bytes"] for row in rows),
                             candidate_requests=sum(row["geometry"][2] > 0 for row in rows),
                             admitted_reads=sum(row["cpu"] > 0 for row in rows),
                             rejected_reads=sum(row["geometry"][2] > 0 and row["cpu"] == 0 for row in rows),
                             threshold_admitted_blocks=sum(row["threshold_blocks"] for row in rows),
                             learned_retain_accepted=sum(row["retained"] is True for row in rows),
                             learned_retain_rejected=sum(row["retained"] is False for row in rows),
                             ttft=latencies(row["ttft"] for row in rows),
                             server_ttft=latencies(row["server_ttft"] for row in rows))
    baseline = groups["lru-always"]
    base_context = {key: value for key, value in contexts["lru-always"].items() if key not in ("policy", "model_sha256")}
    for name, rows in groups.items():
        require({key: value for key, value in contexts[name].items() if key not in ("policy", "model_sha256")} == base_context
                and summary[name]["block_bytes"] == summary["lru-always"]["block_bytes"],
                root / "eval" / name, "四组评价的环境或块布局不同")
        for index, (left, right) in enumerate(zip(baseline, rows)):
            require(left["tokens"] == right["tokens"], f"{name}/{index:04}", "输出 token ID 序列与 lru-always 不一致")
    comparisons = {}
    for control, treatment in (("lru-always", "lru-gate"), ("learned-always", "learned-gate"),
                               ("lru-always", "learned-always"), ("lru-gate", "learned-gate"),
                               ("lru-always", "learned-gate")):
        pairs = list(zip(groups[control], groups[treatment]))
        by_phase = defaultdict(list)
        for pair in pairs:
            by_phase[pair[0]["phase"]].append(pair)
        comparison = dict(control=control, treatment=treatment, all=paired(pairs),
                          by_phase={phase: paired(subset) for phase, subset in by_phase.items()})
        if control.split("-")[0] == treatment.split("-")[0]:
            matched = [(left, right) for left, right in pairs if left["geometry"] == right["geometry"] and left["cpu"] > 0]
            comparison["same_candidate_gate"] = {
                "admitted": paired([(left, right) for left, right in matched if right["cpu"] > 0]),
                "rejected": paired([(left, right) for left, right in matched if right["cpu"] == 0]),
                "different_geometry": sum(left["geometry"] != right["geometry"] for left, right in pairs),
            }
        comparisons[f"{treatment}_vs_{control}"] = comparison
    return dict(input_and_output_tokens_verified=len(baseline), groups=summary, paired=comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="联合采集输出根目录（含 plans、model、calibration 和 eval）")
    args = parser.parse_args()
    print(json.dumps(review(args.root), ensure_ascii=False, indent=2, allow_nan=False))
