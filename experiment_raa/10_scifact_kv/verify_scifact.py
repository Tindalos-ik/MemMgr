"""离线复核真实请求、原始 Prometheus、策略决定及逐 token 生成。"""

import argparse
from collections import Counter
import importlib
from itertools import zip_longest
import json
import math
from pathlib import Path

from memmgr.long_term.index import sha256_file
from memmgr.short_term.reuse import History, Predictor

from .prepare_scifact import read, replay_plan, verify
from .run_scifact import BLOCK, CPU_BYTES, GPU_BYTES, MODEL_HASHES, ORDERS, POLICIES, TIMED_MODULE

joint = importlib.import_module("experiment_raa.06_kv_joint.review_joint")
require, read_rows, counters = joint.require, joint.read_rows, joint.counters
TIMINGS = ("feature_ns", "score_ns", "select_ns", "evict_ns")
TIMING_FIELDS = (*TIMINGS, "audit_ns", "feature_rows", "scored_rows", "score_calls", "select_calls", "evict_calls")
PLAN_FIELDS = ("index", "query_id", "phase", "request_gap", "prior_prefix_tokens", "prompt_sha256")


def audit_request(events, row, history, resident, prefixes, predictor, policy, capacity, location):
    """用 row 和驻留历史复算 events，核对缓存容量、阈值和保留决定。"""
    begin = events[0]
    tokens = row["prompt_token_ids"]
    keys = begin["keys"]
    require(begin["kind"] == "begin" and begin["prompt_sha256"] == row["prompt_sha256"]
            and begin["prompt_tokens"] == len(tokens) and len(keys) == len(tokens) // BLOCK,
            location, "审计输入 token 或完整块数不一致")
    request_id = begin["request_id"]
    by_key, by_prefix = prefixes
    for index, key in enumerate(keys, 1):
        prefix = tuple(tokens[:index * BLOCK])
        require(by_key.setdefault(key, prefix) == prefix and by_prefix.setdefault(prefix, key) == key,
                location, "块键与实际 token 前缀对应关系不一致")
    history.observe(keys, BLOCK, resident)
    features = [history.features(key) for key in keys]
    require(begin["features"] == features and begin["clock"] == history.clock == row["index"],
            location, "请求时钟或特征不一致")
    scores = dict(zip(keys, predictor.score(features), strict=True)) if predictor else {}
    if predictor:
        limit = next((index for index, key in enumerate(keys) if scores[key] < predictor.threshold), len(keys))
    else:
        limit = len(keys)
    admitted = keys[:limit]
    require(begin["admitted"] == admitted, location, "连续准入与冻结阈值不一致")
    selected = set(admitted)
    lookup = retain = timing = None
    pending, loading = set(), set()
    used = Counter()
    for event in events[1:]:
        kind = event["kind"]
        require(event["request_id"] == request_id, location, "混入其他请求事件")
        if kind == "lookup":
            require(lookup is None and event["prompt"] == len(tokens) and event["gpu"] % BLOCK == 0,
                    location, "重复查找或 GPU 命中几何不合法")
            for key in keys[event["gpu"] // BLOCK:]:
                if key not in resident:
                    break
                loading.add(key)
            require(event["external"] == event["admitted"] == len(loading) * BLOCK and event["estimate"] is None,
                    location, "CPU 连续命中与已驻留块不一致")
            lookup = event
        elif kind == "retain":
            require(policy == "learned" and retain is None and timing is None, location, "意外的保留决定")
            occupied = resident.keys() | pending
            missing = [key for key in admitted if key not in occupied]
            required = max(0, len(occupied) + len(missing) - capacity)
            victims = event["victims"]
            require(required == event["required"] == len(victims) == len(set(victims))
                    and set(victims) <= resident.keys() - set(keys), location, "保留决定的候选块或空位不合法")
            incoming = sum(scores[key] for key in missing)
            displaced = sum(predictor.score([history.features(key) for key in victims]))
            accepted = bool(selected) and (not required or incoming > displaced)
            require(event["incoming"] == incoming and event["displaced"] == displaced and event["accepted"] is accepted,
                    location, "保留与冻结模型的进入及移出评分不一致")
            if not accepted:
                selected.clear()
            retain = event
        elif kind == "store":
            new, evicted = set(event["keys"]), set(event["evicted"])
            require(timing is None and (policy != "learned" or retain is not None)
                    and len(new) == len(event["keys"]) and len(evicted) == len(event["evicted"])
                    and new <= selected and not new.intersection(resident.keys() | pending)
                    and evicted <= resident.keys() - loading
                    and len(evicted) == max(0, len(resident) + len(pending) + len(new) - capacity)
                    and (policy != "learned" or not evicted.intersection(keys)), location, "写入或淘汰不合法")
            used["select_calls"] += 1
            used["evict_calls"] += bool(evicted)
            for key in evicted:
                used["evicted"] += 1
                used["evicted_unread"] += not resident.pop(key)
            pending.update(new)
        elif kind == "stored":
            for key in event["keys"]:
                require(key in pending, location, "写入完成缺少对应预约")
                pending.remove(key)
                resident[key] = False
                used["stored"] += 1
        elif kind == "loaded":
            for key in event["keys"]:
                require(key in loading and key in resident, location, "回迁完成缺少准入或驻留块")
                loading.remove(key)
                resident[key] = True
                used["loaded"] += 1
        elif kind == "timing":
            require(timing is None and all(type(event[field]) is int and event[field] >= 0 for field in TIMING_FIELDS),
                    location, "重复或无效的分段计时")
            require(event["feature_ns"] > 0 and event["feature_rows"] == event["scored_rows"]
                    and (event["score_ns"] > 0) == (event["score_calls"] > 0) == bool(predictor)
                    and (event["scored_rows"] > 0) == bool(predictor)
                    and all(event[field] == used[field] for field in ("select_calls", "evict_calls"))
                    and (event["select_ns"] > 0) == (event["select_calls"] > 0)
                    and (event["evict_ns"] > 0) == (event["evict_calls"] > 0), location, "计时调用数与审计不一致")
            timing = {field: event[field] for field in TIMING_FIELDS}
        else:
            raise ValueError(f"{location}: unexpected audit event {kind}")
    require(lookup is not None and timing is not None and not pending and not loading,
            location, "缺少查找、计时或未完成迁移")
    return request_id, lookup, timing, used, len(admitted), retain["accepted"] if retain else None


def verify_sequence(path, plan, events, previous, predictor, policy, config, capacity):
    """核对 path 对应的一个独立序列，previous 是同服务上一序列的末尾快照。"""
    initial = counters(path / "initial.prom")
    require(read(path / "reset.json") == {"success": True}
            and (previous == initial if previous is not None else not any(initial.values())),
            path, "缓存重置报告或序列边界计数不一致")
    previous = initial
    requests = read_rows(path / "requests.jsonl")
    require(len(requests) == len(plan), path, "保存请求数与计划不符")
    events_by_request = {}
    prior_id = None
    for event in events:
        request_id = event["request_id"]
        if request_id != prior_id:
            require(request_id not in events_by_request, path, "不同请求的审计事件交错")
            events_by_request[request_id] = []
            prior_id = request_id
        events_by_request[request_id].append(event)
    require(list(events_by_request) == [row["request_id"] for row in requests], path, "审计请求顺序或数量不一致")
    resident = {}
    history = History()
    prefixes = ({}, {})
    reviewed = []
    for row, record in zip(plan, requests, strict=True):
        i = row["index"]
        location = f"{path}/{i:02}"
        rid, lookup, timing, used, admitted, retained = audit_request(
            events_by_request[record["request_id"]], row, history, resident, prefixes, predictor, policy, capacity, location)
        current = counters(path / f"{i:02}.prom")
        delta = {key: current[key] - previous[key] for key in current}
        previous = current
        length = len(row["prompt_token_ids"])
        require(record["request_id"] == rid and record["client_request_id"] == f"{path.name}-{i:02}"
                and all(record[key] == row[key] for key in PLAN_FIELDS)
                and record["usage"]["prompt_tokens"] == length and delta == record["metrics"],
                location, "请求计划、记录或 Prometheus 增量不一致")
        require(all(value >= 0 for value in delta.values()) and delta["requests"] == delta["ttft_count"] == 1
                and delta["preemptions"] == 0 and delta["queries"] == length and delta["gpu"] == lookup["gpu"]
                and delta["cpu"] == lookup["admitted"] == used["loaded"] * BLOCK
                and delta["gpu"] + delta["cpu"] + delta["computed"] == length
                and delta["gpu"] + delta["cpu"] <= row["prior_prefix_tokens"]
                and delta["load_bytes"] == used["loaded"] * config["block_bytes"]
                and delta["store_bytes"] == used["stored"] * config["block_bytes"], location, "前缀、重算或迁移计数不一致")
        ids = record["generated_token_ids"]
        finish = record["finish_reason"]
        require(1 <= len(ids) <= config["max_output"] and all(type(token) is int and token >= 0 for token in ids)
                and len(ids) == record["usage"]["completion_tokens"] and record["usage"]["total_tokens"] == length + len(ids)
                and (finish == "stop" or (finish == "length" and len(ids) == config["max_output"])),
                location, "缺少完整生成 token ID 或使用量不符")
        require(0 < record["client_ttft_seconds"] <= record["total_seconds"] < math.inf and delta["ttft"] > 0,
                location, "TTFT 无效")
        reviewed.append(dict(record, prompt_tokens=length, timing=timing, admitted_blocks=admitted, retained=retained,
                             evicted=used["evicted"], evicted_unread=used["evicted_unread"]))
    require(counters(path / "final.prom") == previous, path, "序列结束后指标发生变化")
    return reviewed, previous, len(resident)


def verify_policy(path, inputs, config, sequence_ids, policy):
    """核对 path 的服务参数和所有 sequence_ids，返回逐请求复核资料。"""
    require(not (path / "audit.jsonl.1").exists(), path, "审计日志轮转")
    init, runtime, *events = read_rows(path / "audit.jsonl")
    context = runtime["context"]
    server = read(path / "server.json")
    argv = server["argv"]
    transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
    extra = transfer["kv_connector_extra_config"]
    options = extra["policy_config"]
    capacity = config["cpu_bytes"] // config["block_bytes"]
    gpu_capacity = config["gpu_bytes"] // config["block_bytes"]
    require(init["kind"] == "init" and runtime["kind"] == "context" and context["identity"] == init["identity"]
            and init["policy"] == context["policy"] == server["policy"] == options["policy"] == policy
            and init["capacity"] == context["cpu_blocks"] == capacity and context["gpu_blocks"] == gpu_capacity
            and context["gpu_bytes"] == options["gpu_bytes"] == config["gpu_bytes"] == GPU_BYTES
            and options["cpu_bytes"] == config["cpu_bytes"] == CPU_BYTES
            and context["identity"]["block_tokens"] == options["block_tokens"] == BLOCK
            and context["max_model_len"] == extra["max_model_len"] == config["max_len"]
            and context["batch_invariant"] is True and context["enforce_eager"] is True
            and options["restore"] == "always" and not options["cost_profile"] and options["tier"] == "cpu"
            and transfer["kv_connector_module_path"] == extra["spec_module_path"] == TIMED_MODULE
            and transfer["kv_connector"] == "TimedConnector" and extra["spec_name"] == "TimedCPUSpec"
            and server["sequences"] == sequence_ids and server["env"] == config["env"], path, "服务身份或固定容量不一致")
    model_dir = Path(config["model_dir"])
    predictor = Predictor(model_dir, context["identity"]) if policy != "lru" else None
    require(init["model_sha256"] == context["model_sha256"] == (MODEL_HASHES["model.joblib"] if predictor else None)
            and (predictor is None or predictor.threshold == config["model_threshold"] == 0.35), path, "冻结模型或阈值有变化")
    cursor = init["time_ns"]
    segments = []
    for event in (runtime, *events):
        require(event["session"] == init["session"] and event["time_ns"] >= cursor, path, "审计会话混杂或时间倒序")
        cursor = event["time_ns"]
        if event["kind"] == "reset":
            segments.append((event, []))
        elif event is not runtime:
            require(bool(segments), path, "重置前发生生成请求")
            segments[-1][1].append(event)
    require(len(segments) == len(sequence_ids), path, "缓存重置次数与序列数不符")
    sequences = {sequence["id"]: sequence for sequence in inputs["sequences"]}
    records, previous, clock, resident = {}, None, -1, 0
    for sequence_id, (reset, events) in zip(sequence_ids, segments, strict=True):
        require(reset["prior_clock"] == clock and reset["cpu_blocks_before"] == resident
                and reset["cpu_blocks_after"] == 0, path / sequence_id, "重置前后的历史或 CPU 边界不一致")
        plan = replay_plan(inputs, sequences[sequence_id])
        records[sequence_id], previous, resident = verify_sequence(path / sequence_id, plan, events, previous,
                                                                  predictor, policy, config, capacity)
        clock = len(plan) - 1
    return records, context, init["session"], init["time_ns"], cursor


def verify_run(root, partial_epochs=None):
    """读取 root 的全部阶段；partial_epochs 只用于单序列或单顺序复核。"""
    config = read(root / "run.json")
    inputs_path, model_path = Path(config["inputs"]), Path(config["model_dir"])
    require(config["model_hashes"] == MODEL_HASHES == {name: sha256_file(model_path / name) for name in MODEL_HASHES}
            and config["inputs_sha256"] == sha256_file(inputs_path / "inputs.json"), root, "冻结输入或模型摘要改变")
    inputs = verify(inputs_path)
    sequences = [sequence["id"] for sequence in inputs["sequences"]]
    groups, reference, sessions = {}, {}, set()
    common_context, prior_end = None, -1
    for epoch in partial_epochs or ORDERS:
        path = root / epoch
        started = read(path / "start.json")
        expected = sequences if epoch == "forward" else sequences[::-1]
        require(started == read(path / "complete.json") and started["order"] == list(ORDERS[epoch])
                and (started["sequences"] == expected or
                     (partial_epochs and len(started["sequences"]) == 1 and started["sequences"][0] in expected)),
                path, "阶段未完成或顺序与计划不一致")
        groups[epoch] = {}
        for policy in started["order"]:
            records, context, session, started_ns, ended_ns = verify_policy(
                path / policy, inputs, config, started["sequences"], policy)
            require(started_ns > prior_end and session not in sessions, path, "策略启动顺序或服务会话不独立")
            prior_end = ended_ns
            sessions.add(session)
            comparable = {key: value for key, value in context.items() if key not in ("policy", "model_sha256")}
            if common_context is None:
                common_context = comparable
            require(comparable == common_context, path / policy, "策略间的硬件或服务上下文不一致")
            groups[epoch][policy] = records
            for sequence, rows in records.items():
                for row in rows:
                    key = (sequence, row["index"])
                    ids = row["generated_token_ids"]
                    baseline, finish = reference.setdefault(key, (ids, row["finish_reason"]))
                    if ids != baseline:
                        i, a, b = next((i, a, b) for i, (a, b) in enumerate(zip_longest(baseline, ids)) if a != b)
                        raise ValueError(f"{path}/{policy}/{key}: output token[{i}] {a} != {b}")
                    require(finish == row["finish_reason"], path / policy, f"终止原因不一致：{key}")
    return config, inputs, groups, common_context


def main():
    """运行离线复核；仅在检查采集器时可指定一个 epoch。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--epoch", choices=ORDERS)
    args = parser.parse_args()
    _, _, groups, _ = verify_run(args.root, [args.epoch] if args.epoch else None)
    count = sum(len(rows) for policies in groups.values() for sequences in policies.values() for rows in sequences.values())
    print(f"已复核 {count} 条真实生成请求及逐 token 结果", flush=True)


if __name__ == "__main__":
    main()
