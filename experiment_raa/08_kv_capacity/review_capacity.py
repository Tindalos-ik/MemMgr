"""复算真实容量组的冻结准入、回迁链、逐 token 一致性及配对 TTFT。"""

import argparse
from collections import Counter
from importlib import import_module
import hashlib
import json
from pathlib import Path

from memmgr.short_term.reuse import HISTORY_LIMIT, History, Predictor


collector = import_module("experiment_raa.08_kv_capacity.run_capacity")
ssd = import_module("experiment_raa.07_kv_joint_ssd.review_ssd_joint")
joint = ssd.joint
require = ssd.require
EVIDENCE = ("audit.jsonl", "requests.jsonl", "resources.jsonl", "disk.json", "server.json", "server.log",
            "complete.json", "failure.json")


def digest(path):
    """计算 path 的字节摘要。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evidence(directory):
    """对 directory 的原始日志和快照建立摘要，排除 SSD KV 内容文件。"""
    paths = [(base / name) for base in (directory, directory / "warmup") for name in EVIDENCE]
    paths.extend(directory.glob("*.prom"))
    paths.extend((directory / "warmup").glob("*.prom"))
    hashes = {str(path.relative_to(directory)): digest(path) for path in sorted(paths) if path.is_file()}
    return dict(sha256=joint.digest(hashes), files_sha256=hashes)


def check_admission(path, tracked, records, predictor):
    """从 path 的请求顺序重建块统计，按 predictor 阈值核对准入并汇总 records。"""
    history = History()
    measured = {record["request_id"] for record in records}
    distinct = set()
    counts = Counter()
    for rid, req in tracked.items():
        begin = req["begin"]
        keys = begin["keys"]
        distinct.update(keys)
        require(len(distinct) <= HISTORY_LIMIT, path, "该计划越过无历史淘汰的复核范围")
        history.observe(keys, 64)
        features = [history.features(key) for key in keys]
        require(begin["clock"] == history.clock and begin["features"] == features, path, "预测特征不一致")
        limit = len(keys)
        if predictor is not None:
            limit = next((i for i, score in enumerate(predictor.score(features)) if score < predictor.threshold), limit)
        require(begin["admitted"] == keys[:limit], path, "连续前缀准入与冻结模型不一致")
        if rid in measured:
            counts["threshold_admitted_blocks"] += limit
            counts["threshold_filtered_blocks"] += len(keys) - limit
            retain = req.get("retain", {})
            counts["retain_capacity_checks"] += retain.get("required", 0) > 0
            counts["retain_capacity_vetoes"] += retain.get("required", 0) > 0 and retain["accepted"] is False
    return dict(counts)


def validate_server(directory, env, reference_argv, expected_model):
    """要求 directory 的 CLI 与 reference_argv 同模型同推理参数，按 env 核对名义容量。"""
    tier, policy = directory.name.split("-")
    path = directory / "server.json"
    server = joint.read_json(path)
    argv = server["argv"]
    require(argv[1:argv.index("--port")] == reference_argv[1:reference_argv.index("--port")]
            and argv[argv.index("--port") + 2:-1] == reference_argv[reference_argv.index("--port") + 2:-1]
            and server["env"] == env["arguments"]["env"], path, "启动参数与 07 不一致")
    require(int(argv[argv.index("--kv-cache-memory-bytes") + 1]) == server["gpu_bytes"] == env["gpu_bytes"],
            path, "GPU 配额不符")
    transfer = json.loads(argv[-1])
    extra = transfer["kv_connector_extra_config"]
    options = extra["policy_config"]
    require(transfer["kv_connector"] == "ManagedConnector"
            and extra["spec_name"] == ("ManagedTieringSpec" if tier == "ssd" else "ManagedCPUSpec")
            and options["tier"] == server["tier"] == tier and options["policy"] == server["policy"] == policy
            and options["restore"] == server["restore"] == "always" and not options["cost_profile"]
            and options["batch_invariant"] is True and options["block_tokens"] == extra["block_size"] == 64
            and Path(options["audit_path"]) == directory / "audit.jsonl"
            and options["cpu_bytes"] == extra["cpu_bytes_to_use"] == server["cpu_bytes"] == env["cpu_bytes"]
            and options["gpu_bytes"] == extra["gpu_bytes"] == env["gpu_bytes"], path, "层、策略或名义容量不符")
    require(options["model_dir"] == ("" if policy == "lru" else str(expected_model)), path, "冻结模型路径不符")
    if tier == "ssd":
        require(options["ssd_bytes"] == env["ssd_bytes"] and Path(options["ssd_root"]) == directory / "ssd"
                and extra["secondary_tiers"][0]["root_dir"] == server["ssd_root"], path, "独立 SSD 目录或容量不符")
    return server, extra


def review_group(directory, epoch, plan, warmup, env, reference_argv, predictor):
    """复核 directory 中 epoch 的完成标记、布局、冻结预测器及 plan/warmup 的真实输出。"""
    marker = joint.read_json(directory / "complete.json")
    require(marker == dict(status="complete", epoch=epoch, group=directory.name,
                           requests=len(plan), warmup_requests=len(warmup)), directory, "组完成标记无效")
    tier, policy = directory.name.split("-")
    server, extra = validate_server(directory, env, reference_argv, directory.parents[1] / "model")
    records = joint.read_rows(directory / "requests.jsonl")
    warm_dir = directory / "warmup"
    warm_records = joint.read_rows(warm_dir / "requests.jsonl")
    audit = directory / "audit.jsonl"
    block_bytes = env["kv_bytes_per_block"]
    tracked, stats, context, window = ssd.audit_run(audit, warm_records, records, tier, policy, block_bytes)
    require(stats["cpu_capacity_bytes"] == stats["cpu_capacity_blocks"] * block_bytes == env["cpu_bytes"]
            and context["gpu_blocks"] == env["gpu_bytes"] // block_bytes
            and context["gpu_bytes"] == env["gpu_bytes"] and context["batch_invariant"] is True
            and extra["model_revision"] == context["identity"]["model_revision"]
            and extra["dtype"] == context["identity"]["dtype"]
            and context["model_sha256"] == (None if policy == "lru" else predictor.sha256), directory,
            "实际块布局或审计模型摘要与配置不符")
    stats.update(check_admission(audit, tracked, records, None if policy == "lru" else predictor))
    warm_rows, _, warm_end = joint.review_prom(warm_dir, warm_records, warmup, tracked, block_bytes, 64, "warm")
    rows, initial, _ = joint.review_prom(directory, records, plan, tracked, block_bytes, 64, "joint")
    require(warm_end == initial and all(len(row["tokens"]) == 8 for row in warm_rows + rows),
            directory, "预热边界或真实输出长度不符")
    for row, record in zip(rows, records, strict=True):
        row["ssd_chain_blocks"] = tracked[record["request_id"]]["chain_blocks"]
    io = ssd.process_io(directory / "resources.jsonl", window)
    new_file_bytes = (stats["ssd_end_files"] - stats["ssd_start_files"]) * stats["ssd_block_bytes"]
    disk = joint.read_json(directory / "disk.json")
    require(io["process_read_bytes"] >= stats["ssd_logical_read_bytes"] and io["process_write_bytes"] >= new_file_bytes
            and disk["files"] == stats["ssd_end_files"] and disk["bytes"] == disk["files"] * stats["ssd_block_bytes"]
            and disk["root"] == server["ssd_root"] and ((disk["root"] is None) == (tier == "cpu")),
            directory, "实际 I/O 或盘面字节与成功传输不符")
    if tier == "ssd":
        require(stats["ssd_capacity_bytes"] == env["ssd_bytes"] and stats["ssd_block_bytes"] == block_bytes,
                directory, "SSD 实际字节布局与配置不符")
    groups = dict(all=rows, any_prefix_hit=[r for r in rows if r["gpu"] or r["cpu"]],
                  gpu_hit=[r for r in rows if r["gpu"]], outer_hit=[r for r in rows if r["cpu"]],
                  ssd_chain=[r for r in rows if r["ssd_chain_blocks"]])
    tokens = [row["tokens"] for row in warm_rows + rows]
    stats.update(io, ssd_new_file_bytes=new_file_bytes, disk_root=disk["root"],
                 ssd_end_bytes=disk["bytes"], gpu_capacity_blocks=context["gpu_blocks"],
                 ssd_recovery=("not_applicable" if tier == "cpu" else "confirmed" if groups["ssd_chain"] else "zero"),
                 ssd_recovered_tokens=stats["ssd_to_gpu_chain_blocks"] * 64,
                 requests=len(rows), output_tokens_sha256=joint.digest(tokens),
                 hit_gpu_tokens=sum(r["gpu"] for r in rows), hit_external_tokens=sum(r["cpu"] for r in rows),
                 computed_tokens=sum(r["computed"] for r in rows),
                 gpu_to_cpu_bytes=sum(r["store_bytes"] for r in rows), cpu_to_gpu_bytes=sum(r["load_bytes"] for r in rows),
                 ttft={subset: {field: joint.latencies(r[field] for r in selected) for field in ("ttft", "server_ttft")}
                       for subset, selected in groups.items()}, evidence=evidence(directory))
    return rows, tokens, stats, context


def failed_group(directory, epoch, env, reference_argv):
    """回报 directory 的失败原因和容量证据，不把 epoch 的部分请求计入指标。"""
    failure = joint.read_json(directory / "failure.json")
    require(failure["epoch"] == epoch and failure["group"] == directory.name, directory, "失败标记与组不符")
    if (directory / "server.json").is_file():
        validate_server(directory, env, reference_argv, directory.parents[1] / "model")
    log = directory / "server.log"
    audit = directory / "audit.jsonl"
    engine_errors = []
    if log.is_file():
        errors = ("CPU KV allocation failed", "CPU allocation failed during SSD promotion",
                  "Cannot reserve current prefix with pinned blocks", "SSD KV capacity exhausted")
        for line in log.read_text(encoding="utf-8").splitlines():
            for error in errors:
                if f"RuntimeError: {error}" in line and error not in engine_errors:
                    engine_errors.append(error)
    events = []
    if audit.is_file():
        # 失败组保留不完整末行，复核只使用已写完的事件。
        events = [json.loads(line) for line in audit.read_bytes().splitlines(keepends=True) if line.endswith(b"\n")]
    capacity = next((row["capacity"] for row in events if row["kind"] == "init"), None)
    last_begin = next((row for row in reversed(events) if row["kind"] == "begin"), None)
    return dict(status="capacity_failure" if engine_errors else "capture_failure", **failure, engine_errors=engine_errors,
                configured_cpu_bytes=env["cpu_bytes"], actual_cpu_blocks=capacity,
                last_begin=({key: last_begin[key] for key in ("request_id", "prompt_tokens", "prompt_sha256")}
                            | {"threshold_admitted_blocks": len(last_begin["admitted"])} if last_begin else None),
                completed_request_records={stage: len(path.read_bytes().splitlines()) if path.is_file() else 0
                                           for stage, path in (("warmup", directory / "warmup/requests.jsonl"),
                                                               ("formal", directory / "requests.jsonl"))},
                evidence=evidence(directory))


def comparisons(groups):
    """按 groups 的同索引 token 输入配对，返回处理减对照的全体及命中回迁差。"""
    contrasts = [(f"{tier}-lru", f"{tier}-admission", "写入准入（LRU 固定）") for tier in ("cpu", "ssd")]
    contrasts += [(f"{tier}-admission", f"{tier}-learned", "保留否决与替换顺序联合变化") for tier in ("cpu", "ssd")]
    contrasts += [(f"{tier}-lru", f"{tier}-learned", "准入、保留否决与替换联合变化") for tier in ("cpu", "ssd")]
    contrasts += [(f"cpu-{policy}", f"ssd-{policy}", "增加 SSD 层") for policy in collector.POLICIES]
    result = {}
    for control, treatment, change in contrasts:
        if control not in groups or treatment not in groups:
            continue
        pairs = list(zip(groups[control], groups[treatment], strict=True))
        result[f"{treatment}_vs_{control}"] = dict(
            change=change, all=ssd.paired(pairs),
            both_prefix_hit=ssd.paired([p for p in pairs if (p[0]["gpu"] or p[0]["cpu"]) and (p[1]["gpu"] or p[1]["cpu"])]),
            treatment_outer_hit=ssd.paired([p for p in pairs if p[1]["cpu"]]),
            treatment_ssd_chain=ssd.paired([p for p in pairs if p[1]["ssd_chain_blocks"]]))
    return result


def common_context(context):
    """从 context 排除政策及 CPU 容量处理变量。"""
    return {key: value for key, value in context.items() if key not in ("policy", "model_sha256", "cpu_blocks")}


def review_reference(source, env):
    """检查 source 中 07 的冻结工件与完整阶段，并取得参考命令与运行上下文。"""
    reference_path = source / "environment.json"
    ref = joint.read_json(reference_path)
    marker = joint.read_json(source / "complete.json")
    require(marker == dict(status="complete", epochs=ref["epochs"], requests_per_group=78, warmup_requests_per_group=1)
            and ref["epochs"] == [list(ssd.GROUPS), list(ssd.GROUPS[::-1])], source, "07 完成标记或顺序不符")
    require(digest(reference_path) == env["source_environment_sha256"]
            and all(env[key] == ref[key] for key in ("packages", "model_config_sha256", "gpu", "filesystem", "kv_bytes_per_block"))
            and all(env["source_sha256"][key] == value for key, value in ref["source_sha256"].items())
            and all(env["artifacts_sha256"][key] == value for key, value in ref["artifacts_sha256"].items())
            and all(env["arguments"][key] == ref["arguments"][key] for key in ("seed", "env", "server", "model")),
            source, "阶段的冻结工件或环境与 07 不一致")
    model_sha256 = ssd.frozen_model(source, ref["artifacts_sha256"])
    reference_group = source / "epoch-0/cpu-lru"
    argv = joint.read_json(reference_group / "server.json")["argv"]
    context, = (event["context"] for event in joint.read_rows(reference_group / "audit.jsonl") if event["kind"] == "context")
    return argv, context, model_sha256


def review(root, source):
    """复核 root 下一个阶段；source 为 07 原始目录，仅在 56MiB 冻结阶段补入其四组。"""
    env = joint.read_json(root / "environment.json")
    args = env["arguments"]
    pressure = args["pressure_policies"]
    cpu_mib = args["cpu_mib"]
    require(cpu_mib in (56, 28, 14) and env["cpu_bytes"] == cpu_mib * collector.MIB
            and env["gpu_bytes"] == 32 * collector.MIB and env["ssd_bytes"] == 4 * 2**30
            and env["workload"] == ("pressure" if pressure else "frozen-78"), root, "阶段标称配置无效")
    argv, context, expected_model = review_reference(source, env)
    hashes = env["artifacts_sha256"]
    require(ssd.frozen_model(root, {key: value for key, value in hashes.items() if key != "plans/pressure.json"})
            == expected_model, root, "本阶段冻结模型与 07 不符")
    predictor = Predictor(root / "model", context["identity"])
    plan, warmup = (joint.read_plan(root, name) for name in ("eval", "warmup"))
    require(len(plan) == 78 and len(warmup) == 1, root, "冻结计划长度无效")
    if pressure:
        pressure_plan = joint.read_plan(root, "pressure")
        require(digest(root / "plans/pressure.json") == hashes["plans/pressure.json"]
                and pressure_plan == collector.pressure_plan(plan), root, "压力计划与声明的派生方法不符")
        plan = pressure_plan
    reference_env = dict(env, cpu_bytes=56 * collector.MIB)
    reference_dir = source / "epoch-0/cpu-lru"
    _, reference_tokens, _, _ = review_group(
        reference_dir, 0, joint.read_plan(source, "eval"), warmup, reference_env, argv, predictor)
    order = list(collector.groups_for(cpu_mib, pressure))
    epochs = [order, order[::-1]]
    require(env["epochs"] == epochs, root, "计划执行顺序无效")
    reused = cpu_mib == 56 and not pressure
    complete = root / "complete.json"
    failures = root / "failures.json"
    require(not (complete.exists() and failures.exists()), root, "阶段的完成与失败标记并存")
    report = dict(status="not_measured", workload=env["workload"], cpu_mib=cpu_mib,
                  paths=dict(root=str(root), source=str(source)),
                  artifacts_sha256=hashes, environment_sha256=digest(root / "environment.json"),
                  source_environment_sha256=env["source_environment_sha256"], epochs={})
    expected_tokens = [reference_tokens[0]] if pressure else reference_tokens
    verified_groups = 0
    sessions, disk_roots = set(), set()
    failed, missing = [], []
    for epoch, captured in enumerate(epochs):
        epoch_name = f"epoch-{epoch}"
        groups, rows_by_group, invalid = {}, {}, {}
        old = ssd.GROUPS if reused else ()
        selected = dict.fromkeys((*captured, *old))
        for group in selected:
            reference = group in old
            directory = (source if reference else root) / epoch_name / group
            done = (directory / "complete.json").is_file()
            bad = (directory / "failure.json").is_file()
            require(not (done and bad) and (not reference or done), directory, "组标记冲突或 07 组缺失")
            if bad:
                failed.append([epoch, group])
                invalid[group] = failed_group(directory, epoch, env, argv)
            elif not done:
                missing.append([epoch, group])
                invalid[group] = dict(status="not_measured")
            else:
                rows, tokens, stats, runtime = review_group(directory, epoch, plan, warmup, env, argv, predictor)
                require(common_context(runtime) == common_context(context), directory, "运行上下文与 07 不同")
                if pressure:
                    require(tokens[0] == expected_tokens[0], directory, "预热输出与 07 不一致")
                    by_case = {}
                    for request, output in zip(plan, tokens[1:], strict=True):
                        require(by_case.setdefault(request["case"], output) == output, directory,
                                f"{request['case']} 的相同输入输出 token 不一致")
                    if len(expected_tokens) == 1:
                        expected_tokens = tokens
                for index, (left, right) in enumerate(zip(expected_tokens, tokens, strict=True)):
                    require(left == right, directory, f"预热/正式第 {index} 条输出 token ID 不一致")
                session = stats["audit_session"]
                require(session not in sessions, directory, "复用了审计会话")
                sessions.add(session)
                disk = stats["disk_root"]
                if disk is not None:
                    require(disk not in disk_roots, directory, "复用了 SSD 缓存目录")
                    disk_roots.add(disk)
                stats["origin"] = "07" if reference else "08"
                groups[group], rows_by_group[group] = stats, rows
                verified_groups += 1
        report["epochs"][epoch_name] = dict(captured_order=captured,
                                            reused_07_order=list(old if epoch == 0 else old[::-1]),
                                            groups=groups, excluded=invalid, paired=comparisons(rows_by_group))
    if complete.is_file():
        marker = joint.read_json(complete)
        require(marker == dict(status="complete", epochs=epochs, requests_per_group=len(plan),
                               warmup_requests_per_group=len(warmup)) and not failed and not missing,
                root, "阶段完成标记与原始组不符")
        report["status"] = "complete"
    elif failures.is_file():
        require(joint.read_json(failures) == dict(status="failed_groups", failures=failed, epochs=epochs)
                and failed and not missing, root, "失败清单与原始组不符")
        report["status"] = "failed_groups"
    report["verified_output_requests_per_group"] = len(plan) if verified_groups else 0
    return report


def main():
    """从 CLI 读取 08 与 07 路径，输出完整复核报告；未全成或存在失败时退出 2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path, default=Path(".agent/kv-joint-ssd/output"))
    args = parser.parse_args()
    result = review(args.root.resolve(), args.source.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
