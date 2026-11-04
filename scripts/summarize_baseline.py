"""从正式执行记录复算请求、批次、资源与索引汇总。

用法：在 WSL 项目根目录运行 `python scripts/summarize_baseline.py EXECUTION_DIR`。
脚本读取执行清单、命令日志和各运行目录中的原始记录，输出写入当前工作目录的
`.agent/results/`。隔离工作树可将复算汇总与已有结果分开保存。
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


# 用于所有请求类报告的固定指标次序。
REQUEST_METRICS = [
    "e2e_ttft_seconds",
    "model_ttft_seconds",
    "e2e_total_seconds",
    "model_total_seconds",
    "queue_wait_seconds",
    "scheduled_total_seconds",
    "query_embedding_seconds",
    "retrieval_seconds",
    "context_assembly_seconds",
    "full_retrieval_seconds",
    "input_tokens_server",
    "output_tokens_server",
    "output_rate_server_model_tokens_per_second",
    "output_rate_server_e2e_tokens_per_second",
    "recall_at_k",
    "ndcg_at_k",
]


def parse_args() -> argparse.Namespace:
    """解析唯一位置参数：包含`commands.json`的执行目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("execution")
    return parser.parse_args()


def stats(values: list[int | float]) -> dict:
    """计算样本量、平均数、中位数与最近秩P95。"""
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
    }


def command_run_paths(execution_dir: Path, command: dict) -> tuple[Path, list[Path]]:
    """优先读取命令清单的运行目录，缺省时从私有目录的日志行取得路径。"""
    log_path = execution_dir / f"{command['label']}.log"
    if command.get("run_directory"):
        return log_path, [Path(command["run_directory"])]
    run_paths = [
        Path(line.strip())
        for line in log_path.read_text().splitlines()
        if line.strip().startswith(".agent/runs/")
    ]
    return log_path, run_paths


def collect_request_metrics(
    run_dir: Path, result: dict, cases: list[dict], allow_missing_batches: bool = False
) -> None:
    """汇总逐请求指标、repeat批次与符合条件的回答样例。

    正式请求按status为ok计入成功统计；warmup单列在phase统计中。
    就地更新 result 的指标字段，并向 cases 追加符合条件的回答样例。
    allow_missing_batches 为真时显式记录中断运行缺失的批次文件，吞吐保持 None。
    """
    request_path = run_dir / "requests.jsonl"
    all_rows = [json.loads(line) for line in request_path.read_text().splitlines()]
    formal_rows = [row for row in all_rows if row.get("phase") == "formal"]
    successful_rows = [row for row in formal_rows if row.get("status") == "ok"]
    repeats = sorted({row["repeat"] for row in formal_rows})
    result.update(
        requests=len(formal_rows),
        successes=len(successful_rows),
        failures=len(formal_rows) - len(successful_rows),
        query_ids=list(dict.fromkeys(row["query_id"] for row in formal_rows)),
        repeats=repeats,
        finish_reason_counts={
            str(reason): sum(
                row.get("finish_reason") == reason for row in successful_rows
            )
            for reason in {row.get("finish_reason") for row in successful_rows}
        },
    )
    result["metrics"] = {
        metric: stats(
            [row[metric] for row in successful_rows if row.get(metric) is not None]
        )
        for metric in REQUEST_METRICS
    }
    result["per_repeat"] = {
        str(repeat): {
            metric: stats(
                [
                    row[metric]
                    for row in successful_rows
                    if row["repeat"] == repeat and row.get(metric) is not None
                ]
            )
            for metric in REQUEST_METRICS
        }
        for repeat in repeats
    }
    result["phase_metrics"] = summarize_request_phases(
        all_rows, formal_rows, successful_rows, repeats
    )
    collect_batch_metrics(
        run_dir, formal_rows, successful_rows, repeats, result, allow_missing_batches
    )
    collect_answer_cases(successful_rows, cases)


def summarize_request_phases(
    all_rows: list[dict],
    formal_rows: list[dict],
    successful_rows: list[dict],
    repeats: list[int],
) -> dict:
    """分别汇总warmup、每轮首条完成的正式记录和剩余正式请求。"""
    first_request_ids = {
        next(
            (
                row.get("request_id")
                for row in formal_rows
                if row.get("repeat") == repeat
            ),
            None,
        )
        for repeat in repeats
    }
    phases = {
        "warmup": [row for row in all_rows if row.get("phase") == "warmup"],
        "first_logged_completion_per_repeat": [
            row for row in successful_rows if row.get("request_id") in first_request_ids
        ],
        "remaining_formal_requests": [
            row
            for row in successful_rows
            if row.get("request_id") not in first_request_ids
        ],
    }
    return {
        phase: {
            metric: stats(
                [row[metric] for row in phase_rows if row.get(metric) is not None]
            )
            for metric in REQUEST_METRICS
        }
        for phase, phase_rows in phases.items()
    }


def collect_batch_metrics(
    run_dir: Path,
    formal_rows: list[dict],
    successful_rows: list[dict],
    repeats: list[int],
    result: dict,
    allow_missing: bool = False,
) -> None:
    """按repeat wall记录复算成功吞吐、尝试吞吐和服务端输出token速率。"""
    result["missing_batch_files"] = []
    batches = []
    for repeat in repeats:
        repeat_path = run_dir / f"repeat-{repeat}.json"
        if allow_missing and not repeat_path.exists():
            result["missing_batch_files"].append(str(repeat_path))
            continue
        repeat_record = json.loads(repeat_path.read_text())
        wall_seconds = repeat_record["wall_seconds"]
        attempted_rows = [row for row in formal_rows if row["repeat"] == repeat]
        completed_rows = [row for row in attempted_rows if row["status"] == "ok"]
        actual_output_tokens = sum(
            row.get("output_tokens_server") or 0 for row in completed_rows
        )
        batches.append(
            {
                "repeat": repeat,
                "source": str(repeat_path),
                "source_sha256": hashlib.sha256(repeat_path.read_bytes()).hexdigest(),
                "wall_seconds": wall_seconds,
                "attempts": len(attempted_rows),
                "successes": len(completed_rows),
                "success_rps": len(completed_rows) / wall_seconds,
                "attempt_rps": len(attempted_rows) / wall_seconds,
                "output_tokens_server_actual": actual_output_tokens,
                "server_output_tokens_per_second": (
                    actual_output_tokens / wall_seconds
                    if any(
                        row.get("output_tokens_server") is not None
                        for row in completed_rows
                    )
                    else None
                ),
            }
        )
    result["batches"] = batches

    total_wall_seconds = sum(batch["wall_seconds"] for batch in batches)
    if result["missing_batch_files"] or total_wall_seconds == 0:
        result["batch_aggregate"] = None
    else:
        result["batch_aggregate"] = {
            "wall_seconds": total_wall_seconds,
            "success_rps": len(successful_rows) / total_wall_seconds,
            "attempt_rps": len(formal_rows) / total_wall_seconds,
            "server_output_tokens_per_second": (
                sum(batch["output_tokens_server_actual"] for batch in batches)
                / total_wall_seconds
                if any(
                    row.get("output_tokens_server") is not None
                    for row in successful_rows
                )
                else None
            ),
        }
    for metric in [
        "wall_seconds",
        "success_rps",
        "attempt_rps",
        "server_output_tokens_per_second",
    ]:
        result["metrics"]["batch_" + metric] = stats(
            [batch[metric] for batch in batches if batch.get(metric) is not None]
        )


def collect_answer_cases(successful_rows: list[dict], cases: list[dict]) -> None:
    """按既有筛选规则收集带来源行内引用且以stop结束的回答样例。"""
    case_fields = [
        "run_id",
        "query_id",
        "query",
        "retrieved_doc_ids",
        "included_doc_ids",
        "prompt",
        "answer",
        "finish_reason",
        "input_tokens_server",
        "output_tokens_server",
    ]
    for row in successful_rows:
        cited_ids = re.findall(r"\[DOC ([^\]]+)\]", row.get("answer", ""))
        included_ids = set(row.get("included_doc_ids", []))
        existing_case_ids = {case["query_id"] for case in cases}
        if (
            cited_ids
            and set(cited_ids) <= included_ids
            and row.get("finish_reason") == "stop"
            and row["query_id"] not in existing_case_ids
        ):
            cases.append({key: row.get(key) for key in case_fields})


def collect_resource_summary(run_dir: Path, result: dict) -> None:
    """从资源JSONL复算显存、WSL、进程、Docker和磁盘观测值。"""
    resource_path = run_dir / "resources.jsonl"
    if not resource_path.exists():
        return

    # 沿用既有统计口径，跳过资源记录的第一行。
    samples = [json.loads(line) for line in resource_path.read_text().splitlines()][1:]
    gpu_samples = []
    for sample in samples:
        try:
            gpu_samples.append(
                [
                    float(value.strip())
                    for value in sample["nvidia_smi_device_wide"].split(",")
                ]
            )
        except (KeyError, ValueError):
            pass

    resources = {
        "samples": len(samples),
        "gpu_samples": len(gpu_samples),
        "cpu_percent": stats(
            [
                sample["cpu_percent"]
                for sample in samples
                if isinstance(sample.get("cpu_percent"), (int, float))
            ]
        ),
        "timestamp_first_unix_seconds": next(
            (sample["timestamp"] for sample in samples if "timestamp" in sample), None
        ),
        "timestamp_last_unix_seconds": next(
            (
                sample["timestamp"]
                for sample in reversed(samples)
                if "timestamp" in sample
            ),
            None,
        ),
        "gpu_device_total_mib": gpu_samples[0][0] if gpu_samples else None,
        "gpu_device_used_max_mib": max(
            (sample[1] for sample in gpu_samples), default=None
        ),
        "gpu_utilization_max_percent": max(
            (sample[2] for sample in gpu_samples), default=None
        ),
        "wsl_memory_used_max_bytes": max(
            (
                sample["wsl_memory_swap"]["used_bytes"]
                for sample in samples
                if "used_bytes" in sample.get("wsl_memory_swap", {})
            ),
            default=None,
        ),
        "wsl_swap_used_max_bytes": max(
            (
                sample["wsl_memory_swap"]["swap_used_bytes"]
                for sample in samples
                if "swap_used_bytes" in sample.get("wsl_memory_swap", {})
            ),
            default=None,
        ),
    }

    for process_tree in ["python_process_tree", "vllm_process_tree"]:
        values = [
            sum(
                process.get("rss_bytes", 0)
                for process in sample[process_tree]
                if isinstance(process, dict)
            )
            for sample in samples
            if isinstance(sample.get(process_tree), list)
        ]
        resources[process_tree + "_rss_max_bytes"] = max(values, default=None)

    container_samples = []
    for sample in samples:
        try:
            container_samples.append(json.loads(sample["chroma_docker_stats"]))
        except (KeyError, ValueError):
            pass
    resources["chroma_cpu_max_percent"] = max(
        (
            float(container["CPUPerc"].strip("%"))
            for container in container_samples
            if container.get("CPUPerc")
        ),
        default=None,
    )
    resources["chroma_memory_usage_raw"] = [
        container.get("MemUsage") for container in container_samples
    ]
    resources["chroma_volume_startup_bytes"] = next(
        (
            sample["chroma_volume_size_bytes"]
            for sample in samples
            if "chroma_volume_size_bytes" in sample
        ),
        None,
    )
    for key in ["read_bytes", "write_bytes"]:
        values = [
            sample["disk_io_cumulative"][key]
            for sample in samples
            if key in sample.get("disk_io_cumulative", {})
        ]
        resources["wsl_disk_" + key + "_delta"] = (
            values[-1] - values[0] if len(values) > 1 else None
        )
    result["resources"] = resources


def collect_command_result(
    execution_dir: Path, command: dict, cases: list[dict]
) -> dict:
    """合并命令、原始运行记录、索引摘要与资源摘要。"""
    log_path, run_paths = command_run_paths(execution_dir, command)
    result = dict(command)
    if not run_paths:
        return result

    run_dir = run_paths[-1]
    result["run_directory"] = str(run_dir)
    result["source_trace"] = {
        "command_log": str(log_path),
        "requests": str(run_dir / "requests.jsonl"),
        "resources": str(run_dir / "resources.jsonl"),
        "config": str(run_dir / "config.json"),
        "code_snapshot": str(run_dir / "code-snapshot.json"),
    }
    source_paths = [
        run_dir / "requests.jsonl",
        run_dir / "resources.jsonl",
        run_dir / "config.json",
    ]
    result["source_trace"]["sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
        if path.exists()
    }

    request_path = run_dir / "requests.jsonl"
    if request_path.exists() and request_path.stat().st_size:
        collect_request_metrics(run_dir, result, cases)
    elif (run_dir / "summary.json").exists():
        result["index_summary"] = json.loads((run_dir / "summary.json").read_text())

    collect_resource_summary(run_dir, result)
    return result


def write_request_summaries(results: list[dict]) -> None:
    """写入逐请求统计CSV和便于阅读的汇总表。"""
    with Path(".agent/results/baseline-summary.csv").open("w", newline="") as output:
        fields = [
            "label",
            "run_directory",
            "repeat",
            "metric",
            "unit",
            "n",
            "mean",
            "median",
            "p95",
            "requests",
            "successes",
            "failures",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in results:
            grouped = {
                "all": item.get("metrics", {}),
                **item.get("per_repeat", {}),
                **item.get("phase_metrics", {}),
            }
            for repeat, metrics in grouped.items():
                for metric, value in metrics.items():
                    unit = (
                        "tokens/second"
                        if "tokens_per_second" in metric
                        else (
                            "requests/second"
                            if metric.endswith("_rps")
                            else (
                                "seconds"
                                if metric.endswith("_seconds")
                                else ("tokens" if "tokens" in metric else "ratio")
                            )
                        )
                    )
                    writer.writerow(
                        {
                            "label": item["label"],
                            "run_directory": item.get("run_directory"),
                            "repeat": repeat,
                            "metric": metric,
                            "unit": unit,
                            **value,
                            "requests": item.get("requests"),
                            "successes": item.get("successes"),
                            "failures": item.get("failures"),
                        }
                    )

    table = [
        "| 实验 | 正式n | 成功/失败 | 重复数 | E2E或检索mean(s) | median(s) | P95(s) | 来源run |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        metric = (
            "e2e_total_seconds"
            if item["label"].startswith("e2e")
            else "full_retrieval_seconds"
        )
        values = item.get("metrics", {}).get(metric)
        if values and values["n"]:
            table.append(
                f"| {item['label']} | {values['n']} | "
                f"{item.get('successes')}/{item.get('failures')} | "
                f"{len(item.get('repeats', []))} | "
                f"{values['mean']:.6f} | {values['median']:.6f} | "
                f"{values['p95']:.6f} | `{item.get('run_directory')}` |"
            )
    Path(".agent/results/baseline-table.txt").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )


def write_import_summaries(results: list[dict]) -> None:
    """按正式集合筛选fresh索引来源并写入导入JSON和CSV。"""
    registry_path = Path(".agent/state/index-run-registry.json")
    imports = []
    formal_collections = {
        result["index_summary"]["collection"]
        for result in results
        if "collection" in result.get("index_summary", {})
    }
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        for collection, run_path in registry.items():
            summary_path = Path(run_path) / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if collection in formal_collections and not summary.get("index_reused"):
                    imports.append({"source_run": run_path, **summary})

    Path(".agent/results/baseline-imports.json").write_text(
        json.dumps(imports, indent=2) + "\n"
    )
    with Path(".agent/results/baseline-import-summary.csv").open(
        "w", newline=""
    ) as output:
        fields = [
            "source_run",
            "tier",
            "documents",
            "chunks",
            "new_chunks_this_attempt",
            "setup_seconds",
            "preparation_seconds",
            "embedding_seconds",
            "write_seconds",
            "full_seconds",
            "measured_wall_seconds",
            "chunks_per_second",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(imports)


def write_batch_summary(results: list[dict]) -> None:
    """写入按repeat记录实际wall与服务端生成token的整批统计CSV。"""
    with Path(".agent/results/baseline-batch-summary.csv").open(
        "w", newline=""
    ) as output:
        fields = [
            "label",
            "run_directory",
            "repeat",
            "wall_seconds",
            "attempts",
            "successes",
            "success_rps",
            "attempt_rps",
            "output_tokens_server_actual",
            "server_output_tokens_per_second",
            "source",
            "source_sha256",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in results:
            for batch in item.get("batches", []):
                writer.writerow(
                    {
                        "label": item["label"],
                        "run_directory": item["run_directory"],
                        **batch,
                    }
                )


def write_outputs(execution_dir: Path, results: list[dict], cases: list[dict]) -> None:
    """写入复算产生的JSON、CSV和表格文件。"""
    Path(".agent/results").mkdir(parents=True, exist_ok=True)
    Path(".agent/results/baseline-results.json").write_text(
        json.dumps({"execution": str(execution_dir), "runs": results}, indent=2) + "\n"
    )
    Path(".agent/results/baseline-cases.json").write_text(
        json.dumps(cases[:3], indent=2, ensure_ascii=False) + "\n"
    )
    write_request_summaries(results)
    write_import_summaries(results)
    write_batch_summary(results)


def main() -> None:
    """读取执行清单，汇总每条命令并写入固定复算产物。"""
    args = parse_args()
    execution_dir = Path(args.execution)
    commands = json.loads((execution_dir / "commands.json").read_text())
    cases = []
    results = [
        collect_command_result(execution_dir, command, cases) for command in commands
    ]
    write_outputs(execution_dir, results, cases)

    for result in results:
        metrics = result.get("metrics", {})
        metric = (
            "e2e_total_seconds"
            if result["label"].startswith("e2e")
            else "full_retrieval_seconds"
        )
        print(
            result["label"],
            result.get("run_directory"),
            json.dumps(metrics.get(metric, result.get("index_summary", {}))),
        )


if __name__ == "__main__":
    main()
