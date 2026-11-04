"""复核端到端运行的原始请求、token预算与计时完整性。

用法：运行 `python scripts/review_run.py RUN_DIR`，其中 RUN_DIR 是单个端到端
运行目录。读取其中的配置和逐请求记录，将审查结果写入同目录的
`integrity-review.json`；失败项以退出码 1 报告。
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def summarize(values: list[float]) -> dict:
    """按最近秩计算数值序列的n、平均数、中位数和P95。

    空序列沿用审查结果中的n=0及空统计值表示。
    """
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": sorted(values)[math.ceil(len(values) * 0.95) - 1],
    }


def validate_query_coverage(
    formal_rows: list[dict], config: dict
) -> list[dict]:
    """按配置检查每次重复的固定问题数量、唯一性和问题集合。"""
    expected_queries = min(
        int(config["experiment"].get("queries", config["dataset"]["query_count"])),
        int(config["dataset"]["query_count"]),
    )
    issues = []
    fixed_query_ids = None
    for repeat in range(1, int(config["experiment"]["repeats"]) + 1):
        query_ids = [
            row["query_id"]
            for row in formal_rows
            if row.get("repeat") == repeat
        ]
        if (
            len(query_ids) != expected_queries
            or len(set(query_ids)) != expected_queries
        ):
            issues.append(
                {
                    "repeat": repeat,
                    "error": "unexpected E2E query coverage",
                    "expected": expected_queries,
                    "actual": len(query_ids),
                }
            )
        if fixed_query_ids is None:
            fixed_query_ids = query_ids
        elif set(query_ids) != set(fixed_query_ids):
            issues.append({"repeat": repeat, "error": "E2E query set changed"})
    return issues


def validate_successful_row(
    row: dict,
    context_limit: int,
    output_limit: int,
    input_budget: int,
) -> list[dict]:
    """检查成功请求的token、回答、计时分解和引用来源。"""
    issues = []
    query_id = row["query_id"]
    server_input_tokens = row.get("input_tokens_server")
    if server_input_tokens is not None:
        if server_input_tokens + output_limit > context_limit:
            issues.append({"query_id": query_id, "error": "server context overflow"})
        if row.get("input_tokens_local") != server_input_tokens:
            issues.append(
                {
                    "query_id": query_id,
                    "error": "local/server prompt token mismatch",
                }
            )
        if server_input_tokens > input_budget:
            issues.append({"query_id": query_id, "error": "input budget overflow"})

    if "run_id" in row and not row.get("usage_reported"):
        issues.append({"query_id": query_id, "error": "usage missing"})

    if "run_id" in row:
        if not row.get("answer", "").strip():
            issues.append({"query_id": query_id, "error": "empty answer"})
        if row.get("finish_reason") not in ["stop", "length"]:
            issues.append(
                {
                    "query_id": query_id,
                    "error": "unexpected finish status",
                    "finish_reason": row.get("finish_reason"),
                }
            )
        if (
            row.get("e2e_ttft_seconds") is None
            or row.get("model_ttft_seconds") is None
            or row["e2e_ttft_seconds"] < row["model_ttft_seconds"]
        ):
            issues.append({"query_id": query_id, "error": "invalid TTFT ordering"})
        if (
            row.get("e2e_total_seconds", 0)
            < row.get("model_total_seconds", 0)
        ):
            issues.append({"query_id": query_id, "error": "invalid elapsed ordering"})

        # E2E TTFT起于应用接纳；队列时间单列，阶段项为embedding、检索与组装耗时。
        stage_seconds = sum(
            row.get(key, 0)
            for key in [
                "query_embedding_seconds",
                "retrieval_seconds",
                "context_assembly_seconds",
            ]
        )
        if (
            row.get("e2e_ttft_seconds") is not None
            and row.get("model_ttft_seconds") is not None
        ):
            residual = (
                row["e2e_ttft_seconds"]
                - row["model_ttft_seconds"]
                - stage_seconds
            )
            if abs(residual) > 0.2:
                issues.append(
                    {
                        "query_id": query_id,
                        "error": "E2E TTFT contains unexplained queue/time",
                        "residual_seconds": residual,
                    }
                )

        cited_ids = re.findall(r"\[DOC ([^\]]+)\]", row.get("answer", ""))
        if not set(cited_ids) <= set(row.get("included_doc_ids", [])):
            issues.append(
                {
                    "query_id": query_id,
                    "error": "answer cites an ID absent from included evidence",
                    "cited": cited_ids,
                }
            )
    return issues


def summarize_metrics(formal_rows: list[dict]) -> dict:
    """按正式成功请求汇总各类计时、检索质量和token相关指标。"""
    metrics = {}
    metric_keys = [
        "e2e_ttft_seconds",
        "model_ttft_seconds",
        "e2e_total_seconds",
        "model_total_seconds",
        "retrieval_seconds",
        "query_embedding_seconds",
        "recall_at_k",
        "ndcg_at_k",
    ]
    for key in metric_keys:
        values = [
            row[key]
            for row in formal_rows
            if row["status"] == "ok" and row.get(key) is not None
        ]
        if values:
            metrics[key] = summarize(values)
    return metrics


def review_run(run_dir: Path) -> None:
    """复核一轮端到端原始记录并写入完整性报告。

    参数 `run_dir` 是已存在的运行目录；报告写入同目录的
    `integrity-review.json`。
    """
    rows = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text().splitlines()
    ]
    formal_rows = [row for row in rows if row.get("phase") == "formal"]
    config = json.loads((run_dir / "config.json").read_text())
    context_limit = config["llm"]["max_model_len"]
    output_limit = config["llm"]["max_tokens"]
    input_budget = config["experiment"]["context_tokens"]

    issues = validate_query_coverage(formal_rows, config)
    request_failures = []
    for row in rows:
        if row["status"] != "ok":
            target = issues if row["status"] == "usage_missing" else request_failures
            target.append(
                {
                    "request_id": row.get("request_id"),
                    "status": row["status"],
                    "error": row.get("error"),
                }
            )
            continue
        issues.extend(
            validate_successful_row(
                row,
                context_limit,
                output_limit,
                input_budget,
            )
        )

    result = {
        "run": str(run_dir),
        "requests": len(rows),
        "formal_requests": len(formal_rows),
        "successful": sum(row["status"] == "ok" for row in formal_rows),
        "issues": issues,
        "query_ids": list(
            dict.fromkeys(row["query_id"] for row in formal_rows)
        ),
        "metrics": summarize_metrics(formal_rows),
    }
    result["request_failures"] = request_failures
    result["finish_reason_counts"] = {
        key: sum(row.get("finish_reason") == key for row in formal_rows)
        for key in sorted({str(row.get("finish_reason")) for row in formal_rows})
    }
    result["truncated_answers"] = sum(
        row.get("finish_reason") == "length" for row in formal_rows
    )
    (run_dir / "integrity-review.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


def main() -> None:
    """解析运行目录参数并执行端到端完整性复核。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    review_run(Path(args.run))


if __name__ == "__main__":
    main()
