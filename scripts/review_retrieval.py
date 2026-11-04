"""检查正式检索请求的覆盖范围与唯一标识。

用法：运行 `python scripts/review_retrieval.py RUN_DIR`，其中 RUN_DIR 是单个
检索运行目录。脚本读取该目录的 `config.json` 与 `requests.jsonl`，在同目录
写入 `integrity-review.json` 并打印正式请求数、失败数和检查问题。
"""

import argparse
import json
from pathlib import Path


def review_retrieval_run(run_dir: Path) -> None:
    """检查各重复轮的问题覆盖、warmup和request ID。

    参数 `run_dir` 指向一轮检索原始结果目录。报告写入该目录的
    `integrity-review.json`；原始请求状态保存在 `requests.jsonl`，报告汇总失败数。
    """
    config = json.loads((run_dir / "config.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text().splitlines()
    ]
    expected_count = int(config["dataset"]["query_count"])
    formal_rows = [row for row in rows if row.get("phase") == "formal"]
    issues = []

    repeats = int(config["experiment"]["repeats"])
    query_ids = None
    for repeat in range(1, repeats + 1):
        repeated_rows = [
            row for row in formal_rows if row.get("repeat") == repeat
        ]
        ids = [row["query_id"] for row in repeated_rows]
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            issues.append(
                {
                    "repeat": repeat,
                    "error": "unexpected retrieval query coverage",
                    "expected": expected_count,
                    "actual": len(ids),
                    "unique": len(set(ids)),
                }
            )
        if query_ids is None:
            query_ids = ids
        elif set(ids) != set(query_ids):
            issues.append({"repeat": repeat, "error": "retrieval query set changed"})

    if int(config["experiment"].get("warmups", 0)) and not any(
        row.get("phase") == "warmup" for row in rows
    ):
        issues.append({"error": "warmup requests missing from raw JSONL"})

    request_ids = [row.get("request_id") for row in rows]
    if None in request_ids or len(request_ids) != len(set(request_ids)):
        issues.append({"error": "missing or duplicate request IDs"})

    result = {
        "run": str(run_dir),
        "requests": len(rows),
        "formal_requests": len(formal_rows),
        "expected_query_count": expected_count,
        "query_ids": query_ids,
        "issues": issues,
        "failed_requests": sum(
            row.get("status") != "ok" for row in formal_rows
        ),
    }
    (run_dir / "integrity-review.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


def main() -> None:
    """解析单个检索运行目录参数并启动完整性检查。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    review_retrieval_run(Path(args.run))


if __name__ == "__main__":
    main()
