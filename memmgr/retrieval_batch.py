"""运行纯检索批次，记录编码、数据库耗时、文档级相关性与整批吞吐。

本模块记录实验阶段、逐请求指标和错误状态；生成模型只在独立 RAG 模块使用。
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import shutil
import statistics
import time
import uuid
from pathlib import Path

from .index import make_embedder
from .retrieval import load_queries_and_qrels, open_collection, score_retrieval
from .run_metadata import write_run_metadata, finish_environment_snapshot


def _prepare_retrieval_output(
    config, tier, concurrency, tier_root, queries, index_manifest
):
    """创建纯检索结果目录和开始快照，返回运行 ID、目录和日志路径。

    保存配置、语料副本、实际查询顺序及索引签名，随后记录初始化时刻。
    """
    experiment = config["experiment"]
    run_id = "retrieval-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output_directory = Path(experiment.get("output_dir", ".agent/runs")) / run_id
    output_directory.mkdir(parents=True)
    (output_directory / "logs").mkdir()
    (output_directory / "requests.jsonl").touch()
    (output_directory / "failures.jsonl").touch()
    (output_directory / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    dataset_out = output_directory / "dataset"
    dataset_out.mkdir()
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels.tsv"):
        shutil.copy2(tier_root / filename, dataset_out / filename)
    write_run_metadata(
        output_directory,
        config,
        tier,
        list(queries),
        index_manifest,
    )
    environment_path = output_directory / "env.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["before"].update(
        {
            "tier": tier,
            "concurrency": concurrency,
            "collection": index_manifest["collection"],
            "query_ids": list(queries),
            "mode": "retrieval-only",
        },
    )
    environment_path.write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    log_path = output_directory / "logs" / "test.log"
    log_path.write_text(
        f"run_id={run_id} tier={tier} concurrency={concurrency} started_monotonic={time.perf_counter()}\n",
        encoding="utf-8",
    )
    return run_id, output_directory, log_path


def _write_retrieval_repeat(
    output_directory, repeat_index, batch, wall, persist_requests=True
):
    """保存一次检索重复的原始请求、失败记录与统计，返回该重复的汇总字典。

    batch 保留提交顺序，wall 提供成功/尝试吞吐分母。
    只对成功请求的非空指标计算统计；全部请求仍保留在 JSONL。
    persist_requests 为假时请求已在 fail-fast 路径逐条持久化，不重复追加。
    """
    successful_requests = [record for record in batch if record["status"] == "ok"]
    if persist_requests:
        with (output_directory / "requests.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            for request_record in batch:
                stream.write(json.dumps(request_record, ensure_ascii=False) + "\n")
        with (output_directory / "failures.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            for request_record in batch:
                if request_record["status"] == "failed":
                    stream.write(json.dumps(request_record, ensure_ascii=False) + "\n")
    repeat_summary = {
        "repeat": repeat_index,
        "requests_attempted": len(batch),
        "successes": len(successful_requests),
        "failures": len(batch) - len(successful_requests),
        "success_rate": len(successful_requests) / max(1, len(batch)),
        "successful_rps": len(successful_requests) / max(wall, 1e-09),
        "attempt_rps": len(batch) / max(wall, 1e-09),
        "wall_seconds": wall,
    }
    for metric in (
        "query_embedding_seconds",
        "retrieval_seconds",
        "full_retrieval_seconds",
        "recall_at_k",
        "ndcg_at_k",
    ):
        values = [
            record[metric]
            for record in successful_requests
            if record.get(metric) is not None
        ]
        repeat_summary[metric + "_n"] = len(values)
        repeat_summary[metric + "_mean"] = statistics.mean(values) if values else None
        repeat_summary[metric + "_median"] = (
            statistics.median(values) if values else None
        )
        repeat_summary[metric + "_p95"] = (
            sorted(values)[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
            if values
            else None
        )
    (output_directory / f"repeat-{repeat_index}.json").write_text(
        json.dumps(repeat_summary, indent=2),
        encoding="utf-8",
    )
    return repeat_summary


def _write_summary_csv(output_directory, summary_rows):
    """把各重复汇总按重复顺序写成 CSV，空结果保留 repeat 表头。"""
    with (output_directory / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(summary_rows[0]) if summary_rows else ["repeat"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


async def run_retrieval_benchmark(config, tier, concurrency):
    """测量指定语料全部查询的编码、检索和相关性，打印结果目录。

    config 决定语料、K、预热和重复数；tier 选择语料级别。
    concurrency 覆盖编码与检索，编码器另用线程锁串行访问。

    批次只协调初始化、预热、正式重复和资源线程；统计由独立函数落盘。
    结束时保存环境，即使批次异常也关闭采样并保留已有请求。
    """
    import threading
    from .resources import sample

    tier_root, queries, qrels = load_queries_and_qrels(config, tier)
    _, embed = make_embedder(config)
    collection, index_manifest = open_collection(config, tier)
    experiment = config["experiment"]
    top_k = int(experiment.get("top_k", 5))
    run_id, output_directory, log_path = _prepare_retrieval_output(
        config,
        tier,
        concurrency,
        tier_root,
        queries,
        index_manifest,
    )
    stop_sampling = threading.Event()
    sampling_thread = threading.Thread(
        target=sample,
        args=(
            output_directory,
            float(config.get("resources", {}).get("sample_interval_seconds", 1)),
            stop_sampling,
            config,
        ),
        daemon=True,
    )
    sampling_thread.start()
    semaphore = asyncio.Semaphore(concurrency)
    embedding_lock = threading.Lock()
    from .batch import append_request, gather_requests

    fail_fast = bool(config.get("_web_fail_fast", False))

    async def run_query(query_id, query, phase, repeat):
        """执行一个检索请求并返回记录字典，异常写入该记录。

        query_id 和 query 是问题 ID 与文本，phase 和 repeat 标识测量阶段。
        应用排队单列，full_retrieval 从接纳后开始；失败时该字段从排队开始计时。
        K 个块按原始 doc_id 去重后与官方 qrels 评分，由外层批量循环统一落盘。
        """
        request_id = uuid.uuid4().hex
        scheduled_start = time.perf_counter()
        request_record = {
            "run_id": run_id,
            "request_id": request_id,
            "query_id": query_id,
            "query": query,
            "tier": tier,
            "concurrency": concurrency,
            "phase": phase,
            "repeat": repeat,
            "status": "failed",
        }
        try:
            async with semaphore:
                admitted_start = time.perf_counter()
                request_record["queue_wait_seconds"] = admitted_start - scheduled_start
                start = time.perf_counter()
                embedding_started = start

                def encode_query():
                    """在共享线程锁内编码当前查询，返回单个向量，供 asyncio 线程池调度。"""
                    with embedding_lock:
                        return embed([query], query=True)[0].tolist()

                query_embedding = await asyncio.to_thread(encode_query)
                embedding_finished = time.perf_counter()
                retrieved_chunks = await asyncio.to_thread(
                    collection.query,
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["metadatas"],
                )
                retrieval_finished = time.perf_counter()
                retrieved_doc_ids = []
                seen_doc_ids = set()
                for chunk_metadata in retrieved_chunks["metadatas"][0]:
                    doc_id = str(chunk_metadata["doc_id"])
                    if doc_id not in seen_doc_ids:
                        retrieved_doc_ids.append(doc_id)
                        seen_doc_ids.add(doc_id)
                recall, ndcg = score_retrieval(
                    retrieved_doc_ids, qrels.get(query_id, set()), top_k
                )
                request_record.update(
                    {
                        "status": "ok",
                        "query_embedding_seconds": embedding_finished
                        - embedding_started,
                        "retrieval_seconds": retrieval_finished - embedding_finished,
                        "full_retrieval_seconds": time.perf_counter() - start,
                        "scheduled_total_seconds": time.perf_counter()
                        - scheduled_start,
                        "retrieved_doc_ids": retrieved_doc_ids,
                        "recall_at_k": recall,
                        "ndcg_at_k": ndcg,
                    },
                )
        except Exception as exc:
            request_record.update(
                {
                    "error": repr(exc),
                    "full_retrieval_seconds": time.perf_counter() - scheduled_start,
                },
            )
        if fail_fast:
            append_request(output_directory, request_record)
            if request_record["status"] == "failed":
                raise RuntimeError(request_record["error"])
        return request_record

    try:
        warmup_count = int(experiment.get("warmups", 0))
        if warmup_count:
            warmups = await gather_requests(
                (
                    run_query(query_id, queries[query_id], "warmup", 0)
                    for query_id in list(queries)[:warmup_count]
                ),
                fail_fast=fail_fast,
            )
            if not fail_fast:
                with (output_directory / "requests.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    for request_record in warmups:
                        stream.write(
                            json.dumps(request_record, ensure_ascii=False) + "\n"
                        )
                with (output_directory / "failures.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    for request_record in warmups:
                        if request_record["status"] == "failed":
                            stream.write(
                                json.dumps(request_record, ensure_ascii=False) + "\n"
                            )
        repeats = int(experiment.get("repeats", 1))
        summary_rows = []
        for repeat_index in range(1, repeats + 1):
            started = time.perf_counter()
            batch = await gather_requests(
                (
                    run_query(query_id, query, "formal", repeat_index)
                    for query_id, query in queries.items()
                ),
                fail_fast=fail_fast,
            )
            wall = time.perf_counter() - started
            summary_rows.append(
                _write_retrieval_repeat(
                    output_directory,
                    repeat_index,
                    batch,
                    wall,
                    persist_requests=not fail_fast,
                )
            )
        _write_summary_csv(output_directory, summary_rows)
    finally:
        stop_sampling.set()
        sampling_thread.join(timeout=5)
        finish_environment_snapshot(output_directory, config)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"finished_monotonic={time.perf_counter()}\n")
    print(output_directory)
