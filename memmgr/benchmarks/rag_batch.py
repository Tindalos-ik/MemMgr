"""组织端到端问答测量：一次 RAG 请求、预热、正式重复及批次汇总。

流式事件解析由 streaming 模块负责；本模块负责请求状态、单调计时和测量结果持久化。
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import shutil
import statistics
import time
import uuid
from pathlib import Path

from ..long_term.index import make_embedder
from ..inference.prompt import assemble_prompt
from ..long_term.retrieval import load_queries_and_qrels, open_collection, score_retrieval
from .run_metadata import write_run_metadata, finish_environment_snapshot
from ..inference.streaming import stream_completion
from .batch import append_request, gather_requests


def _select_queries(experiment, all_queries, query_override):
    """确定本轮问题及其顺序，返回 ID 到文本的映射。

    experiment.queries 按文件顺序截取问题；显式 query_override 生成稳定摘要 ID，
    用于单条 ask。截取范围和空集合行为遵循 Python 切片。
    """
    query_limit = int(experiment.get("queries", len(all_queries)))
    queries = dict(list(all_queries.items())[:query_limit])
    if query_override is not None:
        query_id = (
            "adhoc-"
            + hashlib.sha256(
                query_override.encode("utf-8"),
            ).hexdigest()[:16]
        )
        queries = {query_id: query_override}
    return queries


def _prepare_run_output(
    config, tier, concurrency, tier_root, queries, index_manifest, budget
):
    """创建本轮目录并保存配置、数据、源码与开始环境，返回运行 ID 和目录。

    queries 的顺序写入元数据；tier_root 下的三份数据复制到本轮目录。
    budget 记录完整提示词上限；初始化日志在资源线程和请求计时开始前写入。
    """
    experiment = config["experiment"]
    output_root = Path(experiment.get("output_dir", ".agent/runs"))
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output_directory = output_root / run_id
    (output_directory / "logs").mkdir(parents=True)
    (output_directory / "requests.jsonl").touch()
    (output_directory / "failures.jsonl").touch()
    (output_directory / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    dataset_directory = output_directory / "dataset"
    dataset_directory.mkdir()
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels.tsv"):
        shutil.copy2(tier_root / filename, dataset_directory / filename)
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
            "context_budget": budget,
        },
    )
    environment_path.write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    (output_directory / "logs" / "run.log").write_text(
        f"run_id={run_id} tier={tier} concurrency={concurrency} started_monotonic={time.perf_counter()}\n",
        encoding="utf-8",
    )
    return run_id, output_directory


def _write_repeat_result(output_directory, repeat_index, result, wall):
    """写入一次正式重复的整批统计 JSON。

    result 是按提交顺序返回的请求记录，wall 是整批单调时钟耗时。
    HTTP 成功按 request_success 计数；服务端实际 token 与缺用量估值分列。
    """
    successful = [record for record in result if record.get("request_success")]
    actual_tokens = sum(
        (
            record["output_tokens_server"]
            for record in successful
            if record.get("output_tokens_server") is not None
        ),
    )
    estimated_tokens = sum(
        (
            record["output_tokens_local_estimate"]
            for record in successful
            if record.get("output_tokens_local_estimate") is not None
        ),
    )
    (output_directory / f"repeat-{repeat_index}.json").write_text(
        json.dumps(
            {
                "repeat": repeat_index,
                "wall_seconds": wall,
                "requests_attempted": len(result),
                "successful_requests": len(successful),
                "failed_requests": len(result) - len(successful),
                "usage_missing": sum(
                    (record["status"] == "usage_missing" for record in successful)
                ),
                "batch_success_rate": len(successful) / max(1, len(result)),
                "batch_successful_rps": len(successful) / max(wall, 1e-09),
                "batch_attempt_rps": len(result) / max(wall, 1e-09),
                "output_tokens_server_actual": actual_tokens,
                "server_output_tokens_per_second": actual_tokens / max(wall, 1e-09),
                "output_tokens_local_estimate_usage_missing": estimated_tokens,
                "local_estimated_output_tokens_per_second_usage_missing": (
                    estimated_tokens / max(wall, 1e-09)
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_summary_csv(output_directory, all_formal, repeat_walls, repeats):
    """按正式重复汇总逐请求指标并写 summary.csv。

    只对成功请求的非空指标计算 n、平均值、中位数与最近秩 P95。
    各重复的实际 wall 提供成功/尝试吞吐分母；空重复保留表头和空统计。
    """
    metrics = [
        "queue_wait_seconds",
        "e2e_ttft_seconds",
        "model_ttft_seconds",
        "e2e_total_seconds",
        "scheduled_total_seconds",
        "model_total_seconds",
        "retrieval_seconds",
        "context_assembly_seconds",
        "output_rate_server_model_tokens_per_second",
        "output_rate_local_estimate_model_tokens_per_second",
        "recall_at_k",
        "ndcg_at_k",
    ]
    repeat_summaries = []
    for repeat_index in range(1, repeats + 1):
        rows = [
            record
            for record in all_formal
            if record["repeat"] == repeat_index and record.get("request_success")
        ]
        attempts = [record for record in all_formal if record["repeat"] == repeat_index]
        repeat_summary = {
            "repeat": repeat_index,
            "requests_attempted": len(attempts),
            "successes": len(rows),
            "success_rate": len(rows) / max(1, len(attempts)),
            "success_rps": len(rows) / max(repeat_walls[repeat_index], 1e-09),
            "attempt_rps": len(attempts) / max(repeat_walls[repeat_index], 1e-09),
            "wall_seconds": repeat_walls[repeat_index],
            "usage_missing": sum(
                (
                    (
                        record["repeat"] == repeat_index
                        and record["status"] == "usage_missing"
                    )
                    for record in all_formal
                ),
            ),
            "failures": sum(
                (
                    record["repeat"] == repeat_index and record["status"] == "failed"
                    for record in all_formal
                ),
            ),
        }
        for metric_name in metrics:
            values = [
                record[metric_name]
                for record in rows
                if record.get(metric_name) is not None
            ]
            repeat_summary[metric_name + "_n"] = len(values)
            repeat_summary[metric_name + "_mean"] = (
                statistics.mean(values) if values else None
            )
            repeat_summary[metric_name + "_median"] = (
                statistics.median(values) if values else None
            )
            repeat_summary[metric_name + "_p95"] = (
                sorted(values)[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
                if values
                else None
            )
        repeat_summaries.append(repeat_summary)
    with (output_directory / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        fields = (
            list(repeat_summaries[0])
            if repeat_summaries
            else ["repeat", "requests", "successes", "failures"]
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repeat_summaries)


async def run_rag_benchmark(config, tier, concurrency, query_override=None):
    """运行一次 RAG 实验，协调模型初始化、预热、正式重复和资源采样。

    参数：
        config 提供模型、查询范围、token 预算及输出目录。
        tier 选择语料级别，concurrency 控制已接纳请求数。
        query_override 指定单条 ask；省略时读取固定问题集合。

    副作用：
        模型在调用时加载，结果逐请求落盘；最后打印结果目录。
        结束时停止资源线程并保存结束环境，异常时保留已写记录。
    """
    import httpx
    from transformers import AutoTokenizer
    from .resources import sample

    tier_root, all_queries, qrels = load_queries_and_qrels(config, tier)
    experiment = config["experiment"]
    queries = _select_queries(experiment, all_queries, query_override)
    llm_config = config["llm"]
    tokenizer_name = llm_config.get("tokenizer", llm_config["model"])
    tokenizer_options = {}
    if llm_config.get("revision"):
        tokenizer_options["revision"] = llm_config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_options)
    _, embed = make_embedder(config)
    collection, index_manifest = open_collection(config, tier)
    top_k = int(experiment.get("top_k", 5))
    max_output_tokens = int(llm_config.get("max_tokens", 128))
    budget = min(
        int(experiment.get("context_tokens", 2048)),
        int(llm_config.get("max_model_len", 2048)) - max_output_tokens,
    )
    if budget < 1:
        raise ValueError("Context/token limits leave no input-token budget")
    run_id, output_directory = _prepare_run_output(
        config,
        tier,
        concurrency,
        tier_root,
        queries,
        index_manifest,
        budget,
    )
    import threading

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
    limits = httpx.Limits(max_connections=concurrency)
    base_url = llm_config.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
    async with httpx.AsyncClient(
        timeout=llm_config.get("timeout", 180),
        limits=limits,
    ) as http:
        fail_fast = bool(config.get("_web_fail_fast", False))

        async def run_request(query_id, query, phase, repeat):
            """执行一个问题，返回检索、生成和计时记录并追加写入 JSONL。

            query_id 和 query 标识问题，phase 和 repeat 标识预热或正式重复。
            信号量覆盖编码、检索和生成；接纳前等待单列，接纳后开始 E2E 计时。

            传输失败记录 failed；传输成功但服务端用量缺失记录 usage_missing。
            服务端实际 token 与本地估值分列，failed 同时写入 failures.jsonl。
            """
            request_start = time.perf_counter()
            request_id = uuid.uuid4().hex
            request_record = {
                "run_id": run_id,
                "request_id": request_id,
                "query_id": query_id,
                "tier": tier,
                "concurrency": concurrency,
                "phase": phase,
                "repeat": repeat,
                "query": query,
                "status": "failed",
            }
            try:
                # 接纳前只累计应用队列；信号量覆盖完整检索到生成链路。
                async with semaphore:
                    acquired = time.perf_counter()
                    request_record["queue_wait_seconds"] = acquired - request_start
                    admitted_start = acquired
                    stage_started = time.perf_counter()

                    def encode_query():
                        """持有共享编码锁生成当前查询的一个向量，供线程池调用，避免多个请求同时进入同一 CPU 编码器。"""
                        with embedding_lock:
                            return embed([query], query=True)[0].tolist()

                    query_embedding = await asyncio.to_thread(encode_query)
                    request_record["query_embedding_seconds"] = (
                        time.perf_counter() - stage_started
                    )
                    stage_started = time.perf_counter()
                    retrieved_chunks = await asyncio.to_thread(
                        collection.query,
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"],
                    )
                    request_record["retrieval_seconds"] = (
                        time.perf_counter() - stage_started
                    )
                    retrieved_doc_ids = []
                    candidate_documents = []
                    for chunk_text, chunk_metadata in zip(
                        retrieved_chunks["documents"][0],
                        retrieved_chunks["metadatas"][0],
                    ):
                        doc_id = str(chunk_metadata["doc_id"])
                        if doc_id not in retrieved_doc_ids:
                            retrieved_doc_ids.append(doc_id)
                            candidate_documents.append((doc_id, chunk_text))
                    request_record["retrieved_doc_ids"] = retrieved_doc_ids
                    stage_started = time.perf_counter()
                    (
                        messages,
                        included_doc_ids,
                        evidence,
                        input_ids,
                        truncated,
                    ) = assemble_prompt(
                        tokenizer,
                        query,
                        candidate_documents,
                        budget,
                    )
                    request_record.update(
                        {
                            "included_doc_ids": included_doc_ids,
                            "prompt": messages,
                            "input_tokens_local": len(input_ids),
                            "evidence_tokens": len(
                                tokenizer.encode(evidence, add_special_tokens=False)
                            ),
                            "truncated": truncated,
                        },
                    )
                    request_record["context_assembly_seconds"] = (
                        time.perf_counter() - stage_started
                    )
                    payload = {
                        "model": llm_config["model"],
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": max_output_tokens,
                        "seed": int(llm_config.get("seed", 42)),
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    (
                        answer,
                        ttft,
                        model_ttft,
                        finish,
                        usage,
                        chunks,
                        model_total,
                    ) = await stream_completion(
                        http,
                        base_url + "/chat/completions",
                        payload,
                        admitted_start,
                    )
                    total = time.perf_counter() - admitted_start
                    scheduled_total = time.perf_counter() - request_start
                    local_output_tokens = len(
                        tokenizer.encode(answer, add_special_tokens=False)
                    )
                    actual_output_tokens = (
                        usage.get("completion_tokens") if usage else None
                    )
                    usage_complete = (
                        usage is not None
                        and usage.get("prompt_tokens") is not None
                        and actual_output_tokens is not None
                    )
                    estimated_output_tokens = (
                        local_output_tokens if actual_output_tokens is None else None
                    )
                    recall, ndcg = (
                        score_retrieval(retrieved_doc_ids, qrels[query_id], top_k)
                        if query_id in qrels
                        else (None, None)
                    )
                    request_record.update(
                        {
                            "status": "ok" if usage_complete else "usage_missing",
                            "request_success": True,
                            "answer": answer,
                            "finish_reason": finish,
                            "input_tokens_server": usage.get("prompt_tokens")
                            if usage
                            else None,
                            "output_tokens_server": actual_output_tokens,
                            "output_tokens_local_estimate": estimated_output_tokens,
                            "input_tokens_local": len(input_ids),
                            "usage_reported": usage is not None,
                            "model_ttft_seconds": model_ttft,
                            "e2e_ttft_seconds": ttft,
                            "model_total_seconds": model_total,
                            "e2e_total_seconds": total,
                            "scheduled_total_seconds": scheduled_total,
                            "stream_chunks": chunks,
                            "output_rate_server_model_tokens_per_second": (
                                actual_output_tokens / max(model_total, 1e-09)
                                if actual_output_tokens is not None
                                else None
                            ),
                            "output_rate_server_e2e_tokens_per_second": (
                                actual_output_tokens / max(total, 1e-09)
                                if actual_output_tokens is not None
                                else None
                            ),
                            "output_rate_local_estimate_model_tokens_per_second": (
                                estimated_output_tokens / max(model_total, 1e-09)
                                if estimated_output_tokens is not None
                                else None
                            ),
                            "output_rate_local_estimate_e2e_tokens_per_second": (
                                estimated_output_tokens / max(total, 1e-09)
                                if estimated_output_tokens is not None
                                else None
                            ),
                            "recall_at_k": recall,
                            "ndcg_at_k": ndcg,
                        },
                    )
            except Exception as exc:
                request_record["error"] = repr(exc)
            append_request(output_directory, request_record)
            if fail_fast and request_record["status"] == "failed":
                raise RuntimeError(request_record["error"])
            return request_record

        try:
            for query_id in list(queries)[: int(experiment.get("warmups", 0))]:
                await run_request(query_id, queries[query_id], "warmup", 0)
            repeats = int(experiment.get("repeats", 1))
            all_formal = []
            repeat_walls = {}
            for repeat_index in range(1, repeats + 1):
                started = time.perf_counter()
                result = await gather_requests(
                    (
                        run_request(query_id, queries[query_id], "formal", repeat_index)
                        for query_id in queries
                    ),
                    fail_fast=fail_fast,
                )
                wall = time.perf_counter() - started
                repeat_walls[repeat_index] = wall
                all_formal += result
                _write_repeat_result(output_directory, repeat_index, result, wall)
            _write_summary_csv(output_directory, all_formal, repeat_walls, repeats)
        finally:
            stop_sampling.set()
            sampling_thread.join(timeout=5)
            finish_environment_snapshot(output_directory, config)
            with (output_directory / "logs" / "run.log").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(f"finished_monotonic={time.perf_counter()}\n")
    print(output_directory)
