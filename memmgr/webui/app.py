"""提供仅本机使用的 MemMgr 同源 Web API、真实业务流与只读报告。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .config import (
    ROOT,
    STATE,
    NAMES,
    QueryInput,
    TaskInput,
    configs,
    effective_config,
    read_json,
    save_json,
    safe_name,
    validate_configs,
    workspace_root,
)
from .tasks import resource_lock, start_task, read_task, cancel_task
from .records import complete_rows, incomplete_tail


@asynccontextmanager
async def lifespan(app):
    """在 app 启停时管理真实资源采样线程与后端日志。"""
    from ..benchmarks.resources import sample

    STATE.mkdir(parents=True, exist_ok=True)
    from .config import save_configs

    if (STATE / "config-transaction.json").exists():
        save_configs(configs())
    for path in (STATE / "requests").glob("*.json"):
        record = read_json(path)
        if record.get("status") == "running":
            record.update(
                status="interrupted",
                error="Backend process restarted before request completion",
                finished_at=time.time(),
            )
            save_json(path, record)
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        STATE / "backend.log", maxBytes=2 * 1024 * 1024, backupCount=1, encoding="utf-8"
    )
    logging.basicConfig(
        handlers=[handler],
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    stop = threading.Event()
    thread = threading.Thread(
        target=sample,
        args=(STATE, 1, stop, configs()["baseline"]),
        kwargs={"max_bytes": 2 * 1024 * 1024, "append": True},
        daemon=True,
    )
    thread.start()
    from .logs import ChromaLogs

    try:
        follower = ChromaLogs(configs()["baseline"]["resources"]["chroma_container"])
    except OSError as exc:
        logging.exception("Chroma log operation failed")
        save_json(
            STATE / "chroma-log-status.json",
            {"available": False, "reason": f"{type(exc).__name__}: {exc}"},
        )
        follower = None
    app.state.chroma_follower = follower
    app.state.chroma_log_lock = threading.Lock()
    for path in (STATE / "tasks").glob("*/task.json"):
        read_task(path.parent.name)
    logging.info("backend startup pid=%s", os.getpid())
    yield
    stop.set()
    if app.state.chroma_follower is not None:
        app.state.chroma_follower.close()
    thread.join(timeout=20)
    logging.info("backend shutdown pid=%s", os.getpid())


app = FastAPI(title="MemMgr local WebUI", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def protect_local(request, call_next):
    """检查 request 的写入来源并为全部真实请求分配日志关联 ID。"""
    origin = request.headers.get("origin")
    allowed = {
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    }
    if (
        request.method not in ("GET", "HEAD", "OPTIONS")
        and origin
        and origin not in allowed
    ):
        return JSONResponse({"detail": "Origin is not allowed"}, status_code=403)
    if (
        request.method not in ("GET", "HEAD", "OPTIONS")
        and request.headers.get("sec-fetch-site") == "cross-site"
    ):
        return JSONResponse(
            {"detail": "Cross-site writes are forbidden"}, status_code=403
        )
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    logging.info("request=%s %s %s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except BlockingIOError as exc:
        response = JSONResponse(
            {"detail": f"Shared GPU/data/config resource is busy: {exc}"},
            status_code=409,
        )
    except FileNotFoundError as exc:
        response = JSONResponse({"detail": str(exc)}, status_code=404)
    except ValueError as exc:
        response = JSONResponse({"detail": str(exc)}, status_code=422)
    except Exception as exc:
        logging.exception("request=%s failed", request_id)
        response = JSONResponse(
            {"detail": f"{type(exc).__name__}: {exc}"}, status_code=502
        )
    response.headers["X-Request-ID"] = request_id
    return response


def chroma():
    """按已保存本机 Chroma 配置创建真实客户端。"""
    import chromadb

    config = configs()["baseline"]["chroma"]
    return chromadb.HttpClient(host=config["host"], port=config["port"])


def page_result(result):
    """把 result 的 Chroma 列式结果转换为保留真实字段的分页记录。"""
    return [
        {
            "id": item,
            "text": result["documents"][i]
            if result.get("documents") is not None
            else None,
            "metadata": result["metadatas"][i]
            if result.get("metadatas") is not None
            else None,
        }
        for i, item in enumerate(result["ids"])
    ]


@app.get("/api/health")
async def health():
    """读取实际 vLLM/Chroma 健康接口并保留不可达的具体错误。"""
    result = {"backend": {"status": "ok", "pid": os.getpid(), "timestamp": time.time()}}
    from ..inference.service_status import health_snapshot
    from .runtime import capture_runtime

    snapshot = configs()
    result.update(await health_snapshot(snapshot["baseline"]))
    runtime = await owned_thread(capture_runtime, snapshot)
    result["vllm"].update(
        configuration_matches=runtime["configuration_matches"],
        restart_required=result["vllm"]["status"] == "ok"
        and not runtime["configuration_matches"],
        configuration_differences=runtime["differences"],
        runtime=runtime,
    )
    return result


@app.get("/api/services")
def services():
    """重新发现两个固定服务的真实健康、启动状态及进程归属。"""
    from .services import NAMES, service_status

    result = {name: service_status(name) for name in NAMES}
    if result["chroma"]["status"] == "running":
        from .logs import ChromaLogs

        with app.state.chroma_log_lock:
            follower = app.state.chroma_follower
            if follower is None or follower.process.poll() is not None:
                if follower is not None:
                    follower.close()
                try:
                    app.state.chroma_follower = ChromaLogs(
                        configs()["baseline"]["resources"]["chroma_container"]
                    )
                except OSError as exc:
                    save_json(
                        STATE / "chroma-log-status.json",
                        {"available": False, "reason": f"{type(exc).__name__}: {exc}"},
                    )
                    app.state.chroma_follower = None
    return {"services": result}


@app.get("/api/kv")
async def kv_status():
    """读取当前 vLLM 的真实 KV 指标及已保存策略配置。"""
    from prometheus_client.parser import text_string_to_metric_families

    snapshot = configs()
    url = snapshot["baseline"]["llm"]["base_url"].removesuffix("/v1") + "/metrics"
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"KV 指标读取失败：{exc}") from exc
    metrics = []
    for family in text_string_to_metric_families(response.text):
        for item in family.samples:
            if "kv_offload" in item.name or item.name in (
                "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                "vllm:external_prefix_cache_hits_total", "vllm:request_prefill_kv_computed_tokens_sum",
            ):
                metrics.append(dict(name=item.name, labels=item.labels, value=item.value))
    return dict(configured=snapshot["server"].get("kv_cache", {"tier": "gpu", "policy": "lru"}),
                metrics=metrics, captured_at=time.time())


@app.post("/api/services/{name}/start")
def service_start(name: str, value: dict):
    """以空 value 请求启动固定 name 服务，保留共享资源冲突及原始错误。"""
    from .services import NAMES, start_service

    if name not in NAMES:
        raise HTTPException(404, "Unknown service")
    if value:
        raise ValueError("Service start requires an empty object")
    result = start_service(name)
    return JSONResponse(
        result, status_code=200 if result.get("result") == "already_running" else 202
    )


@app.get("/api/resources")
def resources(
    since: float = 0, until: float = 1e20, limit: int = Query(600, ge=1, le=3600)
):
    """读取 since/until UNIX 秒范围内至多 limit 条真实资源观测。"""
    rows = []
    path = STATE / "resources.jsonl"
    if path.exists():
        from collections import deque

        rows = list(
            deque(
                (
                    row
                    for row in complete_rows(path)
                    if row.get("timestamp") is not None
                    and since <= row["timestamp"] <= until
                ),
                maxlen=limit,
            )
        )
    return {
        "items": rows,
        "units": {
            "timestamp": "unix_seconds",
            "cpu_percent": "percent, whole WSL host",
            "wsl_memory_swap": "bytes, WSL host",
            "nvidia_smi_device_wide": "memory.total MiB,memory.used MiB,utilization.gpu percent; whole GPU",
        },
        "since": since,
        "until": until,
        "retention": "latest bounded 2 MiB real sampling window",
        "incomplete_tail": incomplete_tail(path),
    }


@app.get("/api/config")
def get_config():
    """读取已保存 baseline/models/server 以及只读字段与生效说明。"""
    return configs()


@app.get("/api/config/schema")
def config_schema():
    """返回可编辑配置字段的现有实际类型及只读元数据。"""
    from .schema import BaselineConfig, ModelSnapshot, ServerConfig

    return {
        "values": configs(),
        "schema": {
            "baseline": BaselineConfig.model_json_schema(),
            "models": ModelSnapshot.model_json_schema(),
            "server": ServerConfig.model_json_schema(),
        },
        "readonly": [
            "baseline.dataset.root",
            "baseline.experiment.output_dir",
            "baseline.chroma",
            "baseline.resources",
            "server.prefix_caching",
            "server.chunked_prefill",
            "server.concurrency_semantics",
        ],
        "restart_required": ["server", "models", "baseline.llm"],
        "task_effect": "New tasks freeze saved baseline/models/server; running tasks never reread global files",
    }


@app.put("/api/config")
def put_configs(values: dict):
    """对 values 中 baseline/models/server 整组校验并事务保存，支持关联字段一起修改。"""
    from .config import save_configs

    if set(values) != set(NAMES):
        raise ValueError("Require baseline, models and server")
    with resource_lock():
        save_configs(values)
    return {
        "values": values,
        "restart_required": True,
        "effect": "New tasks consume frozen saved configuration; restart vLLM for server/model changes",
    }


@app.put("/api/config/{name}")
def put_config(name: str, value: dict):
    """验证并原子保存 name 的完整 value，资源争用立即拒绝。"""
    if name not in NAMES:
        raise HTTPException(404, "Unknown config")
    with resource_lock():
        all_configs = configs()
        all_configs[name] = value
        validate_configs(all_configs)
        from .config import save_configs

        save_configs(all_configs)
    return {
        "name": name,
        "value": value,
        "restart_required": name in ("server", "models"),
        "effect": "New tasks use saved configuration; vLLM changes require explicit service restart",
    }


@app.get("/api/workspaces")
def workspaces():
    """发现正式只读数据和所有隔离工作区的真实文件与索引状态。"""
    names = ["formal"] + sorted(
        p.name for p in (STATE / "workspaces").glob("*") if p.is_dir()
    )
    items = []
    for name in names:
        root = workspace_root(name)
        item = {
            "name": name,
            "readonly": name == "formal",
            "path": str(root),
            "prepared": (root / "metadata.json").exists(),
            "tiers": {},
        }
        if item["prepared"]:
            item["metadata"] = read_json(root / "metadata.json")
        for tier in ("small", "medium", "full"):
            item["tiers"][tier] = {
                "corpus_exists": (root / tier / "corpus.jsonl").exists()
            }
            if item["tiers"][tier]["corpus_exists"]:
                try:
                    from ..long_term.retrieval import open_collection

                    collection, manifest = open_collection(
                        effective_config(configs(), name, STATE / "unused"), tier
                    )
                    item["tiers"][tier].update(
                        indexed=True, collection=collection.name, manifest=manifest
                    )
                except Exception as exc:
                    item["tiers"][tier].update(
                        indexed=False, error=f"{type(exc).__name__}: {exc}"
                    )
        items.append(item)
    return {"items": items}


@app.post("/api/workspaces")
def create_workspace(value: dict):
    """创建 value.name 指定的空隔离目录，复用已校验官方压缩包以避免重复下载。"""
    name = safe_name(value["name"])
    if name == "formal":
        raise ValueError("formal is reserved")
    with resource_lock():
        root = STATE / "workspaces" / name
        root.mkdir(parents=True, exist_ok=False)
        archive = workspace_root("formal") / "scifact.zip"
        if archive.exists():
            import shutil

            shutil.copyfile(archive, root / "scifact.zip")
    return {"name": name, "path": str(root), "readonly": False}


@app.get("/api/tasks")
def tasks():
    """读取所有自有任务，后端重启后保留运行或中断精确状态。"""
    return {
        "items": sorted(
            [read_task(p.parent.name) for p in (STATE / "tasks").glob("*/task.json")],
            key=lambda x: x["created_at"],
            reverse=True,
        )
    }


@app.post("/api/tasks")
def task_create(value: TaskInput):
    """按 value 请求创建真实受约束任务并冻结当前配置。"""
    return start_task(value)


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str):
    """读取 task_id 的状态、真实日志进度和固定配置。"""
    return read_task(task_id)


@app.post("/api/tasks/{task_id}/cancel")
def task_cancel(task_id: str):
    """取消 task_id 所属的已核验独立进程组。"""
    return cancel_task(task_id)


@app.get("/api/collections")
def collections(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    """用 Chroma 服务端 limit/offset 分页发现集合。"""
    client = chroma()
    return {
        "items": [
            {"name": c.name, "count": c.count(), "metadata": c.metadata}
            for c in client.list_collections(limit=limit, offset=offset)
        ],
        "limit": limit,
        "offset": offset,
        "total": client.count_collections(),
    }


@app.get("/api/collections/{name}")
def collection_page(
    name: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    id: str | None = None,
    text: str | None = None,
):
    """对 name 集合执行服务端分页；id 精确过滤，text 使用服务端 contains。"""
    collection = chroma().get_collection(safe_name(name))
    kwargs = {"limit": limit, "offset": offset, "include": ["documents", "metadatas"]}
    if id:
        kwargs["ids"] = [id]
    if text:
        kwargs["where_document"] = {"$contains": text}
    return {
        "items": page_result(collection.get(**kwargs)),
        "limit": limit,
        "offset": offset,
        "total": collection.count(),
        "total_scope": "whole_collection",
        "metadata": collection.metadata,
    }


def retrieve(value, expected=None):
    """按 value 真实编码查询；expected 非空时校验所选集合身份一致。"""
    from ..long_term.index import make_embedder
    from ..long_term.retrieval import open_collection

    config = effective_config(configs(), value.workspace, STATE / "unused")
    started = time.perf_counter()
    collection, manifest = open_collection(config, value.tier)
    if expected and collection.name != expected:
        raise ValueError(
            "Selected collection does not match workspace/tier embedding identity"
        )
    _, embed = make_embedder(config)
    ready = time.perf_counter()
    vector = embed([value.query or value.question], query=True).tolist()
    encoded = time.perf_counter()
    result = collection.query(
        query_embeddings=vector,
        n_results=min(value.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    completed = time.perf_counter()
    hits = [
        {
            "id": item,
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i],
        }
        for i, item in enumerate(result["ids"][0])
    ]
    return {
        "hits": hits,
        "collection": collection.name,
        "identity": manifest["identity"],
        "distance_direction": "lower_is_better",
        "distance_metric": "cosine",
        "setup_seconds": ready - started,
        "query_embedding_seconds": encoded - ready,
        "retrieval_seconds": completed - encoded,
    }


@app.post("/api/collections/{name}/query")
def collection_query(name: str, value: QueryInput):
    """验证 name 对应 value 工作区身份并执行正式 embedding 规则的查询。"""
    with resource_lock():
        return retrieve(value, safe_name(name))


def memory_collection(namespace, create=False):
    """读取 namespace 的专有记忆集合；create 只在新增写入时允许创建。"""
    from ..long_term.memory import _collection
    from chromadb.errors import NotFoundError

    safe_name(namespace)
    config = memory_config()
    try:
        collection = _collection(config, namespace, create=create)
    except NotFoundError as exc:
        raise HTTPException(404, f"Memory namespace not found: {namespace}") from exc
    if collection.metadata.get(
        "namespace"
    ) != namespace or not collection.name.startswith("memmgr_web_memory_"):
        raise ValueError("Memory namespace identity mismatch")
    return collection


def memory_config():
    """返回固定专有记忆前缀和真实本地快照的业务配置。"""
    config = effective_config(configs(), "formal", STATE / "unused")
    config["chroma"]["collection_prefix"] = "memmgr_web"
    return config


@app.get("/api/memory")
def memory_namespaces():
    """从真实持久集合发现 Web 记忆 namespace，支持重启后恢复选择。"""
    client = chroma()
    items = []
    offset = 0
    while True:
        batch = client.list_collections(limit=100, offset=offset)
        if not batch:
            break
        items.extend(
            {"namespace": c.metadata["namespace"], "name": c.name, "count": c.count()}
            for c in batch
            if c.name.startswith("memmgr_web_memory_")
            and c.metadata
            and "namespace" in c.metadata
        )
        offset += len(batch)
    return {"items": items}


@app.get("/api/memory/{namespace}")
def memory_page(
    namespace: str, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
):
    """读取 namespace 的服务端 limit/offset 记忆分页。"""
    with resource_lock():
        collection = memory_collection(namespace)
        from ..long_term.memory import list_items

        return {
            "items": page_result(list_items(memory_config(), namespace, limit, offset)),
            "total": collection.count(),
            "limit": limit,
            "offset": offset,
        }


@app.post("/api/memory/{namespace}/query")
def memory_query(namespace: str, value: QueryInput):
    """以正式查询前缀和归一化向量查询 namespace 真实记忆。"""
    from ..long_term.memory import query

    with resource_lock():
        collection = memory_collection(namespace)
        result = query(
            memory_config(), namespace, value.query or value.question, value.top_k
        )
        return {
            "hits": [
                {
                    "id": item,
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result["distances"][0][i],
                }
                for i, item in enumerate(result["ids"][0])
            ],
            "distance_direction": "lower_is_better",
        }


def memory_write(namespace, value, memory_id=None):
    """新增或更新 namespace 的 value 文本及 metadata；memory_id 非空要求原记录存在。"""
    from ..long_term import memory

    identifier = memory_id or value["id"]
    if not isinstance(identifier, str) or not identifier or len(identifier) > 512:
        raise ValueError("Invalid memory ID")
    safe_name(namespace)
    with resource_lock():
        config = memory_config()
        if memory_id:
            memory_collection(namespace)
            memory.update(
                config,
                namespace,
                identifier,
                value["text"],
                metadata=value.get("metadata"),
            )
        else:
            memory.add(
                config,
                namespace,
                value["text"],
                identifier,
                metadata=value.get("metadata"),
            )
        result = memory.get(config, namespace, identifier)
        return {"item": page_result(result)[0], "readback_verified": True}


@app.post("/api/memory/{namespace}")
def memory_add(namespace: str, value: dict):
    """新增 namespace 的 value 记录并校验真实持久读回。"""
    return memory_write(namespace, value)


@app.put("/api/memory/{namespace}/{memory_id:path}")
def memory_update(namespace: str, memory_id: str, value: dict):
    """更新 namespace/memory_id 的 value 并核对读回。"""
    return memory_write(namespace, value, memory_id)


@app.delete("/api/memory/{namespace}/{memory_id:path}")
def memory_delete(namespace: str, memory_id: str, confirm: bool = False):
    """确认 confirm 后删除 namespace/memory_id，并验证其真实消失。"""
    if not confirm:
        raise ValueError("Deletion requires confirm=true")
    with resource_lock():
        from ..long_term.memory import delete

        memory_collection(namespace)
        delete(memory_config(), namespace, memory_id)
    return {"id": memory_id, "deleted": True, "readback_verified": True}


@app.get("/api/requests")
def requests_list(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    """分页返回持久问答记录，不重新生成回答。"""
    paths = sorted((STATE / "requests").glob("*.json"), reverse=True)
    return {
        "items": [read_json(p) for p in paths[offset : offset + limit]],
        "total": len(paths),
    }


@app.get("/api/requests/{request_id}")
def request_detail(request_id: str):
    """读取 request_id 的真实提示词、证据、用量和错误记录。"""
    return read_json(STATE / "requests" / f"{safe_name(request_id)}.json")


async def owned_thread(function, *args, **kwargs):
    """等待 function(*args, **kwargs) 的真实线程结束，取消时保持资源锁直到线程退出。"""
    import anyio

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            await task
        raise


@app.post("/api/qa")
async def qa(value: QueryInput, request: Request):
    """按 value 执行真实检索、提示词预算和流式问答，以 SSE 逐阶段发送并持久化。"""
    from ..inference.prompt import assemble_prompt
    from ..inference.streaming import stream_completion

    lock = resource_lock()
    lock.__enter__()
    record = None
    try:
        from .runtime import capture_runtime

        snapshot = configs()
        validate_configs(snapshot)
        from scripts.run_baseline import code_snapshot

        snapshot["source_sha256"] = code_snapshot()
        request_id = f"{time.time_ns()}_{uuid.uuid4().hex}"
        record = {
            "id": request_id,
            "http_request_id": request.state.request_id,
            "created_at": time.time(),
            "status": "running",
            "input": value.model_dump(),
            "snapshot": snapshot,
        }
        path = STATE / "requests" / f"{request_id}.json"
        save_json(path, record)
        snapshot["runtime"] = await owned_thread(
            capture_runtime, snapshot, require_llm=True
        )
        save_json(path, record)
    except BaseException as exc:
        try:
            if record is not None:
                record.update(
                    status="interrupted"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=time.time(),
                )
                save_json(path, record)
        finally:
            lock.__exit__(None, None, None)
        raise

    async def events():
        """将本次真实请求阶段与正文排入 SSE；请求断开会结束生成并保留记录。"""
        queue = asyncio.Queue(maxsize=128)

        async def emit(event, data):
            """将 event 及 data 推入有界事件队列。"""
            await queue.put((event, data))

        async def run():
            """运行问答业务并保存完整真实结果；异常保存原始类型及消息。"""
            admitted = time.perf_counter()
            try:
                await emit("request", {"id": request_id})
                await emit("stage", {"stage": "retrieval", "status": "running"})
                found = await owned_thread(retrieve, value)
                record["retrieval"] = found
                await emit("retrieval", found)
                start = time.perf_counter()
                from transformers import AutoTokenizer

                config = effective_config(
                    record["snapshot"], value.workspace, STATE / "unused"
                )
                tokenizer = await owned_thread(
                    AutoTokenizer.from_pretrained,
                    config["llm"]["tokenizer"],
                    local_files_only=True,
                )
                seen = set()
                candidates = []
                for hit in found["hits"]:
                    doc_id = hit["metadata"]["doc_id"]
                    if doc_id not in seen:
                        candidates.append((doc_id, hit["text"]))
                        seen.add(doc_id)
                budget = min(
                    value.context_tokens,
                    config["llm"]["max_model_len"] - config["llm"]["max_tokens"],
                )
                messages, included, evidence, ids, truncated = assemble_prompt(
                    tokenizer, value.question or value.query, candidates, budget
                )
                prompt = {
                    "messages": messages,
                    "rendered_prompt": tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ),
                    "included_doc_ids": included,
                    "evidence": evidence,
                    "budget": budget,
                    "requested_budget": value.context_tokens,
                    "input_tokens_local": len(ids),
                    "evidence_tokens": len(
                        tokenizer.encode(evidence, add_special_tokens=False)
                    ),
                    "truncated": truncated,
                    "context_assembly_seconds": time.perf_counter() - start,
                }
                record["prompt"] = prompt
                await emit("prompt", prompt)
                payload = {
                    "model": config["llm"]["model"],
                    "messages": messages,
                    "max_tokens": config["llm"]["max_tokens"],
                    "temperature": 0,
                    "seed": config["llm"]["seed"],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }

                async def delta(text):
                    """把真实 text 正文块转发至当前 SSE 连接。"""
                    await emit("delta", {"text": text})

                async with httpx.AsyncClient(
                    timeout=config["llm"]["timeout"], trust_env=False
                ) as client:
                    (
                        answer,
                        e2e_ttft,
                        model_ttft,
                        finish,
                        usage,
                        chunks,
                        total,
                    ) = await stream_completion(
                        client,
                        config["llm"]["base_url"] + "/chat/completions",
                        payload,
                        admitted,
                        on_delta=delta,
                    )
                record.update(
                    status="ok" if usage else "usage_missing",
                    answer=answer,
                    usage=usage,
                    finish_reason=finish,
                    stream_chunks=chunks,
                    e2e_ttft_seconds=e2e_ttft,
                    model_ttft_seconds=model_ttft,
                    model_total_seconds=total,
                    e2e_total_seconds=time.perf_counter() - admitted,
                    finished_at=time.time(),
                )
                save_json(path, record)
                await emit("done", record)
            except asyncio.CancelledError:
                record.update(
                    status="interrupted",
                    error="Client stream disconnected",
                    finished_at=time.time(),
                )
                save_json(path, record)
                raise
            except Exception as exc:
                record.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=time.time(),
                )
                logging.exception("qa=%s failed", request_id)
                save_json(path, record)
                await emit("error", {"id": request_id, "detail": record["error"]})
            finally:
                if asyncio.current_task().cancelling():
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait(None)
                else:
                    await queue.put(None)

        operation = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                name, data = event
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            if not operation.done():
                operation.cancel()
            import anyio

            with anyio.CancelScope(shield=True):
                with contextlib.suppress(asyncio.CancelledError):
                    await operation
            lock.__exit__(None, None, None)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/logs")
def log_sources():
    """列出经过实际调查的固定服务日志来源及任务日志。"""
    return {
        "items": [
            {"name": n}
            for n in ["backend", "vllm", "chroma", "service-vllm", "service-chroma"]
        ]
        + [
            {"name": "task-" + p.parent.name}
            for p in (STATE / "tasks").glob("*/task.json")
        ]
    }


@app.get("/api/logs/{source}")
def logs(
    source: str,
    cursor: int = Query(0, ge=0),
    limit: int = Query(32768, ge=1, le=65536),
    identity: str | None = None,
):
    """从 source 的真实日志读取有界字节，cursor 越界或 identity 轮换时重置。"""
    metadata = {}
    if source == "backend":
        path = STATE / "backend.log"
    elif source in ("service-vllm", "service-chroma"):
        from .services import service_path

        path = service_path(source.removeprefix("service-")) / "service.log"
        if not path.exists():
            return {
                "available": False,
                "reason": "No startup log exists; service has not been started by this WebUI",
                "text": None,
                "cursor": cursor,
            }
    elif source == "vllm":
        path = ROOT / "environment" / "vllm-server.log"
        if not path.exists():
            return {
                "available": False,
                "reason": "Running vLLM was started in a foreground terminal; no owned readable log file exists in environment/vllm-server.log",
                "text": None,
                "cursor": cursor,
            }
    elif source == "chroma":
        path = STATE / "chroma.log"
        status = read_json(STATE / "chroma-log-status.json")
        metadata = {
            "following_available": status["available"],
            "reason": status.get("reason"),
        }
        if not path.exists():
            return {**status, "text": None, "cursor": cursor, "log_path": str(path)}
    elif source.startswith("task-"):
        path = STATE / "tasks" / safe_name(source[5:]) / "task.log"
    else:
        raise HTTPException(404, "Unknown log source")
    stat = path.stat()
    current = f"{stat.st_dev}:{stat.st_ino}"
    reset = cursor > stat.st_size or (identity is not None and identity != current)
    start = 0 if reset else cursor
    with path.open("rb") as stream:
        stream.seek(start)
        chunk = stream.read(limit)
    return {
        **metadata,
        "available": True,
        "text": chunk.decode("utf-8", errors="replace"),
        "cursor": start + len(chunk),
        "reset": reset,
        "identity": current,
        "size": stat.st_size,
    }


@app.get("/api/reports")
def reports():
    """列出仓库已有真实报告，不改写任何正式实验产物。"""
    return {
        "items": [
            {"name": p.name, "size": p.stat().st_size}
            for p in (ROOT / "reports").glob("*.md")
        ]
    }


@app.get("/api/reports/{name}")
def report(name: str):
    """读取 allowlist 内 name 报告，并解析真实 Markdown 表格供图表使用。"""
    allowed = {p.name: p for p in (ROOT / "reports").glob("*.md")}
    if name not in allowed:
        raise HTTPException(404, "Unknown report")
    markdown = allowed[name].read_text(encoding="utf-8")
    tables = []
    block = []
    for line in markdown.splitlines() + [""]:
        if line.startswith("|"):
            block.append([cell.strip() for cell in line.strip("|").split("|")])
        elif block:
            if len(block) > 2:
                tables.append(
                    {
                        "headers": block[0],
                        "rows": block[2:],
                        "source": "markdown",
                        "values": "verbatim; units are in headers",
                    }
                )
            block = []
    return {"name": name, "markdown": markdown, "tables": tables}


def run_paths():
    """建立允许读取的正式历史与 Web 隔离运行目录映射，禁止任意路径访问。"""
    import hashlib

    paths = {}
    for root in [ROOT / ".agent" / "runs", STATE / "tasks"]:
        if root.exists():
            for file in root.rglob("requests.jsonl"):
                path = file.parent.resolve()
                if path.is_relative_to(root.resolve()):
                    paths[hashlib.sha256(str(path).encode()).hexdigest()[:24]] = path
    return paths


def read_page(path, offset, limit):
    """顺序读取 path 的 JSONL 第 offset 起 limit 行，避免加载全部请求和资源。"""
    import itertools

    if not path.exists():
        return None
    return list(itertools.islice(complete_rows(path), offset, offset + limit))


def run_summary(path):
    """读取 path 下实际存在的 JSON 或 CSV summary，空数值保留 None。"""
    import csv

    if (path / "summary.json").exists():
        return read_json(path / "summary.json")
    if not (path / "summary.csv").exists():
        return None
    rows = []
    with (path / "summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    key: None
                    if value == ""
                    else float(value)
                    if re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", value)
                    else value
                    for key, value in row.items()
                }
            )
    return {
        "source": "summary.csv",
        "rows": rows,
        "units": "Original metric field suffixes; seconds/tokens/ratios and counts are preserved",
    }


@app.get("/api/runs")
def runs():
    """发现真实 requests 产物所在运行目录，缺失历史时明确返回空列表。"""
    return {
        "items": [
            {
                "id": key,
                "name": path.name,
                "path": str(path),
                "summary": run_summary(path),
            }
            for key, path in run_paths().items()
        ]
    }


@app.get("/api/runs/{run_id}")
def run_detail(
    run_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)
):
    """分页读取 run_id 的真实原始请求/资源并返回原始 summary 与配置。"""
    paths = run_paths()
    if run_id not in paths:
        raise HTTPException(404, "Unknown run")
    path = paths[run_id]
    config_path = path / "config.json"
    return {
        "id": run_id,
        "name": path.name,
        "path": str(path),
        "summary": run_summary(path),
        "config": read_json(config_path) if config_path.exists() else None,
        "requests": read_page(path / "requests.jsonl", offset, limit),
        "resources": read_page(path / "resources.jsonl", offset, limit),
        "offset": offset,
        "limit": limit,
        "incomplete_tail": {
            "requests": incomplete_tail(path / "requests.jsonl"),
            "resources": incomplete_tail(path / "resources.jsonl"),
        },
    }


@app.get("/api/runs/{run_id}/metrics")
def run_metrics(run_id: str):
    """复用正式汇总模块计算 run_id 的真实请求、批次、资源和回答实例，缺失批次保留 None。"""
    from scripts.summarize_baseline import (
        collect_request_metrics,
        collect_resource_summary,
    )

    detail = run_detail(run_id, 0, 1)
    path = run_paths()[run_id]
    if incomplete_tail(path / "requests.jsonl") or incomplete_tail(
        path / "resources.jsonl"
    ):
        raise HTTPException(
            409, "Run has an uncommitted JSONL tail; metrics require completed lines"
        )
    summary = {}
    examples = []
    if any(complete_rows(path / "requests.jsonl")):
        collect_request_metrics(path, summary, examples, allow_missing_batches=True)
    else:
        summary = {
            "metrics": {},
            "requests": 0,
            "successes": 0,
            "batch_aggregate": None,
            "batches": [],
        }
    collect_resource_summary(path, summary)
    detail.pop("requests")
    detail["n"] = summary.get("successes")
    detail["metrics"] = {
        key: {
            **item,
            "unit": (
                "requests/second"
                if key.endswith("_rps")
                else "seconds"
                if key.endswith("_seconds")
                else "tokens/second"
                if "tokens_per_second" in key
                else "tokens"
                if "tokens" in key
                else "ratio"
            ),
            "population": "formal status=ok only; per-request descriptive statistics; batch metrics use actual repeat wall seconds",
        }
        for key, item in summary["metrics"].items()
    }
    detail["batch_aggregate"] = summary.get("batch_aggregate")
    detail["batches"] = summary.get("batches")
    detail["missing_batch_files"] = summary.get("missing_batch_files", [])
    detail["resources"] = summary.get("resources")
    detail["resource_population"] = (
        "Whole run sampling, including warmups; GPU whole device MiB/percent; WSL and process RSS bytes; missing values null"
    )
    detail["examples"] = examples
    detail["complete"] = (path / "summary.csv").exists() or (
        path / "summary.json"
    ).exists()
    return detail


@app.post("/api/compare")
def compare(value: dict):
    """比较 value.run_ids 的真实正式请求指标，复用报告 stats 并保留缺失与口径警告。"""
    identifiers = value["run_ids"]
    if not isinstance(identifiers, list) or not 2 <= len(identifiers) <= 8:
        raise ValueError("Select 2 to 8 runs")
    result = [run_metrics(identifier) for identifier in identifiers]
    differences = []
    first = result[0]["config"]
    for item in result[1:]:
        if item["config"] != first:
            differences.append(
                {
                    "left": result[0]["id"],
                    "right": item["id"],
                    "left_config": first,
                    "right_config": item["config"],
                }
            )
    return {
        "runs": result,
        "config_differences": differences,
        "warnings": [
            "Repeated queries are not independent samples; no significance test is performed.",
            "Different configurations, model/server snapshots, phases, units or sample populations may be statistically incompatible. Missing metrics remain null.",
        ],
        "missing_value": None,
    }


if (ROOT / "web" / "dist").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "web" / "dist", html=True), name="web")
