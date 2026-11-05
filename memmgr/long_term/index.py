"""将文献分块、编码并写入 Chroma，维护可复用的索引签名和导入清单。

签名覆盖语料内容、文本规范、模型版本与分块配置；集合名由身份摘要确定，持久化检索采用 cosine 距离。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def clean_text(text):
    """合并连续空白并清除首尾空白，空值按空字符串处理，返回规范化文本。"""
    return re.sub("\\s+", " ", text or "").strip()


def document_text(document):
    """把文献标题和正文以一个空格连接，再执行统一空白规范，返回待分块文本。"""
    return clean_text(
        (document.get("title", "") + " " + document.get("text", "")).strip(),
    )


def sha256_file(path):
    """按 1 MiB 分块读取文件并返回 SHA256，避免为整份语料建立额外内存副本。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def index_identity(tier_root, tier, embedding, chunk_tokens, overlap):
    """构建索引身份字典。

    tier_root 为某一级语料目录；身份包含绝对目录、corpus.jsonl 的真实摘要、模型
    revision、块长/重叠与文本格式版本。字典序列化结果决定集合名称。
    """
    return {
        "tier": tier,
        "dataset": str(Path(tier_root).resolve()),
        "corpus_sha256": sha256_file(Path(tier_root) / "corpus.jsonl"),
        "embedding_model": embedding["model"],
        "revision": embedding.get("revision"),
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": overlap,
        "text_format": "title-space-text-clean-whitespace-v1",
        "index_format": 3,
    }


def index_collection_name(prefix, tier, digest):
    """由前缀、语料级别和摘要生成最多 63 字符的集合名，优先保留摘要后缀以区分数据版本。"""
    suffix = f"_{tier}_{digest}"
    if len(suffix) > 62:
        suffix = f"_{tier[: 62 - len(digest) - 1]}_{digest}"
    return f"{prefix[: 63 - len(suffix)]}{suffix}"


def validate_chunk_config(max_tokens, overlap):
    """验证块长为正且 0 ≤ overlap < max_tokens；合法配置返回 None，越界抛出 ValueError。"""
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("Require max_tokens > 0 and 0 <= overlap < max_tokens")


def chunk_document(text, tokenizer, max_tokens=384, overlap=32):
    """用指定 tokenizer 按固定窗口分块并解码，返回文本块列表。

    步长为 max_tokens-overlap；编码时省略特殊 token，空文档仍返回一个空块，并以一个块 ID 表示空文档。
    """
    validate_chunk_config(max_tokens, overlap)
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max_tokens - overlap
    chunks = []
    for start in range(0, len(ids), step):
        chunk_text = ids[start : start + max_tokens]
        if not chunk_text:
            break
        chunks.append(tokenizer.decode(chunk_text, skip_special_tokens=True))
        if start + max_tokens >= len(ids):
            break
    return chunks or [""]


def make_embedder(config):
    """根据模型 ID、revision 和设备加载 SentenceTransformer，返回模型与编码闭包。

    闭包接收文本序列及 query 标记，输出归一化 NumPy 向量；仅查询添加配置中的检索指令。
    """
    from sentence_transformers import SentenceTransformer

    embedding_config = config["embedding"]
    model_options = {"device": embedding_config.get("device", "cpu")}
    if embedding_config.get("revision"):
        model_options["revision"] = embedding_config["revision"]
    embedding_model = SentenceTransformer(
        embedding_config.get("snapshot_path", embedding_config["model"]),
        **model_options,
    )

    def embed(texts, query=False):
        """将一批文本编码为归一化向量。

        query=True 时为每条查询加官方前缀，正文保持原内容；batch_size 取配置值，返回 NumPy
        数组以便调用者选择持久化格式。
        """
        prefix = embedding_config.get("query_instruction", QUERY_PREFIX)
        return embedding_model.encode(
            [prefix + t if query else t for t in texts],
            batch_size=int(embedding_config.get("batch_size", 64)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return (embedding_model, embed)


def index_tier(config, tier):
    """导入指定语料级别，校验数量与试检索后返回导入指标。

    已完成且身份一致的集合直接返回清单；部分导入按文本块 ID 跳过已有记录后继续。新增块按批编码/写入，结束时校验
    expected=count 并执行查询，最后持久化清单。阶段计时分别覆盖文本准备、编码和 add 返回，全流程含初始化及验证。
    """
    import chromadb

    tier_root = Path(config["dataset"].get("root", "data/scifact")) / tier
    import_started = time.perf_counter()
    corpus = {
        r["_id"]: r
        for r in map(json.loads, (tier_root / "corpus.jsonl").open(encoding="utf-8"))
    }
    embedding_config = config["embedding"]
    chunk_tokens = int(embedding_config.get("chunk_tokens", 384))
    overlap = int(embedding_config.get("chunk_overlap", 32))
    batch_size = int(embedding_config.get("batch_size", 64))
    validate_chunk_config(chunk_tokens, overlap)
    print(
        json.dumps(
            {
                "stage": "embedding_model_loading",
                "tier": tier,
                "documents_total": len(corpus),
            }
        ),
        flush=True,
    )
    embedding_model, embed = make_embedder(config)
    tokenizer = embedding_model.tokenizer
    chroma_config = config["chroma"]
    client = chromadb.HttpClient(
        host=chroma_config.get("host", "127.0.0.1"),
        port=int(chroma_config.get("port", 8001)),
    )
    identity = index_identity(
        tier_root,
        tier,
        embedding_config,
        chunk_tokens,
        overlap,
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    collection_name = index_collection_name(
        chroma_config.get("collection_prefix", "memmgr"),
        tier,
        digest,
    )
    manifest_directory = tier_root.parent / "index-manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_directory / f"{collection_name}.json"
    existing_names = client.list_collections()
    existing_names = {
        item if isinstance(item, str) else item.name for item in existing_names
    }
    # 已完成清单直接复用；部分写入按块 ID 续传，保护已落盘数据。
    if collection_name in existing_names:
        collection = client.get_collection(collection_name)
        if collection.metadata.get("memmgr:identity") != json.dumps(
            identity, sort_keys=True
        ):
            raise RuntimeError(
                f"Collection {collection_name} has an unexpected identity; refusing to reuse or overwrite",
            )
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("identity") != identity
                or manifest.get("collection") != collection_name
            ):
                raise RuntimeError("Existing index manifest identity differs")
            if collection.count() != manifest["expected_chunk_count"]:
                raise RuntimeError("Existing index count differs from manifest")
            return manifest
        existing_ids = set(collection.get(include=["metadatas"])["ids"])
    else:
        collection = client.create_collection(
            collection_name,
            metadata={
                "hnsw:space": "cosine",
                "memmgr:identity": json.dumps(identity, sort_keys=True),
            },
        )
        existing_ids = set()
    setup_done = time.perf_counter()
    preparation_seconds = embedding_seconds = write_seconds = 0.0
    new_chunk_count = 0
    expected_chunks = 0
    pending_ids = []
    pending_documents = []
    pending_metadata = []

    def flush():
        """提交当前待写批次，并累计编码耗时、写入耗时和本轮新增块数。

        空批次直接返回；成功后更新已存在 ID 并清空缓冲，后续迭代据此保持断点续传语义。
        """
        nonlocal embedding_seconds, write_seconds, new_chunk_count
        nonlocal pending_ids, pending_documents, pending_metadata
        if not pending_ids:
            return
        stage_started = time.perf_counter()
        print(
            json.dumps(
                {
                    "stage": "embedding",
                    "tier": tier,
                    "pending_chunks": len(pending_ids),
                    "written_chunks": new_chunk_count,
                }
            ),
            flush=True,
        )
        vectors = embed(pending_documents)
        encoding_finished = time.perf_counter()
        collection.add(
            ids=pending_ids,
            documents=pending_documents,
            metadatas=pending_metadata,
            embeddings=vectors.tolist(),
        )
        write_finished = time.perf_counter()
        embedding_seconds += encoding_finished - stage_started
        write_seconds += write_finished - encoding_finished
        new_chunk_count += len(pending_ids)
        print(
            json.dumps(
                {
                    "stage": "index_write_completed",
                    "tier": tier,
                    "written_chunks": new_chunk_count,
                    "expected_chunks_seen": expected_chunks,
                }
            ),
            flush=True,
        )
        existing_ids.update(pending_ids)
        pending_ids = []
        pending_documents = []
        pending_metadata = []

    for doc_id, document in corpus.items():
        stage_started = time.perf_counter()
        text = document_text(document)
        document_chunks = chunk_document(text, tokenizer, chunk_tokens, overlap)
        expected_chunks += len(document_chunks)
        for chunk_index, chunk_text in enumerate(document_chunks):
            chunk_id = f"{doc_id}::{chunk_index}"
            if chunk_id not in existing_ids:
                pending_ids.append(chunk_id)
                pending_documents.append(chunk_text)
                pending_metadata.append(
                    {
                        "doc_id": str(doc_id),
                        "chunk_index": chunk_index,
                        "title": clean_text(document.get("title", "")),
                    },
                )
        preparation_seconds += time.perf_counter() - stage_started
        if len(pending_ids) >= batch_size:
            flush()
    flush()
    print(
        json.dumps(
            {
                "stage": "index_validation",
                "tier": tier,
                "expected_chunks": expected_chunks,
            }
        ),
        flush=True,
    )
    observed_chunks = collection.count()
    if observed_chunks != expected_chunks:
        raise RuntimeError(
            f"Indexed chunk count mismatch: expected {expected_chunks}, observed {observed_chunks}",
        )
    first_doc_id, first_document = next(iter(corpus.items()))
    probe_query = (
        clean_text(first_document.get("title", "")) or document_text(first_document)
    )[:256]
    probe_start = time.perf_counter()
    probe_vector = embed([probe_query], query=True)[0].tolist()
    probe = collection.query(
        query_embeddings=[probe_vector],
        n_results=1,
        include=["metadatas", "distances"],
    )
    probe_result = {
        "query": probe_query,
        "returned_doc_ids": [m["doc_id"] for m in probe["metadatas"][0]],
        "distances": probe["distances"][0],
        "success": bool(probe["ids"][0]),
        "seconds": time.perf_counter() - probe_start,
    }
    if not probe_result["success"]:
        raise RuntimeError("Chroma probe query returned no indexed chunk")
    import_finished = time.perf_counter()
    manifest_data = {
        "tier": tier,
        "collection": collection_name,
        "identity": identity,
        "documents": len(corpus),
        "chunks": observed_chunks,
        "expected_chunk_count": expected_chunks,
        "new_chunks_this_attempt": new_chunk_count,
        "probe_query": probe_result,
        "preparation_seconds": preparation_seconds,
        "embedding_seconds": embedding_seconds,
        "write_seconds": write_seconds,
        "setup_seconds": setup_done - import_started,
        "full_seconds": import_finished - import_started,
        "chunks_per_second": observed_chunks
        / max(1e-09, import_finished - import_started),
        "embedding_model": embedding_config["model"],
        "embedding_revision": embedding_config.get("revision"),
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": overlap,
        "timing_definition": (
            "setup includes corpus/model/client/collection initialization; stage "
            "durations sum measured per-batch prep/embedding/write, excluding surrounding"
            " setup"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2),
        encoding="utf-8",
    )
    return manifest_data
