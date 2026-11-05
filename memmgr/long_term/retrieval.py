"""为检索和生成测量提供共同的数据读取、索引定位与相关性评分。

文本块候选以原始文档 ID 去重，排序和官方 qrels 评分规则由本模块统一维护。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .index import index_identity, index_collection_name


def score_retrieval(retrieved, relevant, k):
    """按原始文档 ID 去重后计算 Recall@k 和二元 nDCG@k，返回两个浮点数。

    retrieved 为候选排名，relevant 为相关文档集合；k 是文本块候选上限。
    空相关集的分母取至少 1，理想增益分母使用 1e-12 下限。
    """
    ranked_doc_ids = []
    seen_doc_ids = set()
    for doc_id in retrieved:
        if doc_id not in seen_doc_ids:
            ranked_doc_ids.append(doc_id)
            seen_doc_ids.add(doc_id)
    hit_count = sum((doc_id in relevant for doc_id in ranked_doc_ids[:k]))
    recall = hit_count / max(1, len(relevant))
    discounted_gain = sum(
        (
            1 / math.log2(i + 2)
            for i, doc_id in enumerate(ranked_doc_ids[:k])
            if doc_id in relevant
        ),
    )
    ideal_gain = sum((1 / math.log2(i + 2) for i in range(min(len(relevant), k))))
    return (recall, discounted_gain / max(ideal_gain, 1e-12))


def read_rows(path):
    """读取 UTF-8 JSONL 为记录列表，按文件顺序保留每条原始记录。"""
    return [json.loads(x) for x in path.open(encoding="utf-8")]


def load_queries_and_qrels(config, tier):
    """读取某一级语料的问题和正相关标签，返回目录、问题映射及相关文档集合。

    问题顺序与 queries.jsonl 一致，相关性读取官方三列表；
    只有分值大于零的文档进入评分集合，调用者决定问题截取范围。
    """
    tier_root = Path(config["dataset"].get("root", "data/scifact")) / tier
    queries = {
        record["_id"]: record["text"]
        for record in read_rows(tier_root / "queries.jsonl")
    }
    qrels = {}
    for line in (tier_root / "qrels.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        query_id, doc_id, relevance_score = line.split("\t")
        if int(relevance_score) > 0:
            qrels.setdefault(query_id, set()).add(doc_id)
    return tier_root, queries, qrels


def open_collection(config, tier):
    """按语料内容和编码配置定位已完成索引，返回 (Chroma集合, 索引清单)。

    配置默认数据根仍为 data/scifact；集合签名与导入模块一致。清单缺失时抛出带 index 提示的
    RuntimeError，调用者只查询该集合。
    """
    import chromadb

    chroma_config = config["chroma"]
    dataset_root = Path(config["dataset"].get("root", "data/scifact"))
    embedding_config = config["embedding"]
    identity = index_identity(
        dataset_root / tier,
        tier,
        embedding_config,
        int(embedding_config.get("chunk_tokens", 384)),
        int(embedding_config.get("chunk_overlap", 32)),
    )
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode(),
    ).hexdigest()[:12]
    expected = index_collection_name(
        chroma_config.get("collection_prefix", "memmgr"),
        tier,
        digest,
    )
    manifest_path = dataset_root / "index-manifests" / f"{expected}.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Index manifest is missing: {manifest_path}; run index --tier {tier}",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = chromadb.HttpClient(
        host=chroma_config.get("host", "127.0.0.1"),
        port=int(chroma_config.get("port", 8001)),
    ).get_collection(
        manifest["collection"],
    )
    if manifest.get("identity") != identity:
        raise RuntimeError("Index manifest identity mismatch")
    if collection.name != expected:
        raise RuntimeError("Index collection name mismatch")
    if collection.metadata.get("memmgr:identity") != json.dumps(
        identity, sort_keys=True
    ):
        raise RuntimeError("Index collection metadata identity mismatch")
    if collection.count() != manifest["expected_chunk_count"]:
        raise RuntimeError("Index collection count mismatch")
    return (collection, manifest)
