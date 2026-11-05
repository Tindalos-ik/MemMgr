"""提供长期记忆命名空间隔离、真实向量 CRUD、服务端分页及写后读回验证。"""

from __future__ import annotations

import hashlib
import uuid
from .index import make_embedder


def collection_name(prefix, namespace):
    """以 prefix 和 namespace 摘要建立稳定集合名，长度遵守 Chroma 限制。"""
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:24]
    suffix = f"_memory_{digest}"
    return f"{(prefix or 'm')[: 63 - len(suffix)]}{suffix}"


def _collection(config, namespace, create=True):
    """按 config 定位 namespace；create 为真允许创建，否则只读取既有集合。"""
    import chromadb

    chroma = config["chroma"]
    client = chromadb.HttpClient(
        host=chroma.get("host", "127.0.0.1"), port=int(chroma.get("port", 8001))
    )
    name = collection_name(chroma.get("collection_prefix", "memmgr"), namespace)
    if create:
        collection = client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine", "namespace": namespace}
        )
    else:
        collection = client.get_collection(name)
    if collection.metadata.get("namespace") != namespace:
        raise ValueError("Memory namespace identity mismatch")
    return collection


def _encoder(config, embed):
    """返回传入 embed 编码器，缺省时按 config 加载共享正式编码规则。"""
    return embed if embed is not None else make_embedder(config)[1]


def _metadata(namespace, metadata):
    """校验 metadata 为 Chroma 标量字典并持久保存真实 namespace。"""
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or not isinstance(v, (str, int, float, bool))
        for k, v in metadata.items()
    ):
        raise ValueError("Metadata must contain string keys and scalar values")
    return {**metadata, "namespace": namespace}


def add(config, namespace, text, memory_id=None, metadata=None, embed=None):
    """按 config 向 namespace 新增 text/memory_id/metadata，embed 可复用编码器；重复 ID 立即失败。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty text is required")
    memory_id = memory_id or str(uuid.uuid4())
    collection = _collection(config, namespace)
    if collection.get(ids=[memory_id])["ids"]:
        raise ValueError("Memory ID already exists")
    metadata = _metadata(namespace, metadata)
    collection.add(
        ids=[memory_id],
        documents=[text],
        metadatas=[metadata],
        embeddings=_encoder(config, embed)([text]).tolist(),
    )
    actual = collection.get(ids=[memory_id], include=["documents", "metadatas"])
    if actual["documents"] != [text] or actual["metadatas"] != [metadata]:
        raise RuntimeError("Memory add readback mismatch")
    return memory_id


def update(config, namespace, memory_id, text, metadata=None, embed=None):
    """更新已存在 namespace/memory_id 的 text/metadata；embed 允许复用，结束前核验读回。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty text is required")
    collection = _collection(config, namespace, create=False)
    if not collection.get(ids=[memory_id])["ids"]:
        raise ValueError("Memory ID does not exist")
    metadata = _metadata(namespace, metadata)
    collection.update(
        ids=[memory_id],
        documents=[text],
        metadatas=[metadata],
        embeddings=_encoder(config, embed)([text]).tolist(),
    )
    actual = collection.get(ids=[memory_id], include=["documents", "metadatas"])
    if actual["documents"] != [text] or actual["metadatas"] != [metadata]:
        raise RuntimeError("Memory update readback mismatch")


def get(config, namespace, memory_id):
    """按 config 只读取得 namespace/memory_id 的真实文本与元数据，不创建集合。"""
    return _collection(config, namespace, create=False).get(
        ids=[memory_id], include=["documents", "metadatas"]
    )


def list_items(config, namespace, limit=None, offset=0):
    """按 config 对 namespace 执行服务端 limit/offset 分页，缺省 limit 保持 CLI 行为。"""
    return _collection(config, namespace, create=False).get(
        limit=limit, offset=offset, include=["documents", "metadatas"]
    )


def delete(config, namespace, memory_id):
    """删除 namespace 内已存在 memory_id，按 config 读取并核实删除持久效果。"""
    collection = _collection(config, namespace, create=False)
    if not collection.get(ids=[memory_id])["ids"]:
        raise ValueError("Memory ID does not exist")
    collection.delete(ids=[memory_id])
    if collection.get(ids=[memory_id])["ids"]:
        raise RuntimeError("Memory deletion readback mismatch")


def query(config, namespace, text, k=5, embed=None):
    """按 config 查询 namespace 的 text 前 k 个邻居，embed 可复用正式前缀及归一化编码器。"""
    collection = _collection(config, namespace, create=False)
    if not collection.count():
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    return collection.query(
        query_embeddings=_encoder(config, embed)([text], query=True).tolist(),
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
