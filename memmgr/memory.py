"""提供长期记忆的命名空间隔离和 Chroma 持久化增删改查接口。

集合名由原始命名空间摘要确定；写入与更新采用正文向量，查询采用带检索指令的向量。
"""

from __future__ import annotations

import hashlib
import uuid

from .index import make_embedder


def collection_name(prefix, namespace):
    """以命名空间 SHA256 的前 24 位建立稳定后缀，裁切前缀后返回符合 63 字符限制的集合名。"""
    digest = hashlib.sha256(namespace.encode('utf-8')).hexdigest()[:24]
    suffix = f'_memory_{digest}'
    return f"{(prefix or 'm')[:63 - len(suffix)]}{suffix}"


def _collection(config, namespace):
    """连接配置的 Chroma 服务，返回该命名空间的持久集合，缺失时按 cosine 距离设置创建。"""
    import chromadb
    chroma_config = config['chroma']
    client = chromadb.HttpClient(
        host=chroma_config.get('host', '127.0.0.1'),
        port=int(chroma_config.get('port', 8001)),
    )
    name = collection_name(
        chroma_config.get('collection_prefix', 'memmgr'),
        namespace,
    )
    return client.get_or_create_collection(
        name,
        metadata={'hnsw:space': 'cosine', 'namespace': namespace},
    )


def add(config, namespace, text, memory_id=None):
    """编码并新增一条记忆，返回记忆 ID。

    调用者可指定 memory_id，缺省生成 UUID；正文与原始 namespace 一起存储，重复 ID 行为交由 Chroma
    处理。
    """
    _, embed = make_embedder(config)
    memory_id = memory_id or str(uuid.uuid4())
    collection = _collection(config, namespace)
    collection.add(
        ids=[memory_id],
        documents=[text],
        metadatas=[{'namespace': namespace}],
        embeddings=embed([text]).tolist(),
    )
    return memory_id


def update(config, namespace, memory_id, text):
    """重新编码指定 ID 的文本并更新正文、向量及 namespace 元数据，返回 None。"""
    _, embed = make_embedder(config)
    collection = _collection(config, namespace)
    collection.update(
        ids=[memory_id],
        documents=[text],
        metadatas=[{'namespace': namespace}],
        embeddings=embed([text]).tolist(),
    )


def get(config, namespace, memory_id):
    """读取指定 ID，返回 Chroma 的 documents/metadatas 结果字典，保留缺失记录的原始返回格式。"""
    return _collection(config, namespace).get(
        ids=[memory_id],
        include=['documents', 'metadatas'],
    )


def list_items(config, namespace):
    """读取当前命名空间全部记录，返回文档、元数据与原始 ID。"""
    return _collection(config, namespace).get(include=['documents', 'metadatas'])


def delete(config, namespace, memory_id):
    """从当前命名空间删除指定 ID，持久化副作用由 Chroma 客户端完成。"""
    _collection(config, namespace).delete(ids=[memory_id])


def query(config, namespace, text, k=5):
    """将查询文本编码后返回前 k 条记忆，结果保留文档、元数据、距离及排名顺序。"""
    _, embed = make_embedder(config)
    return _collection(config, namespace).query(
        query_embeddings=embed([text], query=True).tolist(),
        n_results=k,
        include=['documents', 'metadatas', 'distances'],
    )
