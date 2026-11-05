"""准备并校验 SciFact 数据，按固定查询构建保留相关文献的嵌套语料。

本模块负责配置读取、官方三列相关性表和文件持久化；抽样种子、原始 ID、下载校验和以及输出格式保持稳定。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
import urllib.error
import zipfile
from pathlib import Path
from urllib.request import urlopen
BEIR_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
HF_CARD = 'https://huggingface.co/datasets/BeIR/scifact/raw/main/README.md'
EXPECTED_MD5 = '5f7d1de60b170fc8027bb7898e2efca1'


def load_config(path):
    """读取 JSON 配置并返回字典。

    要求 dataset、embedding、chroma、llm、experiment 五个顶层节；缺节时抛出
    ValueError，具体参数由消费者解释。
    """
    with open(path, encoding='utf-8') as stream:
        config = json.load(stream)
    required_sections = ('dataset', 'embedding', 'chroma', 'llm', 'experiment')
    absent = [section for section in required_sections if section not in config]
    if absent:
        raise ValueError(f"Missing config sections: {', '.join(absent)}")
    return config


def prepare(config):
    """下载并校验官方压缩包，生成各级语料及元数据并返回统计字典。

    config 提供数据目录、抽样种子、问题数和各级规模。每级先纳入所有相关文献，再扩展随机候选；按规模累积形成嵌套集合。缓存压缩包仍校验
    MD5，下载用临时文件完成后原子替换，失败保留明确异常。输出 JSONL、三列 qrels 和 metadata.json。
    """
    dataset_config = config['dataset']
    dataset_root = Path(dataset_config.get('root', 'data/scifact'))
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / 'scifact.zip'
    downloaded_at_utc = None
    if not archive_path.exists():
        errors = []
        for url in (BEIR_URL,):
            descriptor, temporary_path = tempfile.mkstemp(prefix='scifact-', suffix='.part', dir=dataset_root)
            try:
                with os.fdopen(descriptor, 'wb') as download_stream, urlopen(url, timeout=120) as response:
                    while (block := response.read(1024 * 1024)):
                        download_stream.write(block)
                downloaded_at_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                os.replace(temporary_path, archive_path)
                break
            except (OSError, urllib.error.URLError) as exc:
                errors.append(f'{url}: {exc}')
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        else:
            raise RuntimeError('SciFact download failed: ' + '; '.join(errors))
    md5_digest = hashlib.md5()
    sha256_digest = hashlib.sha256()
    with archive_path.open('rb') as stream:
        while (block := stream.read(1024 * 1024)):
            md5_digest.update(block)
            sha256_digest.update(block)
    md5 = md5_digest.hexdigest()
    sha256 = sha256_digest.hexdigest()
    if md5 != EXPECTED_MD5:
        raise ValueError(
            f'SciFact archive MD5 mismatch: expected {EXPECTED_MD5}, got {md5}',
        )
    source_root = dataset_root / 'scifact'
    if not (source_root / 'corpus.jsonl').exists():
        with zipfile.ZipFile(archive_path) as zip_archive:
            zip_archive.extractall(dataset_root)
    corpus_by_id = _jsonl(source_root / 'corpus.jsonl')
    queries_by_id = _jsonl(source_root / 'queries.jsonl')
    relevance_by_query = _qrels(source_root / 'qrels' / 'test.tsv')
    seed = int(dataset_config.get('seed', 211))
    eligible_query_ids = sorted(set(queries_by_id) & set(relevance_by_query))
    query_count = int(dataset_config.get('query_count', len(eligible_query_ids)))
    selected_query_ids = sorted(
        random.Random(seed).sample(
            eligible_query_ids,
            min(query_count, len(eligible_query_ids)),
        ),
    )
    relevant = {
        query_id: {
            doc_id
            for doc_id, relevance_score in relevance_by_query[query_id].items() if relevance_score > 0
        }
        for query_id in selected_query_ids
    }
    # 每级先纳入所选查询的全部相关文献，再扩充背景语料。
    required_doc_ids = set().union(*relevant.values()) if relevant else set()
    tiers = dataset_config.get(
        'tiers',
        {'small': 100, 'medium': 1000, 'full': len(corpus_by_id)},
    )
    candidate_docs_by_tier = {}
    for tier, size in tiers.items():
        candidates = [
            doc_id
            for doc_id in sorted(corpus_by_id) if doc_id not in required_doc_ids
        ]
        random.Random(seed).shuffle(candidates)
        candidate_docs_by_tier[tier] = (
            set(required_doc_ids)
            | set(candidates[:max(0, int(size) - len(required_doc_ids))])
        )
    selected_doc_ids = set(required_doc_ids)
    tier_metadata = {}
    for tier, size in sorted(tiers.items(), key=lambda item: int(item[1])):
        selected_doc_ids |= candidate_docs_by_tier[tier]
        documents = {doc_id: corpus_by_id[doc_id] for doc_id in sorted(selected_doc_ids)}
        destination = dataset_root / tier
        destination.mkdir(exist_ok=True)
        _write_jsonl(destination / 'corpus.jsonl', documents)
        _write_jsonl(
            destination / 'queries.jsonl',
            {
                query_id: queries_by_id[query_id]
                for query_id in selected_query_ids
            },
        )
        _write_qrels(
            destination / 'qrels.tsv',
            {
                query_id: relevance_by_query[query_id]
                for query_id in selected_query_ids
            },
        )
        relevant_pair_count = sum(map(len, relevant.values()))
        covered_pair_count = sum(
            (
                len(relevant[query_id] & set(documents))
                for query_id in selected_query_ids
            ),
        )
        tier_metadata[tier] = {
            'documents': len(documents),
            'queries': len(selected_query_ids),
            'queries_with_relevant_doc': sum(
                (
                    bool(relevant[query_id] & set(documents))
                    for query_id in selected_query_ids
                ),
            ),
            'qrel_doc_coverage': covered_pair_count / max(1, relevant_pair_count),
        }
    metadata = {
        'source': BEIR_URL,
        'metadata_source': HF_CARD,
        'license': 'cc-by-sa-4.0',
        'downloaded_at_utc': downloaded_at_utc,
        'archive_md5': md5,
        'archive_sha256': sha256,
        'expected_archive_md5': EXPECTED_MD5,
        'expected_source_counts': {'corpus_documents': 5183, 'queries': 1109, 'test_qrel_queries': 300},
        'actual_counts': {
            'corpus_documents': len(corpus_by_id),
            'queries': len(queries_by_id),
            'test_qrel_queries': len(relevance_by_query),
            'test_qrel_pairs': sum(map(len, relevance_by_query.values())),
        },
        'seed': seed,
        'query_ids': selected_query_ids,
        'qrels': 'official BEIR SciFact test.tsv (3 columns)',
        'tiers': tier_metadata,
    }
    (dataset_root / 'metadata.json').write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )
    return metadata


def _jsonl(path):
    """读取 UTF-8 JSONL，返回以原始 _id 为键的记录字典，保留文件出现顺序。"""
    with path.open(encoding='utf-8') as stream:
        return {row['_id']: row for row in map(json.loads, stream)}


def _qrels(path):
    """读取官方三列制表符相关性表，跳过表头，返回 query_id → doc_id → 整数分值。

    零相关分值同样保留；格式错误直接给出解析异常，供上层定位数据问题。
    """
    relevance_by_query = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('query-id'):
                continue
            qid, docid, score = line.rstrip('\n').split('\t')
            relevance_by_query.setdefault(qid, {})[docid] = int(score)
    return relevance_by_query


def _write_jsonl(path, data):
    """将记录字典按当前插入顺序写成 UTF-8 JSONL，每条记录占一行并保留原始 ID。"""
    with path.open('w', encoding='utf-8') as stream:
        for row in data.values():
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_qrels(path, data):
    """按查询和文档的当前顺序写出官方三列相关性表，包含 query-id 表头。"""
    with path.open('w', encoding='utf-8') as stream:
        stream.write('query-id\tcorpus-id\tscore\n')
        for query_id, documents in data.items():
            for doc_id, score in documents.items():
                stream.write(f'{query_id}\t{doc_id}\t{score}\n')
