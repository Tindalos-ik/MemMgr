"""为每次测量保存源码、配置、数据与运行环境的可追溯快照。

开始与结束环境分别保存，源码摘要覆盖包内全部 Python 文件。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    """分块计算给定 Path 的 SHA256，返回十六进制摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprints() -> dict:
    """按文件名排序计算当前 memmgr 包所有 Python 文件的摘要，返回项目相对路径到摘要的字典。"""
    source_root = Path(__file__).parent
    return {
        str(p.relative_to(source_root.parent).as_posix()): sha256_file(p)
        for p in sorted(source_root.glob("*.py"))
    }


def environment_snapshot(config: dict) -> dict:
    """采集时间、解释器、平台、核心包版本和配置中的模型信息，返回可 JSON 序列化字典。

    缺失包版本记为 None；单调时钟和 UTC 分别支持耗时归因与跨记录定位。
    """
    versions = {}
    for package in (
        "sentence-transformers",
        "transformers",
        "chromadb-client",
        "httpx",
        "torch",
        "vllm",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "monotonic": time.monotonic(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "web_snapshot": config.get("_web_snapshot"),
        "package_versions": versions,
        "llm": {
            "model": config["llm"].get("model"),
            "revision": config["llm"].get("revision"),
            "tokenizer": config["llm"].get("tokenizer"),
            "max_model_len": config["llm"].get("max_model_len"),
            "max_tokens": config["llm"].get("max_tokens"),
        },
        "embedding": {
            "model": config["embedding"].get("model"),
            "revision": config["embedding"].get("revision"),
            "device": config["embedding"].get("device"),
        },
    }


def write_run_metadata(
    output_directory: Path,
    config: dict,
    tier: str,
    query_ids: list[str],
    index_manifest: dict,
):
    """为输出目录写入配置、数据签名、运行清单与开始环境。

    query_ids 保留本轮实际顺序；语料/问题/qrels 内容校验和与索引清单一并记录。配置副本附加可复算来源信息，输入
    config 字典保持原值。
    """
    config_bytes = json.dumps(config, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    code_hashes = code_fingerprints()
    environment = environment_snapshot(config)
    run_config = json.loads(json.dumps(config))
    run_config["_run_reproducibility"] = {
        "source_config_sha256": config_hash,
        "code_sha256": code_hashes,
    }
    (output_directory / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dataset_root = Path(config["dataset"].get("root", "data/scifact")) / tier
    files = {
        name: sha256_file(dataset_root / name)
        for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv")
    }
    subset = {
        "tier": tier,
        "query_ids": query_ids,
        "files": files,
        "index_collection": index_manifest.get("collection"),
    }
    subset_signature = hashlib.sha256(
        json.dumps(subset, sort_keys=True, ensure_ascii=False).encode(),
    ).hexdigest()
    (output_directory / "dataset.json").write_text(
        json.dumps(
            {
                **subset,
                "subset_signature_sha256": subset_signature,
                "index_manifest": index_manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": output_directory.name,
                "config_sha256": config_hash,
                "code_sha256": code_hashes,
                "model": environment["llm"],
                "embedding": environment["embedding"],
                "dataset_subset_signature_sha256": subset_signature,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_directory / "env.json").write_text(
        json.dumps({"before": environment}, indent=2),
        encoding="utf-8",
    )


def finish_environment_snapshot(output_directory: Path, config: dict):
    """为该轮 env.json 增补 after 环境快照，保留已写入的 before 和其他字段。"""
    path = output_directory / "env.json"
    snapshots = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    snapshots["after"] = environment_snapshot(config)
    path.write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
