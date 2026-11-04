"""测量指定语料索引的首次导入或已存在索引复用。

用法：在 WSL 项目根目录运行
`python scripts/index_measured.py --config CONFIG --tier {small,medium,full}`。
脚本连接配置中的Chroma服务，在 `experiment.output_dir` 下创建运行目录，
保存数据、环境、源码摘要、资源与索引结果；fresh导入来源写入
`.agent/state/index-run-registry.json`，reuse 运行关联已有来源。
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memmgr.index import index_collection_name, index_identity, index_tier
from memmgr.resources import sample


def parse_args() -> argparse.Namespace:
    """解析必需的配置文件和语料级别参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True)
    return parser.parse_args()


def snapshot_application_code() -> dict[str, str]:
    """返回 `memmgr/*.py` 的相对路径与SHA256映射。"""
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("memmgr").glob("*.py"))
    }


def prepare_run_files(
    run_dir: Path, config: dict, tier: str, dataset: Path
) -> None:
    """在新运行目录保存配置、环境、源码、依赖和数据ID快照。

    同时创建空的请求与失败记录文件，保持索引运行目录的标准结构。
    """
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    environment = {
        "python": sys.version,
        "tier": tier,
        "server_config": json.loads(
            Path("config/server.json").read_text()
        ),
    }
    (run_dir / "env.json").write_text(json.dumps(environment, indent=2))
    (run_dir / "code-snapshot.json").write_text(
        json.dumps(snapshot_application_code(), indent=2)
    )
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    (run_dir / "packages.json").write_text(json.dumps(packages, indent=2))
    (run_dir / "logs").mkdir()
    (run_dir / "requests.jsonl").touch()
    (run_dir / "failures.jsonl").touch()

    dataset_ids = {
        "document_ids": [
            json.loads(line)["_id"]
            for line in (dataset / "corpus.jsonl").read_text().splitlines()
        ],
        "query_ids": [
            json.loads(line)["_id"]
            for line in (dataset / "queries.jsonl").read_text().splitlines()
        ],
    }
    (run_dir / "dataset-ids.json").write_text(json.dumps(dataset_ids, indent=2))


def resolve_collection(
    config: dict, tier: str, dataset: Path
) -> tuple[str, Path, bool]:
    """按索引身份计算集合名，并判断当前清单是否表示已有索引。

    返回集合名、对应清单路径及fresh/reuse标记；现有集合以索引清单是否存在识别。
    """
    embedding = config["embedding"]
    identity = index_identity(
        dataset,
        tier,
        embedding,
        int(embedding["chunk_tokens"]),
        int(embedding["chunk_overlap"]),
    )
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:12]
    collection = index_collection_name(
        config["chroma"]["collection_prefix"], tier, digest
    )
    manifest = dataset.parent / "index-manifests" / f"{collection}.json"
    return collection, manifest, manifest.exists()


def main() -> None:
    """建立索引运行目录、启动资源采样并调用应用索引入口。

    fresh导入记录完整索引耗时并登记来源；reuse运行记录本次查找耗时和原来源。
    采样线程在成功或异常退出时停止。
    """
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    run_name = (
        f"index-{args.tier}-{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    run_dir = Path(config["experiment"]["output_dir"]) / run_name
    run_dir.mkdir(parents=True)
    dataset = Path(config["dataset"]["root"]) / args.tier
    prepare_run_files(run_dir, config, args.tier, dataset)
    collection, manifest, reused = resolve_collection(config, args.tier, dataset)

    registry_path = Path(".agent/state/index-run-registry.json")
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    stop_sampling = threading.Event()
    sampler = threading.Thread(
        target=sample,
        args=(
            run_dir,
            float(config["resources"]["sample_interval_seconds"]),
            stop_sampling,
            config,
        ),
        daemon=True,
    )
    sampler.start()

    try:
        started = time.monotonic()
        index_result = index_tier(config, args.tier)
        wall_seconds = time.monotonic() - started
        if reused:
            summary = {
                "index_reused": True,
                "collection": collection,
                "lookup_wall_seconds": wall_seconds,
                "original_manifest": str(manifest),
                "first_measured_run": registry.get(collection),
                "note": (
                    "No fresh import measured; original manifest timings are "
                    "not attributed to this lookup run."
                ),
            }
        else:
            summary = {
                **index_result,
                "index_reused": False,
                "measured_wall_seconds": wall_seconds,
                "original_manifest": str(manifest),
            }
            registry[collection] = str(run_dir)
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text(json.dumps(registry, indent=2) + "\n")

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    except Exception as error:
        (run_dir / "failures.jsonl").write_text(
            json.dumps({"status": "failed", "error": repr(error)}) + "\n"
        )
        raise
    finally:
        stop_sampling.set()
        sampler.join(timeout=10)
        print(run_dir, flush=True)


if __name__ == "__main__":
    main()
