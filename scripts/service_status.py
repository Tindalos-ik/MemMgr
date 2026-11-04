"""读取本机服务健康状态与 Chroma 集合数量。

用法：在 WSL 项目根目录运行 `python scripts/service_status.py`。
脚本查询本机 vLLM 健康与模型接口、Chroma心跳和集合计数，结果保存至
`.agent/state/service-status.json` 并同时打印到标准输出。
"""

import datetime
import json
import time
from pathlib import Path

import chromadb
import httpx


def main() -> None:
    """采集本机服务状态和集合计数，并保存一份带耗时的 JSON 快照。

    计时从创建 Chroma 客户端前开始；HTTP 健康检查沿用 10 秒超时。
    """
    started = time.monotonic()
    chroma = chromadb.HttpClient(host="127.0.0.1", port=8001)
    collections = []

    for item in chroma.list_collections():
        name = item if isinstance(item, str) else item.name
        collection = chroma.get_collection(name, embedding_function=None)
        collections.append({"name": name, "count": collection.count()})

    health = httpx.get("http://127.0.0.1:8000/health", timeout=10)
    health.raise_for_status()
    models = httpx.get("http://127.0.0.1:8000/v1/models", timeout=10)
    models.raise_for_status()

    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "vllm_health_status": health.status_code,
        "vllm_models": models.json(),
        "chroma_heartbeat": chroma.heartbeat(),
        "collections": collections,
        "mutated_data": False,
    }
    output = Path(".agent/state/service-status.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
