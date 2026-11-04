"""从明确配置构造受约束 vLLM argv，使用本地固定模型快照启动服务。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .models import local_snapshot


def launch(server_path, models_path):
    """读取 server_path/models_path 的固定配置并 exec 当前 Python 的 vLLM 服务。"""
    server = json.loads(Path(server_path).read_text())
    models = json.loads(Path(models_path).read_text())
    item = models[server["model"]]
    if item["revision"] != server["revision"]:
        raise ValueError("server/models revision mismatch")
    snapshot = local_snapshot(server["model"], item["revision"])
    if server["host"] not in ("127.0.0.1", "localhost"):
        raise ValueError("Only loopback service binding is allowed")
    if not 0 < float(server["gpu_memory_utilization"]) < 1:
        raise ValueError("Invalid GPU memory utilization")
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(snapshot),
        "--served-model-name",
        server["model"],
        "--host",
        server["host"],
        "--port",
        str(server["port"]),
        "--dtype",
        server["dtype"],
        "--max-model-len",
        str(server["max_model_len"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
    ]
    if server["enforce_eager"]:
        argv.append("--enforce-eager")
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_USE_FLASHINFER_SAMPLER",
    ):
        if server[key] not in ("0", "1"):
            raise ValueError(f"Invalid {key}")
        os.environ[key] = server[key]
    print(json.dumps({"server": server, "models": models, "argv": argv}), flush=True)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="config/server.json")
    parser.add_argument("--models", default="config/models.json")
    args = parser.parse_args()
    launch(args.server, args.models)
