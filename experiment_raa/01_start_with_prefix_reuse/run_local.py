"""在独立 WSL 临时环境下载固定模型、启动自有服务并运行前缀复用实验。"""

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(sys.prefix).parent
os.environ.update(HF_HOME=str(RUNTIME / "hf"),
                  HF_XET_CACHE=str(RUNTIME / "xet"),
                  VLLM_CACHE_ROOT=str(RUNTIME / "vllm-cache"),
                  TRITON_CACHE_DIR=str(RUNTIME / "triton-cache"),
                  TORCHINDUCTOR_CACHE_DIR=str(RUNTIME / "torch-cache"),
                  OMP_NUM_THREADS="2", VLLM_USE_V2_MODEL_RUNNER="0",
                  VLLM_USE_FLASHINFER_SAMPLER="0")
from huggingface_hub import snapshot_download
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
model_path = snapshot_download(MODEL, revision=REVISION, cache_dir=RUNTIME / "hf/hub",
                               allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("Model downloaded:", model_path, flush=True)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
        "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", "18080",
        "--dtype", "float16", "--max-model-len", "2048", "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.65", "--enforce-eager", "--enable-prefix-caching",
        "--generation-config", "vllm"]
environment = dict(model=MODEL, revision=REVISION, server_argv=argv,
                   packages={name: importlib.metadata.version(name)
                             for name in ("vllm", "torch", "transformers", "httpx", "prometheus-client")},
                   gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True))
(ROOT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
with (ROOT / "server.log").open("w", encoding="utf-8") as log:
    with httpx.Client(trust_env=False, timeout=2) as client:
        # 端口占用时直接停止，避免连接其他人的服务。
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 18080))
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError("vLLM exited; inspect server.log")
                try:
                    response = client.get("http://127.0.0.1:18080/health")
                except httpx.ConnectError:
                    response = None
                if response is not None:
                    response.raise_for_status()
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("vLLM startup exceeded 600 seconds")
                time.sleep(1)
            subprocess.run([sys.executable, str(ROOT / "prefix_baseline.py"),
                            "--tokenizer", model_path, "--output", str(ROOT / "results")], check=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
