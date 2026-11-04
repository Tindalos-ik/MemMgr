# 用途：以前台进程启动本机 Qwen vLLM OpenAI兼容服务。
# 用法：`bash scripts/start-vllm.sh`；可用 `PROJECT_ROOT` 和 `PYTHON` 覆盖项目目录与解释器。
# 前提：WSL内可连接GPU，`config/models.json`及固定缓存中的模型快照已准备完成。
# 服务监听127.0.0.1；启动参数与运行环境变量沿用本次部署配置。
set -euo pipefail
cd "${PROJECT_ROOT:-$(dirname "$(realpath "$0")")/..}"
export HF_HOME=/home/tindalos/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=2
PYTHON="${PYTHON:-/home/tindalos/mycode/vLLM/.venv/bin/python}"
REVISION="$($PYTHON -c '
import json

with open("config/models.json") as models_file:
    print(json.load(models_file)["Qwen/Qwen2.5-1.5B-Instruct"]["revision"])
')"
exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct --revision "$REVISION" \
  --tokenizer-revision "$REVISION" --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype half --enforce-eager \
  --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.80 \
  --generation-config vllm --no-enable-log-requests
