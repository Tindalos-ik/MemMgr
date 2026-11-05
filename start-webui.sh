# """从项目根目录启动已构建的本机 WebUI；PYTHON 可指定已有 Python 解释器。"""
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
if [[ -z "${PYTHON:-}" && -f .env.webui ]]; then
  source .env.webui
fi

if [[ ! -f web/dist/index.html ]]; then
  printf '%s\n' 'WebUI 前端尚未构建。请先运行：cd web && npm ci && npm run build' >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
PYTHON="${PYTHON:-.venv/bin/python}"
exec "$PYTHON" -m uvicorn memmgr.webui.app:app --host 127.0.0.1 --port 8765
