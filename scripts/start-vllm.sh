# """按 config/server.json 与 config/models.json 启动本机固定快照 vLLM；参数为可选配置路径。"""
set -euo pipefail
cd "${PROJECT_ROOT:-$(dirname "$(realpath "$0")")/..}"
if [[ -z "${PYTHON:-}" && -f .env ]]; then
  source .env
fi
export OMP_NUM_THREADS=2
PYTHON="${PYTHON:-.venv/bin/python}"
exec "$PYTHON" -m memmgr.inference.server --server "${1:-config/server.json}" --models "${2:-config/models.json}"
