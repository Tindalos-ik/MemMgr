# 用途：管理 compose 中的 Chroma 服务，保留命名卷中的持久化数据。
# 用法：`bash scripts/chroma.sh [up|stop|restart|status]`；省略参数时执行 up。
# 前提：WSL 可调用 Docker Compose v2；`PROJECT_ROOT` 可指定项目目录。
set -euo pipefail
cd "${PROJECT_ROOT:-$(dirname "$(realpath "$0")")/..}"
case "${1:-up}" in
  up)
    docker compose -f compose.yaml up -d --wait --wait-timeout 180
    ;;
  stop)
    docker compose -f compose.yaml stop
    ;;
  restart)
    docker compose -f compose.yaml restart chroma
    docker compose -f compose.yaml up -d --wait --wait-timeout 180
    ;;
  status)
    docker compose -f compose.yaml ps
    ;;
  *)
    printf 'Usage: bash scripts/chroma.sh {up|stop|restart|status}\n' >&2
    exit 2
    ;;
esac
