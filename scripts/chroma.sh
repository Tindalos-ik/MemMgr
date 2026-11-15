# """管理 Chroma 持久服务；参数为 up/stop/restart/status，PROJECT_ROOT 指定项目目录。"""
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
