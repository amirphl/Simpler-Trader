#!/usr/bin/env bash
# Serve both pivot v1 (/) and pivot v2 (/v2) from the same FastAPI app.
#   v1 page  → http://<host>:<port>/
#   v2 page  → http://<host>:<port>/v2
#   v1 API   → POST /api/pivots
#   v2 API   → POST /api/pivots/v2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: run_pivot_v2_server.sh [--source binance|csv] [--csv-path PATH] [--host HOST] [--port PORT]

Serves pivot v1 (/) and pivot v2 (/v2) from the same webserver.pivot_app:app instance.
EOF
}

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env.pivots}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9093}"
PIVOT_CANDLE_SOURCE="${PIVOT_CANDLE_SOURCE:-binance}"
PIVOT_CANDLE_CSV_PATH="${PIVOT_CANDLE_CSV_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 1; }
      PIVOT_CANDLE_SOURCE="$2"
      shift 2
      ;;
    --csv-path)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 1; }
      PIVOT_CANDLE_CSV_PATH="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 1; }
      HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 1; }
      PORT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export PIVOT_CANDLE_SOURCE
export PIVOT_CANDLE_CSV_PATH

echo "Pivot v1 → http://${HOST}:${PORT}/"
echo "Pivot v2 → http://${HOST}:${PORT}/v2"

cd "$REPO_ROOT"
exec uvicorn webserver.pivot_app:app --host "$HOST" --port "$PORT"
