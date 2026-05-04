#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: run_bos_choch_server.sh [--source binance|csv] [--csv-path PATH] [--host HOST] [--port PORT]
EOF
}

# Optional: source environment overrides
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env.bos_choch}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9094}"
BOS_CHOCH_CANDLE_SOURCE="${BOS_CHOCH_CANDLE_SOURCE:-binance}"
BOS_CHOCH_CANDLE_CSV_PATH="${BOS_CHOCH_CANDLE_CSV_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 1; }
      BOS_CHOCH_CANDLE_SOURCE="$2"
      shift 2
      ;;
    --csv-path)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 1; }
      BOS_CHOCH_CANDLE_CSV_PATH="$2"
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

export BOS_CHOCH_CANDLE_SOURCE
export BOS_CHOCH_CANDLE_CSV_PATH

cd "$REPO_ROOT"
exec uvicorn webserver.bos_choch_app:app --host "$HOST" --port "$PORT"
