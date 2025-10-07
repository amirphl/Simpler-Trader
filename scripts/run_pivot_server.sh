#!/usr/bin/env bash
set -euo pipefail

# Optional: source environment overrides
ENV_FILE="${ENV_FILE:-.env.pivots}"
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
      PIVOT_CANDLE_SOURCE="$2"
      shift 2
      ;;
    --csv-path)
      PIVOT_CANDLE_CSV_PATH="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export PIVOT_CANDLE_SOURCE
export PIVOT_CANDLE_CSV_PATH

exec uvicorn webserver.pivot_app:app --host "$HOST" --port "$PORT"
