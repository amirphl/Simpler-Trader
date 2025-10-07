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

exec uvicorn webserver.pivot_app:app --host "$HOST" --port "$PORT"
