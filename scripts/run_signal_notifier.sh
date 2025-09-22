#!/bin/bash
# Convenience wrapper for the signal notifier. Populate the environment
# variables below before running (defaults are placeholders).

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

is_truthy() {
  case "${1,,}" in
    true|1|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

# --- Strategy selection ----------------------------------------------------
export STRATEGY_TIMEFRAME="${STRATEGY_TIMEFRAME:-15m}"
export SYMBOLS="${SYMBOLS:-}"                 # comma-separated list (optional)
export TOP_N="${TOP_N:-100}"
export LOOKBACK="${LOOKBACK:-10}"
export EPSILON_MINUTES="${EPSILON_MINUTES:-0.1}"
export SIGNAL_STATE_FILE="${SIGNAL_STATE_FILE:-./data/signal_state.json}"
export SIGNAL_STATE_DISABLE="${SIGNAL_STATE_DISABLE:-false}"  # set true to disable state tracking
export SIGNAL_DRY_RUN="${SIGNAL_DRY_RUN:-false}"

# Strategy-specific knobs (override as needed)
export STRATEGY_LEVERAGE="${STRATEGY_LEVERAGE:-5.0}"
export STRATEGY_WINDOW_SIZE="${STRATEGY_WINDOW_SIZE:-5}"
export STRATEGY_TAKE_PROFIT_PCT="${STRATEGY_TAKE_PROFIT_PCT:-0.02}"
export VOLUME_WINDOW="${VOLUME_WINDOW:-20}"
export MAX_VOLUME_SCORE="${MAX_VOLUME_SCORE:-3.0}"

# Binance proxy settings (optional)
export PROXY="${PROXY:-http://127.0.0.1:12334}"
export HTTP_PROXY="${HTTP_PROXY:-}"
export HTTPS_PROXY="${HTTPS_PROXY:-}"

# Telegram configuration (required)
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
export TELEGRAM_PROXY="${TELEGRAM_PROXY:-http://127.0.0.1:12334}"               # optional

# --- Run notifier ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${TELEGRAM_BOT_TOKEN}" || -z "${TELEGRAM_CHAT_ID}" ]]; then
  echo "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python executable not found (looked for python3/python)." >&2
    exit 1
  fi
fi

CLI_ARGS=(
  --timeframe "${STRATEGY_TIMEFRAME}"
  --lookback "${LOOKBACK}"
  --epsilon-minutes "${EPSILON_MINUTES}"
  --leverage "${STRATEGY_LEVERAGE}"
  --window-size "${STRATEGY_WINDOW_SIZE}"
  --take-profit-pct "${STRATEGY_TAKE_PROFIT_PCT}"
  --volume-window "${VOLUME_WINDOW}"
  --max-volume-score "${MAX_VOLUME_SCORE}"
  --telegram-token "${TELEGRAM_BOT_TOKEN}"
  --telegram-chat-id "${TELEGRAM_CHAT_ID}"
)

if [[ -n "${SYMBOLS}" ]]; then
  CLI_ARGS+=(--symbols "${SYMBOLS}")
else
  CLI_ARGS+=(--top-n "${TOP_N}")
fi

if [[ -n "${TELEGRAM_PROXY}" ]]; then
  CLI_ARGS+=(--telegram-proxy "${TELEGRAM_PROXY}")
fi

if [[ -n "${PROXY}" ]]; then
  CLI_ARGS+=(--proxy "${PROXY}")
fi
if [[ -n "${HTTP_PROXY}" ]]; then
  CLI_ARGS+=(--http-proxy "${HTTP_PROXY}")
fi
if [[ -n "${HTTPS_PROXY}" ]]; then
  CLI_ARGS+=(--https-proxy "${HTTPS_PROXY}")
fi

if is_truthy "${SIGNAL_STATE_DISABLE}"; then
  CLI_ARGS+=(--no-state)
elif [[ -n "${SIGNAL_STATE_FILE}" ]]; then
  CLI_ARGS+=(--state-file "${SIGNAL_STATE_FILE}")
fi

if is_truthy "${SIGNAL_DRY_RUN}"; then
  CLI_ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" -m cmd.signal_notifier.main "${CLI_ARGS[@]}" "$@"
