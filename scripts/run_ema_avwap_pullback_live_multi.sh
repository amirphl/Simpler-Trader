#!/usr/bin/env bash

# Run one independent EMA + AVWAP Pullback live coordinator per symbol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

MODE="blocking"
CONFIG_FILE="${CONFIG_FILE:-configs/live_trading.ema_avwap_pullback.env}"
SYMBOLS_INPUT="${SYMBOLS:-${EMA_AVWAP_SYMBOLS:-}}"
DATA_ROOT="${EMA_AVWAP_DATA_ROOT:-./data/ema_avwap_pullback}"
LOG_ROOT="${EMA_AVWAP_LOG_ROOT:-./logs/ema_avwap_pullback}"
PID_ROOT="${EMA_AVWAP_PID_ROOT:-./data/ema_avwap_pullback/pids}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ema_avwap_pullback_live_multi.sh --symbols ETHUSDT,BTCUSDT [options] [-- extra live args]

Options:
  --symbols LIST       Comma-separated symbols. If omitted, uses SYMBOLS from env/config.
  --mode MODE         blocking or async. Default: blocking.
  --config-file PATH  Base .env config. Default: configs/live_trading.ema_avwap_pullback.env
  --data-root PATH    Per-symbol state/db root. Default: ./data/ema_avwap_pullback
  --log-root PATH     Per-symbol log root. Default: ./logs/ema_avwap_pullback
  --pid-root PATH     Per-symbol pid root. Default: ./data/ema_avwap_pullback/pids
  --python PATH       Python executable. Default: python3
  -h, --help          Show this help.

Modes:
  blocking  Starts all symbols concurrently, streams each process stdout/stderr with
            a [SYMBOL] prefix, and stops all child processes on Ctrl-C.
  async     Starts each symbol with nohup in the background, writes pid files, prints
            per-symbol log paths, and exits.

Examples:
  scripts/run_ema_avwap_pullback_live_multi.sh --symbols ETHUSDT,BTCUSDT --mode blocking

  scripts/run_ema_avwap_pullback_live_multi.sh \
    --config-file configs/live_trading.ema_avwap_pullback.env \
    --symbols ETHUSDT,SOLUSDT,XRPUSDT \
    --mode async \
    -- --live
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --symbols)
      SYMBOLS_INPUT="${2:-}"
      shift 2
      ;;
    --symbols=*)
      SYMBOLS_INPUT="${1#--symbols=}"
      shift
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --config-file)
      CONFIG_FILE="${2:-}"
      shift 2
      ;;
    --config-file=*)
      CONFIG_FILE="${1#--config-file=}"
      shift
      ;;
    --data-root)
      DATA_ROOT="${2:-}"
      shift 2
      ;;
    --data-root=*)
      DATA_ROOT="${1#--data-root=}"
      shift
      ;;
    --log-root)
      LOG_ROOT="${2:-}"
      shift 2
      ;;
    --log-root=*)
      LOG_ROOT="${1#--log-root=}"
      shift
      ;;
    --pid-root)
      PID_ROOT="${2:-}"
      shift 2
      ;;
    --pid-root=*)
      PID_ROOT="${1#--pid-root=}"
      shift
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN="${1#--python=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

read_config_value() {
  local key="$1"
  local path="$2"
  local line value

  [ -f "${path}" ] || return 1
  while IFS= read -r line; do
    line="$(trim "${line}")"
    [ -z "${line}" ] && continue
    case "${line}" in
      \#*) continue ;;
    esac
    line="${line%%#*}"
    line="$(trim "${line}")"
    case "${line}" in
      "${key}"=*)
        value="${line#*=}"
        value="$(trim "${value}")"
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        printf '%s\n' "${value}"
        return 0
        ;;
    esac
  done < "${path}"
  return 1
}

if [ -z "${SYMBOLS_INPUT}" ]; then
  SYMBOLS_INPUT="$(read_config_value SYMBOLS "${CONFIG_FILE}" || true)"
fi

if [ -z "${SYMBOLS_INPUT}" ]; then
  echo "Error: no symbols provided. Use --symbols ETHUSDT,BTCUSDT or set SYMBOLS in ${CONFIG_FILE}." >&2
  exit 1
fi

case "${MODE}" in
  blocking|async) ;;
  *)
    echo "Error: --mode must be 'blocking' or 'async'." >&2
    exit 1
    ;;
esac

if [ -n "${CONFIG_FILE}" ] && [ ! -f "${CONFIG_FILE}" ]; then
  echo "Error: config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
elif [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

IFS=',' read -r -a RAW_SYMBOLS <<< "${SYMBOLS_INPUT}"
SYMBOLS_TO_RUN=()
for raw_symbol in "${RAW_SYMBOLS[@]}"; do
  symbol="$(trim "${raw_symbol}")"
  symbol="$(printf '%s' "${symbol}" | tr '[:lower:]' '[:upper:]')"
  [ -z "${symbol}" ] && continue
  SYMBOLS_TO_RUN+=("${symbol}")
done

if [ "${#SYMBOLS_TO_RUN[@]}" -eq 0 ]; then
  echo "Error: symbol list is empty after parsing: ${SYMBOLS_INPUT}" >&2
  exit 1
fi

mkdir -p "${DATA_ROOT}" "${LOG_ROOT}" "${PID_ROOT}"

WORKER_PIDS=()
TAIL_PIDS=()

LIVE_MODE=false
for arg in "${EXTRA_ARGS[@]:-}"; do
  if [ "${arg}" = "--live" ]; then
    LIVE_MODE=true
    break
  fi
done

if [ "${LIVE_MODE}" = true ] && [ "${LIVE_TRADING_CONFIRM:-}" != "YES" ]; then
  echo "LIVE TRADING MODE ENABLED - REAL MONEY WILL BE TRADED"
  read -r -p "Type YES to start ${#SYMBOLS_TO_RUN[@]} live process(es): " response
  if [ "${response}" != "YES" ]; then
    echo "Live trading cancelled"
    exit 0
  fi
  export LIVE_TRADING_CONFIRM=YES
fi

safe_name() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9._-' '_'
}

symbol_paths() {
  local symbol="$1"
  local safe_symbol
  safe_symbol="$(safe_name "${symbol}")"

  SYMBOL_DATA_DIR="${DATA_ROOT}/${safe_symbol}"
  SYMBOL_LOG_DIR="${LOG_ROOT}/${safe_symbol}"
  SYMBOL_PID_FILE="${PID_ROOT}/${safe_symbol}.pid"
  SYMBOL_STATE_FILE="${SYMBOL_DATA_DIR}/state.json"
  SYMBOL_POSITIONS_DB="${SYMBOL_DATA_DIR}/positions.db"
  SYMBOL_KLINES_DB="${SYMBOL_DATA_DIR}/klines.db"
  SYMBOL_LOG_FILE="${SYMBOL_LOG_DIR}/live.log"
  SYMBOL_STDOUT_FILE="${SYMBOL_LOG_DIR}/stdout.log"
}

build_command() {
  local symbol="$1"
  symbol_paths "${symbol}"

  CMD=(
    "${PYTHON_BIN}" -m cmd.live_trading.ema_avwap_pullback_main
    --config-file "${CONFIG_FILE}"
    --symbols "${symbol}"
    --state-file "${SYMBOL_STATE_FILE}"
    --positions-db "${SYMBOL_POSITIONS_DB}"
    --klines-db "${SYMBOL_KLINES_DB}"
    --log-file "${SYMBOL_LOG_FILE}"
  )
  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
}

stop_blocking_children() {
  local pid
  for pid in "${TAIL_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

start_symbol() {
  local symbol="$1"
  symbol_paths "${symbol}"
  mkdir -p "${SYMBOL_DATA_DIR}" "${SYMBOL_LOG_DIR}"
  build_command "${symbol}"

  if [ -f "${SYMBOL_PID_FILE}" ]; then
    local old_pid
    old_pid="$(cat "${SYMBOL_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "[${symbol}] already running pid=${old_pid}; skipping"
      echo "[${symbol}] pid_file=${SYMBOL_PID_FILE}"
      return 0
    fi
  fi

  if [ "${MODE}" = "async" ]; then
    nohup "${CMD[@]}" > "${SYMBOL_STDOUT_FILE}" 2>&1 &
    local pid="$!"
    printf '%s\n' "${pid}" > "${SYMBOL_PID_FILE}"
    echo "[${symbol}] started pid=${pid}"
    echo "[${symbol}] log=${SYMBOL_LOG_FILE}"
    echo "[${symbol}] stdout=${SYMBOL_STDOUT_FILE}"
    echo "[${symbol}] state=${SYMBOL_STATE_FILE}"
    disown "${pid}" 2>/dev/null || true
    return 0
  fi

  : > "${SYMBOL_STDOUT_FILE}"
  "${CMD[@]}" > "${SYMBOL_STDOUT_FILE}" 2>&1 &
  local worker_pid="$!"
  WORKER_PIDS+=("${worker_pid}")
  printf '%s\n' "${worker_pid}" > "${SYMBOL_PID_FILE}"
  echo "[${symbol}] started pid=${worker_pid}"
  echo "[${symbol}] log=${SYMBOL_LOG_FILE}"

  tail -n +1 -F "${SYMBOL_STDOUT_FILE}" 2>/dev/null \
    | awk -v prefix="[${symbol}] " '{ print prefix $0; fflush(); }' &
  TAIL_PIDS+=("$!")
}

echo "=== EMA + AVWAP Pullback Multi-Symbol Live Trading ==="
echo "Mode: ${MODE}"
echo "Config: ${CONFIG_FILE}"
echo "Symbols: ${SYMBOLS_TO_RUN[*]}"
echo "Data root: ${DATA_ROOT}"
echo "Log root: ${LOG_ROOT}"

for symbol in "${SYMBOLS_TO_RUN[@]}"; do
  start_symbol "${symbol}"
done

if [ "${MODE}" = "async" ]; then
  echo "All symbols started asynchronously. PID files are in ${PID_ROOT}."
  exit 0
fi

trap stop_blocking_children INT TERM EXIT

status=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
