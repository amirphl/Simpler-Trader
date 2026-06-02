#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/backtest.ema_avwap_pullback.env}"

cd "${PROJECT_ROOT}"

if [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  set -a
  source "${CONFIG_FILE}"
  set +a
fi

python3 -m cmd.backtest.main --strategy ema_avwap_pullback "$@"
