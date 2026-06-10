#!/usr/bin/env bash

# EMA + AVWAP Pullback live-trading launcher

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== EMA + AVWAP Pullback Live Trading ===${NC}"
echo "Project root: ${PROJECT_ROOT}"

if [ ! -d "venv" ]; then
  echo -e "${RED}Error: Virtual environment not found${NC}"
  echo "Create one with: python3 -m venv venv"
  exit 1
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

EXTRA_ARGS=()
if [ -n "${CONFIG_FILE:-}" ]; then
  if [ -f "${CONFIG_FILE}" ]; then
    echo -e "${YELLOW}Using configuration file: ${CONFIG_FILE}${NC}"
    EXTRA_ARGS+=(--config-file "${CONFIG_FILE}")
  else
    echo -e "${RED}Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
  fi
fi

mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/logs"

echo -e "${GREEN}Starting EMA + AVWAP Pullback live coordinator...${NC}"
python3 -m cmd.live_trading.ema_avwap_pullback_main "${EXTRA_ARGS[@]}" "$@"
