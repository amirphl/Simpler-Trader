#!/usr/bin/env bash

# Strong Trend Stair live-trading launcher

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Minimal color helpers
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Strong Trend Stair Live Trading ===${NC}"
echo "Project root: ${PROJECT_ROOT}"

# Require virtualenv
if [ ! -d "venv" ]; then
  echo -e "${RED}Error: Virtual environment not found${NC}"
  echo "Create one with: python3 -m venv venv"
  exit 1
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Optional config override via ENV or flag: CONFIG_FILE=/path/to/env
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

# Ensure runtime dirs exist
mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/logs"

echo -e "${GREEN}Starting Strong Trend Stair live coordinator...${NC}"
python3 -m cmd.live_trading.strong_trend_stair_main "${EXTRA_ARGS[@]}" "$@"
