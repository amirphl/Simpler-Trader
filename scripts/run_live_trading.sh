#!/usr/bin/env bash

# Live Trading Bot Startup Script
# This script runs the live trading bot with proper environment setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Live Trading Bot ===${NC}"
echo "Project root: ${PROJECT_ROOT}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Optional config file forwarding to main.py
EXTRA_ARGS=()
if [ -n "${CONFIG_FILE:-}" ] && [ -f "${CONFIG_FILE}" ]; then
    echo -e "${YELLOW}Using configuration file: ${CONFIG_FILE}${NC}"
    EXTRA_ARGS+=(--config-file "${CONFIG_FILE}")
elif [ -n "${CONFIG_FILE:-}" ]; then
    echo -e "${YELLOW}Warning: Config file not found: ${CONFIG_FILE}${NC}"
    echo "Using strategy-specific defaults in cmd.live_trading.main"
fi

# Ensure data and logs directories exist
mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/logs"

# Run the live trading bot
echo -e "${GREEN}Starting live trading bot...${NC}"
python3 -m cmd.live_trading.main "${EXTRA_ARGS[@]}" "$@"
