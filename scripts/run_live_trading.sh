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

# Check if config file exists
if [ -z "${CONFIG_FILE:-}" ]; then
    CONFIG_FILE="${PROJECT_ROOT}/configs/live_trading.env"
fi

if [ -f "${CONFIG_FILE}" ]; then
    echo -e "${YELLOW}Loading configuration from: ${CONFIG_FILE}${NC}"
    set -a
    source "${CONFIG_FILE}"
    set +a
else
    echo -e "${YELLOW}Warning: Config file not found: ${CONFIG_FILE}${NC}"
    echo "Using command-line arguments or environment variables"
fi

# Ensure data and logs directories exist
mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/logs"

# Run the live trading bot
echo -e "${GREEN}Starting live trading bot...${NC}"
python -m cmd.live_trading.main "$@"

