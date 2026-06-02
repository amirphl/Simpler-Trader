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

# Resolve strategy-specific entrypoint.
read_strategy_name() {
    local config_path="$1"
    if [ ! -f "${config_path}" ]; then
        return 1
    fi

    local line
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -z "${line}" ] && continue
        case "${line}" in
            \#*) continue ;;
        esac
        line="${line%%#*}"
        case "${line}" in
            STRATEGY_NAME=*)
                local value="${line#STRATEGY_NAME=}"
                value="${value%\"}"
                value="${value#\"}"
                value="${value%\'}"
                value="${value#\'}"
                value="${value#"${value%%[![:space:]]*}"}"
                value="${value%"${value##*[![:space:]]}"}"
                printf '%s\n' "${value}"
                return 0
                ;;
        esac
    done < "${config_path}"

    return 1
}

resolve_strategy_module() {
    local strategy_name="$1"
    case "${strategy_name}" in
        heiken_ashi)
            printf '%s\n' "cmd.live_trading.heiken_ashi_main"
            ;;
        strong_trend_stair)
            printf '%s\n' "cmd.live_trading.strong_trend_stair_main"
            ;;
        pinbar_magic_v3|"")
            printf '%s\n' "cmd.live_trading.pinbar_magic_v3_main"
            ;;
        *)
            echo -e "${YELLOW}Warning: Unknown strategy '${strategy_name}', defaulting to pinbar_magic_v3${NC}" >&2
            printf '%s\n' "cmd.live_trading.pinbar_magic_v3_main"
            ;;
    esac
}

# Optional config file forwarding to strategy-specific main module.
EXTRA_ARGS=()
if [ -n "${CONFIG_FILE:-}" ] && [ -f "${CONFIG_FILE}" ]; then
    echo -e "${YELLOW}Using configuration file: ${CONFIG_FILE}${NC}"
    EXTRA_ARGS+=(--config-file "${CONFIG_FILE}")
elif [ -n "${CONFIG_FILE:-}" ]; then
    echo -e "${YELLOW}Warning: Config file not found: ${CONFIG_FILE}${NC}"
    echo "Using strategy-specific defaults from the selected live trading module"
fi

STRATEGY_NAME="${LIVE_STRATEGY_NAME:-${STRATEGY_NAME:-}}"
if [ -z "${STRATEGY_NAME}" ] && [ -n "${CONFIG_FILE:-}" ] && [ -f "${CONFIG_FILE}" ]; then
    STRATEGY_NAME="$(read_strategy_name "${CONFIG_FILE}" || true)"
fi

LIVE_TRADING_MODULE="$(resolve_strategy_module "${STRATEGY_NAME}")"

# Ensure data and logs directories exist
mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/logs"

# Run the live trading bot
echo -e "${GREEN}Starting live trading bot...${NC}"
echo "Module: ${LIVE_TRADING_MODULE}"
python3 -m "${LIVE_TRADING_MODULE}" "${EXTRA_ARGS[@]}" "$@"
