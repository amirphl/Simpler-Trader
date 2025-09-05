# Configuration Files

This directory contains scenario-specific configuration files.

## ETH 15m Scenario

**File:** `eth_15m.env`

**Configuration:**
- Symbol: ETHUSDT
- Timeframe: 15m
- Window Size: 2
- Leverage: 1.0
- Take Profit: 0.3 (30%)
- Stop Loss: Percent mode (0.5% below entry)
- Exchange Fee: 0.04% per side (0.0004)
- Wick Filter: disabled
- Bollinger Filter: disabled (period 20, stddev 2.0)
- Volume Filter: enabled
- Stochastic Filter: enabled (K20 > K100)
- Start: 2025-01-01T00:00:00Z
- End: Current time (updated dynamically)
- Proxy: http://127.0.0.1:12334

**Usage:**

1. **Using the script:**
   ```bash
   ./scripts/run_eth_15m.sh
   ```

2. **Using the direct command script:**
   ```bash
   ./scripts/run_eth_15m_direct.sh
   ```

3. **Manual command:**
   ```bash
   # Load config and run
   set -a
   source configs/eth_15m.env
   set +a
   export BACKTEST_END=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
   python -m cmd.backtest.main
   ```

4. **Direct CLI (no .env file):**
   ```bash
   python -m cmd.backtest.main \
     --symbol ETHUSDT \
     --timeframe 15m \
     --window-size 2 \
     --leverage 1.0 \
     --take-profit-pct 0.3 \
    --stop-loss-mode percent \
    --stop-loss-pct 0.005 \
    --volume-filter \
    --stoch-enabled \
    --stoch-first-line k \
    --stoch-first-period 20 \
    --stoch-second-line k \
    --stoch-second-period 100 \
    --stoch-comparison gt \
    --bollinger-period 20 \
    --bollinger-stddev 2.0 \
    --exchange-fee-pct 0.0004 \
     --start 2025-01-01T00:00:00Z \
     --end $(date -u +"%Y-%m-%dT%H:%M:%SZ") \
     --initial-capital 10000.0 \
     --proxy http://127.0.0.1:12334 \
     --store-path ./data/eth_15m_candles.db \
     --stats-output ./results/eth_15m_stats.json \
     --plot-output ./results/eth_15m_plot.html \
     --show-plot
   ```

## Creating New Configurations

1. Copy `eth_15m.env` to a new file (e.g., `btc_1h.env`)
2. Update the values for your scenario
3. Create a corresponding script in `../scripts/` if needed

## Live Trading

**Files:** `live_trading.env.example` (copy to `live_trading.env`)

**Key settings:**
- `EXCHANGE`, `TRADING_MODE`, `API_KEY`, `API_SECRET`, `API_PASSPHRASE`, `TESTNET`
- `TIMEFRAME`, `TOP_M_SYMBOLS`, `TOP_N_SIGNALS`, `PRICE_CHANGE_THRESHOLD`
- Heiken Ashi: `HEIKEN_ASHI_CANDLES`, `LEVERAGE`, `TAKE_PROFIT_PCT`
- Position/risk: `MARGIN_MODE`, `DISABLE_SYMBOL_HOURS`, `POSITION_SIZE_USDT`, `MAX_CONCURRENT_POSITIONS`, `MAX_POSITION_SIZE_PCT`
- Persistence/logging: `STATE_FILE`, `POSITIONS_DB`, `LOG_FILE`
- Scheduling: `CANDLE_READY_DELAY_SECONDS`
- Network: `HTTP_PROXY`, `HTTPS_PROXY`, `PROXY`
- Notifications: `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_PROXY`, `TELEGRAM_TIMEOUT`

**Usage:**
```bash
cp configs/live_trading.env.example configs/live_trading.env
# edit values
./scripts/run_live_trading.sh
```
