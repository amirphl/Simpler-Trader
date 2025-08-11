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

