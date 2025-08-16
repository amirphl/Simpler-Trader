# Backtest CLI Usage Guide

This guide shows how to use the integrated backtest CLI that downloads candles, runs strategies, generates statistics, and creates visualizations.

## Quick Start

### Using Command-Line Arguments

```bash
python -m cmd.backtest.main \
  --symbol BTCUSDT \
  --timeframe 1h \
  --window-size 5 \
  --leverage 2.0 \
  --take-profit-pct 0.02 \
  --start 2024-01-01T00:00:00Z \
  --end 2024-02-01T00:00:00Z \
  --initial-capital 10000.0 \
  --stats-output ./backtest_stats.json \
  --plot-output ./backtest_plot.html \
  --show-plot
```

### Using Environment Variables

Create a `.env` file (or export variables):

```bash
export STRATEGY_SYMBOL=BTCUSDT
export STRATEGY_TIMEFRAME=1h
export STRATEGY_WINDOW_SIZE=5
export STRATEGY_LEVERAGE=2.0
export STRATEGY_TAKE_PROFIT_PCT=0.02
export BACKTEST_START=2024-01-01T00:00:00Z
export BACKTEST_END=2024-02-01T00:00:00Z
export BACKTEST_INITIAL_CAPITAL=10000.0
export STATS_OUTPUT=./backtest_stats.json
export PLOT_OUTPUT=./backtest_plot.html
export SHOW_PLOT=true
```

Then run:

```bash
python -m cmd.backtest.main
```

### Mixed: Environment Variables + CLI Overrides

Environment variables provide defaults, CLI arguments override them:

```bash
# .env sets defaults
export STRATEGY_SYMBOL=BTCUSDT
export STRATEGY_TIMEFRAME=1h
# ... other vars ...

# CLI overrides specific values
python -m cmd.backtest.main --symbol ETHUSDT --leverage 3.0
```

## Configuration Options

### Strategy Parameters

- `--symbol` / `STRATEGY_SYMBOL`: Trading pair (e.g., BTCUSDT, ETHUSDT)
- `--timeframe` / `STRATEGY_TIMEFRAME`: Binance interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
- `--window-size` / `STRATEGY_WINDOW_SIZE`: Number of candles to check for bearish pattern
- `--leverage` / `STRATEGY_LEVERAGE`: Leverage multiplier (e.g., 2.0 for 2x)
- `--take-profit-pct` / `STRATEGY_TAKE_PROFIT_PCT`: Take profit as decimal (0.02 = 2%)
- `--doji-size` / `STRATEGY_DOJI_SIZE`: Doji detection threshold (default: 0.05)

### Backtest Parameters

- `--start` / `BACKTEST_START`: Start datetime in ISO8601 format (UTC)
- `--end` / `BACKTEST_END`: End datetime in ISO8601 format (UTC)
- `--initial-capital` / `BACKTEST_INITIAL_CAPITAL`: Starting capital amount
- `--override-download`: Re-download all candles (ignores existing data)

### Storage Parameters

- `--store-kind` / `STORE_KIND`: Storage type (`sqlite` or `csv`)
- `--store-path` / `STORE_PATH`: Path to storage file/database

### Network Parameters (Optional)

- `--http-proxy` / `HTTP_PROXY`: HTTP proxy URL
- `--https-proxy` / `HTTPS_PROXY`: HTTPS proxy URL
- `--proxy` / `PROXY`: Shortcut for both HTTP and HTTPS proxy

### Output Parameters

- `--stats-output` / `STATS_OUTPUT`: Path to write statistics JSON file
- `--plot-output` / `PLOT_OUTPUT`: Path to save plot (supports .html, .png, .svg, .pdf)
- `--show-plot`: Display plot in browser (or set `SHOW_PLOT=true`)
- `--no-stochastic`: Hide stochastic oscillator subplot
- `--no-equity`: Hide equity curve subplot

## Output Files

### Statistics JSON

The statistics file contains comprehensive backtest results:

```json
{
  "strategy": "EngulfingStrategy",
  "config": {
    "start": "2024-01-01T00:00:00+00:00",
    "end": "2024-02-01T00:00:00+00:00",
    "initial_capital": 10000.0,
    ...
  },
  "statistics": {
    "total_trades": 15,
    "winning_trades": 9,
    "losing_trades": 6,
    "win_rate": 0.6,
    "net_profit": 1250.50,
    "net_profit_pct": 12.51,
    "sharpe_ratio": 1.85,
    "max_drawdown_pct": 5.23,
    ...
  },
  "trades": [
    {
      "entry_time": "2024-01-05T10:00:00+00:00",
      "exit_time": "2024-01-05T14:00:00+00:00",
      "pnl": 150.25,
      "return_pct": 1.50,
      "notes": "Take Profit at 42500.00",
      "metadata": {
        "entry_price": 42000.0,
        "exit_price": 42500.0,
        "stop_loss": 41800.0,
        "take_profit": 42840.0,
        "leverage": 2.0
      }
    },
    ...
  ]
}
```

### Plot Output

The plot is an interactive HTML file (or image) showing:
- Candlestick chart with volume
- Entry/exit markers
- Stop loss and take profit lines
- Stochastic oscillator
- Equity curve
- Summary statistics in title

## Example Workflow

1. **First run** (downloads candles and runs backtest):
   ```bash
   python -m cmd.backtest.main \
     --symbol BTCUSDT \
     --timeframe 1h \
     --window-size 5 \
     --leverage 2.0 \
     --take-profit-pct 0.02 \
     --start 2024-01-01T00:00:00Z \
     --end 2024-02-01T00:00:00Z \
     --initial-capital 10000.0 \
     --stats-output ./results/stats.json \
     --plot-output ./results/plot.html
   ```

2. **Subsequent runs** (reuses downloaded candles):
   ```bash
   # Same command - automatically uses cached candles
   python -m cmd.backtest.main ...
   ```

3. **Force re-download**:
   ```bash
   python -m cmd.backtest.main ... --override-download
   ```

## Dependencies

- **Required**: Standard library only (no external deps for core functionality)
- **Optional for plotting**: `plotly` (install with `pip install plotly`)
- **Optional for image export**: `kaleido` (install with `pip install kaleido`)

## Troubleshooting

### Plotly Not Available

If you see "Plotly not available", install it:
```bash
pip install plotly
```

### Proxy Issues

If you need to use a proxy:
```bash
python -m cmd.backtest.main ... --proxy http://127.0.0.1:7890
```

Or set environment variable:
```bash
export PROXY=http://127.0.0.1:7890
python -m cmd.backtest.main ...
```

### Missing Configuration

The CLI will report missing required parameters:
```
ValueError: Missing required configuration: symbol, timeframe, window_size
```

Ensure all required parameters are provided via CLI args or environment variables.

