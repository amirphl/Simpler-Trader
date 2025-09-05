# Scalp Test - Cryptocurrency Backtesting Framework

A comprehensive, production-ready backtesting framework for cryptocurrency trading strategies. Built with Python, featuring automatic candle data management, pattern recognition, and interactive visualization.

## 🚀 Features

- **Automated Candle Management**: Downloads and caches historical candles from Binance with intelligent gap-filling
- **Pattern Recognition**: 15+ candlestick patterns including Doji, Engulfing, Stars, Harami, and more
- **Strategy Framework**: Clean architecture with extensible strategy base classes
- **Comprehensive Statistics**: Win rate, Sharpe ratio, drawdown, CAGR, profit factor, and more
- **Interactive Visualization**: Scrollable, zoomable charts with entry/exit markers, indicators, and equity curves
- **Multiple Strategies**: Bullish Engulfing, Bullish Pinbar, and Pin Bar Magic — each with its own control panel
- **Live Trading**: Automated trading with Heiken Ashi reversal strategy, position management, and risk controls
- **Proxy Support**: Built-in HTTP/HTTPS proxy configuration for network flexibility
- **Multiple Storage Backends**: SQLite (recommended) or CSV for candle data
- **Type Safety**: Full type hints with Pyright configuration
- **Web Control Panel**: FastAPI-powered UI with REST + WebSocket endpoints
- **Lean Core**: Backtesting/downloader modules rely on the Python standard library; optional UI/plotting layers add FastAPI and Plotly only when needed

## 📋 Requirements

- Python 3.10 or higher
- Optional: `plotly` and `kaleido` for visualization (see Installation)

## 🔧 Installation

### Basic Installation

The core framework requires no external dependencies:

```bash
git clone <repository-url>
cd scalp-test
```

### Optional: Install Visualization Dependencies

For interactive plotting and chart generation:

```bash
pip install -r requirements.txt
```

This installs:
- `plotly>=5.18.0` - Interactive plotting
- `kaleido>=0.2.1` - Image export (PNG/SVG/PDF)

## 🎯 Quick Start

### Example: ETH 15m Backtest

Run a complete backtest with a single command:

```bash
./scripts/run_eth_15m_direct.sh
```

Or manually:

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
  --start 2025-01-01T00:00:00Z \
  --end $(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --initial-capital 10000.0 \
  --exchange-fee-pct 0.0004 \
  --proxy http://127.0.0.1:12334 \
  --stats-output ./results/eth_15m_stats.json \
  --plot-output ./results/eth_15m_plot.html \
  --show-plot
```

## 🖥️ Web Control Panel

Prefer a UI instead of the CLI? Launch the FastAPI-powered control panel:

```bash
pip install -r requirements.txt
# local-only mode (no TLS, binds 127.0.0.1:9092)
python -m cmd.web.main --local
# or use the helper script
./scripts/run_web.sh --local
```

Then open `http://127.0.0.1:9092` to:

- Configure Engulfing strategy parameters via a web form.
- Submit backtests and receive a job ID immediately.
- Watch status updates and final statistics/trades over a live WebSocket stream.
- Retrieve historical backtests anytime via the UI or REST endpoints.
- Tail logs via your process manager (e.g., `journalctl -u backtest-web.service -f`) to review structured entries such as job submission, start, completion, and failures.
- All web-specific logs are also emitted to stdout/stderr (respecting `WEB_LOG_LEVEL`), so local runs show the same structured messages without extra setup.
- The landing page now offers dedicated panels for the Bullish Engulfing and Bullish Pinbar strategies to keep their forms focused.

### API Surface

- `POST /api/backtests` — submit a job. Returns `{job_id, status}` immediately.
- `GET /api/backtests/{job_id}` — check status.
- `GET /api/backtests/{job_id}/result` — fetch stored stats/trades when ready (persists across restarts).
- `WS /ws/backtests/{job_id}` — receive status transitions and the final report in real time.

All job artifacts are stored under `results/web_backtests`, so runs continue even if the browser closes.

### Production Deployment

1. Start the FastAPI service on a loopback port (e.g., `python -m cmd.web.main --host 127.0.0.1 --port 9000`).
2. Install the nginx file `deploy/nginx/balut.jaazebeh.ir.conf` (or adapt it to your domain), update the upstream if needed, then symlink it into `/etc/nginx/sites-enabled/`.
3. Obtain certificates under `/etc/letsencrypt/live/jaazebeh.ir/` (as referenced in the config).
4. Reload nginx and ensure port `9092/tcp` is open on the firewall.
5. Set `WEB_TRUSTED_HOSTS="trade.jaazebeh.ir"` and `WEB_FORCE_HTTPS=true` on the server (defaults already enforce these).

#### Systemd Service (Recommended)

1. Copy the provided unit file:
   ```bash
   sudo install -m 644 deploy/systemd/backtest-web.service /etc/systemd/system/backtest-web.service
   ```
2. Adjust the file to match your deployment paths/user (`WorkingDirectory`, `ExecStart`, `User`/`Group`).
3. Create an environment file for secrets and overrides (optional):
   ```bash
   sudo install -d /etc/scalp-test
   sudo nano /etc/scalp-test/web.env    # export WEB_TRUSTED_HOSTS=..., etc.
   ```
4. Reload systemd and enable the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now backtest-web.service
   ```
5. Tail logs:
   ```bash
   journalctl -fu backtest-web.service
   ```

### Live Trading Bot

**⚠️ WARNING: Live trading involves real money. Always test with testnet first!**

Run the automated live trading bot with Heiken Ashi reversal strategy:

```bash
# Configure the bot
cp configs/live_trading.env.example configs/live_trading.env
# Edit configs/live_trading.env with your API keys and parameters

# Run the bot (starts in testnet mode by default)
./scripts/run_live_trading.sh
```

Or use command-line arguments:

```bash
# Weex Futures Trading (IMPLEMENTED)
python -m cmd.live_trading.main \
  --exchange weex \
  --trading-mode futures \
  --timeframe 15m \
  --candle-ready-delay-seconds 30 \
  --telegram-enabled \
  --telegram-token "$TELEGRAM_BOT_TOKEN" \
  --telegram-chat-id "$TELEGRAM_CHAT_ID" \
  --api-key "your_key" \
  --api-secret "your_secret" \
  --testnet \
  --top-m-symbols 100 \
  --top-n-signals 5 \
  --price-change-threshold 2.0 \
  --heiken-ashi-candles 3 \
  --leverage 10 \
  --take-profit-pct 1.0 \
  --margin-mode isolated \
  --position-size-usdt 100.0

# Weex Spot Trading (IMPLEMENTED)
python -m cmd.live_trading.main \
  --exchange weex \
  --trading-mode spot \
  --timeframe 1h \
  --api-key "your_key" \
  --api-secret "your_secret" \
  --testnet \
  --position-size-usdt 50.0
```

**Supported Exchanges**:
- ✅ **Weex** (Futures & Spot) - Fully implemented
- ⏳ Binance - Interface ready, needs implementation
- ⏳ Bybit - Interface ready, needs implementation

Config & notifications:
- Environment file: copy `configs/live_trading.env.example` to `configs/live_trading.env` and fill in exchange keys (including `API_PASSPHRASE` if required), timeframe, sizing, and `CANDLE_READY_DELAY_SECONDS`.
- Proxies: set `HTTP_PROXY` / `HTTPS_PROXY` or `PROXY` for both the exchange and Binance data fetches.
- Telegram (optional): set `TELEGRAM_ENABLED=true` plus `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to receive executed trade alerts; `TELEGRAM_PROXY` and `TELEGRAM_TIMEOUT` are supported.

See `WEEX_QUICK_START.md` for Weex setup guide and `LIVE_TRADING_USAGE.md` for complete documentation.

### Example: Send Live Signals to Telegram

```bash
python -m cmd.signal_notifier.main \
  --strategy scalping_fvg \
  --timeframe 15m \
  --top-n 100 \
  --live-candles 400 \
  --poll-interval 20 \
  --telegram-token "$TELEGRAM_BOT_TOKEN" \
  --telegram-chat-id "$TELEGRAM_CHAT_ID"
```

Environment variables (optional):

```bash
export TELEGRAM_BOT_TOKEN=123456:ABCDEF
export TELEGRAM_CHAT_ID=-1001234567890
export TELEGRAM_PROXY=http://127.0.0.1:7890  # optional
export SYMBOLS=BTCUSDT,ETHUSDT              # override auto-discoved universe
export SIGNAL_STATE_FILE=./data/signal_state.json
```

The notifier keeps a state file so each signal is only sent once. Use `--dry-run` to preview messages without pushing to Telegram or `--no-state` to resend everything each cycle.

## 📖 Usage

### Command-Line Interface

The main backtest CLI supports both environment variables and command-line arguments:

```bash
python -m cmd.backtest.main [OPTIONS]
```

#### Key Options

**Strategy Parameters:**
- `--symbol SYMBOL` - Trading pair (e.g., BTCUSDT, ETHUSDT)
- `--timeframe TIMEFRAME` - Binance interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
- `--window-size N` - Number of candles to check for bearish pattern
- `--leverage LEVERAGE` - Leverage multiplier (e.g., 2.0 for 2x)
- `--take-profit-pct PCT` - Take profit as decimal (0.02 = 2%)
- `--stop-loss-mode {percent,close,low,open,body}` - Stop-loss placement strategy
- `--stop-loss-pct PCT` - Fractional stop loss when mode is `percent`
- `--exchange-fee-pct PCT` - Exchange fee per side (decimal, default 0.0004)
- `--[no-]skip-wick-filter` - Toggle rejection of long upper-wick engulfing candles
- `--[no-]skip-bollinger-cross` - Toggle rejection of Bollinger upper-band pierces
- `--bollinger-period N` - Period for Bollinger band filter
- `--bollinger-stddev MULT` - Stddev multiplier for Bollinger filter
- `--volume-filter / --no-volume-filter` - Enable or disable volume pressure filter
- `--stoch-enabled / --no-stoch-enabled` - Enable or disable stochastic comparison
- `--stoch-first-line {k,d}` / `--stoch-first-period N` / `--stoch-first-threshold VALUE`
- `--stoch-second-line {k,d}` / `--stoch-second-period N` / `--stoch-second-threshold VALUE`
- `--stoch-comparison {gt,lt}` - Determines which stochastic leg must be greater
- `--stoch-d-smoothing N` - %D smoothing length (when using D legs)

**Backtest Parameters:**
- `--start START` - Start datetime (ISO8601, UTC)
- `--end END` - End datetime (ISO8601, UTC)
- `--initial-capital AMOUNT` - Starting capital

**Network:**
- `--proxy URL` - HTTP/HTTPS proxy URL
- `--http-proxy URL` - HTTP proxy only
- `--https-proxy URL` - HTTPS proxy only

**Output:**
- `--stats-output PATH` - Statistics JSON file path
- `--plot-output PATH` - Plot file path (HTML/PNG/SVG/PDF)
- `--show-plot` - Display plot in browser

See `BACKTEST_USAGE.md` for complete documentation.

### Environment Variables

All configuration can be provided via environment variables:

```bash
export STRATEGY_SYMBOL=BTCUSDT
export STRATEGY_TIMEFRAME=1h
export STRATEGY_WINDOW_SIZE=5
export STRATEGY_LEVERAGE=2.0
export STRATEGY_TAKE_PROFIT_PCT=0.02
export STRATEGY_STOP_LOSS_MODE=percent
export STRATEGY_STOP_LOSS_PCT=0.005
export STRATEGY_EXCHANGE_FEE_PCT=0.0004
export STRATEGY_SKIP_WICK_FILTER=false
export STRATEGY_SKIP_BB_FILTER=false
export STRATEGY_BB_PERIOD=20
export STRATEGY_BB_STDDEV=2.0
export STRATEGY_VOLUME_FILTER_ENABLED=true
export STRATEGY_STOCH_ENABLED=true
export STRATEGY_STOCH_FIRST_LINE=k
export STRATEGY_STOCH_FIRST_PERIOD=20
export STRATEGY_STOCH_FIRST_THRESHOLD=
export STRATEGY_STOCH_SECOND_LINE=k
export STRATEGY_STOCH_SECOND_PERIOD=100
export STRATEGY_STOCH_SECOND_THRESHOLD=
export STRATEGY_STOCH_COMPARISON=gt
export STRATEGY_STOCH_D_SMOOTHING=3
export BACKTEST_START=2024-01-01T00:00:00Z
export BACKTEST_END=2024-02-01T00:00:00Z
export BACKTEST_INITIAL_CAPITAL=10000.0
export PROXY=http://127.0.0.1:7890

python -m cmd.backtest.main
```

**Web Server Environment Variables:**

```bash
export WEB_FORCE_HTTPS=true              # disable when using --local
export WEB_TRUSTED_HOSTS=balut.jaazebeh.ir,trade.jaazebeh.ir,localhost,127.0.0.1
export WEB_ALLOWED_ORIGINS=https://balut.jaazebeh.ir:9091,http://localhost:9092
export WEB_LOG_LEVEL=info                # uvicorn log level
export WEB_CANDLE_PROXY=                  # optional fallback for both http/https proxies (e.g. http://127.0.0.1:12334)
export WEB_CANDLE_HTTP_PROXY=             # override for HTTP-only proxy, takes precedence over WEB_CANDLE_PROXY
export WEB_CANDLE_HTTPS_PROXY=            # override for HTTPS-only proxy, takes precedence over WEB_CANDLE_PROXY
```

### Programmatic Usage

```python
from datetime import datetime, timezone
from pathlib import Path

from backtest import (
    BacktestRunConfig,
    BaseBacktester,
    EngulfingStrategy,
    EngulfingStrategyConfig,
    StopLossMode,
)
from candle_downloader.binance import BinanceClient, BinanceClientConfig
from candle_downloader.downloader import CandleDownloader
from candle_downloader.storage import build_store

# Setup
store = build_store("sqlite", Path("./data/candles.db"))
client = BinanceClient(BinanceClientConfig(proxies={"https": "http://127.0.0.1:7890"}))
downloader = CandleDownloader(client=client, store=store)

# Create strategy
strategy = EngulfingStrategy(
    EngulfingStrategyConfig(
        symbol="BTCUSDT",
        timeframe="1h",
        window_size=5,
        leverage=2.0,
        take_profit_pct=0.02,
        stop_loss_mode=StopLossMode.PERCENT,
        stop_loss_pct=0.005,
        skip_large_upper_wick=True,
        skip_bollinger_cross=False,
        bollinger_period=20,
        bollinger_stddev=2.0,
        enable_volume_pressure_filter=True,
        enable_stochastic_filter=True,
        stochastic_first_line="k",
        stochastic_first_period=20,
        stochastic_second_line="k",
        stochastic_second_period=100,
        stochastic_comparison="gt",
    )
)

# Run backtest
runner = BaseBacktester(strategy=strategy, downloader=downloader, store=store)
report = runner.run(
    BacktestRunConfig(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        initial_capital=10_000.0,
    )
)

# Access results
print(f"Total Trades: {report.statistics.total_trades}")
print(f"Win Rate: {report.statistics.win_rate*100:.2f}%")
print(f"Net P&L: {report.statistics.net_profit:+.2f}")
print(f"Sharpe Ratio: {report.statistics.sharpe_ratio:.2f}")

# Plot results
from backtest import plot_backtest_from_store, show_plot

fig = plot_backtest_from_store(report, store, "BTCUSDT", "1h")
show_plot(fig)
```

## 🏗️ Architecture

The framework follows Clean Architecture principles with clear separation of concerns:

### Core Modules

- **`candle_downloader/`**: Binance integration, candle storage, and data management
  - `binance.py` - Binance API client with proxy support
  - `downloader.py` - Intelligent candle synchronization
  - `storage.py` - SQLite and CSV storage backends
  - `models.py` - Domain models (Candle, etc.)

- **`backtest/`**: Backtesting engine and strategies
  - `base.py` - Base backtester and strategy interfaces
  - `engulfing_strategy.py` - Bullish Engulfing + Stochastic strategy
  - `patterns.py` - 15+ candlestick pattern detection
  - `plotter.py` - Interactive visualization with Plotly

- **`live_trading/`**: Live trading automation
  - `exchange.py` - Exchange interface (abstract base class)
  - `models.py` - Live trading data structures
  - `scanner.py` - Symbol scanning and top mover detection
  - `heiken_ashi.py` - Heiken Ashi calculations and reversal detection
  - `strategy.py` - Live trading strategy engine
  - `position_manager.py` - Position execution and state persistence
  - `coordinator.py` - Main orchestrator with scheduling

- **`cmd/`**: Command-line interfaces
  - `backtest/main.py` - Main backtest CLI
  - `candle_downloader/main.py` - Standalone candle downloader
  - `live_trading/main.py` - Live trading bot CLI

### Design Principles

- **Interface-Driven**: All components interact through interfaces
- **Dependency Injection**: Explicit dependencies, easy to test
- **Immutable Data**: Dataclasses with frozen=True where appropriate
- **Type Safety**: Full type hints throughout
- **Error Handling**: Explicit error checking and meaningful messages

## 📊 Strategy: Bullish Engulfing + Stochastic

The included `EngulfingStrategy` implements a long-only strategy:

**Entry Conditions:**
1. Previous candle is "Bullish Engulfing" pattern
2. N previous candles (before the engulfing candle) are bearish
3. **Optional Stochastic Filter**: configurable comparison between any two stochastic legs (default K(20) > K(100)); can be disabled.

**Exit Conditions:**
- **Take Profit**: Price reaches entry × (1 + take_profit_pct)
- **Stop Loss**: Configurable per trade — percent drop from entry (default 0.5%), engulfing candle close, engulfing candle low, engulfing candle open, or a fraction of the engulfing candle body.
- **Fees**: PnL automatically subtracts a configurable per-side exchange fee (default 0.04%) on both entry and exit.

**Optional Filters:**
- Skip signals where the engulfing candle's upper wick is larger than its body.
- Skip signals where the engulfing candle pierces the Bollinger upper band (period/stddev configurable).
- Enable or disable the volume-pressure exhaustion filter (window + threshold configurable).

**Configuration:**
- `symbol`: Trading pair
- `timeframe`: Candle interval
- `window_size`: Number of bearish candles to check
- `leverage`: Position leverage multiplier
- `take_profit_pct`: Take profit percentage (decimal)
- `stop_loss_mode`: `percent`, `close`, `low`, `open`, or `body`
- `stop_loss_pct`: Fraction used when `stop_loss_mode=percent` and as the body fraction when `stop_loss_mode=body`
- `exchange_fee_pct`: Maker/taker fee per side expressed as a decimal (default 0.0004)
- `skip_large_upper_wick`: Toggle wick-exhaustion filter
- `skip_bollinger_cross`: Toggle Bollinger upper-band filter
- `bollinger_period`: Period for Bollinger calculation
- `bollinger_stddev`: Standard deviation multiplier for Bollinger bands

## 📈 Statistics and Metrics

The framework calculates comprehensive performance metrics:

- **Trade Statistics**: Total trades, winning/losing trades, win rate
- **Profit Metrics**: Gross profit, gross loss, net profit, profit factor
- **Risk Metrics**: Maximum drawdown, Sharpe ratio
- **Time Metrics**: Average trade duration, CAGR (Compound Annual Growth Rate)
- **Expectancy**: Average profit per trade

All statistics are exported to JSON and displayed in interactive plots.

## 🎨 Visualization

Interactive plots include:

- **Candlestick Chart**: Full OHLCV data with volume bars
- **Entry/Exit Markers**: Visual indicators for trade signals
- **Stop Loss/Take Profit Lines**: Horizontal lines showing exit levels
- **Stochastic Oscillator**: K(20) and K(100) with overbought/oversold levels
- **Equity Curve**: Capital progression over time
- **Summary Statistics**: Win rate, P&L, Sharpe ratio in chart title

Plots are scrollable, zoomable, and exportable to HTML, PNG, SVG, or PDF.

## 📁 Project Structure

```
scalp-test/
├── backtest/                 # Backtesting engine
│   ├── base.py              # Base classes and statistics
│   ├── engulfing_strategy.py # Bullish Engulfing strategy
│   ├── patterns.py          # Candlestick pattern detection
│   └── plotter.py           # Visualization
├── candle_downloader/        # Data management
│   ├── binance.py           # Binance API client
│   ├── downloader.py        # Candle synchronization
│   ├── models.py            # Domain models
│   └── storage.py           # Storage backends
├── live_trading/             # Live trading automation
│   ├── exchange.py          # Exchange interface
│   ├── models.py            # Trading data structures
│   ├── scanner.py           # Symbol scanner
│   ├── heiken_ashi.py       # Heiken Ashi calculations
│   ├── strategy.py          # Strategy engine
│   ├── position_manager.py  # Position management
│   ├── coordinator.py       # Main coordinator
│   └── exchanges/           # Exchange implementations
│       └── README.md        # Implementation guide
├── cmd/                      # CLI entry points
│   ├── backtest/            # Main backtest CLI
│   ├── candle_downloader/   # Candle downloader CLI
│   └── live_trading/        # Live trading CLI
├── configs/                  # Scenario configurations
│   ├── eth_15m.env          # ETH 15m scenario
│   └── live_trading.env.example # Live trading config example
├── scripts/                  # Helper scripts
│   ├── run_eth_15m.sh       # Run ETH scenario (with .env)
│   ├── run_eth_15m_direct.sh # Run ETH scenario (direct)
│   └── run_live_trading.sh  # Run live trading bot
├── data/                     # Candle data storage (auto-created)
├── results/                  # Backtest results (auto-created)
├── logs/                     # Log files (auto-created)
├── requirements.txt          # Optional dependencies
├── pyrightconfig.json        # Type checking configuration
├── README.md                 # This file
├── BACKTEST_USAGE.md        # Backtest documentation
└── LIVE_TRADING_USAGE.md    # Live trading documentation
```

## 🔍 Pattern Detection

The framework includes 15+ candlestick patterns:

- **Reversal Patterns**: Doji, Hammer, Inverted Hammer, Hanging Man, Shooting Star
- **Engulfing Patterns**: Bullish/Bearish Engulfing
- **Star Patterns**: Morning Star, Evening Star
- **Harami Patterns**: Bullish/Bearish Harami
- **Other Patterns**: Piercing Line, Dark Cloud Cover, Bullish/Bearish Kicker, Bullish Belt

Patterns are detected using the same logic as the TradingView Pinescript study.

## 🛠️ Development

### Type Checking

The project uses Pyright for static type checking:

```bash
# Install Pyright
npm install -g pyright

# Run type checking
pyright
```

Configuration is in `pyrightconfig.json`.

### Adding New Strategies

1. Extend `BacktestStrategy`:

```python
from backtest import BacktestStrategy, BacktestContext, TradePerformance

class MyStrategy(BacktestStrategy):
    def symbols(self):
        return ["BTCUSDT"]
    
    def timeframes(self):
        return ["1h"]
    
    def run(self, context: BacktestContext) -> Sequence[TradePerformance]:
        # Your strategy logic here
        candles = context.data["BTCUSDT"]["1h"]
        # ... analyze and generate trades
        return trades
```

2. Use with `BaseBacktester`:

```python
strategy = MyStrategy()
backtester = BaseBacktester(strategy=strategy, downloader=downloader, store=store)
report = backtester.run(config)
```

### Adding New Patterns

Patterns are detected in `backtest/patterns.py`. The `detect_candle_patterns()` function returns `CandlePatternSignals` with boolean flags for each pattern.

## 📝 Configuration Files

### Scenario Configs

Pre-configured scenarios are in `configs/`:

- `eth_15m.env` - ETH 15m scenario with specific parameters

See `configs/README.md` for details.

### Environment Variables

All configuration can be provided via environment variables. See `BACKTEST_USAGE.md` for the complete list.

## 🐛 Troubleshooting

### Plotly Not Available

If you see "Plotly not available", install it:

```bash
pip install plotly
```

### Proxy Issues

Ensure your proxy is accessible:

```bash
# Test proxy
curl -x http://127.0.0.1:12334 https://api.binance.com/api/v3/ping
```

### Missing Candles

The downloader automatically fills gaps. To force re-download:

```bash
python -m cmd.backtest.main ... --override-download
```

### Type Checking Errors

Ensure you're using Python 3.10+ and that `pyrightconfig.json` is in the project root.

## 📚 Documentation

- **`BACKTEST_USAGE.md`**: Complete CLI and configuration reference
- **`LIVE_TRADING_USAGE.md`**: Live trading setup, configuration, and usage guide
- **`WEEX_QUICK_START.md`**: Quick start guide for Weex exchange
- **`live_trading/exchanges/WEEX_SETUP.md`**: Detailed Weex setup and configuration
- **`configs/README.md`**: Configuration file documentation
- **`live_trading/exchanges/README.md`**: Exchange implementation guide
- **Code**: Full docstrings and type hints throughout

## 🤝 Contributing

Contributions are welcome! Please ensure:

- Code follows existing patterns and style
- Type hints are included
- Docstrings are added for public APIs
- Tests are added for new features (when applicable)

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Binance API for historical data
- TradingView Pinescript community for pattern definitions
- Plotly for excellent visualization tools

---

**Built with ❤️ for cryptocurrency traders and developers**
