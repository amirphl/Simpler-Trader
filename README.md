# Simpler Trader

Crypto strategy research, backtesting, signal monitoring, and live-trading tooling.

This repository is a Python workspace for testing trading ideas against Binance candle data, running a local FastAPI backtest panel, and operating strategy-specific live coordinators against supported exchange adapters. It is intended for research and controlled live testing, not unattended production trading.

## What Is Included

- Backtesting for `engulfing`, `pinbar_magic_v3`, `stochastic_fsm`, and `strong_trend_stair`.
- Binance candle ingestion with PostgreSQL-backed storage.
- A local web control panel for submitting and monitoring backtests.
- Live trading coordinators for `heiken_ashi`, `pinbar_magic_v3`, and `strong_trend_stair`.
- Bitunix futures adapter code plus helper scripts for positions, orders, and TP/SL management.
- Telegram signal notification for live engulfing-pattern scans.
- Experimental market-structure tools for pivots, BOS/CHOCH, and liquidity zones.

## Safety First

Live trading code can place real orders when configured with live credentials and live mode. Treat every config file under `configs/` as sensitive once copied from an example template.

- Do not commit real API keys, Telegram tokens, or account identifiers.
- Prefer exchange testnet or dry-run flows before any live run.
- Review strategy settings, leverage, position sizing, and margin mode before starting a coordinator.
- Local `.env` files can be read by CLI defaults and may appear in `--help` output, so avoid sharing terminal logs from credentialed machines.
- This repo does not provide high-availability monitoring, automatic key rotation, alert escalation, or operational guardrails expected in production systems.

## Requirements

- Python 3.10 or newer.
- PostgreSQL for candle storage.
- `pip` and a virtual environment.
- Node/npm only for JavaScript linting or future web asset work; the current web UI is plain checked-in static assets.

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your shell does not expose `python` after activating the virtual environment, use `python3` in the commands below.

## Project Layout

```text
.
|-- backtest/              # Backtest engine, reports, plotting, and strategies
|-- candle_downloader/     # Binance candle client, downloader, and PostgreSQL store
|-- cmd/                   # Python module entrypoints
|-- configs/               # Live-trading .env templates and config notes
|-- experiments/           # Pivot, liquidity-zone, BOS/CHOCH, and CSV helpers
|-- live_trading/          # Coordinators, exchange adapters, position management
|-- scripts/               # Launchers, candle utilities, and Bitunix helper CLIs
|-- signal_notifier/       # Telegram signal scanner and notifier
|-- tests/                 # Unit tests
|-- web/ui/                # Static browser UI
`-- webserver/             # FastAPI app and backtest job manager
```

## Database Setup

Candle storage is PostgreSQL-only.

```bash
createdb scalp_test

export CANDLE_DB_HOST=localhost
export CANDLE_DB_PORT=5432
export CANDLE_DB_USER=postgres
export CANDLE_DB_PASSWORD=postgres
export CANDLE_DB_NAME=scalp_test
```

Supported database environment variables:

- `CANDLE_DATABASE_URL` or `DATABASE_URL`
- `CANDLE_DB_HOST`, `CANDLE_DB_PORT`, `CANDLE_DB_USER`, `CANDLE_DB_PASSWORD`, `CANDLE_DB_NAME`
- `CANDLE_DB_SSLMODE`, `CANDLE_DB_MIN_POOL_SIZE`, `CANDLE_DB_MAX_POOL_SIZE`
- Matching `POSTGRES_*` aliases
- `CANDLE_DB_ENV_FILE` or `POSTGRES_ENV_FILE` pointing to an env file

The `candles` table and index are created automatically by the storage layer.

## Backtesting

Backtests download missing Binance candles into PostgreSQL, load the requested range, run the selected strategy, write statistics JSON, and optionally save an interactive Plotly chart.

Basic example:

```bash
python -m cmd.backtest.main \
  --strategy pinbar_magic_v3 \
  --symbol BTCUSDT \
  --timeframe 1h \
  --start 2025-01-01T00:00:00Z \
  --end 2025-02-01T00:00:00Z \
  --initial-capital 10000 \
  --store-kind postgres \
  --stats-output results/pinbar_magic_v3_stats.json \
  --plot-output results/pinbar_magic_v3.html
```

Supported strategy names:

- `engulfing`
- `pinbar_magic_v3`
- `stochastic_fsm`
- `strong_trend_stair`

Useful flags:

- `--override-download` redownloads candles even when local rows exist.
- `--store-path path/to/db.env` loads PostgreSQL settings from an env file.
- `--http-proxy`, `--https-proxy`, or `--proxy` route Binance requests through a proxy.
- `--show-plot` opens the generated Plotly chart in a browser.
- `--no-stochastic` and `--no-equity` hide chart subplots.

Strategy settings can be supplied with CLI flags or environment variables. Run a strategy-specific help command for the exact options:

```bash
python -m cmd.backtest.main --strategy engulfing --help
python -m cmd.backtest.main --strategy pinbar_magic_v3 --help
python -m cmd.backtest.main --strategy stochastic_fsm --help
python -m cmd.backtest.main --strategy strong_trend_stair --help
```

## Web Control Panel

Start the local FastAPI panel:

```bash
./scripts/run_web.sh --local
```

Then open:

```text
http://127.0.0.1:9092
```

`--local` binds to `127.0.0.1:9092`, disables HTTPS enforcement, and trusts localhost. Without `--local`, configure these environment variables as needed:

- `WEB_HOST`
- `WEB_PORT`
- `WEB_LOG_LEVEL`
- `WEB_FORCE_HTTPS`
- `WEB_TRUSTED_HOSTS`
- `WEB_ALLOWED_ORIGINS`

Main API surfaces:

- `POST /api/backtests`
- `GET /api/backtests/{job_id}`
- `GET /api/backtests/{job_id}/result`
- `WS /ws/backtests/{job_id}`

Additional experiment servers have dedicated launch scripts:

```bash
./scripts/run_pivot_server.sh
./scripts/run_bos_choch_server.sh
./scripts/run_liquidity_zone_server.sh
```

## Candle Utilities

Download candles directly to CSV:

```bash
python scripts/download_candles_to_csv.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --start 2025-01-01T00:00:00Z \
  --end 2025-01-07T00:00:00Z \
  --output data/BTCUSDT-15m.csv
```

Check PostgreSQL candle completeness and optionally redownload missing data:

```bash
python scripts/check_missing_candles.py \
  --db-kind postgres \
  --pg-host localhost \
  --pg-port 5432 \
  --pg-user postgres \
  --pg-password postgres \
  --pg-db scalp_test \
  --symbol BTCUSDT \
  --timeframe 15m \
  --start-date 2025-01-01T00:00:00Z \
  --end-date 2025-02-01T00:00:00Z \
  --redownload-from-binance
```

## Live Trading

Create a strategy config from a template:

```bash
cp configs/live_trading.heiken_ashi.env.example configs/live_trading.heiken_ashi.env
cp configs/live_trading.pinbar_magic_v3.env.example configs/live_trading.pinbar_magic_v3.env
cp configs/live_trading.strong_trend_stair.env.example configs/live_trading.strong_trend_stair.env
```

Edit only the strategy file you plan to run. See [configs/README.md](configs/README.md) for the full variable list and resolution order.

Run through the generic launcher:

```bash
CONFIG_FILE=configs/live_trading.pinbar_magic_v3.env ./scripts/run_live_trading.sh
```

Or run a strategy module directly:

```bash
python -m cmd.live_trading.heiken_ashi_main --config-file configs/live_trading.heiken_ashi.env
python -m cmd.live_trading.pinbar_magic_v3_main --config-file configs/live_trading.pinbar_magic_v3.env
python -m cmd.live_trading.strong_trend_stair_main --config-file configs/live_trading.strong_trend_stair.env
```

Config precedence for live trading is:

1. CLI arguments
2. OS environment variables
3. Strategy config file
4. Built-in defaults

Common live-trading settings include:

- `STRATEGY_NAME`
- `EXCHANGE`
- `TRADING_MODE`
- `API_KEY`, `API_SECRET`, `API_PASSPHRASE`
- `TESTNET`
- `LEVERAGE`
- `POSITION_SIZE_USDT`
- `MAX_CONCURRENT_POSITIONS`
- `MARGIN_MODE`
- `STATE_FILE`, `POSITIONS_DB`, `KLINES_DB`, `LOG_FILE`
- `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

Use `--help` on a specific module before changing live settings:

```bash
python -m cmd.live_trading.pinbar_magic_v3_main --help
```

## Signal Notifier

Monitor Binance symbols for engulfing signals and send Telegram notifications:

```bash
python -m cmd.signal_notifier.main \
  --timeframe 15m \
  --top-n 100 \
  --telegram-token "$TELEGRAM_BOT_TOKEN" \
  --telegram-chat-id "$TELEGRAM_CHAT_ID"
```

Use `--dry-run` to log signals without sending Telegram messages:

```bash
python -m cmd.signal_notifier.main --timeframe 15m --symbols BTCUSDT,ETHUSDT --dry-run
```

## Bitunix Helper Scripts

These scripts use `BITUNIX_API_KEY` and `BITUNIX_API_SECRET` unless `--key` and `--secret` are provided.

```bash
export BITUNIX_API_KEY=...
export BITUNIX_API_SECRET=...
```

Common helpers:

- `scripts/bitunix_list_positions.py`
- `scripts/bitunix_place_position_tpsl.py`
- `scripts/bitunix_modify_position_tpsl.py`
- `scripts/bitunix_update_stop_loss.py`
- `scripts/bitunix_order_smoke_test.py`

Example:

```bash
python scripts/bitunix_list_positions.py --symbol BTCUSDT
```

Place position-level TP/SL:

```bash
python scripts/bitunix_place_position_tpsl.py \
  --symbol BTCUSDT \
  --position-id 123456 \
  --sl-price 27000 \
  --sl-stop-type MARK_PRICE
```

## Testing And Checks

Run the unit test suite:

```bash
python -m pytest
```

Run the existing type checker configuration if Pyright is installed:

```bash
pyright
```

Run npm tooling:

```bash
npm install
npm exec eslint .
```

## Deployment Notes

The `deploy/` directory contains example service configuration:

- `deploy/systemd/backtest-web.service`
- `deploy/nginx/balut.jaazebeh.ir.conf`

Treat these as starting points. Before exposing the web panel, review trusted hosts, CORS, HTTPS settings, firewall rules, process supervision, logs, and credential storage.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
