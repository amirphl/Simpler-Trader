# Scalp Test

`scalp-test` is a crypto strategy research and execution workspace with:
- backtesting engines,
- a candle downloader,
- a web backtest control panel,
- and live trading runners.

## What Is Included

- Backtest strategies:
  - `engulfing`
  - `pinbar_magic_v2`
  - `stochastic_fsm` (dedicated CLI)
- Web UI strategies:
  - Engulfing
  - Pinbar
  - Pin Bar Magic v1
  - Pin Bar Magic v2
  - Stochastic FSM
- Live trading strategies:
  - `heiken_ashi`
  - `pinbar_magic_v2` (ETHUSDT-focused coordinator)
- Candle storage:
  - PostgreSQL only (with pooled connections via `psycopg_pool`)

## Requirements

- Python 3.10+
- PostgreSQL (default local settings shown below)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Candle Database (PostgreSQL)

The project now uses PostgreSQL for candle persistence.

Default connection values:

- host: `localhost`
- port: `5432`
- user: `postgres`
- password: `postgres`
- database: `scalp_test`
- sslmode: `prefer`

Equivalent `psql` command:

```bash
psql --host localhost --port 5432 --username postgres -d scalp_test --password
```

Supported env keys (either prefix works):

- `CANDLE_DB_HOST` / `POSTGRES_HOST`
- `CANDLE_DB_PORT` / `POSTGRES_PORT`
- `CANDLE_DB_USER` / `POSTGRES_USER`
- `CANDLE_DB_PASSWORD` / `POSTGRES_PASSWORD`
- `CANDLE_DB_NAME` / `POSTGRES_DB`
- `CANDLE_DB_SSLMODE` / `POSTGRES_SSLMODE`
- `CANDLE_DB_MIN_POOL_SIZE` / `POSTGRES_MIN_POOL_SIZE`
- `CANDLE_DB_MAX_POOL_SIZE` / `POSTGRES_MAX_POOL_SIZE`
- `CANDLE_DATABASE_URL` or `DATABASE_URL`

## Backtesting

### Main Backtest CLI

Run:

```bash
python -m cmd.backtest.main --help
```

Supported `--strategy` values in this CLI:

- `engulfing`
- `pinbar_magic_v2`

Example (`pinbar_magic_v2`):

```bash
python -m cmd.backtest.main \
  --strategy pinbar_magic_v2 \
  --symbol BTCUSDT \
  --timeframe 1h \
  --start 2025-01-01T00:00:00Z \
  --end 2025-02-01T00:00:00Z \
  --initial-capital 10000 \
  --store-kind postgres
```

### Stochastic FSM Backtest CLI

```bash
python -m cmd.backtest.stochastic_fsm --help
```

This strategy has its own dedicated runner with its own option set.

### Plotting Existing Stats

```bash
python -m cmd.plot.main --help
```

## Candle Downloader

```bash
python -m cmd.candle_downloader.main --help
```

Example:

```bash
python -m cmd.candle_downloader.main \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2025-01-01T00:00:00Z \
  --end 2025-02-01T00:00:00Z \
  --store-kind postgres
```

## Web Backtest Control Panel

Start locally:

```bash
python -m cmd.web.main --local
# or
./scripts/run_web.sh --local
```

Then open `http://127.0.0.1:9092`.

The web app exposes:

- `POST /api/backtests`
- `GET /api/backtests/{job_id}`
- `GET /api/backtests/{job_id}/result`
- `WS /ws/backtests/{job_id}`

## Live Trading

Start from config file:

```bash
cp configs/live_trading.heiken_ashi.env.example configs/live_trading.heiken_ashi.env
cp configs/live_trading.pinbar_magic_v2.env.example configs/live_trading.pinbar_magic_v2.env
# edit values
./scripts/run_live_trading.sh --strategy-name heiken_ashi
# or
./scripts/run_live_trading.sh --strategy-name pinbar_magic_v2
```

Or direct CLI:

```bash
python -m cmd.live_trading.main --help
```

Key live-trading flags:

- `--config-file` (optional; default is strategy-specific `./configs/live_trading.<strategy>.env`)
- `--strategy-name {heiken_ashi,pinbar_magic_v2}`
- `--exchange {weex,binance,bybit,bitunix}`
- `--timeframe ...`

Pin Bar Magic v2 live mode runs with an ETHUSDT-focused coordinator and executes logic per closed candle, plus periodic trailing/position checks between candles.
Risk and behavior parameters remain configurable from CLI/env/config file (EMA/SMA periods, trailing, entry activation, Friday close, etc.).

Config precedence in `cmd.live_trading.main`:

1. CLI args
2. OS environment variables
3. `--config-file` values

## Data Validation Utility

Use this script to detect missing candles and optionally compare DB candles against Binance candle-by-candle:

```bash
python3 scripts/check_missing_candles.py --help
```

Example (Postgres + Binance re-check):

```bash
python3 scripts/check_missing_candles.py \
  --db-kind postgres \
  --pg-host localhost \
  --pg-port 5432 \
  --pg-user postgres \
  --pg-password postgres \
  --pg-db scalp_test \
  --symbol BTCUSDT \
  --timeframe 15m \
  --start-date 2024-11-02T00:00:00Z \
  --end-date 2025-02-01T00:00:00Z \
  --redownload-from-binance
```

Exit code:

- `0`: no missing candles and no mismatches
- `1`: missing candles or mismatches found

## Project Layout

```text
scalp-test/
├── backtest/
├── candle_downloader/
├── cmd/
│   ├── backtest/
│   ├── candle_downloader/
│   ├── live_trading/
│   ├── plot/
│   ├── signal_notifier/
│   └── web/
├── configs/
├── live_trading/
│   ├── strategy_shared.py
│   ├── heiken_ashi_strategy.py
│   ├── pinbar_magic_strategy.py
│   ├── pinbar_magic_coordinator.py
│   ├── position_manager.py
│   ├── coordinator.py
│   └── exchanges/
├── scripts/
├── web/
└── webserver/
```

## Notes

- Candle persistence is PostgreSQL-only.
- Keep API keys and secrets out of git-tracked files.
- `configs/live_trading.heiken_ashi.env` and `configs/live_trading.pinbar_magic_v2.env` are local runtime configs and should be treated as sensitive.
