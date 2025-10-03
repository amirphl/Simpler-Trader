# Scalp Test

`scalp-test` is a crypto strategy research and execution workspace covering:

- **Backtesting** (CLI + web control panel)
- **Candle ingestion** (Binance → PostgreSQL)
- **Live trading** coordinators (Binance / Bitunix)
- **Utilities** (position tools, candle validation, plotting)

## Scope

Covered:
- Backtests for `engulfing`, `pinbar_magic_v2`, and `stochastic_fsm`
- Web UI to launch/monitor backtests
- Live trading: `heiken_ashi`, `pinbar_magic_v2`, `strong_trend_stair` (Bitunix/Binance)
- Bitunix helper CLIs for stop/TP/SL management

Not covered:
- Spot trading
- Multi-exchange order routing/hedging
- Production-grade deployment (k8s, systemd) — local/VM expected
- Automatic credential/key management

## Requirements

- Python 3.10+
- PostgreSQL (for candle storage/backtests)
- Node/npm only if you rebuild the web UI assets (prebuilt bundle is checked in)
- mkcert or OpenSSL (optional) if you need local HTTPS

Install deps:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (local)

1) **Candle DB** (PostgreSQL)
```bash
createdb scalp_test
export CANDLE_DB_HOST=localhost CANDLE_DB_USER=postgres CANDLE_DB_PASSWORD=postgres
```
Env keys (either prefix): `CANDLE_DB_*` / `POSTGRES_*` (`HOST,PORT,USER,PASSWORD,DB,SSLMODE,MIN_POOL_SIZE,MAX_POOL_SIZE`, or `CANDLE_DATABASE_URL`).

2) **Download candles (example)**
```bash
python -m cmd.candle_downloader.main \
  --symbol BTCUSDT --interval 15m \
  --start 2025-01-01T00:00:00Z --end 2025-02-01T00:00:00Z \
  --store-kind postgres
```

3) **Backtest (CLI)**
```bash
python -m cmd.backtest.main \
  --strategy pinbar_magic_v2 \
  --symbol BTCUSDT --timeframe 1h \
  --start 2025-01-01T00:00:00Z --end 2025-02-01T00:00:00Z \
  --initial-capital 10000 --store-kind postgres
```

4) **Web backtest panel**
```bash
python -m cmd.web.main --local   # disables HTTPS redirect, binds 127.0.0.1:9092
# or
./scripts/run_web.sh --local
open http://127.0.0.1:9092
```
If your browser forces HTTPS and you want to serve HTTPS locally, start uvicorn with a self-signed cert:
```bash
WEB_FORCE_HTTPS=true uvicorn webserver.app:app \
  --host 0.0.0.0 --port 9092 \
  --ssl-keyfile 127.0.0.1-key.pem --ssl-certfile 127.0.0.1.pem
```

5) **Live trading**
- Copy a config and edit secrets:
  ```bash
  cp configs/live_trading.pinbar_magic_v2.env.example configs/live_trading.pinbar_magic_v2.env
  cp configs/live_trading.strong_trend_stair.env.example configs/live_trading.strong_trend_stair.env
  ```
- Run a coordinator (Bitunix/Binance, depends on config):
  ```bash
  CONFIG_FILE=configs/live_trading.strong_trend_stair.env \
  ./scripts/run_strong_trend_stair_live.sh
  ```
  Logs: `tail -f logs/strong_trend_stair.log`

## Bitunix helper CLIs (scripts/)

- `bitunix_update_stop_loss.py` – update TP/SL order by order id.
- `bitunix_place_position_tpsl.py` – place TP/SL by position id.
- `bitunix_modify_position_tpsl.py` – modify TP/SL by position id.
- `bitunix_list_positions.py` – list open/pending positions.

Run from repo root (fish example):
```bash
env BITUNIX_API_KEY=... BITUNIX_API_SECRET=... \
python3 scripts/bitunix_place_position_tpsl.py --symbol BTCUSDT --position-id 123456 --sl-price 27000 --sl-stop-type MARK_PRICE
```

## Web API surfaces

- `POST /api/backtests`
- `GET /api/backtests/{job_id}`
- `GET /api/backtests/{job_id}/result`
- `WS /ws/backtests/{job_id}`

Trusted hosts & CORS: see `WEB_TRUSTED_HOSTS`, `WEB_ALLOWED_ORIGINS` in `webserver/app.py`. Set `WEB_FORCE_HTTPS=false` (or `--local`) to avoid HTTPS redirects in local dev.

## Project Layout

```text
scalp-test/
├── backtest/             # backtest engines & strategies
├── candle_downloader/    # ingest candles to DB
├── cmd/                  # entrypoints (backtest, live_trading, web, etc.)
├── configs/              # *.env templates for live trading
├── live_trading/         # exchange adapters, coordinators, strategies
├── scripts/              # helper launchers & Bitunix tools
├── web/                  # compiled web UI assets
└── webserver/            # FastAPI app for web panel
```

## Notes & Safety

- Keep API keys/secrets out of git.
- Live trading configs (`configs/live_trading.*.env`) are sensitive.
- Candle store is PostgreSQL-only; no SQLite fallback.
- This repo is aimed at research/live testing; hardening (monitoring, HA, alerting) is out of scope.
