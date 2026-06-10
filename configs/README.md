# Configs Directory

This directory contains runtime configuration files.

## Files

- `backtest.ema_avwap_pullback.env.example`: template for EMA + AVWAP pullback backtests
- `live_trading.ema_avwap_pullback.env.example`: template for EMA + AVWAP pullback live trading
- `live_trading.heiken_ashi.env.example`: template for Heiken Ashi strategy
- `live_trading.pinbar_magic_v3.env.example`: template for PinBar Magic v3 strategy
- `live_trading.strong_trend_stair.env.example`: template for Strong Trend Stair strategy
- `postgres.env.example`: template for web/backtest candle PostgreSQL settings
- `live_trading.heiken_ashi.env`: local runtime config for Heiken Ashi (do not commit secrets)
- `live_trading.ema_avwap_pullback.env`: local runtime config for EMA + AVWAP pullback (do not commit secrets)
- `live_trading.pinbar_magic_v3.env`: local runtime config for PinBar Magic v3 (do not commit secrets)
- `live_trading.strong_trend_stair.env`: local runtime config for Strong Trend Stair (do not commit secrets)
- `postgres.env`: local runtime config for web/backtest candle PostgreSQL settings (do not commit secrets)

## Live Trading Config

Use strategy-specific templates:

```bash
cp configs/live_trading.heiken_ashi.env.example configs/live_trading.heiken_ashi.env
cp configs/live_trading.ema_avwap_pullback.env.example configs/live_trading.ema_avwap_pullback.env
cp configs/live_trading.pinbar_magic_v3.env.example configs/live_trading.pinbar_magic_v3.env
cp configs/live_trading.strong_trend_stair.env.example configs/live_trading.strong_trend_stair.env
```

Then run:

```bash
./scripts/run_live_trading.sh
# or
python -m cmd.live_trading.heiken_ashi_main
python -m cmd.live_trading.ema_avwap_pullback_main
python -m cmd.live_trading.pinbar_magic_v3_main
python -m cmd.live_trading.strong_trend_stair_main
# explicit file override:
python -m cmd.live_trading.pinbar_magic_v3_main --config-file ./configs/live_trading.pinbar_magic_v3.env
```

For multiple EMA + AVWAP symbols without creating one config per symbol:

```bash
scripts/run_ema_avwap_pullback_live_multi.sh \
  --config-file configs/live_trading.ema_avwap_pullback.env \
  --symbols ETHUSDT,BTCUSDT,SOLUSDT \
  --mode blocking
```

`blocking` streams prefixed logs to your terminal. `async` starts each symbol in
the background and writes per-symbol pid/log/state files under `./data` and
`./logs`.

## Config Resolution Order (Live Trading)

In each strategy-specific live trading main module, values are resolved in this order:

1. CLI arguments
2. OS environment variables
3. Values inside strategy config file (`--config-file` if set, otherwise `./configs/live_trading.<strategy>.env`)

## Important Variables

## Backtest Config

Example:

```bash
cp configs/backtest.ema_avwap_pullback.env.example configs/backtest.ema_avwap_pullback.env
./scripts/run_ema_avwap_pullback_backtest.sh
```

### Strategy Selection

- `STRATEGY_NAME=heiken_ashi` or `ema_avwap_pullback` or `pinbar_magic_v3` or `strong_trend_stair`
- `TIMEFRAME=...`

`STRATEGY_NAME` still matters when using `./scripts/run_live_trading.sh`, because the script uses it to choose the correct `cmd.live_trading.*_main` module.

### Pin Bar Magic v3 Variables

- `PINBAR_SYMBOLS=ETHUSDT` (single-symbol live flow; ETH default)
- `EQUITY_RISK_PCT`
- `ATR_MULTIPLE`
- `TRAIL_POINTS` (ticks)
- `TRAIL_OFFSET` (ticks)
- `SYMBOL_MINTICK`
- `SLOW_SMA_PERIOD`
- `MEDIUM_EMA_PERIOD`
- `FAST_EMA_PERIOD`
- `ATR_PERIOD`
- `ENTRY_CANCEL_BARS`
- `ENTRY_ACTIVATION_MODE=next_bar|same_bar`
- `USE_STOP_FILL_OPEN_GAP=true|false`
- `ENABLE_FRIDAY_CLOSE=true|false`
- `FRIDAY_CLOSE_HOUR_UTC`
- `ENABLE_EMA_CROSS_CLOSE=true|false`
- `RISK_EQUITY_INCLUDE_UNREALIZED=true|false`
- `RISK_EQUITY_MARK_SOURCE=close|open|hl2|ohlc4`
- `POLL_INTERVAL_SECONDS`
- `TRAILING_CHECK_INTERVAL_SECONDS`

Note:
- PinBar Magic v3 does not rely on symbol scanning in the live coordinator.
- `TOP_M_SYMBOLS`, `TOP_N_SIGNALS`, and `PRICE_CHANGE_THRESHOLD` are scanner-oriented knobs used by Heiken Ashi flow, not the ETH-only PinBar coordinator path.

### EMA + AVWAP Pullback Variables

- `SYMBOLS=ETHUSDT`
- `EQUITY_RISK_PCT`
- `EMA_LENGTH`
- `CONSECUTIVE_COUNT`
- `EMA_VALIDATION_MODE=body|wick`
- `SETUP_WAITING_REPLACEMENT_MODE=keep_waiting|replace_waiting`
- `POSITION_SIZING_MODE=risk_distance|risk_amount_per_price`
- `AVWAP_MULTIPLIER_1`, `AVWAP_MULTIPLIER_2`, `AVWAP_MULTIPLIER_3`
- `RIGID_STOP_LOSS_PCT`
- `TRAILING_ACTIVATION_THRESHOLD_PCT`
- `TRAILING_GAP_PCT`
- `ENTRY_CANCEL_BARS`
- `MAX_HISTORY_BARS`
- `EMERGENCY_CLOSE_ON_STOP_FAILURE=true|false`
- `ALLOW_DYNAMIC_STOP_WIDENING=true|false`
- `POLL_INTERVAL_SECONDS`
- `TRAILING_CHECK_INTERVAL_SECONDS`

### Exchange / Risk / Scheduling

- `EXCHANGE`, `TRADING_MODE`, `API_KEY`, `API_SECRET`, `API_PASSPHRASE`, `TESTNET`
- `LEVERAGE`, `TAKE_PROFIT_PCT`
- `POSITION_SIZE_USDT`, `MAX_CONCURRENT_POSITIONS`, `MAX_POSITION_SIZE_PCT`
- `MARGIN_MODE`, `DISABLE_SYMBOL_HOURS`
- `CANDLE_READY_DELAY_SECONDS`, `EXECUTION_INTERVAL_MINUTES`
- `STATE_FILE`, `POSITIONS_DB`, `KLINES_DB`, `LOG_FILE`
- `KLINES_DB` is treated as an optional `.env` file for candle PostgreSQL settings, not as a SQLite candle database path.

### Strong Trend Stair Variables

- `SYMBOL` (single symbol)
- `LEVERAGE`, `MARGIN_MODE`, `POSITION_SIZE_USDT`
- `TICK_INTERVAL_SECONDS`
- `HARD_STOP_LOSS_PCT`
- `TRAIL_START_PCT`
- `TRAIL_OFFSET_PCT`
- `REVERSE_ON_OPPOSITE_SIGNAL=true|false`
- `EMA_FAST_LEN`, `EMA_MID_LEN`, `EMA_SLOW_LEN`
- `SLOPE_LOOKBACK`
- `ST_ATR_LEN`, `ST_FACTOR`
- `DI_LEN`, `ADX_SMOOTH`, `ADX_MIN`

### Candle DB (PostgreSQL)

For the web server, create a local config file:

```bash
cp configs/postgres.env.example configs/postgres.env
```

Then run:

```bash
python -m cmd.web.main --local
```

By default, the web command reads `./configs/postgres.env`. To use another file:

```bash
python -m cmd.web.main --local --postgres-config-file ./configs/postgres.local.env
```

Defaults:

- host: `localhost`
- port: `5432`
- user: `postgres`
- password: `postgres`
- db: `scalp_test`

Supported env keys:

- `CANDLE_DB_HOST`, `CANDLE_DB_PORT`, `CANDLE_DB_USER`, `CANDLE_DB_PASSWORD`, `CANDLE_DB_NAME`
- `CANDLE_DB_SSLMODE`, `CANDLE_DB_MIN_POOL_SIZE`, `CANDLE_DB_MAX_POOL_SIZE`
- `CANDLE_DATABASE_URL` (or `DATABASE_URL`)

The same settings can also be provided with `POSTGRES_*` aliases.

### Candle Download Proxy

For the web server, proxy defaults can live in the same `configs/postgres.env` file:

```env
WEB_CANDLE_HTTPS_PROXY=http://127.0.0.1:7890
```

Or set both HTTP and HTTPS with one value:

```env
WEB_CANDLE_PROXY=http://127.0.0.1:7890
```

Values entered in the web UI request still override these defaults.

### Network / Telegram

- `HTTP_PROXY`, `HTTPS_PROXY`, `PROXY`
- `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_PROXY`, `TELEGRAM_TIMEOUT`

## Candle Consistency Check

You can validate missing candles and compare DB values with Binance:

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

## Security

Never commit real credentials in:
- `configs/live_trading.heiken_ashi.env`
- `configs/live_trading.ema_avwap_pullback.env`
- `configs/live_trading.pinbar_magic_v3.env`
- `configs/live_trading.strong_trend_stair.env`
