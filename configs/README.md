# Configs Directory

This directory contains runtime configuration files.

## Files

- `live_trading.heiken_ashi.env.example`: template for Heiken Ashi strategy
- `live_trading.pinbar_magic_v2.env.example`: template for PinBar Magic v2 strategy
- `live_trading.heiken_ashi.env`: local runtime config for Heiken Ashi (do not commit secrets)
- `live_trading.pinbar_magic_v2.env`: local runtime config for PinBar Magic v2 (do not commit secrets)

## Live Trading Config

Use strategy-specific templates:

```bash
cp configs/live_trading.heiken_ashi.env.example configs/live_trading.heiken_ashi.env
cp configs/live_trading.pinbar_magic_v2.env.example configs/live_trading.pinbar_magic_v2.env
```

Then run:

```bash
./scripts/run_live_trading.sh
# or
python -m cmd.live_trading.main --strategy-name heiken_ashi
python -m cmd.live_trading.main --strategy-name pinbar_magic_v2
# explicit file override:
python -m cmd.live_trading.main --strategy-name pinbar_magic_v2 --config-file ./configs/live_trading.pinbar_magic_v2.env
```

## Config Resolution Order (Live Trading)

In `cmd.live_trading.main`, values are resolved in this order:

1. CLI arguments
2. OS environment variables
3. Values inside strategy config file (`--config-file` if set, otherwise `./configs/live_trading.<strategy>.env`)

## Important Variables

### Strategy Selection

- `STRATEGY_NAME=heiken_ashi` or `pinbar_magic_v2`
- `TIMEFRAME=...`

### Pin Bar Magic v2 Variables

- `PINBAR_SYMBOLS=ETHUSDT` (single-symbol live flow; ETH default)
- `EQUITY_RISK_PCT`
- `ATR_MULTIPLE`
- `TRAIL_POINTS`
- `TRAIL_OFFSET`
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

Note:
- PinBar Magic v2 no longer relies on symbol scanning in the live coordinator.
- `TOP_M_SYMBOLS`, `TOP_N_SIGNALS`, and `PRICE_CHANGE_THRESHOLD` are scanner-oriented knobs used by Heiken Ashi flow, not the ETH-only PinBar coordinator path.

### Exchange / Risk / Scheduling

- `EXCHANGE`, `TRADING_MODE`, `API_KEY`, `API_SECRET`, `API_PASSPHRASE`, `TESTNET`
- `LEVERAGE`, `TAKE_PROFIT_PCT`
- `POSITION_SIZE_USDT`, `MAX_CONCURRENT_POSITIONS`, `MAX_POSITION_SIZE_PCT`
- `MARGIN_MODE`, `DISABLE_SYMBOL_HOURS`
- `CANDLE_READY_DELAY_SECONDS`, `EXECUTION_INTERVAL_MINUTES`

### Candle DB (PostgreSQL)

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
- `configs/live_trading.pinbar_magic_v2.env`
