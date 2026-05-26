"""Strong trend stair trailing live-trading entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import _shared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live trading bot for the strong trend stair strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "Optional .env-style config file. "
            "If omitted, strategy-specific default is used "
            "(./configs/live_trading.strong_trend_stair.env)."
        ),
    )
    parser.add_argument(
        "--exchange",
        choices=["binance", "bybit", "bitunix"],
        help="Exchange to use for trading",
    )
    parser.add_argument(
        "--trading-mode",
        choices=["spot", "futures"],
        default="futures",
        help="Trading mode (for exchanges that support both)",
    )
    parser.add_argument("--api-key", help="Exchange API key")
    parser.add_argument("--api-secret", help="Exchange API secret")
    parser.add_argument(
        "--api-passphrase", help="Exchange API passphrase (if required)"
    )
    parser.add_argument(
        "--testnet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use testnet for testing",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading mode (WARNING: trades real money)",
    )
    parser.add_argument(
        "--timeframe",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        help="Trading timeframe (must be divisible, e.g., 1h runs at 00:00, 01:00, ...)",
    )
    parser.add_argument(
        "--strategy-name",
        choices=["strong_trend_stair"],
        default="strong_trend_stair",
        help="Live strategy implementation to execute",
    )
    parser.add_argument(
        "--symbol",
        default="ETHUSDT",
        help="Single symbol for strong_trend_stair strategy.",
    )
    parser.add_argument(
        "--tick-interval-seconds",
        type=float,
        default=1.0,
        help="Tick loop interval for strong_trend_stair strategy.",
    )
    parser.add_argument(
        "--hard-stop-loss-pct",
        type=float,
        default=5.0,
        help="Hard stop loss percent on price move for strong_trend_stair strategy.",
    )
    parser.add_argument(
        "--trail-start-pct",
        type=float,
        default=2.0,
        help="Profit threshold percent where stair trailing starts.",
    )
    parser.add_argument(
        "--trail-offset-pct",
        type=float,
        default=1.0,
        help="Trailing offset percent from favorable move.",
    )
    parser.add_argument(
        "--ema-fast-len",
        type=int,
        default=50,
        help="Fast EMA length for trend filter.",
    )
    parser.add_argument(
        "--ema-mid-len",
        type=int,
        default=100,
        help="Mid EMA length for trend filter.",
    )
    parser.add_argument(
        "--ema-slow-len",
        type=int,
        default=200,
        help="Slow EMA length for trend filter.",
    )
    parser.add_argument(
        "--slope-lookback",
        type=int,
        default=10,
        help="Lookback candles for slow EMA slope check.",
    )
    parser.add_argument(
        "--st-atr-len",
        type=int,
        default=10,
        help="Supertrend ATR length.",
    )
    parser.add_argument(
        "--st-factor",
        type=float,
        default=3.0,
        help="Supertrend factor.",
    )
    parser.add_argument(
        "--di-len",
        type=int,
        default=14,
        help="DI length.",
    )
    parser.add_argument(
        "--adx-smooth",
        type=int,
        default=14,
        help="ADX smoothing length.",
    )
    parser.add_argument(
        "--adx-min",
        type=float,
        default=20.0,
        help="Minimum ADX threshold for trend qualification.",
    )
    parser.add_argument(
        "--reverse-on-opposite-signal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable immediate reversal when opposite trend signal appears.",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=10,
        help="Leverage multiplier",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=1.0,
        help="Take profit percentage (default: 1 percent)",
    )
    parser.add_argument(
        "--margin-mode",
        choices=["isolated", "cross"],
        default="isolated",
        help="Margin mode for positions",
    )
    parser.add_argument(
        "--disable-symbol-hours",
        type=float,
        default=24,
        help="Hours to disable symbol after trade (0 = no disable)",
    )
    parser.add_argument(
        "--position-size-usdt",
        type=float,
        default=100.0,
        help="Position size in USDT",
    )
    parser.add_argument(
        "--max-entry-notional-usdt",
        type=float,
        default=15.0,
        help="Maximum entry notional in USDT before leverage is applied",
    )
    parser.add_argument(
        "--max-concurrent-positions",
        type=int,
        default=5,
        help="Maximum number of concurrent positions",
    )
    parser.add_argument(
        "--max-position-size-pct",
        type=float,
        default=10.0,
        help="Maximum position size as fraction of balance",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("./data/live_trading_state.json"),
        help="Path for state persistence",
    )
    parser.add_argument(
        "--positions-db",
        type=Path,
        default=Path("./data/live_trading_positions.db"),
        help="Path for positions database",
    )
    parser.add_argument(
        "--klines-db",
        type=Path,
        default=None,
        help=(
            "Optional .env file path for candle Postgres settings "
            "(CANDLE_DB_* / POSTGRES_* vars). "
            "Defaults to resolved strategy config file."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("./logs/live_trading.log"),
        help="Path for log file",
    )
    parser.add_argument(
        "--candle-ready-delay-seconds",
        type=int,
        default=30,
        help="Seconds to wait after timeframe close before running strategy",
    )
    parser.add_argument(
        "--exchange-base-url",
        default="",
        help="Optional override for exchange API base URL (used for bitunix)",
    )
    parser.add_argument(
        "--execution-interval-minutes",
        type=int,
        default=5,
        help="How often to run the strategy loop",
    )
    parser.add_argument("--http-proxy", help="HTTP proxy")
    parser.add_argument("--https-proxy", help="HTTPS proxy")
    parser.add_argument("--proxy", help="Proxy for both HTTP and HTTPS")
    parser.add_argument(
        "--telegram-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Telegram notifications for executed trades",
    )
    parser.add_argument("--telegram-token", help="Telegram bot token")
    parser.add_argument("--telegram-chat-id", help="Telegram chat or channel ID")
    parser.add_argument("--telegram-proxy", help="Proxy URL for Telegram requests")
    parser.add_argument(
        "--telegram-timeout",
        type=float,
        default=10.0,
        help="Timeout for Telegram requests in seconds",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def _force_strategy(argv: Optional[List[str]], strategy: str) -> List[str]:
    args = list(argv or [])
    out: List[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == "--strategy-name":
            skip_next = True
            continue
        if token.startswith("--strategy-name="):
            continue
        out.append(token)
    out.extend(["--strategy-name", strategy])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    return _shared.run_main(
        parser=build_parser(),
        strategy_name="strong_trend_stair",
        argv=_force_strategy(effective_argv, "strong_trend_stair"),
    )


if __name__ == "__main__":
    sys.exit(main())
