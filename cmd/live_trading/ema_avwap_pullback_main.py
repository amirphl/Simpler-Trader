"""EMA + AVWAP pullback live-trading entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import _shared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live trading bot for the EMA + AVWAP pullback strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "Optional .env-style config file. If omitted, strategy-specific "
            "default is used (./configs/live_trading.ema_avwap_pullback.env)."
        ),
    )
    parser.add_argument("--exchange", choices=["binance", "bybit", "bitunix"])
    parser.add_argument("--trading-mode", choices=["spot", "futures"], default="futures")
    parser.add_argument("--api-key")
    parser.add_argument("--api-secret")
    parser.add_argument("--api-passphrase")
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
    )
    parser.add_argument(
        "--strategy-name",
        choices=["ema_avwap_pullback"],
        default="ema_avwap_pullback",
    )
    parser.add_argument(
        "--symbols",
        default="ETHUSDT",
        help="Comma-separated symbols for EMA + AVWAP live trading",
    )
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--take-profit-pct", type=float, default=1.0)
    parser.add_argument(
        "--margin-mode", choices=["isolated", "cross"], default="isolated"
    )
    parser.add_argument("--position-size-usdt", type=float, default=100.0)
    parser.add_argument("--max-entry-notional-usdt", type=float, default=15.0)
    parser.add_argument("--max-concurrent-positions", type=int, default=1)
    parser.add_argument("--max-position-size-pct", type=float, default=10.0)
    parser.add_argument("--disable-symbol-hours", type=float, default=0.0)

    parser.add_argument("--equity-risk-pct", type=float, default=1.0)
    parser.add_argument("--ema-length", type=int, default=55)
    parser.add_argument("--consecutive-count", type=int, default=4)
    parser.add_argument(
        "--ema-validation-mode", choices=["body", "wick"], default="body"
    )
    parser.add_argument(
        "--setup-waiting-replacement-mode",
        choices=["keep_waiting", "replace_waiting"],
        default="keep_waiting",
    )
    parser.add_argument(
        "--position-sizing-mode",
        choices=["risk_distance", "risk_amount_per_price"],
        default="risk_distance",
    )
    parser.add_argument("--avwap-multiplier-1", type=float, default=1.0)
    parser.add_argument("--avwap-multiplier-2", type=float, default=2.0)
    parser.add_argument("--avwap-multiplier-3", type=float, default=3.0)
    parser.add_argument("--rigid-stop-loss-pct", type=float, default=0.0)
    parser.add_argument(
        "--trailing-activation-threshold-pct", type=float, default=0.0
    )
    parser.add_argument("--trailing-gap-pct", type=float, default=1.0)
    parser.add_argument("--maker-fee-pct", type=float, default=0.0002)
    parser.add_argument("--taker-fee-pct", type=float, default=0.0006)
    parser.add_argument("--entry-slippage-pct", type=float, default=0.0)
    parser.add_argument("--exit-slippage-pct", type=float, default=0.0)
    parser.add_argument(
        "--use-gap-cross-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--entry-cancel-bars", type=int, default=1)
    parser.add_argument("--max-history-bars", type=int, default=512)
    parser.add_argument("--minimum-balance-usdt", type=float, default=0.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-retry-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--emergency-close-on-stop-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force-close a just-filled position if no protective stop can be confirmed",
    )
    parser.add_argument(
        "--allow-dynamic-stop-widening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow AVWAP band stop updates to widen when the backtest model widens",
    )
    parser.add_argument("--min-stop-update-pct", type=float, default=0.0)

    parser.add_argument(
        "--trailing-tick-timeframe",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        default="1m",
    )
    parser.add_argument(
        "--use-trailing-tick-emulation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--trailing-check-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("./data/ema_avwap_pullback_live_trading_state.json"),
    )
    parser.add_argument(
        "--positions-db",
        type=Path,
        default=Path("./data/ema_avwap_pullback_live_trading_positions.db"),
    )
    parser.add_argument("--klines-db", type=Path, default=None)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("./logs/ema_avwap_pullback_live_trading.log"),
    )
    parser.add_argument("--candle-ready-delay-seconds", type=int, default=30)
    parser.add_argument("--execution-interval-minutes", type=int, default=5)
    parser.add_argument("--exchange-base-url", default="")
    parser.add_argument("--http-proxy")
    parser.add_argument("--https-proxy")
    parser.add_argument("--proxy")
    parser.add_argument(
        "--telegram-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--telegram-token")
    parser.add_argument("--telegram-chat-id")
    parser.add_argument("--telegram-proxy")
    parser.add_argument("--telegram-timeout", type=float, default=10.0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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
        strategy_name="ema_avwap_pullback",
        argv=_force_strategy(effective_argv, "ema_avwap_pullback"),
    )


if __name__ == "__main__":
    sys.exit(main())
