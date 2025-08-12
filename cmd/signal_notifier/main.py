from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from candle_downloader.binance import BinanceClient, BinanceClientConfig
from signal_notifier import SignalNotifier, SignalNotifierSettings, TelegramClient, TelegramConfig
from signal_notifier.engulfing_logic import EngulfingSignalConfig, EngulfingSignalDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor Binance symbols and push live engulfing signals to Telegram.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timeframe", required=True, help="Binance interval, e.g. 1m, 5m, 1h.")
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols to monitor (overrides top-n lookup).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="If --symbols is not provided, automatically monitor the top N Binance USDT pairs.",
    )
    parser.add_argument("--history-limit", type=int, default=100, help="Maximum candles to keep per symbol.")
    parser.add_argument(
        "--epsilon-minutes",
        type=float,
        default=0.1,
        help="Extra minutes to wait after the timeframe before polling Binance again.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("./data/signal_state.json"),
        help="Path used to remember last sent signal timestamps.",
    )
    parser.add_argument("--no-state", action="store_true", help="Disable state tracking (send all signals).")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without sending to Telegram.")

    # Strategy-specific knobs
    parser.add_argument("--window-size", type=int, default=5, help="Number of bearish candles required before the engulfing bar.")
    parser.add_argument("--volume-window", type=int, default=20, help="Candles considered when computing volume pressure.")
    parser.add_argument(
        "--max-volume-score",
        type=float,
        default=3.0,
        help="Skip signals when the engulfing candle looks excessively exhaustive.",
    )

    # Network configuration
    parser.add_argument("--http-proxy", dest="http_proxy", help="HTTP proxy for Binance.")
    parser.add_argument("--https-proxy", dest="https_proxy", help="HTTPS proxy for Binance.")
    parser.add_argument("--proxy", dest="proxy", help="Shortcut to set both HTTP/HTTPS proxies.")

    # Telegram configuration
    parser.add_argument("--telegram-token", help="Telegram bot token (see BotFather).")
    parser.add_argument("--telegram-chat-id", help="Target chat or channel ID.")
    parser.add_argument("--telegram-proxy", help="Proxy URL for Telegram requests (optional).")
    parser.add_argument("--telegram-timeout", type=float, default=10.0, help="Telegram request timeout in seconds.")

    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def load_env_config() -> Dict[str, str]:
    env = os.getenv
    return {
        "timeframe": env("TIMEFRAME", ""),
        "symbols": env("SYMBOLS", ""),
        "top_n": env("TOP_N", ""),
        "history_limit": env("HISTORY_LIMIT", ""),
        "epsilon_minutes": env("EPSILON_MINUTES", ""),
        "state_file": env("SIGNAL_STATE_FILE", ""),
        "telegram_token": env("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": env("TELEGRAM_CHAT_ID", ""),
        "telegram_proxy": env("TELEGRAM_PROXY", ""),
        "window_size": env("STRATEGY_WINDOW_SIZE", ""),
        "volume_window": env("VOLUME_WINDOW", ""),
        "max_volume_score": env("MAX_VOLUME_SCORE", ""),
    }


def apply_env_defaults(args: argparse.Namespace, config: Dict[str, str]) -> argparse.Namespace:
    if config["timeframe"]:
        args.timeframe = config["timeframe"]
    if config["symbols"]:
        args.symbols = config["symbols"]
    if config["top_n"]:
        args.top_n = int(config["top_n"])
    if config["history_limit"]:
        args.history_limit = int(config["history_limit"])
    if config["epsilon_minutes"]:
        args.epsilon_minutes = float(config["epsilon_minutes"])
    if config["telegram_token"]:
        args.telegram_token = config["telegram_token"]
    if config["telegram_chat_id"]:
        args.telegram_chat_id = config["telegram_chat_id"]
    if config["telegram_proxy"]:
        args.telegram_proxy = config["telegram_proxy"]
    if config["state_file"]:
        args.state_file = Path(config["state_file"])
    if config["window_size"]:
        args.window_size = int(config["window_size"])
    if config["volume_window"]:
        args.volume_window = int(config["volume_window"])
    if config["max_volume_score"]:
        args.max_volume_score = float(config["max_volume_score"])
    return args


def build_telegram_client(args: argparse.Namespace, logger: logging.Logger) -> TelegramClient:
    token = args.telegram_token or ""
    chat_id = args.telegram_chat_id or ""
    if not token:
        raise ValueError("Telegram bot token is required (--telegram-token or TELEGRAM_BOT_TOKEN).")
    if not chat_id:
        raise ValueError("Telegram chat ID is required (--telegram-chat-id or TELEGRAM_CHAT_ID).")

    config = TelegramConfig(
        bot_token=token,
        chat_id=chat_id,
        proxy=args.telegram_proxy,
        timeout=args.telegram_timeout,
    )
    return TelegramClient(config, logger=logger)


def resolve_symbols_argument(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [token.strip().upper() for token in value.split(",") if token.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("signal_notifier")

    env_config = load_env_config()
    args = apply_env_defaults(args, env_config)

    if not args.timeframe:
        parser.error("--timeframe is required.")

    telegram_logger = logging.getLogger("signal_notifier.telegram")
    telegram_client = build_telegram_client(args, telegram_logger)

    proxy_map: Dict[str, str] = {}
    if args.proxy:
        proxy_map["http"] = args.proxy
        proxy_map["https"] = args.proxy
    if args.http_proxy:
        proxy_map["http"] = args.http_proxy
    if args.https_proxy:
        proxy_map["https"] = args.https_proxy

    binance_logger = logging.getLogger("signal_notifier.binance")
    binance_client = BinanceClient(BinanceClientConfig(proxies=proxy_map or None), logger=binance_logger)

    detector_config = EngulfingSignalConfig(
        timeframe=args.timeframe,
        window_size=args.window_size,
        volume_window=args.volume_window,
        max_volume_pressure_score=args.max_volume_score,
    )
    detector = EngulfingSignalDetector(detector_config)

    settings = SignalNotifierSettings(
        timeframe=args.timeframe,
        symbols=resolve_symbols_argument(args.symbols),
        top_symbols=args.top_n,
        history_limit=args.history_limit,
        poll_epsilon_minutes=args.epsilon_minutes,
        state_file=None if args.no_state else args.state_file,
        dry_run=args.dry_run,
    )

    notifier = SignalNotifier(
        binance_client=binance_client,
        telegram_client=telegram_client,
        detector=detector,
        settings=settings,
        logger=logger,
    )

    notifier.run()
    binance_client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

