"""CLI for live trading."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from signal_notifier import TelegramClient, TelegramConfig

from live_trading import LiveTradingConfig
from live_trading.coordinator import LiveTradingCoordinator
from live_trading.exchange import ExchangeConfig, MarginMode
from live_trading.pinbar_magic_coordinator import (
    PinBarMagicCoordinatorV2,
    PinBarMagicCoordinatorV2Config,
)

DEFAULT_CONFIG_BY_STRATEGY: Dict[str, Path] = {
    "heiken_ashi": Path("./configs/live_trading.heiken_ashi.env"),
    "pinbar_magic_v2": Path("./configs/live_trading.pinbar_magic_v2.env"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live trading bot with configurable strategy engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "Optional .env-style config file. "
            "If omitted, strategy-specific default is used "
            "(./configs/live_trading.<strategy>.env)."
        ),
    )

    # Exchange settings
    parser.add_argument(
        "--exchange",
        choices=["binance", "bybit", "weex", "bitunix"],
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

    # Trading parameters
    parser.add_argument(
        "--timeframe",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        help="Trading timeframe (must be divisible, e.g., 1h runs at 00:00, 01:00, ...)",
    )
    parser.add_argument(
        "--strategy-name",
        choices=["heiken_ashi", "pinbar_magic_v2"],
        default="pinbar_magic_v2",
        help="Live strategy implementation to execute",
    )
    parser.add_argument(
        "--top-m-symbols",
        type=int,
        default=100,
        help="Number of top symbols to scan by volume",
    )
    parser.add_argument(
        "--top-n-signals",
        type=int,
        default=5,
        help="Number of top movers to consider for trading",
    )
    parser.add_argument(
        "--price-change-threshold",
        type=float,
        default=2.0,
        help="Minimum price change percentage",
    )

    # Signal parameters
    parser.add_argument(
        "--heiken-ashi-candles",
        type=int,
        default=3,
        help="Number of HA candles before reversal",
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

    # Pin Bar Magic V2 parameters
    parser.add_argument(
        "--pinbar-symbols",
        default="",
        help="Comma-separated symbols for PinBar strategy; empty means scanner-based selection.",
    )
    parser.add_argument(
        "--equity-risk-pct",
        type=float,
        default=3.0,
        help="Risk per trade as %% of equity",
    )
    parser.add_argument(
        "--atr-multiple", type=float, default=0.5, help="ATR multiplier for risk stop"
    )
    parser.add_argument(
        "--trail-points", type=float, default=1.0, help="Trailing activation distance"
    )
    parser.add_argument(
        "--trail-offset", type=float, default=1.0, help="Trailing stop offset"
    )
    parser.add_argument(
        "--slow-sma-period", type=int, default=50, help="Slow SMA period"
    )
    parser.add_argument(
        "--medium-ema-period", type=int, default=18, help="Medium EMA period"
    )
    parser.add_argument(
        "--fast-ema-period", type=int, default=6, help="Fast EMA period"
    )
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period")
    parser.add_argument(
        "--entry-cancel-bars", type=int, default=3, help="Pending entry timeout in bars"
    )
    parser.add_argument(
        "--entry-activation-mode",
        choices=["next_bar", "same_bar"],
        default="next_bar",
        help="Pending stop-entry activation timing",
    )
    parser.add_argument(
        "--trailing-tick-timeframe",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        default="15m",
        help="Tick emulation timeframe for trailing model",
    )
    parser.add_argument(
        "--use-trailing-tick-emulation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable close-based tick emulation for trailing logic",
    )
    parser.add_argument(
        "--use-stop-fill-open-gap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow stop entries to fill at candle open on gap-through",
    )
    parser.add_argument(
        "--enable-friday-close",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force close positions on configured Friday hour",
    )
    parser.add_argument(
        "--friday-close-hour-utc",
        type=int,
        default=16,
        help="UTC hour for Friday forced close",
    )
    parser.add_argument(
        "--enable-ema-cross-close",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Close on fast/medium EMA cross",
    )
    parser.add_argument(
        "--risk-equity-include-unrealized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include unrealized PnL in risk equity",
    )
    parser.add_argument(
        "--risk-equity-mark-source",
        choices=["close", "open", "hl2", "ohlc4"],
        default="close",
        help="Mark price source for unrealized-risk equity calculations",
    )

    # Position management
    parser.add_argument(
        "--margin-mode",
        choices=["isolated", "cross"],
        default="isolated",
        help="Margin mode for positions",
    )
    parser.add_argument(
        "--disable-symbol-hours",
        type=int,
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

    # Data persistence
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

    # Scheduling
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

    # Network
    parser.add_argument("--http-proxy", help="HTTP proxy")
    parser.add_argument("--https-proxy", help="HTTPS proxy")
    parser.add_argument("--proxy", help="Proxy for both HTTP and HTTPS")

    # Notifications
    parser.add_argument(
        "--telegram-enabled",
        action="store_true",
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

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser


def _load_env_file(path: Optional[Path]) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a .env-style file."""
    if path is None or not path.exists():
        return {}

    values: Dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].rstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        # Ignore malformed lines/file read issues and fall back to environment.
        return {}
    return values


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes", "on")


def _parse_symbols_csv(value: str) -> tuple[str, ...]:
    if not value.strip():
        return tuple()
    symbols = [part.strip().upper() for part in value.split(",")]
    return tuple(symbol for symbol in symbols if symbol)


def resolve_config_file(args: argparse.Namespace) -> Path:
    """Resolve config file path from CLI or strategy-specific default."""
    if args.config_file is not None:
        return args.config_file
    strategy = str(getattr(args, "strategy_name", "pinbar_magic_v2")).strip().lower()
    if strategy in DEFAULT_CONFIG_BY_STRATEGY:
        return DEFAULT_CONFIG_BY_STRATEGY[strategy]
    return DEFAULT_CONFIG_BY_STRATEGY["pinbar_magic_v2"]


def load_env_config(config_file: Optional[Path] = None) -> Dict[str, str]:
    """Load configuration from config file and environment variables.

    Precedence: OS environment > config file > empty.
    """
    file_values = _load_env_file(config_file)
    env = os.getenv

    def read_env(*keys: str) -> str:
        for key in keys:
            value = env(key, "")
            if value != "":
                return value
            file_value = file_values.get(key, "")
            if file_value != "":
                return file_value
        return ""

    return {
        "exchange": read_env("EXCHANGE"),
        "trading_mode": read_env("TRADING_MODE"),
        "api_key": read_env("API_KEY"),
        "api_secret": read_env("API_SECRET"),
        "api_passphrase": read_env("API_PASSPHRASE", "PASS_PHRASE"),
        "testnet": read_env("TESTNET"),
        "timeframe": read_env("TIMEFRAME"),
        "strategy_name": read_env("STRATEGY_NAME", "LIVE_STRATEGY_NAME"),
        "pinbar_symbols": read_env("PINBAR_SYMBOLS"),
        "top_m_symbols": read_env("TOP_M_SYMBOLS"),
        "top_n_signals": read_env("TOP_N_SIGNALS"),
        "price_change_threshold": read_env("PRICE_CHANGE_THRESHOLD"),
        "heiken_ashi_candles": read_env("HEIKEN_ASHI_CANDLES"),
        "leverage": read_env("LEVERAGE"),
        "take_profit_pct": read_env("TAKE_PROFIT_PCT"),
        "equity_risk_pct": read_env("EQUITY_RISK_PCT", "STRATEGY_EQUITY_RISK_PCT"),
        "atr_multiple": read_env("ATR_MULTIPLE", "STRATEGY_ATR_MULTIPLE"),
        "trail_points": read_env("TRAIL_POINTS", "STRATEGY_TRAIL_POINTS"),
        "trail_offset": read_env("TRAIL_OFFSET", "STRATEGY_TRAIL_OFFSET"),
        "slow_sma_period": read_env("SLOW_SMA_PERIOD", "STRATEGY_SLOW_SMA_PERIOD"),
        "medium_ema_period": read_env(
            "MEDIUM_EMA_PERIOD", "STRATEGY_MEDIUM_EMA_PERIOD"
        ),
        "fast_ema_period": read_env("FAST_EMA_PERIOD", "STRATEGY_FAST_EMA_PERIOD"),
        "atr_period": read_env("ATR_PERIOD", "STRATEGY_ATR_PERIOD"),
        "entry_cancel_bars": read_env(
            "ENTRY_CANCEL_BARS", "STRATEGY_ENTRY_CANCEL_BARS"
        ),
        "entry_activation_mode": read_env(
            "ENTRY_ACTIVATION_MODE", "STRATEGY_ENTRY_ACTIVATION_MODE"
        ),
        "trailing_tick_timeframe": read_env(
            "TRAILING_TICK_TIMEFRAME", "STRATEGY_TRAILING_TICK_TIMEFRAME"
        ),
        "use_trailing_tick_emulation": read_env(
            "USE_TRAILING_TICK_EMULATION", "STRATEGY_USE_TRAILING_TICK_EMULATION"
        ),
        "use_stop_fill_open_gap": read_env(
            "USE_STOP_FILL_OPEN_GAP", "STRATEGY_USE_STOP_FILL_OPEN_GAP"
        ),
        "enable_friday_close": read_env(
            "ENABLE_FRIDAY_CLOSE", "STRATEGY_ENABLE_FRIDAY_CLOSE"
        ),
        "friday_close_hour_utc": read_env(
            "FRIDAY_CLOSE_HOUR_UTC", "STRATEGY_FRIDAY_CLOSE_HOUR_UTC"
        ),
        "enable_ema_cross_close": read_env(
            "ENABLE_EMA_CROSS_CLOSE", "STRATEGY_ENABLE_EMA_CROSS_CLOSE"
        ),
        "risk_equity_include_unrealized": read_env(
            "RISK_EQUITY_INCLUDE_UNREALIZED",
            "STRATEGY_RISK_EQUITY_INCLUDE_UNREALIZED",
        ),
        "risk_equity_mark_source": read_env(
            "RISK_EQUITY_MARK_SOURCE", "STRATEGY_RISK_EQUITY_MARK_SOURCE"
        ),
        "margin_mode": read_env("MARGIN_MODE"),
        "disable_symbol_hours": read_env("DISABLE_SYMBOL_HOURS"),
        "position_size_usdt": read_env("POSITION_SIZE_USDT"),
        "max_concurrent_positions": read_env("MAX_CONCURRENT_POSITIONS"),
        "max_position_size_pct": read_env("MAX_POSITION_SIZE_PCT"),
        "state_file": read_env("STATE_FILE"),
        "positions_db": read_env("POSITIONS_DB"),
        "klines_db": read_env("KLINES_DB"),
        "log_file": read_env("LOG_FILE"),
        "candle_ready_delay_seconds": read_env("CANDLE_READY_DELAY_SECONDS"),
        "execution_interval_minutes": read_env("EXECUTION_INTERVAL_MINUTES"),
        "exchange_base_url": read_env("EXCHANGE_BASE_URL"),
        "http_proxy": read_env("HTTP_PROXY"),
        "https_proxy": read_env("HTTPS_PROXY"),
        "proxy": read_env("PROXY"),
        "telegram_enabled": read_env("TELEGRAM_ENABLED"),
        "telegram_token": read_env("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": read_env("TELEGRAM_CHAT_ID"),
        "telegram_proxy": read_env("TELEGRAM_PROXY"),
        "telegram_timeout": read_env("TELEGRAM_TIMEOUT"),
        "log_level": read_env("LOG_LEVEL"),
    }


def apply_env_defaults(
    args: argparse.Namespace, config: Dict[str, str]
) -> argparse.Namespace:
    """Apply environment variable defaults to arguments."""
    if config["exchange"]:
        args.exchange = config["exchange"]
    if config["trading_mode"]:
        args.trading_mode = config["trading_mode"]
    if config["api_key"]:
        args.api_key = config["api_key"]
    if config["api_secret"]:
        args.api_secret = config["api_secret"]
    if config["api_passphrase"]:
        args.api_passphrase = config["api_passphrase"]
    if config["testnet"]:
        args.testnet = config["testnet"].lower() in ("true", "1", "yes")
    if config["timeframe"]:
        args.timeframe = config["timeframe"]
    if config["strategy_name"]:
        args.strategy_name = config["strategy_name"]
    if config["pinbar_symbols"]:
        args.pinbar_symbols = config["pinbar_symbols"]
    if config["top_m_symbols"]:
        args.top_m_symbols = int(config["top_m_symbols"])
    if config["top_n_signals"]:
        args.top_n_signals = int(config["top_n_signals"])
    if config["price_change_threshold"]:
        args.price_change_threshold = float(config["price_change_threshold"])
    if config["heiken_ashi_candles"]:
        args.heiken_ashi_candles = int(config["heiken_ashi_candles"])
    if config["leverage"]:
        args.leverage = int(config["leverage"])
    if config["take_profit_pct"]:
        args.take_profit_pct = float(config["take_profit_pct"])
    if config["equity_risk_pct"]:
        args.equity_risk_pct = float(config["equity_risk_pct"])
    if config["atr_multiple"]:
        args.atr_multiple = float(config["atr_multiple"])
    if config["trail_points"]:
        args.trail_points = float(config["trail_points"])
    if config["trail_offset"]:
        args.trail_offset = float(config["trail_offset"])
    if config["slow_sma_period"]:
        args.slow_sma_period = int(config["slow_sma_period"])
    if config["medium_ema_period"]:
        args.medium_ema_period = int(config["medium_ema_period"])
    if config["fast_ema_period"]:
        args.fast_ema_period = int(config["fast_ema_period"])
    if config["atr_period"]:
        args.atr_period = int(config["atr_period"])
    if config["entry_cancel_bars"]:
        args.entry_cancel_bars = int(config["entry_cancel_bars"])
    if config["entry_activation_mode"]:
        args.entry_activation_mode = config["entry_activation_mode"]
    if config["trailing_tick_timeframe"]:
        args.trailing_tick_timeframe = config["trailing_tick_timeframe"]
    if config["use_trailing_tick_emulation"]:
        args.use_trailing_tick_emulation = _parse_bool(
            config["use_trailing_tick_emulation"]
        )
    if config["use_stop_fill_open_gap"]:
        args.use_stop_fill_open_gap = _parse_bool(config["use_stop_fill_open_gap"])
    if config["enable_friday_close"]:
        args.enable_friday_close = _parse_bool(config["enable_friday_close"])
    if config["friday_close_hour_utc"]:
        args.friday_close_hour_utc = int(config["friday_close_hour_utc"])
    if config["enable_ema_cross_close"]:
        args.enable_ema_cross_close = _parse_bool(config["enable_ema_cross_close"])
    if config["risk_equity_include_unrealized"]:
        args.risk_equity_include_unrealized = _parse_bool(
            config["risk_equity_include_unrealized"]
        )
    if config["risk_equity_mark_source"]:
        args.risk_equity_mark_source = config["risk_equity_mark_source"]
    if config["margin_mode"]:
        args.margin_mode = config["margin_mode"]
    if config["disable_symbol_hours"]:
        args.disable_symbol_hours = float(config["disable_symbol_hours"])
    if config["position_size_usdt"]:
        args.position_size_usdt = float(config["position_size_usdt"])
    if config["max_concurrent_positions"]:
        args.max_concurrent_positions = int(config["max_concurrent_positions"])
    if config["max_position_size_pct"]:
        args.max_position_size_pct = float(config["max_position_size_pct"])
    if config["state_file"]:
        args.state_file = Path(config["state_file"])
    if config["positions_db"]:
        args.positions_db = Path(config["positions_db"])
    if config["klines_db"]:
        args.klines_db = Path(config["klines_db"])
    if config["log_file"]:
        args.log_file = Path(config["log_file"])
    if config["candle_ready_delay_seconds"]:
        args.candle_ready_delay_seconds = int(config["candle_ready_delay_seconds"])
    if config["execution_interval_minutes"]:
        args.execution_interval_minutes = int(config["execution_interval_minutes"])
    if config["exchange_base_url"]:
        args.exchange_base_url = config["exchange_base_url"]
    if config["http_proxy"]:
        args.http_proxy = config["http_proxy"]
    if config["https_proxy"]:
        args.https_proxy = config["https_proxy"]
    if config["proxy"]:
        args.proxy = config["proxy"]
    if config["telegram_enabled"]:
        args.telegram_enabled = _parse_bool(config["telegram_enabled"])
    if config["telegram_token"]:
        args.telegram_token = config["telegram_token"]
    if config["telegram_chat_id"]:
        args.telegram_chat_id = config["telegram_chat_id"]
    if config["telegram_proxy"]:
        args.telegram_proxy = config["telegram_proxy"]
    if config["telegram_timeout"]:
        args.telegram_timeout = float(config["telegram_timeout"])
    if config["log_level"]:
        args.log_level = config["log_level"]

    return args


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """Setup logging configuration."""
    # Ensure log directory exists
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("live_trading")


def create_exchange(args: argparse.Namespace, logger: logging.Logger):
    """Create exchange client based on configuration.

    Note: This will need to be implemented for specific exchanges.
    For now, it raises an error to indicate implementation is needed.
    """
    api_key = args.api_key or ""
    api_secret = args.api_secret or ""
    api_passphrase = args.api_passphrase or ""

    if not api_key:
        raise ValueError("API key is required (--api-key or API_KEY env var)")
    if not api_secret:
        raise ValueError("API secret is required (--api-secret or API_SECRET env var)")

    # Determine testnet mode
    testnet = args.testnet and not args.live

    # Build proxy configuration
    proxies: Optional[Dict[str, str]] = None
    if args.proxy:
        proxies = {"http": args.proxy, "https": args.proxy}
    elif args.http_proxy or args.https_proxy:
        proxies = {}
        if args.http_proxy:
            proxies["http"] = args.http_proxy
        if args.https_proxy:
            proxies["https"] = args.https_proxy

    exchange_config = ExchangeConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        proxies=proxies,
        passphrase=api_passphrase or None,
        base_url=args.exchange_base_url or None,
    )

    # Import and instantiate exchange client
    if args.exchange == "weex":
        from live_trading.exchanges import WeexExchange, WeexTradingMode

        trading_mode = (
            WeexTradingMode.SPOT
            if args.trading_mode == "spot"
            else WeexTradingMode.FUTURES
        )

        logger.info(f"Using Weex exchange in {trading_mode.value} mode")
        return WeexExchange(exchange_config, trading_mode, logger)

    elif args.exchange == "binance":
        logger.warning(
            "Binance exchange implementation not yet available. "
            "Please implement BinanceExchange class in live_trading/exchanges/binance.py"
        )
        raise NotImplementedError("Binance exchange not yet implemented")
    elif args.exchange == "bybit":
        logger.warning(
            "Bybit exchange implementation not yet available. "
            "Please implement BybitExchange class in live_trading/exchanges/bybit.py"
        )
        raise NotImplementedError("Bybit exchange not yet implemented")
    elif args.exchange == "bitunix":
        from live_trading.exchanges import BitunixExchange

        logger.info("Using Bitunix exchange")
        return BitunixExchange(exchange_config, logger)
    else:
        raise ValueError(f"Unknown exchange: {args.exchange}")


def create_telegram_client(
    config: LiveTradingConfig, logger: logging.Logger
) -> Optional[TelegramClient]:
    """Create Telegram client if notifications are enabled."""
    if not config.telegram_enabled:
        return None

    logger.info("Initializing Telegram client for chat %s", config.telegram_chat_id)
    telegram_logger = logging.getLogger("live_trading.telegram")
    telegram_config = TelegramConfig(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        proxy=config.telegram_proxy,
        timeout=config.telegram_timeout_seconds,
    )
    telegram_logger.info(
        "Telegram notifications enabled for chat %s", config.telegram_chat_id
    )
    return TelegramClient(telegram_config, telegram_logger)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = build_parser()
    # Parse once to discover strategy/config-file, then re-parse with env/file defaults
    pre_args, _ = parser.parse_known_args(argv)
    resolved_config_file = resolve_config_file(pre_args)
    env_config = load_env_config(resolved_config_file)
    env_defaults = apply_env_defaults(parser.parse_args([]), env_config)
    parser.set_defaults(**vars(env_defaults))
    args = parser.parse_args(argv)
    args.config_file = resolved_config_file
    if args.klines_db is None:
        args.klines_db = resolved_config_file

    # Setup logging
    logger = setup_logging(args)
    if args.config_file.exists():
        logger.info("Loaded config file: %s", args.config_file)
    else:
        logger.info("Config file not found (using env/CLI only): %s", args.config_file)

    # Validate required arguments
    if not args.exchange:
        parser.error("--exchange is required")
    if not args.timeframe:
        parser.error("--timeframe is required")

    # Warn about live trading
    if args.live:
        logger.warning("=" * 80)
        logger.warning("LIVE TRADING MODE ENABLED - REAL MONEY WILL BE TRADED")
        logger.warning("=" * 80)
        response = input("Are you sure you want to continue? Type 'YES' to confirm: ")
        if response != "YES":
            logger.info("Live trading cancelled")
            return 0

    try:
        # Create exchange client
        logger.info(f"Initializing {args.exchange} exchange client")
        exchange = create_exchange(args, logger)

        # Create live trading configuration
        margin_mode = (
            MarginMode.ISOLATED if args.margin_mode == "isolated" else MarginMode.CROSS
        )

        config = LiveTradingConfig(
            exchange_name=args.exchange,
            api_key=args.api_key or "",
            api_secret=args.api_secret or "",
            testnet=args.testnet and not args.live,
            strategy_name=args.strategy_name,
            timeframe=args.timeframe,
            top_m_symbols=args.top_m_symbols,
            top_n_signals=args.top_n_signals,
            price_change_threshold_pct=args.price_change_threshold,
            pinbar_symbols=_parse_symbols_csv(args.pinbar_symbols),
            heiken_ashi_candles_before=args.heiken_ashi_candles,
            leverage=args.leverage,
            take_profit_pct=args.take_profit_pct,
            equity_risk_pct=args.equity_risk_pct,
            atr_multiple=args.atr_multiple,
            trail_points=args.trail_points,
            trail_offset=args.trail_offset,
            slow_sma_period=args.slow_sma_period,
            medium_ema_period=args.medium_ema_period,
            fast_ema_period=args.fast_ema_period,
            atr_period=args.atr_period,
            entry_cancel_bars=args.entry_cancel_bars,
            entry_activation_mode=args.entry_activation_mode,
            trailing_tick_timeframe=args.trailing_tick_timeframe,
            use_trailing_tick_emulation=args.use_trailing_tick_emulation,
            use_stop_fill_open_gap=args.use_stop_fill_open_gap,
            enable_friday_close=args.enable_friday_close,
            friday_close_hour_utc=args.friday_close_hour_utc,
            enable_ema_cross_close=args.enable_ema_cross_close,
            risk_equity_include_unrealized=args.risk_equity_include_unrealized,
            risk_equity_mark_source=args.risk_equity_mark_source,
            margin_mode=margin_mode,
            disable_symbol_hours=args.disable_symbol_hours,
            position_size_usdt=args.position_size_usdt,
            max_concurrent_positions=args.max_concurrent_positions,
            max_position_size_pct=args.max_position_size_pct,
            state_file=args.state_file,
            positions_db=args.positions_db,
            klines_db=args.klines_db,
            log_file=args.log_file,
            candle_ready_delay_seconds=args.candle_ready_delay_seconds,
            execution_interval_minutes=args.execution_interval_minutes,
            telegram_enabled=args.telegram_enabled,
            telegram_bot_token=args.telegram_token or "",
            telegram_chat_id=args.telegram_chat_id or "",
            telegram_proxy=args.telegram_proxy or None,
            telegram_timeout_seconds=args.telegram_timeout,
        )

        logger.info("Live trading configuration:")
        logger.info(
            f"  Exchange: {config.exchange_name} ({'testnet' if config.testnet else 'LIVE'})"
        )
        logger.info(f"  Strategy: {config.strategy_name}")
        logger.info(f"  Timeframe: {config.timeframe}")
        logger.info(f"  Top M symbols: {config.top_m_symbols}")
        logger.info(f"  Top N signals: {config.top_n_signals}")
        logger.info(f"  Price change threshold: {config.price_change_threshold_pct}%")
        logger.info(f"  HA candles before: {config.heiken_ashi_candles_before}")
        logger.info(f"  Leverage: {config.leverage}x")
        logger.info(f"  Take profit: {config.take_profit_pct}%")
        logger.info(f"  Margin mode: {config.margin_mode.value}")
        logger.info(f"  Disable symbol hours: {config.disable_symbol_hours}")
        logger.info(f"  Position size: {config.position_size_usdt} USDT")
        logger.info(f"  Max concurrent positions: {config.max_concurrent_positions}")
        logger.info(
            f"  Candle ready delay: {config.candle_ready_delay_seconds} seconds"
        )
        logger.info(
            f"  Execution interval: {config.execution_interval_minutes} minutes"
        )
        if config.telegram_enabled:
            logger.info(
                "  Telegram notifications: enabled (chat %s)", config.telegram_chat_id
            )
        else:
            logger.info("  Telegram notifications: disabled")

        telegram_client = create_telegram_client(config, logger)

        # Create and run coordinator
        if config.strategy_name == "pinbar_magic_v2":
            pinbar_cfg = PinBarMagicCoordinatorV2Config(
                symbols=config.pinbar_symbols
                if config.pinbar_symbols
                else ("ETHUSDT",),
                timeframe=config.timeframe,
                top_m_symbols=config.top_m_symbols,
                poll_interval_seconds=5.0,
                trailing_check_interval_seconds=5.0,
                leverage=config.leverage,
                margin_mode=config.margin_mode,
                max_concurrent_positions=config.max_concurrent_positions,
                equity_risk_pct=config.equity_risk_pct,
                atr_multiple=config.atr_multiple,
                trail_points=config.trail_points,
                trail_offset=config.trail_offset,
                slow_sma_period=config.slow_sma_period,
                medium_ema_period=config.medium_ema_period,
                fast_ema_period=config.fast_ema_period,
                atr_period=config.atr_period,
                entry_cancel_bars=config.entry_cancel_bars,
                entry_activation_mode=config.entry_activation_mode,
                enable_friday_close=config.enable_friday_close,
                friday_close_hour_utc=config.friday_close_hour_utc,
                enable_ema_cross_close=config.enable_ema_cross_close,
                risk_equity_include_unrealized=config.risk_equity_include_unrealized,
                risk_equity_mark_source=config.risk_equity_mark_source,
                use_stop_fill_open_gap=config.use_stop_fill_open_gap,
                disable_symbol_hours=config.disable_symbol_hours,
            )
            coordinator = PinBarMagicCoordinatorV2(
                exchange=exchange,
                config=pinbar_cfg,
                logger=logger,
            )
            coordinator.run_forever()
        else:
            coordinator = LiveTradingCoordinator(
                config, exchange, telegram_client=telegram_client, logger=logger
            )
            coordinator.run()

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
