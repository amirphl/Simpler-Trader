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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live trading bot with Heiken Ashi reversal strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Exchange settings
    parser.add_argument(
        "--exchange",
        choices=["binance", "bybit", "weex"],
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
    parser.add_argument("--api-passphrase", help="Exchange API passphrase (if required)")
    parser.add_argument(
        "--testnet",
        action="store_true",
        default=True,
        help="Use testnet for testing (default: True)",
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


def load_env_config() -> Dict[str, str]:
    """Load configuration from environment variables."""
    env = os.getenv
    return {
        "exchange": env("EXCHANGE", ""),
        "trading_mode": env("TRADING_MODE", ""),
        "api_key": env("API_KEY", ""),
        "api_secret": env("API_SECRET", ""),
        "api_passphrase": env("API_PASSPHRASE", "") or env("PASS_PHRASE", ""),
        "testnet": env("TESTNET", ""),
        "timeframe": env("TIMEFRAME", ""),
        "top_m_symbols": env("TOP_M_SYMBOLS", ""),
        "top_n_signals": env("TOP_N_SIGNALS", ""),
        "price_change_threshold": env("PRICE_CHANGE_THRESHOLD", ""),
        "heiken_ashi_candles": env("HEIKEN_ASHI_CANDLES", ""),
        "leverage": env("LEVERAGE", ""),
        "take_profit_pct": env("TAKE_PROFIT_PCT", ""),
        "margin_mode": env("MARGIN_MODE", ""),
        "disable_symbol_hours": env("DISABLE_SYMBOL_HOURS", ""),
        "position_size_usdt": env("POSITION_SIZE_USDT", ""),
        "max_concurrent_positions": env("MAX_CONCURRENT_POSITIONS", ""),
        "max_position_size_pct": env("MAX_POSITION_SIZE_PCT", ""),
        "state_file": env("STATE_FILE", ""),
        "positions_db": env("POSITIONS_DB", ""),
        "log_file": env("LOG_FILE", ""),
        "candle_ready_delay_seconds": env("CANDLE_READY_DELAY_SECONDS", ""),
        "http_proxy": env("HTTP_PROXY", ""),
        "https_proxy": env("HTTPS_PROXY", ""),
        "telegram_enabled": env("TELEGRAM_ENABLED", ""),
        "telegram_token": env("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": env("TELEGRAM_CHAT_ID", ""),
        "telegram_proxy": env("TELEGRAM_PROXY", ""),
        "telegram_timeout": env("TELEGRAM_TIMEOUT", ""),
        "log_level": env("LOG_LEVEL", ""),
    }


def apply_env_defaults(args: argparse.Namespace, config: Dict[str, str]) -> argparse.Namespace:
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
    if config["log_file"]:
        args.log_file = Path(config["log_file"])
    if config["candle_ready_delay_seconds"]:
        args.candle_ready_delay_seconds = int(config["candle_ready_delay_seconds"])
    if config["http_proxy"]:
        args.http_proxy = config["http_proxy"]
    if config["https_proxy"]:
        args.https_proxy = config["https_proxy"]
    if config["telegram_enabled"]:
        args.telegram_enabled = config["telegram_enabled"].lower() in ("true", "1", "yes", "on")
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
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
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
    else:
        raise ValueError(f"Unknown exchange: {args.exchange}")


def create_telegram_client(config: LiveTradingConfig, logger: logging.Logger) -> Optional[TelegramClient]:
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
    telegram_logger.info("Telegram notifications enabled for chat %s", config.telegram_chat_id)
    return TelegramClient(telegram_config, telegram_logger)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Load environment config and apply defaults
    env_config = load_env_config()
    args = apply_env_defaults(args, env_config)
    
    # Setup logging
    logger = setup_logging(args)
    
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
        margin_mode = MarginMode.ISOLATED if args.margin_mode == "isolated" else MarginMode.CROSS
        
        config = LiveTradingConfig(
            exchange_name=args.exchange,
            api_key=args.api_key or "",
            api_secret=args.api_secret or "",
            testnet=args.testnet and not args.live,
            timeframe=args.timeframe,
            top_m_symbols=args.top_m_symbols,
            top_n_signals=args.top_n_signals,
            price_change_threshold_pct=args.price_change_threshold,
            heiken_ashi_candles_before=args.heiken_ashi_candles,
            leverage=args.leverage,
            take_profit_pct=args.take_profit_pct,
            margin_mode=margin_mode,
            disable_symbol_hours=args.disable_symbol_hours,
            position_size_usdt=args.position_size_usdt,
            max_concurrent_positions=args.max_concurrent_positions,
            max_position_size_pct=args.max_position_size_pct,
            state_file=args.state_file,
            positions_db=args.positions_db,
            log_file=args.log_file,
            candle_ready_delay_seconds=args.candle_ready_delay_seconds,
            telegram_enabled=args.telegram_enabled,
            telegram_bot_token=args.telegram_token or "",
            telegram_chat_id=args.telegram_chat_id or "",
            telegram_proxy=args.telegram_proxy or None,
            telegram_timeout_seconds=args.telegram_timeout,
        )
        
        logger.info("Live trading configuration:")
        logger.info(f"  Exchange: {config.exchange_name} ({'testnet' if config.testnet else 'LIVE'})")
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
        logger.info(f"  Candle ready delay: {config.candle_ready_delay_seconds} seconds")
        if config.telegram_enabled:
            logger.info("  Telegram notifications: enabled (chat %s)", config.telegram_chat_id)
        else:
            logger.info("  Telegram notifications: disabled")

        telegram_client = create_telegram_client(config, logger)
        
        # Create and run coordinator
        coordinator = LiveTradingCoordinator(config, exchange, telegram_client=telegram_client, logger=logger)
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
