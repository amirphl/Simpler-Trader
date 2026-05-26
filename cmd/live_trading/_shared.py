"""CLI for live trading."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from signal_notifier import TelegramClient, TelegramConfig

from live_trading import LiveTradingConfig
from live_trading.coordinator import LiveTradingCoordinator
from live_trading.exchange import ExchangeConfig, MarginMode
from live_trading.pinbar_magic_coordinator_v3 import (
    PinBarMagicCoordinatorV3,
    PinBarMagicCoordinatorV3Config,
)
from live_trading.strong_trend_stair_coordinator import (
    StrongTrendStairConfig,
    StrongTrendStairCoordinator,
)

DEFAULT_CONFIG_BY_STRATEGY: Dict[str, Path] = {
    "heiken_ashi": Path("./configs/live_trading.heiken_ashi.env"),
    "pinbar_magic_v3": Path("./configs/live_trading.pinbar_magic_v3.env"),
    "strong_trend_stair": Path("./configs/live_trading.strong_trend_stair.env"),
}
ALLOWED_STRATEGIES = tuple(DEFAULT_CONFIG_BY_STRATEGY.keys())


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


def _parse_int(value: str, key: str) -> int:
    try:
        return int(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {key}: {value!r}") from exc


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number for {key}: {value!r}") from exc


def _parse_symbols_csv(value: str) -> tuple[str, ...]:
    if not value.strip():
        return tuple()
    symbols = [part.strip().upper() for part in value.split(",")]
    return tuple(symbol for symbol in symbols if symbol)


def resolve_config_file(args: argparse.Namespace) -> Path:
    """Resolve config file path from CLI or strategy-specific default."""
    if args.config_file is not None:
        return args.config_file
    strategy = str(getattr(args, "strategy_name", "pinbar_magic_v3")).strip().lower()
    if strategy in DEFAULT_CONFIG_BY_STRATEGY:
        return DEFAULT_CONFIG_BY_STRATEGY[strategy]
    return DEFAULT_CONFIG_BY_STRATEGY["pinbar_magic_v3"]


def _normalize_strategy_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in DEFAULT_CONFIG_BY_STRATEGY:
        return normalized
    return "pinbar_magic_v3"


def _load_heiken_ashi_env_config(read_env: Callable[..., str]) -> Dict[str, str]:
    return {
        "exchange": read_env("EXCHANGE"),
        "trading_mode": read_env("TRADING_MODE"),
        "api_key": read_env("API_KEY"),
        "api_secret": read_env("API_SECRET"),
        "api_passphrase": read_env("API_PASSPHRASE", "PASS_PHRASE"),
        "testnet": read_env("TESTNET"),
        "timeframe": read_env("TIMEFRAME"),
        "strategy_name": read_env("STRATEGY_NAME", "LIVE_STRATEGY_NAME"),
        "leverage": read_env("LEVERAGE"),
        "take_profit_pct": read_env("TAKE_PROFIT_PCT"),
        "margin_mode": read_env("MARGIN_MODE"),
        "disable_symbol_hours": read_env("DISABLE_SYMBOL_HOURS"),
        "position_size_usdt": read_env("POSITION_SIZE_USDT"),
        "max_entry_notional_usdt": read_env("MAX_ENTRY_NOTIONAL_USDT"),
        "max_concurrent_positions": read_env("MAX_CONCURRENT_POSITIONS"),
        "max_position_size_pct": read_env("MAX_POSITION_SIZE_PCT"),
        "state_file": read_env("STATE_FILE"),
        "positions_db": read_env("POSITIONS_DB"),
        "klines_db": read_env("KLINES_DB"),
        "log_file": read_env("LOG_FILE"),
        "candle_ready_delay_seconds": read_env("CANDLE_READY_DELAY_SECONDS"),
        "execution_interval_minutes": read_env("EXECUTION_INTERVAL_MINUTES"),
        "poll_interval_seconds": read_env("POLL_INTERVAL_SECONDS"),
        "trailing_check_interval_seconds": read_env("TRAILING_CHECK_INTERVAL_SECONDS"),
        "exchange_base_url": read_env("EXCHANGE_BASE_URL"),
        "http_proxy": read_env("HTTP_PROXY"),
        "https_proxy": read_env("HTTPS_PROXY"),
        "proxy": read_env("PROXY"),
        "telegram_enabled": read_env("TELEGRAM_ENABLED"),
        "telegram_token": read_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN"),
        "telegram_chat_id": read_env("TELEGRAM_CHAT_ID"),
        "telegram_proxy": read_env("TELEGRAM_PROXY"),
        "telegram_timeout": read_env("TELEGRAM_TIMEOUT"),
        "log_level": read_env("LOG_LEVEL"),
        "top_m_symbols": read_env("TOP_M_SYMBOLS"),
        "top_n_signals": read_env("TOP_N_SIGNALS"),
        "price_change_threshold": read_env("PRICE_CHANGE_THRESHOLD"),
        "heiken_ashi_candles": read_env("HEIKEN_ASHI_CANDLES"),
    }


def _load_pinbar_magic_v3_env_config(read_env: Callable[..., str]) -> Dict[str, str]:
    return {
        "exchange": read_env("EXCHANGE"),
        "trading_mode": read_env("TRADING_MODE"),
        "api_key": read_env("API_KEY"),
        "api_secret": read_env("API_SECRET"),
        "api_passphrase": read_env("API_PASSPHRASE", "PASS_PHRASE"),
        "testnet": read_env("TESTNET"),
        "timeframe": read_env("TIMEFRAME"),
        "strategy_name": read_env("STRATEGY_NAME", "LIVE_STRATEGY_NAME"),
        "leverage": read_env("LEVERAGE"),
        "take_profit_pct": read_env("TAKE_PROFIT_PCT"),
        "margin_mode": read_env("MARGIN_MODE"),
        "disable_symbol_hours": read_env("DISABLE_SYMBOL_HOURS"),
        "position_size_usdt": read_env("POSITION_SIZE_USDT"),
        "max_entry_notional_usdt": read_env("MAX_ENTRY_NOTIONAL_USDT"),
        "max_concurrent_positions": read_env("MAX_CONCURRENT_POSITIONS"),
        "max_position_size_pct": read_env("MAX_POSITION_SIZE_PCT"),
        "state_file": read_env("STATE_FILE"),
        "positions_db": read_env("POSITIONS_DB"),
        "klines_db": read_env("KLINES_DB"),
        "log_file": read_env("LOG_FILE"),
        "candle_ready_delay_seconds": read_env("CANDLE_READY_DELAY_SECONDS"),
        "execution_interval_minutes": read_env("EXECUTION_INTERVAL_MINUTES"),
        "poll_interval_seconds": read_env("POLL_INTERVAL_SECONDS"),
        "trailing_check_interval_seconds": read_env("TRAILING_CHECK_INTERVAL_SECONDS"),
        "exchange_base_url": read_env("EXCHANGE_BASE_URL"),
        "http_proxy": read_env("HTTP_PROXY"),
        "https_proxy": read_env("HTTPS_PROXY"),
        "proxy": read_env("PROXY"),
        "telegram_enabled": read_env("TELEGRAM_ENABLED"),
        "telegram_token": read_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN"),
        "telegram_chat_id": read_env("TELEGRAM_CHAT_ID"),
        "telegram_proxy": read_env("TELEGRAM_PROXY"),
        "telegram_timeout": read_env("TELEGRAM_TIMEOUT"),
        "log_level": read_env("LOG_LEVEL"),
        "pinbar_symbols": read_env("PINBAR_SYMBOLS"),
        "equity_risk_pct": read_env("EQUITY_RISK_PCT", "STRATEGY_EQUITY_RISK_PCT"),
        "atr_multiple": read_env("ATR_MULTIPLE", "STRATEGY_ATR_MULTIPLE"),
        "trail_points": read_env("TRAIL_POINTS", "STRATEGY_TRAIL_POINTS"),
        "trail_offset": read_env("TRAIL_OFFSET", "STRATEGY_TRAIL_OFFSET"),
        "symbol_mintick": read_env("SYMBOL_MINTICK", "STRATEGY_SYMBOL_MINTICK"),
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
            "USE_TRAILING_TICK_EMULATION",
            "STRATEGY_USE_TRAILING_TICK_EMULATION",
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
        "max_entry_notional_usdt": read_env(
            "MAX_ENTRY_NOTIONAL_USDT", "STRATEGY_MAX_ENTRY_NOTIONAL_USDT"
        ),
    }


def _load_strong_trend_stair_env_config(read_env: Callable[..., str]) -> Dict[str, str]:
    return {
        "exchange": read_env("EXCHANGE"),
        "trading_mode": read_env("TRADING_MODE"),
        "api_key": read_env("API_KEY"),
        "api_secret": read_env("API_SECRET"),
        "api_passphrase": read_env("API_PASSPHRASE", "PASS_PHRASE"),
        "testnet": read_env("TESTNET"),
        "timeframe": read_env("TIMEFRAME"),
        "strategy_name": read_env("STRATEGY_NAME", "LIVE_STRATEGY_NAME"),
        "leverage": read_env("LEVERAGE"),
        "take_profit_pct": read_env("TAKE_PROFIT_PCT"),
        "margin_mode": read_env("MARGIN_MODE"),
        "disable_symbol_hours": read_env("DISABLE_SYMBOL_HOURS"),
        "position_size_usdt": read_env("POSITION_SIZE_USDT"),
        "max_entry_notional_usdt": read_env("MAX_ENTRY_NOTIONAL_USDT"),
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
        "telegram_token": read_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN"),
        "telegram_chat_id": read_env("TELEGRAM_CHAT_ID"),
        "telegram_proxy": read_env("TELEGRAM_PROXY"),
        "telegram_timeout": read_env("TELEGRAM_TIMEOUT"),
        "log_level": read_env("LOG_LEVEL"),
        "symbol": read_env("SYMBOL"),
        "tick_interval_seconds": read_env("TICK_INTERVAL_SECONDS"),
        "hard_stop_loss_pct": read_env(
            "HARD_STOP_LOSS_PCT", "STRATEGY_HARD_STOP_LOSS_PCT"
        ),
        "trail_start_pct": read_env("TRAIL_START_PCT", "STRATEGY_TRAIL_START_PCT"),
        "trail_offset_pct": read_env("TRAIL_OFFSET_PCT", "STRATEGY_TRAIL_OFFSET_PCT"),
        "ema_fast_len": read_env("EMA_FAST_LEN"),
        "ema_mid_len": read_env("EMA_MID_LEN"),
        "ema_slow_len": read_env("EMA_SLOW_LEN"),
        "slope_lookback": read_env("SLOPE_LOOKBACK"),
        "st_atr_len": read_env("ST_ATR_LEN"),
        "st_factor": read_env("ST_FACTOR"),
        "di_len": read_env("DI_LEN"),
        "adx_smooth": read_env("ADX_SMOOTH"),
        "adx_min": read_env("ADX_MIN"),
        "reverse_on_opposite_signal": read_env("REVERSE_ON_OPPOSITE_SIGNAL"),
    }


STRATEGY_ENV_LOADERS: Dict[str, Callable[[Callable[..., str]], Dict[str, str]]] = {
    "heiken_ashi": _load_heiken_ashi_env_config,
    "pinbar_magic_v3": _load_pinbar_magic_v3_env_config,
    "strong_trend_stair": _load_strong_trend_stair_env_config,
}


def load_env_config(
    config_file: Optional[Path] = None,
    strategy_name: str = "pinbar_magic_v3",
) -> Dict[str, str]:
    """Load configuration from config file and environment variables.

    Precedence: OS environment > config file > empty.
    """
    file_values = _load_env_file(config_file)
    env = os.getenv

    def read_env(*keys: str) -> str:
        # Global precedence is OS env > config file across all aliases.
        for key in keys:
            value = env(key, "")
            if value != "":
                return value
        for key in keys:
            file_value = file_values.get(key, "")
            if file_value != "":
                return file_value
        return ""

    selected = _normalize_strategy_name(strategy_name)
    strategy_loader = STRATEGY_ENV_LOADERS.get(selected)
    if strategy_loader is not None:
        return strategy_loader(read_env)
    return {}


def _apply_heiken_ashi_env_defaults(
    args: argparse.Namespace, config: Dict[str, str]
) -> None:
    if config["exchange"]:
        args.exchange = config["exchange"].strip().lower()
    if config["trading_mode"]:
        args.trading_mode = config["trading_mode"].strip().lower()
    if config["api_key"]:
        args.api_key = config["api_key"]
    if config["api_secret"]:
        args.api_secret = config["api_secret"]
    if config["api_passphrase"]:
        args.api_passphrase = config["api_passphrase"]
    if config["testnet"]:
        args.testnet = _parse_bool(config["testnet"])
    if config["timeframe"]:
        args.timeframe = config["timeframe"].strip().lower()
    if config["strategy_name"]:
        args.strategy_name = config["strategy_name"].strip().lower()
    if config["leverage"]:
        args.leverage = _parse_int(config["leverage"], "LEVERAGE")
    if config["take_profit_pct"]:
        args.take_profit_pct = _parse_float(
            config["take_profit_pct"], "TAKE_PROFIT_PCT"
        )
    if config["margin_mode"]:
        args.margin_mode = config["margin_mode"].strip().lower()
    if config["disable_symbol_hours"]:
        args.disable_symbol_hours = _parse_float(
            config["disable_symbol_hours"], "DISABLE_SYMBOL_HOURS"
        )
    if config["position_size_usdt"]:
        args.position_size_usdt = _parse_float(
            config["position_size_usdt"], "POSITION_SIZE_USDT"
        )
    if config["max_entry_notional_usdt"]:
        args.max_entry_notional_usdt = _parse_float(
            config["max_entry_notional_usdt"], "MAX_ENTRY_NOTIONAL_USDT"
        )
    if config["max_concurrent_positions"]:
        args.max_concurrent_positions = _parse_int(
            config["max_concurrent_positions"], "MAX_CONCURRENT_POSITIONS"
        )
    if config["max_position_size_pct"]:
        args.max_position_size_pct = _parse_float(
            config["max_position_size_pct"], "MAX_POSITION_SIZE_PCT"
        )
    if config["state_file"]:
        args.state_file = Path(config["state_file"])
    if config["positions_db"]:
        args.positions_db = Path(config["positions_db"])
    if config["klines_db"]:
        args.klines_db = Path(config["klines_db"])
    if config["log_file"]:
        args.log_file = Path(config["log_file"])
    if config["candle_ready_delay_seconds"]:
        args.candle_ready_delay_seconds = _parse_int(
            config["candle_ready_delay_seconds"], "CANDLE_READY_DELAY_SECONDS"
        )
    if config["execution_interval_minutes"]:
        args.execution_interval_minutes = _parse_int(
            config["execution_interval_minutes"], "EXECUTION_INTERVAL_MINUTES"
        )
    if config["poll_interval_seconds"]:
        args.poll_interval_seconds = _parse_float(
            config["poll_interval_seconds"], "POLL_INTERVAL_SECONDS"
        )
    if config["trailing_check_interval_seconds"]:
        args.trailing_check_interval_seconds = _parse_float(
            config["trailing_check_interval_seconds"],
            "TRAILING_CHECK_INTERVAL_SECONDS",
        )
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
        args.telegram_timeout = _parse_float(
            config["telegram_timeout"], "TELEGRAM_TIMEOUT"
        )
    if config["log_level"]:
        args.log_level = config["log_level"].strip().upper()
    if config.get("top_m_symbols"):
        args.top_m_symbols = _parse_int(config["top_m_symbols"], "TOP_M_SYMBOLS")
    if config.get("top_n_signals"):
        args.top_n_signals = _parse_int(config["top_n_signals"], "TOP_N_SIGNALS")
    if config.get("price_change_threshold"):
        args.price_change_threshold = _parse_float(
            config["price_change_threshold"], "PRICE_CHANGE_THRESHOLD"
        )
    if config.get("heiken_ashi_candles"):
        args.heiken_ashi_candles = _parse_int(
            config["heiken_ashi_candles"], "HEIKEN_ASHI_CANDLES"
        )


def _apply_pinbar_magic_v3_env_defaults(
    args: argparse.Namespace, config: Dict[str, str]
) -> None:
    if config["exchange"]:
        args.exchange = config["exchange"].strip().lower()
    if config["trading_mode"]:
        args.trading_mode = config["trading_mode"].strip().lower()
    if config["api_key"]:
        args.api_key = config["api_key"]
    if config["api_secret"]:
        args.api_secret = config["api_secret"]
    if config["api_passphrase"]:
        args.api_passphrase = config["api_passphrase"]
    if config["testnet"]:
        args.testnet = _parse_bool(config["testnet"])
    if config["timeframe"]:
        args.timeframe = config["timeframe"].strip().lower()
    if config["strategy_name"]:
        args.strategy_name = config["strategy_name"].strip().lower()
    if config["leverage"]:
        args.leverage = _parse_int(config["leverage"], "LEVERAGE")
    if config["take_profit_pct"]:
        args.take_profit_pct = _parse_float(
            config["take_profit_pct"], "TAKE_PROFIT_PCT"
        )
    if config["margin_mode"]:
        args.margin_mode = config["margin_mode"].strip().lower()
    if config["disable_symbol_hours"]:
        args.disable_symbol_hours = _parse_float(
            config["disable_symbol_hours"], "DISABLE_SYMBOL_HOURS"
        )
    if config["position_size_usdt"]:
        args.position_size_usdt = _parse_float(
            config["position_size_usdt"], "POSITION_SIZE_USDT"
        )
    if config["max_entry_notional_usdt"]:
        args.max_entry_notional_usdt = _parse_float(
            config["max_entry_notional_usdt"], "MAX_ENTRY_NOTIONAL_USDT"
        )
    if config["max_concurrent_positions"]:
        args.max_concurrent_positions = _parse_int(
            config["max_concurrent_positions"], "MAX_CONCURRENT_POSITIONS"
        )
    if config["max_position_size_pct"]:
        args.max_position_size_pct = _parse_float(
            config["max_position_size_pct"], "MAX_POSITION_SIZE_PCT"
        )
    if config["state_file"]:
        args.state_file = Path(config["state_file"])
    if config["positions_db"]:
        args.positions_db = Path(config["positions_db"])
    if config["klines_db"]:
        args.klines_db = Path(config["klines_db"])
    if config["log_file"]:
        args.log_file = Path(config["log_file"])
    if config["candle_ready_delay_seconds"]:
        args.candle_ready_delay_seconds = _parse_int(
            config["candle_ready_delay_seconds"], "CANDLE_READY_DELAY_SECONDS"
        )
    if config["execution_interval_minutes"]:
        args.execution_interval_minutes = _parse_int(
            config["execution_interval_minutes"], "EXECUTION_INTERVAL_MINUTES"
        )
    if config["poll_interval_seconds"]:
        args.poll_interval_seconds = _parse_float(
            config["poll_interval_seconds"], "POLL_INTERVAL_SECONDS"
        )
    if config["trailing_check_interval_seconds"]:
        args.trailing_check_interval_seconds = _parse_float(
            config["trailing_check_interval_seconds"],
            "TRAILING_CHECK_INTERVAL_SECONDS",
        )
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
        args.telegram_timeout = _parse_float(
            config["telegram_timeout"], "TELEGRAM_TIMEOUT"
        )
    if config["log_level"]:
        args.log_level = config["log_level"].strip().upper()
    if config.get("pinbar_symbols"):
        args.pinbar_symbols = config["pinbar_symbols"]
    if config.get("equity_risk_pct"):
        args.equity_risk_pct = _parse_float(
            config["equity_risk_pct"], "EQUITY_RISK_PCT"
        )
    if config.get("atr_multiple"):
        args.atr_multiple = _parse_float(config["atr_multiple"], "ATR_MULTIPLE")
    if config.get("trail_points"):
        args.trail_points = _parse_float(config["trail_points"], "TRAIL_POINTS")
    if config.get("trail_offset"):
        args.trail_offset = _parse_float(config["trail_offset"], "TRAIL_OFFSET")
    if config.get("symbol_mintick"):
        args.symbol_mintick = _parse_float(config["symbol_mintick"], "SYMBOL_MINTICK")
    if config.get("slow_sma_period"):
        args.slow_sma_period = _parse_int(config["slow_sma_period"], "SLOW_SMA_PERIOD")
    if config.get("medium_ema_period"):
        args.medium_ema_period = _parse_int(
            config["medium_ema_period"], "MEDIUM_EMA_PERIOD"
        )
    if config.get("fast_ema_period"):
        args.fast_ema_period = _parse_int(config["fast_ema_period"], "FAST_EMA_PERIOD")
    if config.get("atr_period"):
        args.atr_period = _parse_int(config["atr_period"], "ATR_PERIOD")
    if config.get("entry_cancel_bars"):
        args.entry_cancel_bars = _parse_int(
            config["entry_cancel_bars"], "ENTRY_CANCEL_BARS"
        )
    if config.get("entry_activation_mode"):
        args.entry_activation_mode = config["entry_activation_mode"].strip().lower()
    if config.get("trailing_tick_timeframe"):
        args.trailing_tick_timeframe = config["trailing_tick_timeframe"].strip().lower()
    if config.get("use_trailing_tick_emulation"):
        args.use_trailing_tick_emulation = _parse_bool(
            config["use_trailing_tick_emulation"]
        )
    if config.get("use_stop_fill_open_gap"):
        args.use_stop_fill_open_gap = _parse_bool(config["use_stop_fill_open_gap"])
    if config.get("enable_friday_close"):
        args.enable_friday_close = _parse_bool(config["enable_friday_close"])
    if config.get("friday_close_hour_utc"):
        args.friday_close_hour_utc = _parse_int(
            config["friday_close_hour_utc"], "FRIDAY_CLOSE_HOUR_UTC"
        )
    if config.get("enable_ema_cross_close"):
        args.enable_ema_cross_close = _parse_bool(config["enable_ema_cross_close"])
    if config.get("risk_equity_include_unrealized"):
        args.risk_equity_include_unrealized = _parse_bool(
            config["risk_equity_include_unrealized"]
        )
    if config.get("risk_equity_mark_source"):
        args.risk_equity_mark_source = config["risk_equity_mark_source"].strip().lower()


def _apply_strong_trend_stair_env_defaults(
    args: argparse.Namespace, config: Dict[str, str]
) -> None:
    if config["exchange"]:
        args.exchange = config["exchange"].strip().lower()
    if config["trading_mode"]:
        args.trading_mode = config["trading_mode"].strip().lower()
    if config["api_key"]:
        args.api_key = config["api_key"]
    if config["api_secret"]:
        args.api_secret = config["api_secret"]
    if config["api_passphrase"]:
        args.api_passphrase = config["api_passphrase"]
    if config["testnet"]:
        args.testnet = _parse_bool(config["testnet"])
    if config["timeframe"]:
        args.timeframe = config["timeframe"].strip().lower()
    if config["strategy_name"]:
        args.strategy_name = config["strategy_name"].strip().lower()
    if config["leverage"]:
        args.leverage = _parse_int(config["leverage"], "LEVERAGE")
    if config["take_profit_pct"]:
        args.take_profit_pct = _parse_float(
            config["take_profit_pct"], "TAKE_PROFIT_PCT"
        )
    if config["margin_mode"]:
        args.margin_mode = config["margin_mode"].strip().lower()
    if config["disable_symbol_hours"]:
        args.disable_symbol_hours = _parse_float(
            config["disable_symbol_hours"], "DISABLE_SYMBOL_HOURS"
        )
    if config["position_size_usdt"]:
        args.position_size_usdt = _parse_float(
            config["position_size_usdt"], "POSITION_SIZE_USDT"
        )
    if config["max_entry_notional_usdt"]:
        args.max_entry_notional_usdt = _parse_float(
            config["max_entry_notional_usdt"], "MAX_ENTRY_NOTIONAL_USDT"
        )
    if config["max_concurrent_positions"]:
        args.max_concurrent_positions = _parse_int(
            config["max_concurrent_positions"], "MAX_CONCURRENT_POSITIONS"
        )
    if config["max_position_size_pct"]:
        args.max_position_size_pct = _parse_float(
            config["max_position_size_pct"], "MAX_POSITION_SIZE_PCT"
        )
    if config["state_file"]:
        args.state_file = Path(config["state_file"])
    if config["positions_db"]:
        args.positions_db = Path(config["positions_db"])
    if config["klines_db"]:
        args.klines_db = Path(config["klines_db"])
    if config["log_file"]:
        args.log_file = Path(config["log_file"])
    if config["candle_ready_delay_seconds"]:
        args.candle_ready_delay_seconds = _parse_int(
            config["candle_ready_delay_seconds"], "CANDLE_READY_DELAY_SECONDS"
        )
    if config["execution_interval_minutes"]:
        args.execution_interval_minutes = _parse_int(
            config["execution_interval_minutes"], "EXECUTION_INTERVAL_MINUTES"
        )
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
        args.telegram_timeout = _parse_float(
            config["telegram_timeout"], "TELEGRAM_TIMEOUT"
        )
    if config["log_level"]:
        args.log_level = config["log_level"].strip().upper()
    if config.get("symbol"):
        args.symbol = config["symbol"].strip().upper()
    if config.get("tick_interval_seconds"):
        args.tick_interval_seconds = _parse_float(
            config["tick_interval_seconds"], "TICK_INTERVAL_SECONDS"
        )
    if config.get("hard_stop_loss_pct"):
        args.hard_stop_loss_pct = _parse_float(
            config["hard_stop_loss_pct"], "HARD_STOP_LOSS_PCT"
        )
    if config.get("trail_start_pct"):
        args.trail_start_pct = _parse_float(
            config["trail_start_pct"], "TRAIL_START_PCT"
        )
    if config.get("trail_offset_pct"):
        args.trail_offset_pct = _parse_float(
            config["trail_offset_pct"], "TRAIL_OFFSET_PCT"
        )
    if config.get("ema_fast_len"):
        args.ema_fast_len = _parse_int(config["ema_fast_len"], "EMA_FAST_LEN")
    if config.get("ema_mid_len"):
        args.ema_mid_len = _parse_int(config["ema_mid_len"], "EMA_MID_LEN")
    if config.get("ema_slow_len"):
        args.ema_slow_len = _parse_int(config["ema_slow_len"], "EMA_SLOW_LEN")
    if config.get("slope_lookback"):
        args.slope_lookback = _parse_int(config["slope_lookback"], "SLOPE_LOOKBACK")
    if config.get("st_atr_len"):
        args.st_atr_len = _parse_int(config["st_atr_len"], "ST_ATR_LEN")
    if config.get("st_factor"):
        args.st_factor = _parse_float(config["st_factor"], "ST_FACTOR")
    if config.get("di_len"):
        args.di_len = _parse_int(config["di_len"], "DI_LEN")
    if config.get("adx_smooth"):
        args.adx_smooth = _parse_int(config["adx_smooth"], "ADX_SMOOTH")
    if config.get("adx_min"):
        args.adx_min = _parse_float(config["adx_min"], "ADX_MIN")
    if config.get("reverse_on_opposite_signal"):
        args.reverse_on_opposite_signal = _parse_bool(
            config["reverse_on_opposite_signal"]
        )


STRATEGY_DEFAULT_APPLIERS: Dict[
    str, Callable[[argparse.Namespace, Dict[str, str]], None]
] = {
    "heiken_ashi": _apply_heiken_ashi_env_defaults,
    "pinbar_magic_v3": _apply_pinbar_magic_v3_env_defaults,
    "strong_trend_stair": _apply_strong_trend_stair_env_defaults,
}


def apply_env_defaults(
    args: argparse.Namespace, config: Dict[str, str], strategy_name: str
) -> argparse.Namespace:
    """Apply environment variable defaults to arguments."""
    selected = _normalize_strategy_name(strategy_name)
    strategy_applier = STRATEGY_DEFAULT_APPLIERS.get(selected)
    if strategy_applier is not None:
        strategy_applier(args, config)
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


def _redact_config_value(key: str, value: object) -> object:
    sensitive_terms = ("key", "secret", "token", "passphrase", "password")
    if any(term in key.lower() for term in sensitive_terms):
        if value:
            return "***REDACTED***"
        return value
    return value


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
    if args.exchange == "binance":
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

    logger.info("Initializing Telegram client")
    telegram_logger = logging.getLogger("live_trading.telegram")
    telegram_config = TelegramConfig(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        proxy=config.telegram_proxy,
        timeout=config.telegram_timeout_seconds,
    )
    telegram_logger.info("Telegram notifications enabled")
    return TelegramClient(telegram_config, telegram_logger)


def build_pinbar_magic_v3_coordinator_config(
    config: LiveTradingConfig,
) -> PinBarMagicCoordinatorV3Config:
    symbols = tuple(symbol.upper() for symbol in config.pinbar_symbols if symbol) or (
        "ETHUSDT",
    )
    return PinBarMagicCoordinatorV3Config(
        symbols=symbols,
        timeframe=config.timeframe,
        trailing_tick_timeframe=config.trailing_tick_timeframe,
        use_trailing_tick_emulation=config.use_trailing_tick_emulation,
        poll_interval_seconds=config.poll_interval_seconds,
        trailing_check_interval_seconds=config.trailing_check_interval_seconds,
        leverage=config.leverage,
        margin_mode=config.margin_mode,
        max_concurrent_positions=config.max_concurrent_positions,
        max_entry_notional_usdt=config.max_entry_notional_usdt,
        equity_risk_pct=config.equity_risk_pct,
        atr_multiple=config.atr_multiple,
        trail_points=config.trail_points,
        trail_offset=config.trail_offset,
        symbol_mintick=config.symbol_mintick,
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
        state_file=config.state_file,
        positions_db=config.positions_db,
        klines_db=config.klines_db,
        log_file=config.log_file,
    )


def build_strong_trend_stair_config(
    args: argparse.Namespace, config: LiveTradingConfig
) -> StrongTrendStairConfig:
    return StrongTrendStairConfig(
        symbol=args.symbol.upper(),
        timeframe=config.timeframe,
        tick_interval_seconds=args.tick_interval_seconds,
        leverage=config.leverage,
        margin_mode=config.margin_mode,
        trade_notional_usd=config.position_size_usdt,
        hard_stop_loss_pct=args.hard_stop_loss_pct,
        trail_start_pct=args.trail_start_pct,
        trail_offset_pct=args.trail_offset_pct,
        ema_fast_len=args.ema_fast_len,
        ema_mid_len=args.ema_mid_len,
        ema_slow_len=args.ema_slow_len,
        slope_lookback=args.slope_lookback,
        st_atr_len=args.st_atr_len,
        st_factor=args.st_factor,
        di_len=args.di_len,
        adx_smooth=args.adx_smooth,
        adx_min=args.adx_min,
        reverse_on_opposite_signal=args.reverse_on_opposite_signal,
    )


def _arg(args: argparse.Namespace, name: str, default: object) -> object:
    return getattr(args, name, default)


def run_main(
    parser: argparse.ArgumentParser,
    strategy_name: str,
    argv: Optional[List[str]] = None,
) -> int:
    """Run the live-trading entrypoint with a strategy-specific parser."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if any(token in {"-h", "--help"} for token in effective_argv):
        parser.parse_args(effective_argv)
        return 0

    # Parse once to discover strategy/config-file, then re-parse with scoped env defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--strategy-name", choices=ALLOWED_STRATEGIES, default=None)
    pre_parser.add_argument("--config-file", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(effective_argv)
    selected_strategy = _normalize_strategy_name(strategy_name)
    resolved_config_file = (
        pre_args.config_file
        if pre_args.config_file is not None
        else DEFAULT_CONFIG_BY_STRATEGY[selected_strategy]
    )
    env_config = load_env_config(resolved_config_file, strategy_name=selected_strategy)
    try:
        env_defaults = apply_env_defaults(
            parser.parse_args([]), env_config, selected_strategy
        )
    except ValueError as exc:
        parser.error(str(exc))
    parser.set_defaults(**vars(env_defaults))
    args = parser.parse_args(effective_argv)
    if args.strategy_name != selected_strategy:
        parser.error(
            f"--strategy-name/config strategy must be {selected_strategy!r} "
            f"for this entrypoint, got {args.strategy_name!r}"
        )
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
            top_m_symbols=int(_arg(args, "top_m_symbols", 100)),
            top_n_signals=int(_arg(args, "top_n_signals", 5)),
            price_change_threshold_pct=float(_arg(args, "price_change_threshold", 2.0)),
            pinbar_symbols=_parse_symbols_csv(str(_arg(args, "pinbar_symbols", ""))),
            heiken_ashi_candles_before=int(_arg(args, "heiken_ashi_candles", 3)),
            leverage=int(_arg(args, "leverage", 10)),
            take_profit_pct=float(_arg(args, "take_profit_pct", 1.0)),
            equity_risk_pct=float(_arg(args, "equity_risk_pct", 3.0)),
            atr_multiple=float(_arg(args, "atr_multiple", 0.5)),
            trail_points=float(_arg(args, "trail_points", 1.0)),
            trail_offset=float(_arg(args, "trail_offset", 1.0)),
            symbol_mintick=float(_arg(args, "symbol_mintick", 1.0)),
            slow_sma_period=int(_arg(args, "slow_sma_period", 50)),
            medium_ema_period=int(_arg(args, "medium_ema_period", 18)),
            fast_ema_period=int(_arg(args, "fast_ema_period", 6)),
            atr_period=int(_arg(args, "atr_period", 14)),
            entry_cancel_bars=int(_arg(args, "entry_cancel_bars", 3)),
            entry_activation_mode=str(_arg(args, "entry_activation_mode", "next_bar")),
            trailing_tick_timeframe=str(_arg(args, "trailing_tick_timeframe", "15m")),
            use_trailing_tick_emulation=bool(
                _arg(args, "use_trailing_tick_emulation", False)
            ),
            use_stop_fill_open_gap=bool(_arg(args, "use_stop_fill_open_gap", True)),
            enable_friday_close=bool(_arg(args, "enable_friday_close", True)),
            friday_close_hour_utc=int(_arg(args, "friday_close_hour_utc", 16)),
            enable_ema_cross_close=bool(_arg(args, "enable_ema_cross_close", True)),
            risk_equity_include_unrealized=bool(
                _arg(args, "risk_equity_include_unrealized", True)
            ),
            risk_equity_mark_source=str(_arg(args, "risk_equity_mark_source", "close")),
            margin_mode=margin_mode,
            disable_symbol_hours=args.disable_symbol_hours,
            position_size_usdt=args.position_size_usdt,
            max_entry_notional_usdt=args.max_entry_notional_usdt,
            max_concurrent_positions=args.max_concurrent_positions,
            max_position_size_pct=args.max_position_size_pct,
            state_file=args.state_file,
            positions_db=args.positions_db,
            klines_db=args.klines_db,
            log_file=args.log_file,
            candle_ready_delay_seconds=args.candle_ready_delay_seconds,
            execution_interval_minutes=args.execution_interval_minutes,
            poll_interval_seconds=float(_arg(args, "poll_interval_seconds", 5.0)),
            trailing_check_interval_seconds=float(
                _arg(args, "trailing_check_interval_seconds", 5.0)
            ),
            telegram_enabled=args.telegram_enabled,
            telegram_bot_token=args.telegram_token or "",
            telegram_chat_id=args.telegram_chat_id or "",
            telegram_proxy=args.telegram_proxy or None,
            telegram_timeout_seconds=args.telegram_timeout,
        )

        logger.info("Live trading configuration (all properties):")
        for key, raw_value in vars(config).items():
            value = raw_value.value if hasattr(raw_value, "value") else raw_value
            logger.info("  %s=%r", key, _redact_config_value(key, value))

        telegram_client = create_telegram_client(config, logger)

        # Create and run coordinator
        if config.strategy_name == "pinbar_magic_v3":
            pinbar_cfg = build_pinbar_magic_v3_coordinator_config(config)
            coordinator = PinBarMagicCoordinatorV3(
                exchange=exchange,
                config=pinbar_cfg,
                telegram_client=telegram_client,
                logger=logger,
            )
            coordinator.run_forever()
        elif config.strategy_name == "strong_trend_stair":
            strong_cfg = build_strong_trend_stair_config(args, config)
            coordinator = StrongTrendStairCoordinator(
                exchange=exchange,
                config=strong_cfg,
                telegram_client=telegram_client,
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
    raise SystemExit(
        "Use one of the strategy entrypoints in cmd/live_trading/*.py instead."
    )
