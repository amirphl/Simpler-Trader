from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from backtest import (
    BacktestRunConfig,
    BaseBacktester,
    EngulfingStrategy,
    EngulfingStrategyConfig,
    PinBarMagicStrategyConfigV2,
    PinBarMagicStrategyV2,
    StopLossMode,
    StochasticRsiFsmConfig,
    StochasticRsiFsmStrategy,
    plot_backtest_from_store,
    save_plot,
    show_plot,
)
from candle_downloader.binance import BinanceClient, BinanceClientConfig
from candle_downloader.downloader import CandleDownloader
from candle_downloader.storage import build_store


def parse_datetime(value: str) -> datetime:
    """Parse ISO8601 datetime string to UTC datetime."""
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid datetime: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_env_config() -> Dict[str, str]:
    """Load configuration from environment variables."""
    return {
        "strategy": os.getenv("BACKTEST_STRATEGY", os.getenv("STRATEGY_KIND", "engulfing")),
        "symbol": os.getenv("STRATEGY_SYMBOL", ""),
        "timeframe": os.getenv("STRATEGY_TIMEFRAME", ""),
        "window_size": os.getenv("STRATEGY_WINDOW_SIZE", ""),
        "leverage": os.getenv("STRATEGY_LEVERAGE", ""),
        "take_profit_pct": os.getenv("STRATEGY_TAKE_PROFIT_PCT", ""),
        "symbols": os.getenv("STRATEGY_SYMBOLS", ""),
        "base_timeframe": os.getenv("STRATEGY_BASE_TIMEFRAME", "1h"),
        "higher_timeframe": os.getenv("STRATEGY_HIGHER_TIMEFRAME", "4h"),
        "higher_timeframe_2": os.getenv("STRATEGY_HIGHER_TIMEFRAME_2", ""),
        "k_period": os.getenv("STRATEGY_K_PERIOD", "8"),
        "k_slowing": os.getenv("STRATEGY_K_SLOWING", "8"),
        "d_period": os.getenv("STRATEGY_D_PERIOD", "1"),
        "use_d_line": os.getenv("STRATEGY_USE_D_LINE", "false"),
        "oversold": os.getenv("STRATEGY_OVERSOLD", "20"),
        "overbought": os.getenv("STRATEGY_OVERBOUGHT", "80"),
        "initial_order_usdt": os.getenv("STRATEGY_INITIAL_ORDER_USDT", "100"),
        "initial_leverage": os.getenv("STRATEGY_INITIAL_LEVERAGE", "3"),
        "martingale_multiplier": os.getenv("STRATEGY_MARTINGALE_MULTIPLIER", "1.1"),
        "martingale_multipliers": os.getenv(
            "STRATEGY_MARTINGALE_MULTIPLIERS", "1.5,2.0,2.5,3.0"
        ),
        "martingale_leverages": os.getenv(
            "STRATEGY_MARTINGALE_LEVERAGES", "3.0,3.0,3.0,3.0"
        ),
        "max_positions": os.getenv("STRATEGY_MAX_POSITIONS", "5"),
        "slippage_pct": os.getenv("STRATEGY_SLIPPAGE_PCT", "0.0002"),
        "maker_fee_pct": os.getenv("STRATEGY_MAKER_FEE_PCT", "0.0002"),
        "taker_fee_pct": os.getenv("STRATEGY_TAKER_FEE_PCT", "0.0006"),
        "funding_rate_pct": os.getenv("STRATEGY_FUNDING_RATE_PCT", "0"),
        "trailing_activation_pct": os.getenv(
            "STRATEGY_TRAILING_ACTIVATION_PCT", "1.5"
        ),
        "trailing_gap_pct": os.getenv("STRATEGY_TRAILING_GAP_PCT", "1.0"),
        "trailing_interval": os.getenv("STRATEGY_TRAILING_INTERVAL", "10"),
        "max_position_days": os.getenv("STRATEGY_MAX_POSITION_DAYS", "30"),
        "aligned_high_stoch_mode": os.getenv(
            "STRATEGY_ALIGNED_HIGH_STOCH_MODE", "v3"
        ),
        "signal_offset": os.getenv("STRATEGY_SIGNAL_OFFSET", "0"),
        "enable_take_profit_check": os.getenv(
            "STRATEGY_ENABLE_TAKE_PROFIT_CHECK", "false"
        ),
        "enable_high_exit_cross": os.getenv("STRATEGY_ENABLE_HIGH_EXIT_CROSS", "false"),
        "use_midsold_filter": os.getenv("STRATEGY_USE_MIDSOLD_FILTER", "false"),
        "enable_reversal_logic": os.getenv("STRATEGY_ENABLE_REVERSAL_LOGIC", "false"),
        "enable_reversal_reentry": os.getenv(
            "STRATEGY_ENABLE_REVERSAL_REENTRY", "false"
        ),
        "enable_grid_martingales": os.getenv(
            "STRATEGY_ENABLE_GRID_MARTINGALES", "true"
        ),
        "grid_martingales_percent": os.getenv(
            "STRATEGY_GRID_MARTINGALES_PERCENT", "3.0"
        ),
        "trailing_use_first_entry_price": os.getenv(
            "STRATEGY_TRAILING_USE_FIRST_ENTRY_PRICE", "true"
        ),
        "trailing_use_close_for_stop_activation": os.getenv(
            "STRATEGY_TRAILING_USE_CLOSE_FOR_STOP_ACTIVATION", "true"
        ),
        "take_profit_use_first_entry_price": os.getenv(
            "STRATEGY_TAKE_PROFIT_USE_FIRST_ENTRY_PRICE", "true"
        ),
        "doji_size": os.getenv("STRATEGY_DOJI_SIZE", "0.05"),
        "stop_loss_mode": os.getenv("STRATEGY_STOP_LOSS_MODE", "percent"),
        "stop_loss_pct": os.getenv("STRATEGY_STOP_LOSS_PCT", "0.005"),
        "exchange_fee_pct": os.getenv("STRATEGY_EXCHANGE_FEE_PCT", "0.0004"),
        "skip_wick_filter": os.getenv("STRATEGY_SKIP_WICK_FILTER", "false"),
        "skip_bollinger_cross": os.getenv("STRATEGY_SKIP_BB_FILTER", "false"),
        "bollinger_period": os.getenv("STRATEGY_BB_PERIOD", "20"),
        "bollinger_stddev": os.getenv("STRATEGY_BB_STDDEV", "2.0"),
        "volume_filter_enabled": os.getenv("STRATEGY_VOLUME_FILTER_ENABLED", "true"),
        "stoch_enabled": os.getenv("STRATEGY_STOCH_ENABLED", "true"),
        "stoch_first_line": os.getenv("STRATEGY_STOCH_FIRST_LINE", "k"),
        "stoch_first_period": os.getenv("STRATEGY_STOCH_FIRST_PERIOD", "20"),
        "stoch_first_threshold": os.getenv("STRATEGY_STOCH_FIRST_THRESHOLD", ""),
        "stoch_second_line": os.getenv("STRATEGY_STOCH_SECOND_LINE", "k"),
        "stoch_second_period": os.getenv("STRATEGY_STOCH_SECOND_PERIOD", "100"),
        "stoch_second_threshold": os.getenv("STRATEGY_STOCH_SECOND_THRESHOLD", ""),
        "stoch_comparison": os.getenv("STRATEGY_STOCH_COMPARISON", "gt"),
        "stoch_d_smoothing": os.getenv("STRATEGY_STOCH_D_SMOOTHING", "3"),
        "equity_risk_pct": os.getenv("STRATEGY_EQUITY_RISK_PCT", "3"),
        "atr_multiple": os.getenv("STRATEGY_ATR_MULTIPLE", "0.5"),
        "trail_points": os.getenv("STRATEGY_TRAIL_POINTS", "1"),
        "trail_offset": os.getenv("STRATEGY_TRAIL_OFFSET", "1"),
        "slow_sma_period": os.getenv("STRATEGY_SLOW_SMA_PERIOD", "50"),
        "medium_ema_period": os.getenv("STRATEGY_MEDIUM_EMA_PERIOD", "18"),
        "fast_ema_period": os.getenv("STRATEGY_FAST_EMA_PERIOD", "6"),
        "atr_period": os.getenv("STRATEGY_ATR_PERIOD", "14"),
        "entry_cancel_bars": os.getenv("STRATEGY_ENTRY_CANCEL_BARS", "3"),
        "entry_activation_mode": os.getenv("STRATEGY_ENTRY_ACTIVATION_MODE", "next_bar"),
        "trailing_tick_timeframe": os.getenv("STRATEGY_TRAILING_TICK_TIMEFRAME", "15m"),
        "use_trailing_tick_emulation": os.getenv("STRATEGY_USE_TRAILING_TICK_EMULATION", "false"),
        "use_stop_fill_open_gap": os.getenv("STRATEGY_USE_STOP_FILL_OPEN_GAP", "true"),
        "enable_friday_close": os.getenv("STRATEGY_ENABLE_FRIDAY_CLOSE", "true"),
        "friday_close_hour_utc": os.getenv("STRATEGY_FRIDAY_CLOSE_HOUR_UTC", "16"),
        "enable_ema_cross_close": os.getenv("STRATEGY_ENABLE_EMA_CROSS_CLOSE", "true"),
        "risk_equity_include_unrealized": os.getenv("STRATEGY_RISK_EQUITY_INCLUDE_UNREALIZED", "true"),
        "risk_equity_mark_source": os.getenv("STRATEGY_RISK_EQUITY_MARK_SOURCE", "close"),
        "start": os.getenv("BACKTEST_START", ""),
        "end": os.getenv("BACKTEST_END", ""),
        "initial_capital": os.getenv("BACKTEST_INITIAL_CAPITAL", ""),
        "warmup_days": os.getenv("BACKTEST_WARMUP_DAYS", "0"),
        "override_download": os.getenv("OVERRIDE_DOWNLOAD", "false"),
        "store_kind": os.getenv("STORE_KIND", "postgres"),
        "store_path": os.getenv("STORE_PATH", ""),
        "http_proxy": os.getenv("HTTP_PROXY", ""),
        "https_proxy": os.getenv("HTTPS_PROXY", ""),
        "proxy": os.getenv("PROXY", ""),
        "stats_output": os.getenv("STATS_OUTPUT", "./backtest_stats.json"),
        "plot_output": os.getenv("PLOT_OUTPUT", ""),
        "show_plot": os.getenv("SHOW_PLOT", "false"),
        "show_stochastic": os.getenv("SHOW_STOCHASTIC", "true"),
        "show_equity": os.getenv("SHOW_EQUITY", "true"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }


def str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    return [float(part) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run backtest strategy with candle download and visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (can be overridden by CLI args):
  STRATEGY_SYMBOL              Trading pair symbol (e.g., BTCUSDT)
  STRATEGY_TIMEFRAME          Binance interval (e.g., 1h, 4h)
  STRATEGY_WINDOW_SIZE        Number of candles to check for bearish pattern
  STRATEGY_LEVERAGE           Leverage multiplier
  STRATEGY_TAKE_PROFIT_PCT    Take profit percentage (e.g., 0.02 for 2%%)
  STRATEGY_DOJI_SIZE          Doji size for pattern detection (default: 0.05)
  STRATEGY_STOP_LOSS_MODE     Stop-loss placement: percent, close, low, open, or body
  STRATEGY_STOP_LOSS_PCT      Fraction for percent-based stop loss (default: 0.005)
  STRATEGY_EXCHANGE_FEE_PCT   Exchange fee per side (decimal, default: 0.0004)
  STRATEGY_SKIP_WICK_FILTER   true/false to skip long upper-wick engulfing candles
  STRATEGY_SKIP_BB_FILTER     true/false to skip when engulfing candle pierces Bollinger upper band
  STRATEGY_BB_PERIOD          Period used for Bollinger filter (default: 20)
  STRATEGY_BB_STDDEV          Stddev multiplier for Bollinger filter (default: 2.0)
  STRATEGY_VOLUME_FILTER_ENABLED  true/false to enable volume pressure filter
  STRATEGY_STOCH_ENABLED      true/false to enable stochastic comparison
  STRATEGY_STOCH_FIRST_LINE   "k" or "d" for first stochastic leg
  STRATEGY_STOCH_FIRST_PERIOD Period for first stochastic leg
  STRATEGY_STOCH_FIRST_THRESHOLD Minimum value required for first leg (optional)
  STRATEGY_STOCH_SECOND_LINE  "k" or "d" for second stochastic leg
  STRATEGY_STOCH_SECOND_PERIOD Period for second stochastic leg
  STRATEGY_STOCH_SECOND_THRESHOLD Minimum value required for second leg (optional)
  STRATEGY_STOCH_COMPARISON   "gt" or "lt" to define comparison between legs
  STRATEGY_STOCH_D_SMOOTHING  %D smoothing length (default: 3)
  BACKTEST_START              Start datetime (ISO8601, UTC)
  BACKTEST_END                End datetime (ISO8601, UTC)
  BACKTEST_INITIAL_CAPITAL    Starting capital
  STORE_KIND                  Storage type: postgres
  STORE_PATH                  Optional .env file path for postgres settings
  HTTP_PROXY                  HTTP proxy URL
  HTTPS_PROXY                 HTTPS proxy URL
  PROXY                       Shortcut for both HTTP and HTTPS proxy
  STATS_OUTPUT                Path to write statistics JSON
  PLOT_OUTPUT                 Path to save plot (HTML/PNG/SVG/PDF)
  SHOW_PLOT                   Set to 'true' to display plot in browser
        """,
    )

    # Strategy parameters
    parser.add_argument(
        "--strategy",
        choices=("engulfing", "pinbar_magic_v2", "stochastic_fsm"),
        help="Backtest strategy to run",
    )
    parser.add_argument("--symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols for stochastic_fsm (e.g., BTCUSDT,ETHUSDT)",
    )
    parser.add_argument("--timeframe", help="Binance interval (e.g., 1h, 4h)")
    parser.add_argument(
        "--window-size", type=int, help="Number of candles to check for bearish pattern"
    )
    parser.add_argument("--leverage", type=float, help="Leverage multiplier")
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        help="Take profit percentage (e.g., 0.02 for 2%%)",
    )
    parser.add_argument(
        "--doji-size", type=float, help="Doji size for pattern detection"
    )
    parser.add_argument(
        "--base-timeframe", help="stochastic_fsm base timeframe (default: 1h)"
    )
    parser.add_argument(
        "--higher-timeframe", help="stochastic_fsm higher timeframe (default: 4h)"
    )
    parser.add_argument(
        "--higher-timeframe-2",
        help="stochastic_fsm optional second higher timeframe (e.g., 1d)",
    )
    parser.add_argument("--k-period", type=int, help="stochastic_fsm K period")
    parser.add_argument("--k-slowing", type=int, help="stochastic_fsm K smoothing")
    parser.add_argument("--d-period", type=int, help="stochastic_fsm D period")
    parser.add_argument(
        "--use-d-line",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm use D line instead of K",
    )
    parser.add_argument("--oversold", type=float, help="stochastic_fsm oversold level")
    parser.add_argument(
        "--overbought", type=float, help="stochastic_fsm overbought level"
    )
    parser.add_argument(
        "--initial-order-usdt",
        type=float,
        help="stochastic_fsm initial order notional in USDT",
    )
    parser.add_argument(
        "--initial-leverage", type=float, help="stochastic_fsm initial leverage"
    )
    parser.add_argument(
        "--martingale-multiplier",
        type=float,
        help="stochastic_fsm scalar martingale multiplier",
    )
    parser.add_argument(
        "--martingale-multipliers",
        help="stochastic_fsm comma-separated multipliers per add step",
    )
    parser.add_argument(
        "--martingale-leverages",
        help="stochastic_fsm comma-separated leverages per add step",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        help="stochastic_fsm max concurrent open symbols",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        help="stochastic_fsm slippage fraction",
    )
    parser.add_argument(
        "--maker-fee-pct", type=float, help="stochastic_fsm maker fee fraction"
    )
    parser.add_argument(
        "--taker-fee-pct", type=float, help="stochastic_fsm taker fee fraction"
    )
    parser.add_argument(
        "--funding-rate-pct",
        type=float,
        help="stochastic_fsm daily funding rate percentage",
    )
    parser.add_argument(
        "--trailing-activation-pct",
        type=float,
        help="stochastic_fsm profit pct to activate trailing",
    )
    parser.add_argument(
        "--trailing-gap-pct",
        type=float,
        help="stochastic_fsm trailing gap pct",
    )
    parser.add_argument(
        "--trailing-interval",
        type=float,
        help="stochastic_fsm minimum seconds between trailing checks",
    )
    parser.add_argument(
        "--max-position-days",
        type=float,
        help="stochastic_fsm maximum holding period in days",
    )
    parser.add_argument(
        "--aligned-high-stoch-mode",
        choices=("v1", "v2", "v3"),
        help="stochastic_fsm higher StochRSI alignment mode",
    )
    parser.add_argument(
        "--signal-offset",
        type=int,
        help="stochastic_fsm additional signal lookback offset",
    )
    parser.add_argument(
        "--enable-take-profit-check",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm enable take-profit check in run loop",
    )
    parser.add_argument(
        "--enable-high-exit-cross",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm enable higher timeframe exit cross",
    )
    parser.add_argument(
        "--use-midsold-filter",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm require midsold filter",
    )
    parser.add_argument(
        "--enable-reversal-logic",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm enable reversal exit logic",
    )
    parser.add_argument(
        "--enable-reversal-reentry",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm enable reversal re-entry",
    )
    parser.add_argument(
        "--enable-grid-martingales",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm require grid distance before martingale add",
    )
    parser.add_argument(
        "--grid-martingales-percent",
        type=float,
        help="stochastic_fsm percent distance for grid martingales",
    )
    parser.add_argument(
        "--trailing-use-first-entry-price",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm base trailing on first entry",
    )
    parser.add_argument(
        "--trailing-use-close-for-stop-activation",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm use close for trailing-stop activation",
    )
    parser.add_argument(
        "--take-profit-use-first-entry-price",
        action=argparse.BooleanOptionalAction,
        help="stochastic_fsm base take-profit on first entry",
    )
    parser.add_argument(
        "--stop-loss-mode",
        choices=("percent", "close", "low", "open", "body"),
        help="Stop-loss placement strategy",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        help="Fraction for percent-based stop loss (e.g., 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--exchange-fee-pct",
        type=float,
        help="Exchange fee per side as a decimal (e.g., 0.0004 = 4 bps)",
    )
    parser.add_argument(
        "--skip-wick-filter",
        action=argparse.BooleanOptionalAction,
        help="Skip signals when engulfing candle upper wick outweighs the body",
    )
    parser.add_argument(
        "--skip-bollinger-cross",
        action=argparse.BooleanOptionalAction,
        help="Skip signals when engulfing candle pierces the Bollinger upper band",
    )
    parser.add_argument(
        "--bollinger-period", type=int, help="Period for Bollinger band filter"
    )
    parser.add_argument(
        "--bollinger-stddev", type=float, help="Stddev multiplier for Bollinger filter"
    )
    parser.add_argument(
        "--volume-filter",
        dest="volume_filter",
        action=argparse.BooleanOptionalAction,
        help="Enable or disable volume pressure filter",
    )
    parser.add_argument(
        "--stoch-enabled",
        dest="stoch_enabled",
        action=argparse.BooleanOptionalAction,
        help="Enable or disable stochastic comparison filter",
    )
    parser.add_argument(
        "--stoch-first-line",
        choices=("k", "d"),
        help="Line type for first stochastic leg (k or d)",
    )
    parser.add_argument(
        "--stoch-first-period", type=int, help="Period for first stochastic leg"
    )
    parser.add_argument(
        "--stoch-first-threshold",
        type=float,
        help="Minimum value for first stochastic leg (optional)",
    )
    parser.add_argument(
        "--stoch-second-line",
        choices=("k", "d"),
        help="Line type for second stochastic leg (k or d)",
    )
    parser.add_argument(
        "--stoch-second-period", type=int, help="Period for second stochastic leg"
    )
    parser.add_argument(
        "--stoch-second-threshold",
        type=float,
        help="Minimum value for second stochastic leg (optional)",
    )
    parser.add_argument(
        "--stoch-comparison",
        choices=("gt", "lt"),
        help="Comparison operator between stochastic legs",
    )
    parser.add_argument(
        "--stoch-d-smoothing",
        type=int,
        help="Smoothing length for %%D (applies when using D line)",
    )
    parser.add_argument(
        "--equity-risk-pct",
        type=float,
        help="Pinbar Magic v2 equity risk percent",
    )
    parser.add_argument(
        "--atr-multiple",
        type=float,
        help="Pinbar Magic v2 ATR multiple",
    )
    parser.add_argument(
        "--trail-points",
        type=float,
        help="Pinbar Magic v2 trailing activation distance (price units)",
    )
    parser.add_argument(
        "--trail-offset",
        type=float,
        help="Pinbar Magic v2 trailing offset distance (price units)",
    )
    parser.add_argument(
        "--slow-sma-period",
        type=int,
        help="Pinbar Magic v2 slow SMA period",
    )
    parser.add_argument(
        "--medium-ema-period",
        type=int,
        help="Pinbar Magic v2 medium EMA period",
    )
    parser.add_argument(
        "--fast-ema-period",
        type=int,
        help="Pinbar Magic v2 fast EMA period",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        help="Pinbar Magic v2 ATR period",
    )
    parser.add_argument(
        "--entry-cancel-bars",
        type=int,
        help="Pinbar Magic v2 bars until pending entry cancellation",
    )
    parser.add_argument(
        "--entry-activation-mode",
        choices=("next_bar", "same_bar"),
        help="Pinbar Magic v2 stop-entry activation mode",
    )
    parser.add_argument(
        "--trailing-tick-timeframe",
        help="Pinbar Magic v2 timeframe used for trailing tick emulation",
    )
    parser.add_argument(
        "--use-trailing-tick-emulation",
        action=argparse.BooleanOptionalAction,
        help="Enable Pinbar Magic v2 trailing tick emulation",
    )
    parser.add_argument(
        "--use-stop-fill-open-gap",
        action=argparse.BooleanOptionalAction,
        help="Enable Pinbar Magic v2 stop fill at open when gap crosses stop",
    )
    parser.add_argument(
        "--enable-friday-close",
        action=argparse.BooleanOptionalAction,
        help="Enable Pinbar Magic v2 Friday close-all behavior",
    )
    parser.add_argument(
        "--friday-close-hour-utc",
        type=int,
        help="Pinbar Magic v2 Friday close hour in UTC",
    )
    parser.add_argument(
        "--enable-ema-cross-close",
        action=argparse.BooleanOptionalAction,
        help="Enable Pinbar Magic v2 EMA cross close-all behavior",
    )
    parser.add_argument(
        "--risk-equity-include-unrealized",
        action=argparse.BooleanOptionalAction,
        help="Include unrealized PnL in Pinbar Magic v2 risk equity",
    )
    parser.add_argument(
        "--risk-equity-mark-source",
        choices=("close", "open", "hl2", "ohlc4"),
        help="Pinbar Magic v2 mark source for unrealized equity",
    )

    # Backtest parameters
    parser.add_argument(
        "--start", type=parse_datetime, help="Start datetime (ISO8601, UTC)"
    )
    parser.add_argument(
        "--end", type=parse_datetime, help="End datetime (ISO8601, UTC)"
    )
    parser.add_argument("--initial-capital", type=float, help="Starting capital")
    parser.add_argument("--warmup-days", type=int, help="Warmup days before start date")
    parser.add_argument(
        "--override-download", action="store_true", help="Re-download all candles"
    )

    # Storage parameters
    parser.add_argument("--store-kind", choices=("postgres",), help="Storage type")
    parser.add_argument(
        "--store-path",
        type=Path,
        help="Optional .env file path for postgres settings",
    )

    # Network parameters
    parser.add_argument("--http-proxy", help="HTTP proxy URL")
    parser.add_argument("--https-proxy", help="HTTPS proxy URL")
    parser.add_argument("--proxy", help="Shortcut for both HTTP and HTTPS proxy")

    # Output parameters
    parser.add_argument(
        "--stats-output", type=Path, help="Path to write statistics JSON"
    )
    parser.add_argument(
        "--plot-output", type=Path, help="Path to save plot (HTML/PNG/SVG/PDF)"
    )
    parser.add_argument(
        "--show-plot", action="store_true", help="Display plot in browser"
    )
    parser.add_argument(
        "--no-stochastic",
        action="store_true",
        help="Hide stochastic oscillator subplot",
    )
    parser.add_argument(
        "--no-equity", action="store_true", help="Hide equity curve subplot"
    )

    parser.add_argument("--log-level", default="INFO", help="Logging level")

    return parser


def resolve_config(
    args: argparse.Namespace, env_config: Dict[str, str]
) -> Dict[str, Any]:
    """Resolve configuration from CLI args and environment, with CLI taking precedence."""
    config: Dict[str, Any] = {}

    # Strategy config
    config["strategy"] = (
        (args.strategy or env_config.get("strategy") or "engulfing").strip().lower()
    )
    config["symbols"] = [
        sym.strip().upper()
        for sym in (args.symbols or env_config.get("symbols", "")).split(",")
        if sym.strip()
    ]
    config["symbol"] = args.symbol or env_config["symbol"] or None
    config["timeframe"] = args.timeframe or env_config["timeframe"] or None
    config["base_timeframe"] = (
        args.base_timeframe or env_config.get("base_timeframe", "1h")
    )
    config["higher_timeframe"] = (
        args.higher_timeframe or env_config.get("higher_timeframe", "4h")
    )
    config["higher_timeframe_2"] = (
        args.higher_timeframe_2 or env_config.get("higher_timeframe_2") or None
    )
    config["k_period"] = (
        args.k_period
        if args.k_period is not None
        else int(env_config.get("k_period", "8"))
    )
    config["k_slowing"] = (
        args.k_slowing
        if args.k_slowing is not None
        else int(env_config.get("k_slowing", "8"))
    )
    config["d_period"] = (
        args.d_period
        if args.d_period is not None
        else int(env_config.get("d_period", "1"))
    )
    if args.use_d_line is not None:
        config["use_d_line"] = args.use_d_line
    else:
        config["use_d_line"] = str_to_bool(env_config.get("use_d_line"), False)
    config["oversold"] = (
        args.oversold
        if args.oversold is not None
        else float(env_config.get("oversold", "20"))
    )
    config["overbought"] = (
        args.overbought
        if args.overbought is not None
        else float(env_config.get("overbought", "80"))
    )
    config["initial_order_usdt"] = (
        args.initial_order_usdt
        if args.initial_order_usdt is not None
        else float(env_config.get("initial_order_usdt", "100"))
    )
    config["initial_leverage"] = (
        args.initial_leverage
        if args.initial_leverage is not None
        else float(env_config.get("initial_leverage", "3"))
    )
    config["martingale_multiplier"] = (
        args.martingale_multiplier
        if args.martingale_multiplier is not None
        else float(env_config.get("martingale_multiplier", "1.1"))
    )
    config["martingale_multipliers"] = parse_float_list(
        args.martingale_multipliers or env_config.get("martingale_multipliers", "")
    )
    config["martingale_leverages"] = parse_float_list(
        args.martingale_leverages or env_config.get("martingale_leverages", "")
    )
    config["max_positions"] = (
        args.max_positions
        if args.max_positions is not None
        else int(env_config.get("max_positions", "5"))
    )
    config["slippage_pct"] = (
        args.slippage_pct
        if args.slippage_pct is not None
        else float(env_config.get("slippage_pct", "0.0002"))
    )
    config["maker_fee_pct"] = (
        args.maker_fee_pct
        if args.maker_fee_pct is not None
        else float(env_config.get("maker_fee_pct", "0.0002"))
    )
    config["taker_fee_pct"] = (
        args.taker_fee_pct
        if args.taker_fee_pct is not None
        else float(env_config.get("taker_fee_pct", "0.0006"))
    )
    config["funding_rate_pct"] = (
        args.funding_rate_pct
        if args.funding_rate_pct is not None
        else float(env_config.get("funding_rate_pct", "0"))
    )
    config["trailing_activation_pct"] = (
        args.trailing_activation_pct
        if args.trailing_activation_pct is not None
        else float(env_config.get("trailing_activation_pct", "1.5"))
    )
    config["trailing_gap_pct"] = (
        args.trailing_gap_pct
        if args.trailing_gap_pct is not None
        else float(env_config.get("trailing_gap_pct", "1.0"))
    )
    config["trailing_interval"] = (
        args.trailing_interval
        if args.trailing_interval is not None
        else float(env_config.get("trailing_interval", "10"))
    )
    config["max_position_days"] = (
        args.max_position_days
        if args.max_position_days is not None
        else float(env_config.get("max_position_days", "30"))
    )
    config["aligned_high_stoch_mode"] = (
        args.aligned_high_stoch_mode
        or env_config.get("aligned_high_stoch_mode", "v3")
    )
    config["signal_offset"] = (
        args.signal_offset
        if args.signal_offset is not None
        else int(env_config.get("signal_offset", "0"))
    )
    if args.enable_take_profit_check is not None:
        config["enable_take_profit_check"] = args.enable_take_profit_check
    else:
        config["enable_take_profit_check"] = str_to_bool(
            env_config.get("enable_take_profit_check"), False
        )
    if args.enable_high_exit_cross is not None:
        config["enable_high_exit_cross"] = args.enable_high_exit_cross
    else:
        config["enable_high_exit_cross"] = str_to_bool(
            env_config.get("enable_high_exit_cross"), False
        )
    if args.use_midsold_filter is not None:
        config["use_midsold_filter"] = args.use_midsold_filter
    else:
        config["use_midsold_filter"] = str_to_bool(
            env_config.get("use_midsold_filter"), False
        )
    if args.enable_reversal_logic is not None:
        config["enable_reversal_logic"] = args.enable_reversal_logic
    else:
        config["enable_reversal_logic"] = str_to_bool(
            env_config.get("enable_reversal_logic"), False
        )
    if args.enable_reversal_reentry is not None:
        config["enable_reversal_reentry"] = args.enable_reversal_reentry
    else:
        config["enable_reversal_reentry"] = str_to_bool(
            env_config.get("enable_reversal_reentry"), False
        )
    if args.enable_grid_martingales is not None:
        config["enable_grid_martingales"] = args.enable_grid_martingales
    else:
        config["enable_grid_martingales"] = str_to_bool(
            env_config.get("enable_grid_martingales"), True
        )
    config["grid_martingales_percent"] = (
        args.grid_martingales_percent
        if args.grid_martingales_percent is not None
        else float(env_config.get("grid_martingales_percent", "3.0"))
    )
    if args.trailing_use_first_entry_price is not None:
        config["trailing_use_first_entry_price"] = args.trailing_use_first_entry_price
    else:
        config["trailing_use_first_entry_price"] = str_to_bool(
            env_config.get("trailing_use_first_entry_price"), True
        )
    if args.trailing_use_close_for_stop_activation is not None:
        config["trailing_use_close_for_stop_activation"] = (
            args.trailing_use_close_for_stop_activation
        )
    else:
        config["trailing_use_close_for_stop_activation"] = str_to_bool(
            env_config.get("trailing_use_close_for_stop_activation"), True
        )
    if args.take_profit_use_first_entry_price is not None:
        config["take_profit_use_first_entry_price"] = (
            args.take_profit_use_first_entry_price
        )
    else:
        config["take_profit_use_first_entry_price"] = str_to_bool(
            env_config.get("take_profit_use_first_entry_price"), True
        )
    config["window_size"] = args.window_size or (
        int(env_config["window_size"]) if env_config["window_size"] else None
    )
    config["leverage"] = args.leverage or (
        float(env_config["leverage"]) if env_config["leverage"] else None
    )
    config["take_profit_pct"] = args.take_profit_pct or (
        float(env_config["take_profit_pct"]) if env_config["take_profit_pct"] else None
    )
    config["doji_size"] = (
        args.doji_size
        if args.doji_size is not None
        else float(env_config.get("doji_size", "0.05"))
    )
    config["stop_loss_mode"] = args.stop_loss_mode or env_config.get(
        "stop_loss_mode", "percent"
    )
    config["stop_loss_pct"] = (
        args.stop_loss_pct
        if args.stop_loss_pct is not None
        else float(env_config.get("stop_loss_pct", "0.005"))
    )
    config["exchange_fee_pct"] = (
        args.exchange_fee_pct
        if args.exchange_fee_pct is not None
        else float(env_config.get("exchange_fee_pct", "0.0004"))
    )
    if args.skip_wick_filter is not None:
        config["skip_wick_filter"] = args.skip_wick_filter
    else:
        config["skip_wick_filter"] = str_to_bool(
            env_config.get("skip_wick_filter"), False
        )
    if args.skip_bollinger_cross is not None:
        config["skip_bollinger_cross"] = args.skip_bollinger_cross
    else:
        config["skip_bollinger_cross"] = str_to_bool(
            env_config.get("skip_bollinger_cross"), False
        )
    config["bollinger_period"] = args.bollinger_period or int(
        env_config.get("bollinger_period", "20")
    )
    config["bollinger_stddev"] = args.bollinger_stddev or float(
        env_config.get("bollinger_stddev", "2.0")
    )
    if args.volume_filter is not None:
        config["volume_filter_enabled"] = args.volume_filter
    else:
        config["volume_filter_enabled"] = str_to_bool(
            env_config.get("volume_filter_enabled"), True
        )
    if args.stoch_enabled is not None:
        config["stoch_enabled"] = args.stoch_enabled
    else:
        config["stoch_enabled"] = str_to_bool(env_config.get("stoch_enabled"), True)
    config["stoch_first_line"] = args.stoch_first_line or env_config.get(
        "stoch_first_line", "k"
    )
    config["stoch_first_period"] = args.stoch_first_period or int(
        env_config.get("stoch_first_period", "20")
    )
    config["stoch_first_threshold"] = (
        args.stoch_first_threshold
        if args.stoch_first_threshold is not None
        else (
            float(env_config["stoch_first_threshold"])
            if env_config.get("stoch_first_threshold")
            else None
        )
    )
    config["stoch_second_line"] = args.stoch_second_line or env_config.get(
        "stoch_second_line", "k"
    )
    config["stoch_second_period"] = args.stoch_second_period or int(
        env_config.get("stoch_second_period", "100")
    )
    config["stoch_second_threshold"] = (
        args.stoch_second_threshold
        if args.stoch_second_threshold is not None
        else (
            float(env_config["stoch_second_threshold"])
            if env_config.get("stoch_second_threshold")
            else None
        )
    )
    config["stoch_comparison"] = args.stoch_comparison or env_config.get(
        "stoch_comparison", "gt"
    )
    config["stoch_d_smoothing"] = args.stoch_d_smoothing or int(
        env_config.get("stoch_d_smoothing", "3")
    )
    config["equity_risk_pct"] = (
        args.equity_risk_pct
        if args.equity_risk_pct is not None
        else float(env_config.get("equity_risk_pct", "3"))
    )
    config["atr_multiple"] = (
        args.atr_multiple
        if args.atr_multiple is not None
        else float(env_config.get("atr_multiple", "0.5"))
    )
    config["trail_points"] = (
        args.trail_points
        if args.trail_points is not None
        else float(env_config.get("trail_points", "1"))
    )
    config["trail_offset"] = (
        args.trail_offset
        if args.trail_offset is not None
        else float(env_config.get("trail_offset", "1"))
    )
    config["slow_sma_period"] = (
        args.slow_sma_period
        if args.slow_sma_period is not None
        else int(env_config.get("slow_sma_period", "50"))
    )
    config["medium_ema_period"] = (
        args.medium_ema_period
        if args.medium_ema_period is not None
        else int(env_config.get("medium_ema_period", "18"))
    )
    config["fast_ema_period"] = (
        args.fast_ema_period
        if args.fast_ema_period is not None
        else int(env_config.get("fast_ema_period", "6"))
    )
    config["atr_period"] = (
        args.atr_period
        if args.atr_period is not None
        else int(env_config.get("atr_period", "14"))
    )
    config["entry_cancel_bars"] = (
        args.entry_cancel_bars
        if args.entry_cancel_bars is not None
        else int(env_config.get("entry_cancel_bars", "3"))
    )
    config["entry_activation_mode"] = (
        args.entry_activation_mode
        or env_config.get("entry_activation_mode", "next_bar")
    )
    config["trailing_tick_timeframe"] = (
        args.trailing_tick_timeframe
        or env_config.get("trailing_tick_timeframe", "15m")
    )
    if args.use_trailing_tick_emulation is not None:
        config["use_trailing_tick_emulation"] = args.use_trailing_tick_emulation
    else:
        config["use_trailing_tick_emulation"] = str_to_bool(
            env_config.get("use_trailing_tick_emulation"), False
        )
    if args.use_stop_fill_open_gap is not None:
        config["use_stop_fill_open_gap"] = args.use_stop_fill_open_gap
    else:
        config["use_stop_fill_open_gap"] = str_to_bool(
            env_config.get("use_stop_fill_open_gap"), True
        )
    if args.enable_friday_close is not None:
        config["enable_friday_close"] = args.enable_friday_close
    else:
        config["enable_friday_close"] = str_to_bool(
            env_config.get("enable_friday_close"), True
        )
    config["friday_close_hour_utc"] = (
        args.friday_close_hour_utc
        if args.friday_close_hour_utc is not None
        else int(env_config.get("friday_close_hour_utc", "16"))
    )
    if args.enable_ema_cross_close is not None:
        config["enable_ema_cross_close"] = args.enable_ema_cross_close
    else:
        config["enable_ema_cross_close"] = str_to_bool(
            env_config.get("enable_ema_cross_close"), True
        )
    if args.risk_equity_include_unrealized is not None:
        config["risk_equity_include_unrealized"] = (
            args.risk_equity_include_unrealized
        )
    else:
        config["risk_equity_include_unrealized"] = str_to_bool(
            env_config.get("risk_equity_include_unrealized"), True
        )
    config["risk_equity_mark_source"] = (
        args.risk_equity_mark_source
        or env_config.get("risk_equity_mark_source", "close")
    )

    # Backtest config
    config["start"] = args.start or (
        parse_datetime(env_config["start"]) if env_config["start"] else None
    )
    config["end"] = args.end or (
        parse_datetime(env_config["end"]) if env_config["end"] else None
    )
    config["initial_capital"] = args.initial_capital or (
        float(env_config["initial_capital"]) if env_config["initial_capital"] else None
    )
    config["override_download"] = args.override_download or (
        env_config.get("override_download", "false").lower() == "true"
    )
    config["warmup_days"] = args.warmup_days or (
        int(env_config["warmup_days"]) if env_config["warmup_days"] else 0
    )

    # Storage config
    config["store_kind"] = "postgres"
    env_store_path = env_config.get("store_path", "")
    config["store_path"] = args.store_path or (Path(env_store_path) if env_store_path else None)

    # Network config
    config["http_proxy"] = args.http_proxy or env_config.get("http_proxy") or None
    config["https_proxy"] = args.https_proxy or env_config.get("https_proxy") or None
    config["proxy"] = args.proxy or env_config.get("proxy") or None

    # Output config
    config["stats_output"] = args.stats_output or Path(
        env_config.get("stats_output", "./backtest_stats.json")
    )
    config["plot_output"] = args.plot_output or (
        Path(env_config["plot_output"]) if env_config.get("plot_output") else None
    )
    config["show_plot"] = args.show_plot or (
        env_config.get("show_plot", "false").lower() == "true"
    )
    config["show_stochastic"] = not args.no_stochastic
    config["show_equity"] = not args.no_equity

    # Logging config
    config["log_level"] = args.log_level or env_config.get("log_level", "INFO")

    return config


def validate_config(config: Dict[str, object]) -> None:
    """Validate that all required configuration is present."""
    strategy = str(config.get("strategy", "engulfing"))
    common_required = ["start", "end", "initial_capital"]
    if strategy == "engulfing":
        strategy_required = ["symbol", "timeframe", "window_size", "leverage", "take_profit_pct"]
    elif strategy == "pinbar_magic_v2":
        strategy_required = ["symbol", "timeframe", "leverage"]
    elif strategy == "stochastic_fsm":
        strategy_required = ["symbols", "base_timeframe", "higher_timeframe"]
    else:
        raise ValueError(
            "Unsupported strategy "
            f"'{strategy}'. Expected one of: engulfing, pinbar_magic_v2, stochastic_fsm"
        )

    required = common_required + strategy_required
    missing = [key for key in required if config.get(key) is None]
    if strategy == "stochastic_fsm" and not config.get("symbols"):
        missing.append("symbols")
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")


def resolve_proxies(config: Dict[str, object]) -> Dict[str, str] | None:
    """Resolve proxy configuration."""
    proxies: Dict[str, str] = {}
    if config.get("proxy"):
        proxies["http"] = str(config["proxy"])
        proxies["https"] = str(config["proxy"])
    if config.get("http_proxy"):
        proxies["http"] = str(config["http_proxy"])
    if config.get("https_proxy"):
        proxies["https"] = str(config["https_proxy"])
    return proxies if proxies else None


def write_stats(report, output_path: Path) -> None:
    """Write comprehensive statistics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report.as_dict(), f, indent=2, default=str)
    logging.info(f"Statistics written to {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for backtest runner."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Load and resolve configuration
    env_config = load_env_config()
    config = resolve_config(args, env_config)
    validate_config(config)

    # Resolve proxies
    proxies = resolve_proxies(config)

    # Build components
    store = build_store(str(config["store_kind"]), config["store_path"])  # type: ignore[arg-type]

    client_logger = logging.getLogger("candle_downloader.binance")
    client = BinanceClient(
        BinanceClientConfig(proxies=proxies or None), logger=client_logger
    )
    downloader = CandleDownloader(client=client, store=store)

    strategy_name = str(config["strategy"])
    if strategy_name == "engulfing":
        strategy = EngulfingStrategy(
            EngulfingStrategyConfig(
                symbol=str(config["symbol"]),
                timeframe=str(config["timeframe"]),
                window_size=int(config["window_size"]),
                leverage=float(config["leverage"]),
                take_profit_pct=float(config["take_profit_pct"]),
                doji_size=float(config["doji_size"]),
                stop_loss_mode=StopLossMode(str(config["stop_loss_mode"])),
                stop_loss_pct=float(config["stop_loss_pct"]),
                skip_large_upper_wick=bool(config["skip_wick_filter"]),
                skip_bollinger_cross=bool(config["skip_bollinger_cross"]),
                bollinger_period=int(config["bollinger_period"]),
                bollinger_stddev=float(config["bollinger_stddev"]),
                enable_volume_pressure_filter=bool(config["volume_filter_enabled"]),
                enable_stochastic_filter=bool(config["stoch_enabled"]),
                stochastic_first_line=str(config["stoch_first_line"]),
                stochastic_first_period=int(config["stoch_first_period"]),
                stochastic_first_threshold=(
                    float(config["stoch_first_threshold"])
                    if config["stoch_first_threshold"] is not None
                    else None
                ),
                stochastic_second_line=str(config["stoch_second_line"]),
                stochastic_second_period=int(config["stoch_second_period"]),
                stochastic_second_threshold=(
                    float(config["stoch_second_threshold"])
                    if config["stoch_second_threshold"] is not None
                    else None
                ),
                stochastic_comparison=str(config["stoch_comparison"]),
                stochastic_d_smoothing=int(config["stoch_d_smoothing"]),
                exchange_fee_pct=float(config["exchange_fee_pct"]),
            )
        )
    elif strategy_name == "pinbar_magic_v2":
        strategy = PinBarMagicStrategyV2(
            PinBarMagicStrategyConfigV2(
                symbol=str(config["symbol"]),
                timeframe=str(config["timeframe"]),
                leverage=float(config["leverage"]),
                equity_risk_pct=float(config["equity_risk_pct"]),
                atr_multiple=float(config["atr_multiple"]),
                trail_points=float(config["trail_points"]),
                trail_offset=float(config["trail_offset"]),
                slow_sma_period=int(config["slow_sma_period"]),
                medium_ema_period=int(config["medium_ema_period"]),
                fast_ema_period=int(config["fast_ema_period"]),
                atr_period=int(config["atr_period"]),
                entry_cancel_bars=int(config["entry_cancel_bars"]),
                trailing_tick_timeframe=str(config["trailing_tick_timeframe"]).strip(),
                use_trailing_tick_emulation=bool(config["use_trailing_tick_emulation"]),
                use_stop_fill_open_gap=bool(config["use_stop_fill_open_gap"]),
                entry_activation_mode=str(config["entry_activation_mode"]).strip().lower(),
                enable_friday_close=bool(config["enable_friday_close"]),
                friday_close_hour_utc=int(config["friday_close_hour_utc"]),
                enable_ema_cross_close=bool(config["enable_ema_cross_close"]),
                risk_equity_include_unrealized=bool(
                    config["risk_equity_include_unrealized"]
                ),
                risk_equity_mark_source=str(config["risk_equity_mark_source"])
                .strip()
                .lower(),
            )
        )
    elif strategy_name == "stochastic_fsm":
        strategy = StochasticRsiFsmStrategy(
            StochasticRsiFsmConfig(
                symbols=list(config["symbols"]),  # type: ignore[arg-type]
                tf_1=str(config["base_timeframe"]),
                tf_2=str(config["higher_timeframe"]),
                tf_3=(
                    str(config["higher_timeframe_2"])
                    if config.get("higher_timeframe_2")
                    else None
                ),
                k_period=int(config["k_period"]),
                k_slowing=int(config["k_slowing"]),
                d_period=int(config["d_period"]),
                use_d_line=bool(config["use_d_line"]),
                oversold=float(config["oversold"]),
                overbought=float(config["overbought"]),
                initial_order_usdt=float(config["initial_order_usdt"]),
                initial_leverage=float(config["initial_leverage"]),
                martingale_multiplier=float(config["martingale_multiplier"]),
                martingale_multipliers=tuple(config["martingale_multipliers"]),  # type: ignore[arg-type]
                martingale_leverages=tuple(config["martingale_leverages"]),  # type: ignore[arg-type]
                max_concurrent_positions=int(config["max_positions"]),
                take_profit_pct=float(config["take_profit_pct"] or 0.02),
                slippage_pct=float(config["slippage_pct"]),
                maker_fee_pct=float(config["maker_fee_pct"]),
                taker_fee_pct=float(config["taker_fee_pct"]),
                funding_rate_per_day_pct=float(config["funding_rate_pct"]),
                trailing_activation_pct=float(config["trailing_activation_pct"]),
                trailing_gap_pct=float(config["trailing_gap_pct"]),
                trailing_check_interval_seconds=float(config["trailing_interval"]),
                max_position_days=float(config["max_position_days"]),
                aligned_high_stoch_mode=str(config["aligned_high_stoch_mode"]),
                signal_offset=int(config["signal_offset"]),
                enable_take_profit_check=bool(config["enable_take_profit_check"]),
                enable_high_exit_cross=bool(config["enable_high_exit_cross"]),
                use_midsold_filter=bool(config["use_midsold_filter"]),
                enable_reversal_logic=bool(config["enable_reversal_logic"]),
                enable_reversal_reentry=bool(config["enable_reversal_reentry"]),
                enable_grid_martingales=bool(config["enable_grid_martingales"]),
                grid_martingales_percent=float(config["grid_martingales_percent"]),
                trailing_use_first_entry_price=bool(
                    config["trailing_use_first_entry_price"]
                ),
                trailing_use_close_for_stop_activation=bool(
                    config["trailing_use_close_for_stop_activation"]
                ),
                take_profit_use_first_entry_price=bool(
                    config["take_profit_use_first_entry_price"]
                ),
            )
        )
    else:  # pragma: no cover - guarded by parser/validation
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    # Create backtester
    backtester = BaseBacktester(strategy=strategy, downloader=downloader, store=store)

    # Run backtest
    backtest_config = BacktestRunConfig(
        start=config["start"],  # type: ignore
        end=config["end"],  # type: ignore
        initial_capital=float(config["initial_capital"]),
        override_download=bool(config["override_download"]),
        warmup_days=int(config["warmup_days"]),
    )

    logging.info("Starting backtest...")
    report = backtester.run(backtest_config)

    # Write statistics
    stats_output: Path = Path(str(config["stats_output"]))
    write_stats(report, stats_output)

    # Print summary
    stats = report.statistics
    logging.info("Backtest completed:")
    logging.info(f"  Total Trades: {stats.total_trades}")
    logging.info(f"  Win Rate: {stats.win_rate * 100:.2f}%")
    logging.info(f"  Net P&L: {stats.net_profit:+.2f} ({stats.net_profit_pct:+.2f}%)")
    logging.info(f"  Sharpe Ratio: {stats.sharpe_ratio:.2f}")
    logging.info(f"  Max Drawdown: {stats.max_drawdown_pct:.2f}%")
    logging.info(f"  CAGR: {stats.cagr_pct:.2f}%")

    # Plot results
    try:
        if strategy_name == "stochastic_fsm":
            logging.info(
                "Skipping plot for stochastic_fsm because current plotter expects a single symbol/timeframe."
            )
            fig = None
        else:
            fig = plot_backtest_from_store(
                report=report,
                store=store,
                symbol=str(config["symbol"]),
                timeframe=str(config["timeframe"]),
                show_stochastic=bool(config["show_stochastic"]),
                show_equity=bool(config["show_equity"]),
                initial_candles=150,  # Show last 150 candles initially for performance
            )

        if fig is not None and config.get("plot_output"):
            plot_output: Path = config["plot_output"]  # type: ignore
            plot_output.parent.mkdir(parents=True, exist_ok=True)
            ext = plot_output.suffix.lower()
            if ext == ".html":
                save_plot(fig, str(plot_output), format="html")
            elif ext in (".png", ".jpg", ".jpeg"):
                save_plot(fig, str(plot_output), format="png")
            elif ext == ".svg":
                save_plot(fig, str(plot_output), format="svg")
            elif ext == ".pdf":
                save_plot(fig, str(plot_output), format="pdf")
            else:
                save_plot(fig, str(plot_output), format="html")
            logging.info(f"Plot saved to {plot_output}")

        if fig is not None and config.get("show_plot"):
            logging.info("Displaying plot in browser...")
            show_plot(fig)

    except ImportError:
        logging.warning("Plotly not available. Install with: pip install plotly")
    except Exception as exc:
        logging.error(f"Failed to create plot: {exc}", exc_info=True)

    # Cleanup
    store.close()
    client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
