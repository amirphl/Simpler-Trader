from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from backtest import (
    BacktestRunConfig,
    BaseBacktester,
    EngulfingStrategy,
    EngulfingStrategyConfig,
    plot_backtest_from_store,
    save_plot,
    show_plot,
)
from candle_downloader.binance import BinanceClient, BinanceClientConfig
from candle_downloader.downloader import CandleDownloader
from candle_downloader.storage import build_store

from backtest.scalping_FVG_strategy import ScalpingFVGStrategy, ScalpingFVGStrategyConfig


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
        "symbol": os.getenv("STRATEGY_SYMBOL", ""),
        "timeframe": os.getenv("STRATEGY_TIMEFRAME", ""),
        "window_size": os.getenv("STRATEGY_WINDOW_SIZE", ""),
        "leverage": os.getenv("STRATEGY_LEVERAGE", ""),
        "take_profit_pct": os.getenv("STRATEGY_TAKE_PROFIT_PCT", ""),
        "doji_size": os.getenv("STRATEGY_DOJI_SIZE", "0.05"),

        "start": os.getenv("BACKTEST_START", ""),
        "end": os.getenv("BACKTEST_END", ""),
        "initial_capital": os.getenv("BACKTEST_INITIAL_CAPITAL", ""),
        "override_download": os.getenv("OVERRIDE_DOWNLOAD", "false"),

        "store_kind": os.getenv("STORE_KIND", "sqlite"),
        "store_path": os.getenv("STORE_PATH", "./data/candles.db"),

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
  BACKTEST_START              Start datetime (ISO8601, UTC)
  BACKTEST_END                End datetime (ISO8601, UTC)
  BACKTEST_INITIAL_CAPITAL    Starting capital
  STORE_KIND                  Storage type: sqlite or csv
  STORE_PATH                  Path to storage file
  HTTP_PROXY                  HTTP proxy URL
  HTTPS_PROXY                 HTTPS proxy URL
  PROXY                       Shortcut for both HTTP and HTTPS proxy
  STATS_OUTPUT                Path to write statistics JSON
  PLOT_OUTPUT                 Path to save plot (HTML/PNG/SVG/PDF)
  SHOW_PLOT                   Set to 'true' to display plot in browser
        """,
    )

    # Strategy parameters
    parser.add_argument("--symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--timeframe", help="Binance interval (e.g., 1h, 4h)")
    parser.add_argument("--window-size", type=int, help="Number of candles to check for bearish pattern")
    parser.add_argument("--leverage", type=float, help="Leverage multiplier")
    parser.add_argument("--take-profit-pct", type=float, help="Take profit percentage (e.g., 0.02 for 2%%)")
    parser.add_argument("--doji-size", type=float, default=0.05, help="Doji size for pattern detection")

    # Backtest parameters
    parser.add_argument("--start", type=parse_datetime, help="Start datetime (ISO8601, UTC)")
    parser.add_argument("--end", type=parse_datetime, help="End datetime (ISO8601, UTC)")
    parser.add_argument("--initial-capital", type=float, help="Starting capital")
    parser.add_argument("--override-download", action="store_true", help="Re-download all candles")

    # Storage parameters
    parser.add_argument("--store-kind", choices=("sqlite", "csv"), help="Storage type")
    parser.add_argument("--store-path", type=Path, help="Path to storage file")

    # Network parameters
    parser.add_argument("--http-proxy", help="HTTP proxy URL")
    parser.add_argument("--https-proxy", help="HTTPS proxy URL")
    parser.add_argument("--proxy", help="Shortcut for both HTTP and HTTPS proxy")

    # Output parameters
    parser.add_argument("--stats-output", type=Path, help="Path to write statistics JSON")
    parser.add_argument("--plot-output", type=Path, help="Path to save plot (HTML/PNG/SVG/PDF)")
    parser.add_argument("--show-plot", action="store_true", help="Display plot in browser")
    parser.add_argument("--no-stochastic", action="store_true", help="Hide stochastic oscillator subplot")
    parser.add_argument("--no-equity", action="store_true", help="Hide equity curve subplot")

    parser.add_argument("--log-level", default="INFO", help="Logging level")

    return parser


def resolve_config(args: argparse.Namespace, env_config: Dict[str, str]) -> Dict[str, object]:
    """Resolve configuration from CLI args and environment, with CLI taking precedence."""
    config: Dict[str, object] = {}

    # Strategy config
    config["symbol"] = args.symbol or env_config["symbol"] or None
    config["timeframe"] = args.timeframe or env_config["timeframe"] or None
    config["window_size"] = args.window_size or (int(env_config["window_size"]) if env_config["window_size"] else None)
    config["leverage"] = args.leverage or (float(env_config["leverage"]) if env_config["leverage"] else None)
    config["take_profit_pct"] = args.take_profit_pct or (
        float(env_config["take_profit_pct"]) if env_config["take_profit_pct"] else None
    )
    config["doji_size"] = args.doji_size or float(env_config.get("doji_size", "0.05"))

    # Backtest config
    config["start"] = args.start or (
        parse_datetime(env_config["start"]) if env_config["start"] else None
    )
    config["end"] = args.end or (parse_datetime(env_config["end"]) if env_config["end"] else None)
    config["initial_capital"] = args.initial_capital or (
        float(env_config["initial_capital"]) if env_config["initial_capital"] else None
    )
    config["override_download"] = args.override_download or (env_config.get("override_download", "false").lower() == "true")

    # Storage config
    config["store_kind"] = args.store_kind or env_config.get("store_kind", "sqlite")
    config["store_path"] = args.store_path or Path(env_config.get("store_path", "./data/candles.db"))

    # Network config
    config["http_proxy"] = args.http_proxy or env_config.get("http_proxy") or None
    config["https_proxy"] = args.https_proxy or env_config.get("https_proxy") or None
    config["proxy"] = args.proxy or env_config.get("proxy") or None

    # Output config
    config["stats_output"] = args.stats_output or Path(env_config.get("stats_output", "./backtest_stats.json"))
    config["plot_output"] = args.plot_output or (Path(env_config["plot_output"]) if env_config.get("plot_output") else None)
    config["show_plot"] = args.show_plot or (env_config.get("show_plot", "false").lower() == "true")
    config["show_stochastic"] = not args.no_stochastic
    config["show_equity"] = not args.no_equity

    # Logging config
    config["log_level"] = args.log_level or env_config.get("log_level", "INFO")

    return config


def validate_config(config: Dict[str, object]) -> None:
    """Validate that all required configuration is present."""
    required = ["symbol", "timeframe", "window_size", "leverage", "take_profit_pct", "start", "end", "initial_capital"]
    missing = [key for key in required if config.get(key) is None]
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
    store_path: Path = Path(config["store_path"])  # type: ignore
    store = build_store(str(config["store_kind"]), store_path)

    client_logger = logging.getLogger("candle_downloader.binance")
    client = BinanceClient(BinanceClientConfig(proxies=proxies or None), logger=client_logger)
    downloader = CandleDownloader(client=client, store=store)

    # Create strategy
    strategy = EngulfingStrategy(
        EngulfingStrategyConfig(
            symbol=str(config["symbol"]),
            timeframe=str(config["timeframe"]),
            window_size=int(config["window_size"]),
            leverage=float(config["leverage"]),
            take_profit_pct=float(config["take_profit_pct"]),
            doji_size=float(config["doji_size"]),
        )
    )

    # strategy = ScalpingFVGStrategy(
    #     ScalpingFVGStrategyConfig(
    #         symbol=str(config["symbol"]),
    #         timeframe=str(config["timeframe"]),
    #         leverage=float(config["leverage"]),
    #         atr_tp_mult=1.0,
    #         atr_sl_mult=0.7,
    #         risk_per_trade_pct=0.01,
    #     )
    # )

    # Create backtester
    backtester = BaseBacktester(strategy=strategy, downloader=downloader, store=store)

    # Run backtest
    backtest_config = BacktestRunConfig(
        start=config["start"],  # type: ignore
        end=config["end"],  # type: ignore
        initial_capital=float(config["initial_capital"]),
        override_download=bool(config["override_download"]),
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
    logging.info(f"  Win Rate: {stats.win_rate*100:.2f}%")
    logging.info(f"  Net P&L: {stats.net_profit:+.2f} ({stats.net_profit_pct:+.2f}%)")
    logging.info(f"  Sharpe Ratio: {stats.sharpe_ratio:.2f}")
    logging.info(f"  Max Drawdown: {stats.max_drawdown_pct:.2f}%")
    logging.info(f"  CAGR: {stats.cagr_pct:.2f}%")

    # Plot results
    try:
        fig = plot_backtest_from_store(
            report=report,
            store=store,
            symbol=str(config["symbol"]),
            timeframe=str(config["timeframe"]),
            show_stochastic=bool(config["show_stochastic"]),
            show_equity=bool(config["show_equity"]),
            initial_candles=150,  # Show last 150 candles initially for performance
        )

        if config.get("plot_output"):
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

        if config.get("show_plot"):
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

