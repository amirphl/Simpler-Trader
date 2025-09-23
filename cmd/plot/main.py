from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backtest import plot_backtest_from_store, save_plot, show_plot
from backtest.base import (
    BacktestReport,
    BacktestRunConfig,
    BacktestStatistics,
    TradePerformance,
)
from candle_downloader.storage import build_store


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot backtest results from cached statistics JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot latest backtest results
  python -m cmd.plot.main --symbol ETHUSDT --timeframe 15m

  # Plot specific stats file
  python -m cmd.plot.main --stats-file ./results/eth_15m_stats.json --symbol ETHUSDT --timeframe 15m

  # Save plot without showing
  python -m cmd.plot.main --symbol BTCUSDT --timeframe 1h --plot-output ./plot.html

  # Hide subplots
  python -m cmd.plot.main --symbol ETHUSDT --timeframe 15m --no-stochastic --no-equity
        """,
    )

    parser.add_argument(
        "--stats-file",
        type=Path,
        help="Path to statistics JSON file (default: latest in ./results)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory to search for latest stats file (default: ./results)",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        required=True,
        help="Binance interval (e.g., 1h, 15m)",
    )
    parser.add_argument(
        "--store-kind",
        choices=("sqlite", "csv"),
        default="sqlite",
        help="Storage type (default: sqlite)",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        help="Path to storage file (default: ./data/candles.db or inferred from stats filename)",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Path to save plot (HTML/PNG/SVG/PDF). If not provided, plot is only displayed.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        default=True,
        help="Display plot in browser (default: True)",
    )
    parser.add_argument(
        "--no-show-plot",
        dest="show_plot",
        action="store_false",
        help="Don't display plot in browser",
    )
    parser.add_argument(
        "--no-stochastic",
        action="store_true",
        help="Hide stochastic oscillator subplot",
    )
    parser.add_argument(
        "--no-equity",
        action="store_true",
        help="Hide equity curve subplot",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Chart height in pixels (default: 900)",
    )
    parser.add_argument(
        "--initial-candles",
        type=int,
        default=150,
        help="Number of candles to show initially (default: 150). Use 0 to show all (may freeze browser with large datasets)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )

    return parser


def load_report_from_json(filepath: Path) -> BacktestReport:
    """Load a BacktestReport from a JSON file.

    Args:
        filepath: Path to the JSON file containing backtest results

    Returns:
        Reconstructed BacktestReport object

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON structure is invalid
    """
    import json
    from datetime import datetime

    if not filepath.exists():
        raise FileNotFoundError(f"Report file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct config
    config_data = data.get("config", {})
    start_raw = config_data.get("start")
    end_raw = config_data.get("end")
    if not start_raw or not end_raw:
        raise ValueError("Invalid report JSON: config.start and config.end are required")
    config = BacktestRunConfig(
        start=datetime.fromisoformat(str(start_raw)),
        end=datetime.fromisoformat(str(end_raw)),
        initial_capital=float(config_data.get("initial_capital", 10_000.0)),
        override_download=bool(config_data.get("override_download", False)),
        max_batch=int(config_data.get("max_batch", 1000)),
        risk_free_rate=float(config_data.get("risk_free_rate", 0.0)),
        warmup_days=int(config_data.get("warmup_days", 30)),
    )

    # Reconstruct statistics
    stats_data = data.get("statistics", {})
    statistics = BacktestStatistics(
        total_trades=int(stats_data.get("total_trades", 0)),
        winning_trades=int(stats_data.get("winning_trades", 0)),
        losing_trades=int(stats_data.get("losing_trades", 0)),
        win_rate=float(stats_data.get("win_rate", 0.0)),
        gross_profit=float(stats_data.get("gross_profit", 0.0)),
        gross_loss=float(stats_data.get("gross_loss", 0.0)),
        net_profit=float(stats_data.get("net_profit", 0.0)),
        net_profit_pct=float(stats_data.get("net_profit_pct", 0.0)),
        average_return_pct=float(stats_data.get("average_return_pct", 0.0)),
        profit_factor=float(stats_data.get("profit_factor", 0.0)),
        expectancy=float(stats_data.get("expectancy", 0.0)),
        max_drawdown_pct=float(stats_data.get("max_drawdown_pct", 0.0)),
        sharpe_ratio=float(stats_data.get("sharpe_ratio", 0.0)),
        cagr_pct=float(stats_data.get("cagr_pct", 0.0)),
        average_trade_duration_sec=float(stats_data.get("average_trade_duration_sec", 0.0)),
        equity_curve=list(stats_data.get("equity_curve", [])),
        extra={
            k: v
            for k, v in stats_data.items()
            if k
            not in {
                "total_trades",
                "winning_trades",
                "losing_trades",
                "win_rate",
                "gross_profit",
                "gross_loss",
                "net_profit",
                "net_profit_pct",
                "average_return_pct",
                "profit_factor",
                "expectancy",
                "max_drawdown_pct",
                "sharpe_ratio",
                "cagr_pct",
                "average_trade_duration_sec",
                "equity_curve",
            }
        },
    )

    # Reconstruct trades
    trades_data = data.get("trades", [])
    trades: list[TradePerformance] = []
    for trade_data in trades_data:
        entry_time_raw = trade_data.get("entry_time")
        exit_time_raw = trade_data.get("exit_time")
        if not entry_time_raw or not exit_time_raw:
            raise ValueError("Invalid report JSON: trade entry_time and exit_time are required")
        trade = TradePerformance(
            entry_time=datetime.fromisoformat(str(entry_time_raw)),
            exit_time=datetime.fromisoformat(str(exit_time_raw)),
            pnl=float(trade_data["pnl"]),
            return_pct=float(trade_data["return_pct"]),
            notes=trade_data.get("notes"),
            metadata=trade_data.get("metadata"),
        )
        trades.append(trade)

    # Reconstruct report
    return BacktestReport(
        strategy_name=str(data.get("strategy", "Unknown")),
        config=config,
        statistics=statistics,
        trades=tuple(trades),
    )


def find_latest_stats_file(directory: Path = Path("./results")) -> Path | None:
    """Find the most recently modified stats JSON file in a directory.

    Args:
        directory: Directory to search in (default: ./results)

    Returns:
        Path to the latest stats file, or None if no files found
    """
    if not directory.exists():
        return None

    stats_files = list(directory.glob("*_stats.json")) + list(directory.glob("stats.json"))
    if not stats_files:
        return None

    # Return the most recently modified file
    return max(stats_files, key=lambda p: p.stat().st_mtime)


def infer_store_path(stats_file: Path, store_kind: str) -> Path:
    """Infer store path from stats filename.

    Args:
        stats_file: Path to stats JSON file
        store_kind: Storage type (sqlite or csv)

    Returns:
        Inferred store path
    """
    # Try to extract symbol/timeframe from filename
    # e.g., eth_15m_stats.json -> ./data/eth_15m_candles.db
    stem = stats_file.stem.replace("_stats", "")
    if store_kind == "sqlite":
        return Path(f"./data/{stem}_candles.db")
    else:
        return Path(f"./data/{stem}_candles.csv")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for plot generator."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Find stats file
    if args.stats_file:
        stats_file = args.stats_file
        if not stats_file.exists():
            logging.error("Stats file not found: %s", stats_file)
            return 1
    else:
        stats_file = find_latest_stats_file(args.results_dir)
        if stats_file is None:
            logging.error("No stats files found in %s", args.results_dir)
            logging.error("Run a backtest first or specify --stats-file")
            return 1
        logging.info("Using latest stats file: %s", stats_file)

    # Load report
    try:
        logging.info("Loading report from %s", stats_file)
        report = load_report_from_json(stats_file)
    except Exception as exc:
        logging.error("Failed to load report: %s", exc, exc_info=True)
        return 1

    # Determine store path
    if args.store_path:
        store_path = args.store_path
    else:
        store_path = infer_store_path(stats_file, args.store_kind)
        logging.info("Inferred store path: %s", store_path)

    # Build store
    try:
        store = build_store(args.store_kind, store_path)
    except Exception as exc:
        logging.error("Failed to open store at %s: %s", store_path, exc)
        logging.error("Make sure the store file exists and matches the stats file")
        return 1

    # Generate plot
    try:
        logging.info("Generating plot...")
        initial_candles: int | None = args.initial_candles if args.initial_candles > 0 else None
        fig = plot_backtest_from_store(
            report=report,
            store=store,
            symbol=args.symbol,
            timeframe=args.timeframe,
            show_stochastic=not args.no_stochastic,
            show_equity=not args.no_equity,
            height=args.height,
            initial_candles=initial_candles,
        )

        # Save plot if requested
        if args.plot_output:
            args.plot_output.parent.mkdir(parents=True, exist_ok=True)
            ext = args.plot_output.suffix.lower()
            if ext == ".html":
                save_plot(fig, str(args.plot_output), format="html")
            elif ext in (".png", ".jpg", ".jpeg"):
                save_plot(fig, str(args.plot_output), format="png")
            elif ext == ".svg":
                save_plot(fig, str(args.plot_output), format="svg")
            elif ext == ".pdf":
                save_plot(fig, str(args.plot_output), format="pdf")
            else:
                save_plot(fig, str(args.plot_output), format="html")
            logging.info("Plot saved to %s", args.plot_output)

        # Show plot if requested
        if args.show_plot:
            logging.info("Displaying plot in browser...")
            show_plot(fig)

        logging.info("Plot generation completed successfully")

    except ImportError:
        logging.error("Plotly is required for plotting. Install with: pip install plotly")
        return 1
    except Exception as exc:
        logging.error("Failed to generate plot: %s", exc, exc_info=True)
        return 1
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
