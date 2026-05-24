from __future__ import annotations

from datetime import datetime
from typing import Any, List, Sequence

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore
    make_subplots = None  # type: ignore

from candle_downloader.models import Candle

from .base import BacktestReport, BacktestRunConfig, TradePerformance

_DEFAULT_MARKER_COLORS = {
    "entry": "#22c55e",
    "exit_gain": "#22c55e",
    "exit_loss": "#ef4444",
    "volume_up": "#26a69a",
    "volume_down": "#ef5350",
    "stop_loss": "#ef4444",
    "take_profit": "#22c55e",
    "equity": "#8b5cf6",
}


def plot_backtest(
    report: BacktestReport,
    candles: Sequence[Candle],
    symbol: str | None = None,
    timeframe: str | None = None,
    show_stochastic: bool = True,
    show_equity: bool = True,
    height: int = 900,
    initial_candles: int | None = 150,
) -> "go.Figure":  # type: ignore
    """Create an interactive, scrollable chart showing backtest performance.

    Args:
        report: The backtest report containing trades and statistics
        candles: The candle data used in the backtest
        symbol: Symbol name (extracted from candles if not provided)
        timeframe: Timeframe (extracted from candles if not provided)
        show_stochastic: Whether to show stochastic oscillator subplot
        show_equity: Whether to show equity curve subplot
        height: Chart height in pixels
        initial_candles: Number of candles to show initially (default: 150).
                        Set to None to show all candles (may cause browser freeze with large datasets)

    Returns:
        Plotly figure object that can be displayed or saved

    Raises:
        ImportError: If plotly is not installed
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for plotting. Install with: pip install plotly"
        )
    if not candles:
        raise ValueError("candles sequence cannot be empty")
    if initial_candles is not None and initial_candles <= 0:
        raise ValueError("initial_candles must be positive when provided")

    # Extract symbol and timeframe from first candle if not provided
    if symbol is None:
        symbol = candles[0].symbol
    if timeframe is None:
        timeframe = candles[0].interval
    run_config = _resolve_report_config(report)

    # Prepare data
    times = [c.open_time for c in candles]
    opens = [c.open for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    # Calculate number of subplots
    num_subplots = 1  # Main candlestick chart
    row_heights = [0.6]  # Main chart takes 60% of height
    subplot_titles = [f"{symbol} - {timeframe}"]

    if show_stochastic:
        num_subplots += 1
        row_heights.append(0.2)
        subplot_titles.append("Stochastic %K")

    if show_equity:
        num_subplots += 1
        row_heights.append(0.2)
        subplot_titles.append("Equity Curve")

    # Create subplots
    fig = make_subplots(
        rows=num_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": True}]] * num_subplots,
    )

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # Volume bars
    colors = [
        _DEFAULT_MARKER_COLORS["volume_up"]
        if closes[i] >= opens[i]
        else _DEFAULT_MARKER_COLORS["volume_down"]
        for i in range(len(candles))
    ]
    fig.add_trace(
        go.Bar(
            x=times,
            y=volumes,
            name="Volume",
            marker_color=colors,
            opacity=0.3,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    trade_points = [
        _build_trade_plot_point(candles, trade, index)
        for index, trade in enumerate(report.trades)
    ]

    # Entry markers (green triangles)
    entry_points = [point for point in trade_points if point["entry_price"] is not None]
    if entry_points:
        fig.add_trace(
            go.Scatter(
                x=[point["trade"].entry_time for point in entry_points],
                y=[point["entry_price"] for point in entry_points],
                mode="markers+text",
                name="Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=_DEFAULT_MARKER_COLORS["entry"],
                    line=dict(width=2, color="darkgreen"),
                ),
                text=[f"E{point['index'] + 1}" for point in entry_points],
                textposition="top center",
                textfont=dict(size=10, color="green"),
                customdata=[
                    [
                        point["trade"].notes or "",
                        point["trade"].pnl,
                        point["trade"].return_pct,
                    ]
                    for point in entry_points
                ],
                hovertemplate=(
                    "<b>Entry</b><br>"
                    + "Time: %{x}<br>"
                    + "Price: %{y:.6f}<br>"
                    + "PnL: %{customdata[1]:+.2f}<br>"
                    + "Return: %{customdata[2]:+.2f}%<br>"
                    + "Notes: %{customdata[0]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Exit markers (red triangles)
    exit_points = [point for point in trade_points if point["exit_price"] is not None]
    if exit_points:
        exit_colors = [
            _DEFAULT_MARKER_COLORS["exit_loss"]
            if point["trade"].pnl < 0
            else _DEFAULT_MARKER_COLORS["exit_gain"]
            for point in exit_points
        ]
        fig.add_trace(
            go.Scatter(
                x=[point["trade"].exit_time for point in exit_points],
                y=[point["exit_price"] for point in exit_points],
                mode="markers+text",
                name="Exit",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=exit_colors,
                    line=dict(width=2, color="darkred"),
                ),
                text=[f"X{point['index'] + 1}" for point in exit_points],
                textposition="bottom center",
                textfont=dict(size=10, color="red"),
                customdata=[
                    [
                        point["trade"].notes or "",
                        point["trade"].pnl,
                        point["trade"].return_pct,
                    ]
                    for point in exit_points
                ],
                hovertemplate=(
                    "<b>Exit</b><br>"
                    + "Time: %{x}<br>"
                    + "Price: %{y:.6f}<br>"
                    + "PnL: %{customdata[1]:+.2f}<br>"
                    + "Return: %{customdata[2]:+.2f}%<br>"
                    + "Notes: %{customdata[0]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Stop loss and take profit lines (if available)
    _add_risk_lines(fig, trade_points)

    # Stochastic oscillator subplot
    stoch_row = 2 if show_stochastic else None
    if show_stochastic:
        stoch_k20, stoch_k100 = (
            _calculate_stochastic_series(candles, 20),
            _calculate_stochastic_series(candles, 100),
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=stoch_k20,
                mode="lines",
                name="Stoch K(20)",
                line=dict(color="blue", width=1.5),
            ),
            row=stoch_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=stoch_k100,
                mode="lines",
                name="Stoch K(100)",
                line=dict(color="orange", width=1.5),
            ),
            row=stoch_row,
            col=1,
        )
        # Add overbought/oversold levels
        fig.add_hline(
            y=80, line_dash="dash", line_color="gray", opacity=0.5, row=stoch_row, col=1
        )
        fig.add_hline(
            y=20, line_dash="dash", line_color="gray", opacity=0.5, row=stoch_row, col=1
        )

    # Equity curve subplot
    equity_row = num_subplots if show_equity else None
    if show_equity and report.statistics.equity_curve:
        equity_times = _build_equity_times(
            run_config,
            report.trades,
            report.statistics.equity_curve,
        )
        fig.add_trace(
            go.Scatter(
                x=equity_times,
                y=report.statistics.equity_curve,
                mode="lines+markers",
                name="Equity",
                line=dict(color=_DEFAULT_MARKER_COLORS["equity"], width=2),
                fill="tozeroy",
                fillcolor="rgba(128, 0, 128, 0.1)",
            ),
            row=equity_row,
            col=1,
        )

    # Update layout
    title_text = f"Backtest: {report.strategy_name} | {symbol} {timeframe}"
    title_text += f"<br><sub>Total Trades: {report.statistics.total_trades} | "
    title_text += f"Win Rate: {report.statistics.win_rate * 100:.1f}% | "
    title_text += f"Net P&L: {report.statistics.net_profit:+.2f} ({report.statistics.net_profit_pct:+.2f}%) | "
    title_text += f"Sharpe: {report.statistics.sharpe_ratio:.2f} | "
    title_text += f"Max DD: {report.statistics.max_drawdown_pct:.2f}%</sub>"

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        height=height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=num_subplots, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(
        title_text="Volume", row=1, col=1, secondary_y=True, showgrid=False
    )

    if show_stochastic:
        fig.update_yaxes(
            title_text="Stochastic %K", range=[0, 100], row=stoch_row, col=1
        )

    if show_equity:
        fig.update_yaxes(title_text="Equity", row=equity_row, col=1)

    # Set initial view to show only a subset of candles for performance
    # This prevents browser freeze with large datasets while keeping all data scrollable
    if initial_candles is not None and len(times) > initial_candles:
        # Show the last N candles initially
        start_idx = max(0, len(times) - initial_candles)
        initial_start = times[start_idx]
        initial_end = times[-1]

        # Add small padding based on visible range
        if initial_end > initial_start:
            visible_range_seconds = (initial_end - initial_start).total_seconds()
            padding_seconds = visible_range_seconds * 0.05  # 5% padding on each side
            initial_start = datetime.fromtimestamp(
                initial_start.timestamp() - padding_seconds, tz=initial_start.tzinfo
            )
            initial_end = datetime.fromtimestamp(
                initial_end.timestamp() + padding_seconds, tz=initial_end.tzinfo
            )
    else:
        # Show all candles if dataset is small or initial_candles is None
        initial_start = times[0] if times else run_config.start
        initial_end = times[-1] if times else run_config.end

    # Enable scrolling and zooming with initial view range
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            range=[initial_start, initial_end],
        ),
        dragmode="pan",
    )

    # Update all x-axes to have the same initial range for synchronized scrolling
    for row in range(1, num_subplots + 1):
        fig.update_xaxes(
            range=[initial_start, initial_end],
            row=row,
            col=1,
        )

    return fig


def _find_candle_at_time(
    candles: Sequence[Candle], target_time: datetime
) -> Candle | None:
    """Find the candle that contains the target time."""
    for candle in candles:
        if candle.open_time <= target_time < candle.close_time:
            return candle
    # Fallback: find closest candle
    if not candles:
        return None
    closest = min(
        candles, key=lambda c: abs((c.open_time - target_time).total_seconds())
    )
    return closest


def _resolve_report_config(report: BacktestReport) -> BacktestRunConfig:
    config = getattr(report, "config", None)
    if config is not None:
        return config

    legacy_config = getattr(report, "run_config", None)
    if legacy_config is not None:
        return legacy_config

    raise AttributeError("BacktestReport must expose either 'config' or 'run_config'")


def _extract_numeric_metadata(
    trade: TradePerformance, key: str
) -> float | None:
    if not trade.metadata or key not in trade.metadata:
        return None

    value = trade.metadata[key]
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _resolve_trade_price(
    candles: Sequence[Candle],
    trade: TradePerformance,
    *,
    metadata_key: str,
    timestamp: datetime,
    candle_attr: str,
) -> float | None:
    price = _extract_numeric_metadata(trade, metadata_key)
    if price is not None and price > 0:
        return price

    candle = _find_candle_at_time(candles, timestamp)
    if candle is None:
        return None
    return float(getattr(candle, candle_attr))


def _build_trade_plot_point(
    candles: Sequence[Candle], trade: TradePerformance, index: int
) -> dict[str, Any]:
    return {
        "index": index,
        "trade": trade,
        "entry_price": _resolve_trade_price(
            candles,
            trade,
            metadata_key="entry_price",
            timestamp=trade.entry_time,
            candle_attr="open",
        ),
        "exit_price": _resolve_trade_price(
            candles,
            trade,
            metadata_key="exit_price",
            timestamp=trade.exit_time,
            candle_attr="close",
        ),
        "stop_loss": _extract_numeric_metadata(trade, "stop_loss"),
        "take_profit": _extract_numeric_metadata(trade, "take_profit"),
    }


def _add_risk_lines(
    fig: "go.Figure", trade_points: Sequence[dict[str, Any]]
) -> None:  # type: ignore
    stop_loss_legend_used = False
    take_profit_legend_used = False

    for point in trade_points:
        trade = point["trade"]
        stop_loss = point["stop_loss"]
        take_profit = point["take_profit"]

        if stop_loss is not None and stop_loss > 0:
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time, trade.exit_time],
                    y=[stop_loss, stop_loss],
                    mode="lines",
                    name="Stop Loss" if not stop_loss_legend_used else None,
                    line=dict(
                        color=_DEFAULT_MARKER_COLORS["stop_loss"],
                        width=1,
                        dash="dash",
                    ),
                    showlegend=not stop_loss_legend_used,
                    hovertemplate=f"Stop Loss: {stop_loss:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            stop_loss_legend_used = True

        if take_profit is not None and take_profit > 0:
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time, trade.exit_time],
                    y=[take_profit, take_profit],
                    mode="lines",
                    name="Take Profit" if not take_profit_legend_used else None,
                    line=dict(
                        color=_DEFAULT_MARKER_COLORS["take_profit"],
                        width=1,
                        dash="dash",
                    ),
                    showlegend=not take_profit_legend_used,
                    hovertemplate=f"Take Profit: {take_profit:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            take_profit_legend_used = True


def _build_equity_times(
    run_config: BacktestRunConfig,
    trades: Sequence[TradePerformance],
    equity_curve: Sequence[float],
) -> List[datetime]:
    if not equity_curve:
        return []

    if len(equity_curve) == len(trades):
        return [trade.exit_time for trade in trades]

    if len(equity_curve) == len(trades) + 1:
        return [run_config.start] + [trade.exit_time for trade in trades]

    if not trades:
        return [run_config.start] * len(equity_curve)

    if len(equity_curve) < len(trades):
        return [trade.exit_time for trade in trades[: len(equity_curve)]]

    times = [run_config.start] + [trade.exit_time for trade in trades]
    last_time = trades[-1].exit_time
    times.extend([last_time] * (len(equity_curve) - len(times)))
    return times


def _calculate_stochastic_series(candles: Sequence[Candle], period: int) -> List[float]:
    """Calculate Stochastic %K series for all candles."""
    result: List[float] = []
    for i in range(len(candles)):
        if i < period - 1:
            result.append(50.0)  # Default neutral value
            continue
        window = candles[i - period + 1 : i + 1]
        highs = [c.high for c in window]
        lows = [c.low for c in window]
        closes = [c.close for c in window]
        highest_high = max(highs)
        lowest_low = min(lows)
        current_close = closes[-1]
        if highest_high == lowest_low:
            result.append(50.0)
        else:
            stoch = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0
            result.append(stoch)
    return result


def save_plot(fig: "go.Figure", filepath: str, format: str = "html") -> None:  # type: ignore
    """Save the plot to a file.

    Args:
        fig: Plotly figure object
        filepath: Path to save the file
        format: File format ('html', 'png', 'svg', 'pdf')
    """
    normalized_format = format.lower()
    if normalized_format == "html":
        fig.write_html(filepath)
    elif normalized_format == "png":
        fig.write_image(filepath, width=1920, height=1080)
    elif normalized_format == "svg":
        fig.write_image(filepath, format="svg", width=1920, height=1080)
    elif normalized_format == "pdf":
        fig.write_image(filepath, format="pdf", width=1920, height=1080)
    else:
        raise ValueError(
            f"Unsupported format: {format}. Use 'html', 'png', 'svg', or 'pdf'"
        )


def show_plot(fig: "go.Figure") -> None:  # type: ignore
    """Display the plot in a browser or notebook.

    Args:
        fig: Plotly figure object
    """
    fig.show()


def plot_backtest_from_store(
    report: BacktestReport,
    store: "CandleStore",  # type: ignore # noqa: F821
    symbol: str,
    timeframe: str,
    show_stochastic: bool = True,
    show_equity: bool = True,
    height: int = 900,
    initial_candles: int | None = 150,
) -> "go.Figure":  # type: ignore
    """Convenience function to plot backtest by loading candles from a store.

    Args:
        report: The backtest report containing trades and statistics
        store: CandleStore instance to load candles from
        symbol: Symbol to load candles for
        timeframe: Timeframe to load candles for
        show_stochastic: Whether to show stochastic oscillator subplot
        show_equity: Whether to show equity curve subplot
        height: Chart height in pixels
        initial_candles: Number of candles to show initially (default: 150)

    Returns:
        Plotly figure object that can be displayed or saved
    """
    run_config = _resolve_report_config(report)
    candles = store.load(symbol, timeframe, run_config.start, run_config.end)
    return plot_backtest(
        report=report,
        candles=candles,
        symbol=symbol,
        timeframe=timeframe,
        show_stochastic=show_stochastic,
        show_equity=show_equity,
        height=height,
        initial_candles=initial_candles,
    )
