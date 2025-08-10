from __future__ import annotations

from datetime import datetime
from typing import List, Sequence

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore
    make_subplots = None  # type: ignore

from candle_downloader.models import Candle

from .base import BacktestReport


def plot_backtest(
    report: BacktestReport,
    candles: Sequence[Candle],
    symbol: str | None = None,
    timeframe: str | None = None,
    show_stochastic: bool = True,
    show_equity: bool = True,
    height: int = 900,
    initial_candles: int = 150,
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
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    if not candles:
        raise ValueError("candles sequence cannot be empty")

    # Extract symbol and timeframe from first candle if not provided
    if symbol is None:
        symbol = candles[0].symbol
    if timeframe is None:
        timeframe = candles[0].interval

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
    colors = ["#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(len(candles))]
    fig.add_trace(
        go.Bar(
            x=times,
            y=volumes,
            name="Volume",
            marker_color=colors,
            opacity=0.3,
            yaxis="y2",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Add entry and exit markers
    entry_times: List[datetime] = []
    entry_prices: List[float] = []
    exit_times: List[datetime] = []
    exit_prices: List[float] = []
    stop_losses: List[float] = []
    take_profits: List[float] = []
    trade_labels: List[str] = []

    for trade in report.trades:
        entry_times.append(trade.entry_time)
        exit_times.append(trade.exit_time)
        if trade.metadata:
            entry_prices.append(trade.metadata.get("entry_price", 0.0))
            exit_prices.append(trade.metadata.get("exit_price", 0.0))
            stop_losses.append(trade.metadata.get("stop_loss", 0.0))
            take_profits.append(trade.metadata.get("take_profit", 0.0))
        else:
            # Fallback: find price from candles
            entry_candle = _find_candle_at_time(candles, trade.entry_time)
            exit_candle = _find_candle_at_time(candles, trade.exit_time)
            entry_prices.append(entry_candle.open if entry_candle else 0.0)
            exit_prices.append(exit_candle.close if exit_candle else 0.0)
            stop_losses.append(0.0)
            take_profits.append(0.0)

        pnl_str = f"{trade.pnl:+.2f}" if trade.pnl else "0.00"
        return_str = f"{trade.return_pct:+.2f}%" if trade.return_pct else "0.00%"
        trade_labels.append(f"Entry<br>PnL: {pnl_str}<br>Return: {return_str}")

    # Entry markers (green triangles)
    if entry_times:
        fig.add_trace(
            go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode="markers+text",
                name="Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#00ff00",
                    line=dict(width=2, color="darkgreen"),
                ),
                text=[f"E{i+1}" for i in range(len(entry_times))],
                textposition="top center",
                textfont=dict(size=10, color="green"),
                hovertemplate="<b>Entry</b><br>" + "Time: %{x}<br>Price: %{y:.2f}<br>" + "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Exit markers (red triangles)
    if exit_times:
        exit_colors = ["#ff0000" if pnl < 0 else "#00ff00" for pnl in [t.pnl for t in report.trades]]
        fig.add_trace(
            go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode="markers+text",
                name="Exit",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=exit_colors,
                    line=dict(width=2, color="darkred"),
                ),
                text=[f"X{i+1}" for i in range(len(exit_times))],
                textposition="bottom center",
                textfont=dict(size=10, color="red"),
                hovertemplate="<b>Exit</b><br>" + "Time: %{x}<br>Price: %{y:.2f}<br>" + "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Stop loss and take profit lines (if available)
    for i, trade in enumerate(report.trades):
        if trade.metadata and "stop_loss" in trade.metadata and "take_profit" in trade.metadata:
            sl = trade.metadata["stop_loss"]
            tp = trade.metadata["take_profit"]
            entry_time = trade.entry_time
            exit_time = trade.exit_time

            # Stop loss line
            if sl > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time, exit_time],
                        y=[sl, sl],
                        mode="lines",
                        name=f"SL {i+1}" if i == 0 else None,
                        line=dict(color="red", width=1, dash="dash"),
                        showlegend=i == 0,
                        hovertemplate=f"Stop Loss: {sl:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            # Take profit line
            if tp > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time, exit_time],
                        y=[tp, tp],
                        mode="lines",
                        name=f"TP {i+1}" if i == 0 else None,
                        line=dict(color="green", width=1, dash="dash"),
                        showlegend=i == 0,
                        hovertemplate=f"Take Profit: {tp:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    # Stochastic oscillator subplot
    stoch_row = 2 if show_stochastic else None
    if show_stochastic:
        stoch_k20, stoch_k100 = _calculate_stochastic_series(candles, 20), _calculate_stochastic_series(candles, 100)
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
        fig.add_hline(y=80, line_dash="dash", line_color="gray", opacity=0.5, row=stoch_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="gray", opacity=0.5, row=stoch_row, col=1)

    # Equity curve subplot
    equity_row = num_subplots if show_equity else None
    if show_equity and report.statistics.equity_curve:
        equity_times = [report.config.start] + [t.exit_time for t in report.trades]
        fig.add_trace(
            go.Scatter(
                x=equity_times,
                y=report.statistics.equity_curve,
                mode="lines+markers",
                name="Equity",
                line=dict(color="purple", width=2),
                fill="tozeroy",
                fillcolor="rgba(128, 0, 128, 0.1)",
            ),
            row=equity_row,
            col=1,
        )

    # Update layout
    title_text = f"Backtest: {report.strategy_name} | {symbol} {timeframe}"
    title_text += f"<br><sub>Total Trades: {report.statistics.total_trades} | "
    title_text += f"Win Rate: {report.statistics.win_rate*100:.1f}% | "
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
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True, showgrid=False)

    if show_stochastic:
        fig.update_yaxes(title_text="Stochastic %K", range=[0, 100], row=stoch_row, col=1)

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
        initial_start = times[0] if times else report.config.start
        initial_end = times[-1] if times else report.config.end

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


def _find_candle_at_time(candles: Sequence[Candle], target_time: datetime) -> Candle | None:
    """Find the candle that contains the target time."""
    for candle in candles:
        if candle.open_time <= target_time < candle.close_time:
            return candle
    # Fallback: find closest candle
    if not candles:
        return None
    closest = min(candles, key=lambda c: abs((c.open_time - target_time).total_seconds()))
    return closest


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
    if format == "html":
        fig.write_html(filepath)
    elif format == "png":
        fig.write_image(filepath, width=1920, height=1080)
    elif format == "svg":
        fig.write_image(filepath, format="svg", width=1920, height=1080)
    elif format == "pdf":
        fig.write_image(filepath, format="pdf", width=1920, height=1080)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'html', 'png', 'svg', or 'pdf'")


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
    initial_candles: int = 150,
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
    candles = store.load(symbol, timeframe, report.config.start, report.config.end)
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

