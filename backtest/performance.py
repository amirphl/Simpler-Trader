from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import NormalDist, mean
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple


_SECONDS_PER_YEAR = 365.25 * 24 * 3600
_EULER_GAMMA = 0.5772156649015329
_EPSILON = 1e-12


class TradeLike(Protocol):
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    metadata: Mapping[str, str | float | int | None] | None

    def duration_seconds(self) -> float:
        ...


@dataclass(frozen=True)
class DeflatedSharpeResult:
    sharpe_ratio_per_trade: float
    deflated_sharpe_ratio: float
    deflated_sharpe_z_score: float
    expected_maximum_sharpe: float
    strategy_trials: int

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "sharpe_ratio_per_trade": self.sharpe_ratio_per_trade,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "deflated_sharpe_z_score": self.deflated_sharpe_z_score,
            "expected_maximum_sharpe": self.expected_maximum_sharpe,
            "strategy_trials": self.strategy_trials,
        }


def build_performance_statistics(
    *,
    trades: Sequence[TradeLike],
    initial_capital: float,
    start: datetime,
    end: datetime,
    risk_free_rate: float = 0.0,
    strategy_trials: int = 1,
) -> Dict[str, Any]:
    """Build detailed trade-level performance statistics.

    The calculations are intentionally based on closed trades only. They do not
    infer mark-to-market equity between exits, which keeps the output honest for
    strategies that do not emit an intrabar equity curve.
    """

    ordered_trades = _ordered_trades(trades)
    start = _ensure_utc(start)
    end = _ensure_utc(end)
    initial_capital = float(initial_capital)

    equity_curve, equity_points = build_equity_curve_points(
        initial_capital=initial_capital,
        trades=ordered_trades,
        start=start,
    )
    final_equity = equity_curve[-1] if equity_curve else initial_capital
    net_profit = final_equity - initial_capital
    net_profit_pct = _safe_pct(net_profit, initial_capital)

    winning = [trade.pnl for trade in ordered_trades if trade.pnl > 0]
    losing = [trade.pnl for trade in ordered_trades if trade.pnl < 0]
    returns = [trade.return_pct / 100.0 for trade in ordered_trades]
    gross_profit = sum(winning)
    gross_loss_abs = abs(sum(losing))
    durations = [max(trade.duration_seconds(), 0.0) for trade in ordered_trades]
    average_duration = mean(durations) if durations else 0.0

    max_drawdown_pct, drawdown_series = compute_drawdown_series(equity_curve)
    drawdown_summary = compute_drawdown_summary(
        equity_points=equity_points,
        end=end,
    )
    exposure_summary = compute_exposure_summary(
        trades=ordered_trades,
        initial_capital=initial_capital,
        start=start,
        end=end,
        net_profit_pct=net_profit_pct,
    )
    fee_slippage_summary = compute_fee_slippage_summary(
        trades=ordered_trades,
        initial_capital=initial_capital,
    )
    cagr_pct = compute_cagr_pct(
        initial_capital=initial_capital,
        final_equity=final_equity,
        start=start,
        end=end,
    )
    sharpe_ratio = compute_sharpe_ratio(
        returns=returns,
        risk_free_rate=risk_free_rate,
        average_period_seconds=average_duration,
    )
    sortino_ratio = compute_sortino_ratio(
        returns=returns,
        risk_free_rate=risk_free_rate,
        average_period_seconds=average_duration,
    )
    calmar_ratio = (cagr_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0
    skewness, excess_kurtosis = compute_skewness_excess_kurtosis(returns)
    deflated = compute_deflated_sharpe_ratio(
        returns=returns,
        risk_free_rate=risk_free_rate,
        average_period_seconds=average_duration,
        strategy_trials=strategy_trials,
    )

    total_trades = len(ordered_trades)
    expectancy = (net_profit / total_trades) if total_trades else 0.0
    average_return_pct = mean(trade.return_pct for trade in ordered_trades) if ordered_trades else 0.0

    return {
        "number_of_trades": total_trades,
        "win_rate": (len(winning) / total_trades) if total_trades else 0.0,
        "average_win": mean(winning) if winning else 0.0,
        "average_loss": mean(losing) if losing else 0.0,
        "average_loss_abs": mean(abs(value) for value in losing) if losing else 0.0,
        "average_return_pct": average_return_pct,
        "expectancy_per_trade": expectancy,
        "expectancy_pct_per_trade": average_return_pct,
        "gross_profit": gross_profit,
        "gross_loss": -gross_loss_abs,
        "net_profit": net_profit,
        "net_profit_pct": net_profit_pct,
        "profit_factor": (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "cagr_pct": cagr_pct,
        "maximum_drawdown_pct": max_drawdown_pct,
        "average_drawdown_pct": drawdown_summary["average_drawdown_pct"],
        "drawdown_series_pct": drawdown_series,
        "time_under_water_pct": drawdown_summary["time_under_water_pct"],
        "time_under_water_days": drawdown_summary["time_under_water_days"],
        "longest_time_under_water_days": drawdown_summary["longest_time_under_water_days"],
        "longest_losing_streak": compute_longest_losing_streak(ordered_trades),
        "monthly_return_distribution_pct": compute_monthly_returns(
            trades=ordered_trades,
            initial_capital=initial_capital,
            start=start,
            end=end,
        ),
        "return_skewness": skewness,
        "return_excess_kurtosis": excess_kurtosis,
        "exposure_time_pct": exposure_summary["exposure_time_pct"],
        "exposure_adjusted_return_pct": exposure_summary["exposure_adjusted_return_pct"],
        "turnover": exposure_summary["turnover"],
        "turnover_pct": exposure_summary["turnover_pct"],
        "fee_drag": fee_slippage_summary["fee_drag"],
        "fee_drag_pct": fee_slippage_summary["fee_drag_pct"],
        "slippage_drag": fee_slippage_summary["slippage_drag"],
        "slippage_drag_pct": fee_slippage_summary["slippage_drag_pct"],
        "final_equity": final_equity,
        "equity_curve": equity_curve,
        **deflated.as_dict(),
    }


def build_equity_curve_points(
    *,
    initial_capital: float,
    trades: Sequence[TradeLike],
    start: datetime,
) -> Tuple[List[float], List[Tuple[datetime, float]]]:
    equity = initial_capital
    equity_curve = [equity]
    equity_points = [(_ensure_utc(start), equity)]

    for trade in _ordered_trades(trades):
        equity += trade.pnl
        equity_curve.append(equity)
        equity_points.append((_ensure_utc(trade.exit_time), equity))

    return equity_curve, equity_points


def compute_drawdown_series(equity_curve: Sequence[float]) -> Tuple[float, List[float]]:
    peak = equity_curve[0] if equity_curve else 0.0
    max_drawdown = 0.0
    drawdowns: List[float] = []
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = _safe_pct(peak - equity, peak)
        max_drawdown = max(max_drawdown, drawdown)
        drawdowns.append(drawdown)
    return max_drawdown, drawdowns


def compute_drawdown_summary(
    *,
    equity_points: Sequence[Tuple[datetime, float]],
    end: datetime,
) -> Dict[str, float]:
    if not equity_points:
        return {
            "average_drawdown_pct": 0.0,
            "time_under_water_pct": 0.0,
            "time_under_water_days": 0.0,
            "longest_time_under_water_days": 0.0,
        }

    peak = equity_points[0][1]
    peak_time = equity_points[0][0]
    underwater_start: datetime | None = None
    current_depth = 0.0
    episode_depths: List[float] = []
    underwater_seconds = 0.0
    longest_underwater_seconds = 0.0

    for timestamp, equity in equity_points[1:]:
        if equity >= peak:
            if underwater_start is not None:
                duration = max((timestamp - underwater_start).total_seconds(), 0.0)
                underwater_seconds += duration
                longest_underwater_seconds = max(longest_underwater_seconds, duration)
                episode_depths.append(current_depth)
                underwater_start = None
                current_depth = 0.0
            peak = equity
            peak_time = timestamp
            continue

        if underwater_start is None:
            underwater_start = peak_time
        drawdown = _safe_pct(peak - equity, peak)
        current_depth = max(current_depth, drawdown)

    if underwater_start is not None:
        duration = max((_ensure_utc(end) - underwater_start).total_seconds(), 0.0)
        underwater_seconds += duration
        longest_underwater_seconds = max(longest_underwater_seconds, duration)
        episode_depths.append(current_depth)

    total_seconds = max((_ensure_utc(end) - equity_points[0][0]).total_seconds(), 0.0)
    return {
        "average_drawdown_pct": mean(episode_depths) if episode_depths else 0.0,
        "time_under_water_pct": _safe_pct(underwater_seconds, total_seconds),
        "time_under_water_days": underwater_seconds / 86400.0,
        "longest_time_under_water_days": longest_underwater_seconds / 86400.0,
    }


def compute_monthly_returns(
    *,
    trades: Sequence[TradeLike],
    initial_capital: float,
    start: datetime,
    end: datetime,
) -> Dict[str, float]:
    ordered = _ordered_trades(trades)
    returns: Dict[str, float] = {}
    equity = initial_capital
    trade_idx = 0

    for month_start, month_end in _iter_months(_ensure_utc(start), _ensure_utc(end)):
        month_key = f"{month_start.year:04d}-{month_start.month:02d}"
        month_start_equity = equity
        while trade_idx < len(ordered) and _ensure_utc(ordered[trade_idx].exit_time) < month_start:
            equity += ordered[trade_idx].pnl
            month_start_equity = equity
            trade_idx += 1
        is_final_month = month_end >= end
        while trade_idx < len(ordered):
            exit_time = _ensure_utc(ordered[trade_idx].exit_time)
            if exit_time < month_end or (is_final_month and exit_time <= end):
                equity += ordered[trade_idx].pnl
                trade_idx += 1
                continue
            break
        returns[month_key] = _safe_pct(equity - month_start_equity, month_start_equity)

    return returns


def compute_exposure_summary(
    *,
    trades: Sequence[TradeLike],
    initial_capital: float,
    start: datetime,
    end: datetime,
    net_profit_pct: float,
) -> Dict[str, float]:
    total_period_seconds = max((_ensure_utc(end) - _ensure_utc(start)).total_seconds(), 0.0)
    intervals = []
    turnover = 0.0

    for trade in trades:
        entry_time = max(_ensure_utc(trade.entry_time), _ensure_utc(start))
        exit_time = min(_ensure_utc(trade.exit_time), _ensure_utc(end))
        if exit_time > entry_time:
            intervals.append((entry_time, exit_time))
        metadata = trade.metadata or {}
        qty = abs(_metadata_float(metadata, "qty"))
        entry_price = _first_metadata_float(metadata, "entry_price", "entry_raw_price")
        exit_price = _first_metadata_float(metadata, "exit_price", "exit_raw_price")
        if qty > 0 and entry_price > 0:
            turnover += qty * entry_price
        if qty > 0 and exit_price > 0:
            turnover += qty * exit_price

    exposure_seconds = _merged_interval_seconds(intervals)
    exposure_time_pct = _safe_pct(exposure_seconds, total_period_seconds)
    exposure_fraction = exposure_time_pct / 100.0
    return {
        "exposure_time_pct": exposure_time_pct,
        "exposure_adjusted_return_pct": (net_profit_pct / exposure_fraction) if exposure_fraction > 0 else 0.0,
        "turnover": turnover / initial_capital if initial_capital > 0 else 0.0,
        "turnover_pct": _safe_pct(turnover, initial_capital),
    }


def compute_fee_slippage_summary(
    *,
    trades: Sequence[TradeLike],
    initial_capital: float,
) -> Dict[str, float]:
    fee_drag = 0.0
    slippage_drag = 0.0
    funding_paid = 0.0
    funding_received = 0.0

    for trade in trades:
        metadata = trade.metadata or {}
        fee_drag += _metadata_float(metadata, "entry_fee")
        fee_drag += _metadata_float(metadata, "exit_fee")
        funding_paid += _metadata_float(metadata, "funding_paid")
        funding_received += _metadata_float(metadata, "funding_received")

        qty = abs(_metadata_float(metadata, "qty"))
        entry_raw = _metadata_float(metadata, "entry_raw_price")
        entry_price = _metadata_float(metadata, "entry_price")
        exit_raw = _metadata_float(metadata, "exit_raw_price")
        exit_price = _metadata_float(metadata, "exit_price")
        if qty > 0 and entry_raw > 0 and entry_price > 0:
            slippage_drag += abs(entry_price - entry_raw) * qty
        if qty > 0 and exit_raw > 0 and exit_price > 0:
            slippage_drag += abs(exit_price - exit_raw) * qty

    return {
        "fee_drag": fee_drag,
        "fee_drag_pct": _safe_pct(fee_drag, initial_capital),
        "slippage_drag": slippage_drag,
        "slippage_drag_pct": _safe_pct(slippage_drag, initial_capital),
        "funding_paid": funding_paid,
        "funding_received": funding_received,
        "funding_pnl": funding_received - funding_paid,
        "funding_pnl_pct": _safe_pct(funding_received - funding_paid, initial_capital),
    }


def compute_longest_losing_streak(trades: Sequence[TradeLike]) -> int:
    longest = 0
    current = 0
    for trade in _ordered_trades(trades):
        if trade.pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def compute_cagr_pct(
    *,
    initial_capital: float,
    final_equity: float,
    start: datetime,
    end: datetime,
) -> float:
    duration_years = max((_ensure_utc(end) - _ensure_utc(start)).total_seconds(), 0.0) / _SECONDS_PER_YEAR
    if duration_years <= 0 or initial_capital <= 0 or final_equity <= 0:
        return 0.0
    return ((final_equity / initial_capital) ** (1.0 / duration_years) - 1.0) * 100.0


def compute_sharpe_ratio(
    *,
    returns: Sequence[float],
    risk_free_rate: float,
    average_period_seconds: float,
) -> float:
    if len(returns) <= 1:
        return 0.0
    avg_return = mean(returns)
    std_dev = _population_std(returns)
    if std_dev <= _EPSILON:
        return 0.0
    periods_per_year = _periods_per_year(average_period_seconds, len(returns))
    rf_per_period = _risk_free_per_period(risk_free_rate, average_period_seconds)
    return ((avg_return - rf_per_period) / std_dev) * math.sqrt(periods_per_year)


def compute_sortino_ratio(
    *,
    returns: Sequence[float],
    risk_free_rate: float,
    average_period_seconds: float,
) -> float:
    if len(returns) <= 1:
        return 0.0
    rf_per_period = _risk_free_per_period(risk_free_rate, average_period_seconds)
    downside = [min(value - rf_per_period, 0.0) for value in returns]
    downside_deviation = math.sqrt(mean(value * value for value in downside))
    if downside_deviation <= _EPSILON:
        return 0.0
    periods_per_year = _periods_per_year(average_period_seconds, len(returns))
    return ((mean(returns) - rf_per_period) / downside_deviation) * math.sqrt(periods_per_year)


def compute_skewness_excess_kurtosis(returns: Sequence[float]) -> Tuple[float, float]:
    if len(returns) < 2:
        return 0.0, 0.0
    avg_return = mean(returns)
    centered = [value - avg_return for value in returns]
    variance = mean(value * value for value in centered)
    if variance <= _EPSILON:
        return 0.0, 0.0
    std_dev = math.sqrt(variance)
    skewness = mean((value / std_dev) ** 3 for value in centered)
    excess_kurtosis = mean((value / std_dev) ** 4 for value in centered) - 3.0
    return skewness, excess_kurtosis


def compute_deflated_sharpe_ratio(
    *,
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    average_period_seconds: float = 0.0,
    strategy_trials: int = 1,
) -> DeflatedSharpeResult:
    """Estimate the Deflated Sharpe Ratio probability.

    This follows the Bailey/Lopez de Prado adjustment at trade frequency. The
    output is a probability-like value in [0, 1], not an annualized Sharpe.
    """

    trials = max(int(strategy_trials), 1)
    if len(returns) <= 1:
        return DeflatedSharpeResult(0.0, 0.0, 0.0, 0.0, trials)

    rf_per_period = _risk_free_per_period(risk_free_rate, average_period_seconds)
    excess_returns = [value - rf_per_period for value in returns]
    avg_excess = mean(excess_returns)
    std_dev = _population_std(excess_returns)
    if std_dev <= _EPSILON:
        return DeflatedSharpeResult(0.0, 0.0, 0.0, 0.0, trials)

    sharpe = avg_excess / std_dev
    skewness, excess_kurtosis = compute_skewness_excess_kurtosis(excess_returns)
    kurtosis = excess_kurtosis + 3.0
    variance_factor = 1.0 - (skewness * sharpe) + (((kurtosis - 1.0) / 4.0) * sharpe * sharpe)
    if variance_factor <= _EPSILON:
        return DeflatedSharpeResult(sharpe, 0.0, 0.0, 0.0, trials)

    expected_maximum_sharpe = _expected_maximum_sharpe(trials, variance_factor, len(returns))
    z_score = (sharpe - expected_maximum_sharpe) * math.sqrt(len(returns) - 1.0) / math.sqrt(variance_factor)
    probability = NormalDist().cdf(z_score)
    return DeflatedSharpeResult(
        sharpe_ratio_per_trade=sharpe,
        deflated_sharpe_ratio=probability,
        deflated_sharpe_z_score=z_score,
        expected_maximum_sharpe=expected_maximum_sharpe,
        strategy_trials=trials,
    )


def _expected_maximum_sharpe(
    trials: int,
    variance_factor: float,
    observations: int,
) -> float:
    if trials <= 1 or observations <= 1:
        return 0.0
    normal = NormalDist()
    adjusted_std = math.sqrt(variance_factor / (observations - 1.0))
    p1 = _clamp_probability(1.0 - (1.0 / trials))
    p2 = _clamp_probability(1.0 - (1.0 / (trials * math.e)))
    return adjusted_std * (
        (1.0 - _EULER_GAMMA) * normal.inv_cdf(p1)
        + _EULER_GAMMA * normal.inv_cdf(p2)
    )


def _iter_months(start: datetime, end: datetime) -> Sequence[Tuple[datetime, datetime]]:
    months: List[Tuple[datetime, datetime]] = []
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    final = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor <= final:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1)
        else:
            next_month = cursor.replace(month=cursor.month + 1)
        month_start = max(cursor, start)
        month_end = min(next_month, end)
        months.append((month_start, month_end))
        cursor = next_month
    return months


def _merged_interval_seconds(intervals: Sequence[Tuple[datetime, datetime]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda item: item[0])
    merged: List[Tuple[datetime, datetime]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return sum((end - start).total_seconds() for start, end in merged)


def _ordered_trades(trades: Sequence[TradeLike]) -> List[TradeLike]:
    return sorted(
        trades,
        key=lambda trade: (_ensure_utc(trade.exit_time), _ensure_utc(trade.entry_time)),
    )


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _safe_pct(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _EPSILON:
        return 0.0
    return (numerator / denominator) * 100.0


def _metadata_float(
    metadata: Mapping[str, str | float | int | None],
    key: str,
) -> float:
    value = metadata.get(key)
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _first_metadata_float(
    metadata: Mapping[str, str | float | int | None],
    *keys: str,
) -> float:
    for key in keys:
        value = _metadata_float(metadata, key)
        if value > 0:
            return value
    return 0.0


def _population_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg_value = mean(values)
    return math.sqrt(mean((value - avg_value) ** 2 for value in values))


def _periods_per_year(average_period_seconds: float, observations: int) -> float:
    if average_period_seconds > 0:
        return _SECONDS_PER_YEAR / average_period_seconds
    return float(max(observations, 1))


def _risk_free_per_period(risk_free_rate: float, average_period_seconds: float) -> float:
    if average_period_seconds <= 0:
        return 0.0
    return risk_free_rate * (average_period_seconds / _SECONDS_PER_YEAR)


def _clamp_probability(value: float) -> float:
    return min(max(value, 1e-9), 1.0 - 1e-9)


__all__ = [
    "DeflatedSharpeResult",
    "TradeLike",
    "build_equity_curve_points",
    "build_performance_statistics",
    "compute_cagr_pct",
    "compute_deflated_sharpe_ratio",
    "compute_drawdown_series",
    "compute_drawdown_summary",
    "compute_exposure_summary",
    "compute_fee_slippage_summary",
    "compute_longest_losing_streak",
    "compute_monthly_returns",
    "compute_sharpe_ratio",
    "compute_skewness_excess_kurtosis",
    "compute_sortino_ratio",
]
