from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Protocol, Sequence, Tuple

from candle_downloader.models import Candle

from .base import BacktestReport, BacktestRunConfig, BaseBacktester, TradePerformance
from .performance import build_performance_statistics


ScoreFunction = Callable[[BacktestReport], float]
BacktesterFactory = Callable[[Mapping[str, Any], float], BaseBacktester]


class BacktesterLike(Protocol):
    def run(self, config: BacktestRunConfig) -> BacktestReport:
        ...


@dataclass(frozen=True)
class EvaluationPeriod:
    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _ensure_utc(self.start)
        end = _ensure_utc(self.end)
        if start >= end:
            raise ValueError("period start must be before end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class OutOfSamplePlan:
    training: EvaluationPeriod
    validation: EvaluationPeriod
    final: EvaluationPeriod

    def __post_init__(self) -> None:
        if self.training.end > self.validation.start:
            raise ValueError("training period must end before validation starts")
        if self.validation.end > self.final.start:
            raise ValueError("validation period must end before final starts")

    def as_dict(self) -> Dict[str, Dict[str, str]]:
        return {
            "training": self.training.as_dict(),
            "validation": self.validation.as_dict(),
            "final": self.final.as_dict(),
        }


@dataclass(frozen=True)
class ParameterCandidate:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class CandidateReport:
    candidate: ParameterCandidate
    score: float
    report: BacktestReport

    def as_dict(self, *, include_report: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "candidate": self.candidate.as_dict(),
            "score": self.score,
        }
        if include_report:
            payload["report"] = self.report.as_dict()
        return payload


@dataclass(frozen=True)
class OutOfSampleEvaluationResult:
    plan: OutOfSamplePlan
    selected_candidate: ParameterCandidate
    training_reports: Sequence[CandidateReport]
    validation_reports: Sequence[CandidateReport]
    final_report: BacktestReport
    selection_metric: str

    def as_dict(self, *, include_reports: bool = True) -> Dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "selection_metric": self.selection_metric,
            "selected_candidate": self.selected_candidate.as_dict(),
            "training_reports": [
                item.as_dict(include_report=include_reports)
                for item in self.training_reports
            ],
            "validation_reports": [
                item.as_dict(include_report=include_reports)
                for item in self.validation_reports
            ],
            "final_report": self.final_report.as_dict() if include_reports else None,
            "final_statistics": self.final_report.statistics.as_dict(),
        }


@dataclass(frozen=True)
class WalkForwardConfig:
    start: datetime
    end: datetime
    train_window: timedelta
    test_window: timedelta
    step: timedelta | None = None
    anchored: bool = False

    def __post_init__(self) -> None:
        start = _ensure_utc(self.start)
        end = _ensure_utc(self.end)
        if start >= end:
            raise ValueError("walk-forward start must be before end")
        if self.train_window <= timedelta(0):
            raise ValueError("train_window must be positive")
        if self.test_window <= timedelta(0):
            raise ValueError("test_window must be positive")
        if self.step is not None and self.step <= timedelta(0):
            raise ValueError("step must be positive when provided")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train: EvaluationPeriod
    test: EvaluationPeriod

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "train": self.train.as_dict(),
            "test": self.test.as_dict(),
        }


@dataclass(frozen=True)
class WalkForwardFoldResult:
    window: WalkForwardWindow
    selected_candidate: ParameterCandidate
    train_score: float
    train_report: BacktestReport
    test_report: BacktestReport

    def as_dict(self, *, include_reports: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "window": self.window.as_dict(),
            "selected_candidate": self.selected_candidate.as_dict(),
            "train_score": self.train_score,
            "test_statistics": self.test_report.statistics.as_dict(),
        }
        if include_reports:
            payload["train_report"] = self.train_report.as_dict()
            payload["test_report"] = self.test_report.as_dict()
        return payload


@dataclass(frozen=True)
class WalkForwardResult:
    folds: Sequence[WalkForwardFoldResult]
    combined_trades: Sequence[TradePerformance]
    combined_statistics: Mapping[str, Any]
    selection_metric: str

    def as_dict(self, *, include_reports: bool = True) -> Dict[str, Any]:
        return {
            "selection_metric": self.selection_metric,
            "folds": [
                fold.as_dict(include_reports=include_reports) for fold in self.folds
            ],
            "combined_statistics": dict(self.combined_statistics),
            "combined_trades": [
                {
                    "index": idx,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct,
                    "notes": trade.notes,
                    "metadata": dict(trade.metadata or {}),
                }
                for idx, trade in enumerate(self.combined_trades)
            ],
        }


@dataclass(frozen=True)
class MarketCase:
    name: str
    symbol: str
    timeframe: str | None = None
    exchange: str = "binance"
    category: str | None = None
    fee_overrides: Mapping[str, float] = field(default_factory=dict)
    start: datetime | None = None
    end: datetime | None = None
    tags: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange": self.exchange,
            "category": self.category,
            "fee_overrides": dict(self.fee_overrides),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class UniverseCaseResult:
    case: MarketCase
    report: BacktestReport

    def as_dict(self, *, include_report: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "case": self.case.as_dict(),
            "statistics": self.report.statistics.as_dict(),
        }
        if include_report:
            payload["report"] = self.report.as_dict()
        return payload


@dataclass(frozen=True)
class RegimeSegment:
    name: str
    start: datetime
    end: datetime
    trend: str
    volatility: str
    observations: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "trend": self.trend,
            "volatility": self.volatility,
            "observations": self.observations,
        }


@dataclass(frozen=True)
class MonteCarloConfig:
    iterations: int = 5000
    seed: int | None = None
    initial_capital: float = 100.0
    horizon_trades: int | None = None
    block_size: int = 5
    drawdown_threshold_pct: float = 30.0
    fee_multipliers: Tuple[float, ...] = (1.0, 2.0, 3.0)
    extra_spread_slippage_pct_range: Tuple[float, float] = (0.0, 0.001)
    missed_fill_probability: float = 0.0
    missed_fill_winner_only: bool = True

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.horizon_trades is not None and self.horizon_trades <= 0:
            raise ValueError("horizon_trades must be positive when provided")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.drawdown_threshold_pct < 0:
            raise ValueError("drawdown_threshold_pct must be non-negative")
        if not self.fee_multipliers:
            raise ValueError("fee_multipliers must not be empty")
        if any(value < 0 for value in self.fee_multipliers):
            raise ValueError("fee_multipliers must be non-negative")
        lo, hi = self.extra_spread_slippage_pct_range
        if lo < 0 or hi < lo:
            raise ValueError("extra spread slippage range must be non-negative")
        if not 0 <= self.missed_fill_probability <= 1:
            raise ValueError("missed_fill_probability must live in [0, 1]")


@dataclass(frozen=True)
class ParameterPerturbationRule:
    relative: float | None = None
    absolute: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False

    def perturb(self, value: Any, rng: random.Random) -> Any:
        base_value = float(value)
        delta = 0.0
        if self.relative is not None:
            delta += base_value * rng.uniform(-self.relative, self.relative)
        if self.absolute is not None:
            delta += rng.uniform(-self.absolute, self.absolute)
        perturbed = base_value + delta
        if self.minimum is not None:
            perturbed = max(perturbed, self.minimum)
        if self.maximum is not None:
            perturbed = min(perturbed, self.maximum)
        if self.integer:
            minimum = self.minimum if self.minimum is not None else 1
            return int(max(round(perturbed), minimum))
        return perturbed


def build_three_way_oos_plan(
    *,
    start: datetime,
    end: datetime,
    training_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> OutOfSamplePlan:
    start = _ensure_utc(start)
    end = _ensure_utc(end)
    if start >= end:
        raise ValueError("start must be before end")
    if not 0 < training_fraction < 1:
        raise ValueError("training_fraction must live in (0, 1)")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must live in (0, 1)")
    if training_fraction + validation_fraction >= 1:
        raise ValueError("training + validation fractions must leave a final period")

    total_seconds = (end - start).total_seconds()
    training_end = start + timedelta(seconds=total_seconds * training_fraction)
    validation_end = training_end + timedelta(seconds=total_seconds * validation_fraction)
    return OutOfSamplePlan(
        training=EvaluationPeriod("training", start, training_end),
        validation=EvaluationPeriod("validation", training_end, validation_end),
        final=EvaluationPeriod("final_out_of_sample", validation_end, end),
    )


def run_out_of_sample_evaluation(
    *,
    build_backtester: BacktesterFactory,
    run_config: BacktestRunConfig,
    plan: OutOfSamplePlan,
    candidates: Sequence[ParameterCandidate] | None = None,
    selection_metric: str = "net_profit_pct",
    scorer: ScoreFunction | None = None,
) -> OutOfSampleEvaluationResult:
    candidate_list = _candidate_list(candidates)
    training_reports = [
        _run_scored_candidate(
            build_backtester=build_backtester,
            candidate=candidate,
            run_config=run_config,
            period=plan.training,
            initial_capital=run_config.initial_capital,
            selection_metric=selection_metric,
            scorer=scorer,
        )
        for candidate in candidate_list
    ]
    validation_reports = [
        _run_scored_candidate(
            build_backtester=build_backtester,
            candidate=candidate,
            run_config=run_config,
            period=plan.validation,
            initial_capital=run_config.initial_capital,
            selection_metric=selection_metric,
            scorer=scorer,
        )
        for candidate in candidate_list
    ]
    selected = max(validation_reports, key=lambda item: item.score).candidate
    final_report = build_backtester(selected.parameters, run_config.initial_capital).run(
        _run_config_for_period(
            run_config=run_config,
            period=plan.final,
            initial_capital=run_config.initial_capital,
        )
    )
    return OutOfSampleEvaluationResult(
        plan=plan,
        selected_candidate=selected,
        training_reports=training_reports,
        validation_reports=validation_reports,
        final_report=final_report,
        selection_metric=selection_metric,
    )


def build_walk_forward_windows(config: WalkForwardConfig) -> List[WalkForwardWindow]:
    windows: List[WalkForwardWindow] = []
    step = config.step or config.test_window
    cursor = config.start
    index = 0

    while True:
        train_start = config.start if config.anchored else cursor
        train_end = cursor + config.train_window
        test_start = train_end
        test_end = test_start + config.test_window
        if test_end > config.end:
            break
        windows.append(
            WalkForwardWindow(
                index=index,
                train=EvaluationPeriod(f"train_{index}", train_start, train_end),
                test=EvaluationPeriod(f"test_{index}", test_start, test_end),
            )
        )
        cursor += step
        index += 1

    return windows


def run_walk_forward(
    *,
    build_backtester: BacktesterFactory,
    run_config: BacktestRunConfig,
    walk_forward_config: WalkForwardConfig,
    candidates: Sequence[ParameterCandidate] | None = None,
    selection_metric: str = "net_profit_pct",
    scorer: ScoreFunction | None = None,
    strategy_trials: int | None = None,
) -> WalkForwardResult:
    candidate_list = _candidate_list(candidates)
    folds: List[WalkForwardFoldResult] = []
    current_equity = run_config.initial_capital
    combined_trades: List[TradePerformance] = []

    for window in build_walk_forward_windows(walk_forward_config):
        scored_candidates = [
            _run_scored_candidate(
                build_backtester=build_backtester,
                candidate=candidate,
                run_config=run_config,
                period=window.train,
                initial_capital=run_config.initial_capital,
                selection_metric=selection_metric,
                scorer=scorer,
            )
            for candidate in candidate_list
        ]
        best = max(scored_candidates, key=lambda item: item.score)
        test_report = build_backtester(best.candidate.parameters, current_equity).run(
            _run_config_for_period(
                run_config=run_config,
                period=window.test,
                initial_capital=current_equity,
            )
        )
        current_equity = _final_equity_from_report(test_report, current_equity)
        combined_trades.extend(test_report.trades)
        folds.append(
            WalkForwardFoldResult(
                window=window,
                selected_candidate=best.candidate,
                train_score=best.score,
                train_report=best.report,
                test_report=test_report,
            )
        )

    combined_trades.sort(key=lambda trade: (trade.exit_time, trade.entry_time))
    if folds:
        stats_start = folds[0].window.test.start
        stats_end = folds[-1].window.test.end
    else:
        stats_start = walk_forward_config.start
        stats_end = walk_forward_config.end
    combined_statistics = build_performance_statistics(
        trades=combined_trades,
        initial_capital=run_config.initial_capital,
        start=stats_start,
        end=stats_end,
        risk_free_rate=run_config.risk_free_rate,
        strategy_trials=strategy_trials or max(len(candidate_list), 1),
    )
    return WalkForwardResult(
        folds=folds,
        combined_trades=combined_trades,
        combined_statistics=combined_statistics,
        selection_metric=selection_metric,
    )


def run_market_universe(
    *,
    build_backtester: Callable[[MarketCase, float], BaseBacktester],
    run_config: BacktestRunConfig,
    cases: Sequence[MarketCase],
) -> List[UniverseCaseResult]:
    results: List[UniverseCaseResult] = []
    for case in cases:
        period = EvaluationPeriod(
            case.name,
            _ensure_utc(case.start) if case.start else run_config.start,
            _ensure_utc(case.end) if case.end else run_config.end,
        )
        report = build_backtester(case, run_config.initial_capital).run(
            _run_config_for_period(
                run_config=run_config,
                period=period,
                initial_capital=run_config.initial_capital,
            )
        )
        results.append(UniverseCaseResult(case=case, report=report))
    return results


def classify_regime_segments(
    candles: Sequence[Candle],
    *,
    trend_lookback: int = 30,
    volatility_lookback: int = 30,
    trend_threshold_pct: float = 5.0,
    high_vol_quantile: float = 0.67,
    low_vol_quantile: float = 0.33,
) -> List[RegimeSegment]:
    if trend_lookback <= 0:
        raise ValueError("trend_lookback must be positive")
    if volatility_lookback <= 1:
        raise ValueError("volatility_lookback must be greater than one")
    if not 0 <= low_vol_quantile <= high_vol_quantile <= 1:
        raise ValueError("volatility quantiles must satisfy 0 <= low <= high <= 1")
    if len(candles) <= max(trend_lookback, volatility_lookback):
        return []

    closes = [candle.close for candle in candles]
    log_returns = [0.0]
    for idx in range(1, len(closes)):
        if closes[idx - 1] <= 0 or closes[idx] <= 0:
            log_returns.append(0.0)
        else:
            log_returns.append(math.log(closes[idx] / closes[idx - 1]))

    raw_items: List[Tuple[int, str, float]] = []
    for idx in range(max(trend_lookback, volatility_lookback), len(candles)):
        base_close = closes[idx - trend_lookback]
        trend_return_pct = ((closes[idx] / base_close) - 1.0) * 100.0 if base_close > 0 else 0.0
        if trend_return_pct >= trend_threshold_pct:
            trend = "bull"
        elif trend_return_pct <= -trend_threshold_pct:
            trend = "bear"
        else:
            trend = "sideways"
        vol_window = log_returns[idx - volatility_lookback + 1 : idx + 1]
        volatility = _population_std(vol_window)
        raw_items.append((idx, trend, volatility))

    vol_values = [item[2] for item in raw_items]
    low_cutoff = _quantile(vol_values, low_vol_quantile)
    high_cutoff = _quantile(vol_values, high_vol_quantile)

    labeled: List[Tuple[int, str, str]] = []
    for idx, trend, volatility in raw_items:
        if volatility >= high_cutoff:
            vol_label = "high_volatility"
        elif volatility <= low_cutoff:
            vol_label = "low_volatility"
        else:
            vol_label = "normal_volatility"
        labeled.append((idx, trend, vol_label))

    segments: List[RegimeSegment] = []
    current_start_idx = labeled[0][0]
    current_trend = labeled[0][1]
    current_vol = labeled[0][2]
    observations = 0

    for idx, trend, vol_label in labeled:
        if trend != current_trend or vol_label != current_vol:
            end_idx = idx - 1
            segments.append(
                _build_regime_segment(
                    candles=candles,
                    start_idx=current_start_idx,
                    end_idx=end_idx,
                    trend=current_trend,
                    volatility=current_vol,
                    observations=observations,
                )
            )
            current_start_idx = idx
            current_trend = trend
            current_vol = vol_label
            observations = 0
        observations += 1

    segments.append(
        _build_regime_segment(
            candles=candles,
            start_idx=current_start_idx,
            end_idx=labeled[-1][0],
            trend=current_trend,
            volatility=current_vol,
            observations=observations,
        )
    )
    return segments


def label_trades_by_regime(
    trades: Sequence[TradePerformance],
    segments: Sequence[RegimeSegment],
) -> List[str]:
    labels: List[str] = []
    for trade in sorted(trades, key=lambda item: (item.exit_time, item.entry_time)):
        exit_time = _ensure_utc(trade.exit_time)
        label = "unknown"
        for segment in segments:
            if segment.start <= exit_time <= segment.end:
                label = segment.name
                break
        labels.append(label)
    return labels


def summarize_trades_by_regime(
    *,
    trades: Sequence[TradePerformance],
    segments: Sequence[RegimeSegment],
    initial_capital: float,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    ordered_trades = sorted(trades, key=lambda item: (item.exit_time, item.entry_time))
    labels = label_trades_by_regime(ordered_trades, segments)
    grouped: Dict[str, List[TradePerformance]] = {}
    for trade, label in zip(ordered_trades, labels):
        grouped.setdefault(label, []).append(trade)

    summaries: Dict[str, Any] = {}
    for label, group in grouped.items():
        group_start = min(trade.entry_time for trade in group)
        group_end = max(trade.exit_time for trade in group)
        summaries[label] = build_performance_statistics(
            trades=group,
            initial_capital=initial_capital,
            start=group_start,
            end=group_end,
            risk_free_rate=risk_free_rate,
        )
    return summaries


def run_trade_reshuffling_monte_carlo(
    trades: Sequence[TradePerformance],
    config: MonteCarloConfig,
) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    pnls = [trade.pnl for trade in _ordered_trades(trades)]
    horizon = config.horizon_trades or len(pnls)
    paths = []
    for _ in range(config.iterations):
        shuffled = list(pnls)
        rng.shuffle(shuffled)
        paths.append(_simulate_pnl_path(shuffled[:horizon], config.initial_capital))
    return _summarize_monte_carlo_paths(paths, config)


def run_block_bootstrap_monte_carlo(
    trades: Sequence[TradePerformance],
    config: MonteCarloConfig,
) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    pnls = [trade.pnl for trade in _ordered_trades(trades)]
    if not pnls:
        return _empty_monte_carlo_summary(config)
    block_size = min(config.block_size, len(pnls))
    blocks = [pnls[idx : idx + block_size] for idx in range(0, len(pnls) - block_size + 1)]
    horizon = config.horizon_trades or len(pnls)
    paths = []
    for _ in range(config.iterations):
        sampled: List[float] = []
        while len(sampled) < horizon:
            sampled.extend(rng.choice(blocks))
        paths.append(_simulate_pnl_path(sampled[:horizon], config.initial_capital))
    return _summarize_monte_carlo_paths(paths, config)


def run_slippage_monte_carlo(
    trades: Sequence[TradePerformance],
    config: MonteCarloConfig,
) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    ordered = _ordered_trades(trades)
    paths = []
    for _ in range(config.iterations):
        adjusted_pnls = [
            _apply_random_execution_costs(trade, rng, config) for trade in ordered
        ]
        paths.append(_simulate_pnl_path(adjusted_pnls, config.initial_capital))
    return _summarize_monte_carlo_paths(paths, config)


def run_regime_monte_carlo(
    trades: Sequence[TradePerformance],
    trade_regimes: Sequence[str],
    config: MonteCarloConfig,
    *,
    target_regime_weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    ordered = _ordered_trades(trades)
    if len(ordered) != len(trade_regimes):
        raise ValueError("trade_regimes must have the same length as trades")
    grouped: Dict[str, List[float]] = {}
    for trade, regime in zip(ordered, trade_regimes):
        grouped.setdefault(regime, []).append(trade.pnl)
    if not grouped:
        return _empty_monte_carlo_summary(config)

    weights = _normalize_regime_weights(grouped, target_regime_weights)
    labels = list(weights.keys())
    cumulative_weights = _cumulative_weights([weights[label] for label in labels])
    horizon = config.horizon_trades or len(ordered)
    paths = []
    for _ in range(config.iterations):
        pnls = []
        for _ in range(horizon):
            label = labels[_weighted_index(rng.random(), cumulative_weights)]
            pnls.append(rng.choice(grouped[label]))
        paths.append(_simulate_pnl_path(pnls, config.initial_capital))
    summary = _summarize_monte_carlo_paths(paths, config)
    summary["target_regime_weights"] = weights
    return summary


def run_remove_best_trades_analysis(
    *,
    trades: Sequence[TradePerformance],
    initial_capital: float,
    start: datetime,
    end: datetime,
    removal_percentages: Sequence[float] = (1.0, 5.0, 10.0),
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    ordered = _ordered_trades(trades)
    results: Dict[str, Any] = {}
    for pct in removal_percentages:
        remove_count = min(len(ordered), int(math.ceil(len(ordered) * (pct / 100.0))))
        removed_ids = {
            id(trade)
            for trade in sorted(ordered, key=lambda item: item.pnl, reverse=True)[
                :remove_count
            ]
        }
        remaining = [trade for trade in ordered if id(trade) not in removed_ids]
        results[f"remove_top_{pct:g}_pct"] = {
            "removed_trade_count": remove_count,
            "remaining_trade_count": len(remaining),
            "statistics": build_performance_statistics(
                trades=remaining,
                initial_capital=initial_capital,
                start=start,
                end=end,
                risk_free_rate=risk_free_rate,
            ),
        }
    return results


def run_monte_carlo_suite(
    *,
    trades: Sequence[TradePerformance],
    config: MonteCarloConfig,
    start: datetime,
    end: datetime,
    risk_free_rate: float = 0.0,
    trade_regimes: Sequence[str] | None = None,
    target_regime_weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    suite: Dict[str, Any] = {
        "historical": build_performance_statistics(
            trades=trades,
            initial_capital=config.initial_capital,
            start=start,
            end=end,
            risk_free_rate=risk_free_rate,
        ),
        "trade_reshuffling": run_trade_reshuffling_monte_carlo(trades, config),
        "block_bootstrap": run_block_bootstrap_monte_carlo(trades, config),
        "slippage": run_slippage_monte_carlo(trades, config),
        "remove_best_trades": run_remove_best_trades_analysis(
            trades=trades,
            initial_capital=config.initial_capital,
            start=start,
            end=end,
            risk_free_rate=risk_free_rate,
        ),
    }
    if trade_regimes is not None:
        suite["regime"] = run_regime_monte_carlo(
            trades,
            trade_regimes,
            config,
            target_regime_weights=target_regime_weights,
        )
    else:
        suite["regime"] = {
            "note": "trade_regimes were not provided; use label_trades_by_regime() or strategy metadata to run regime Monte Carlo"
        }
    return suite


def default_ema_avwap_parameter_perturbation_rules() -> Dict[str, ParameterPerturbationRule]:
    return {
        "ema_length": ParameterPerturbationRule(relative=0.15, minimum=2, integer=True),
        "consecutive_count": ParameterPerturbationRule(absolute=1, minimum=1, integer=True),
        "avwap_multiplier_1": ParameterPerturbationRule(relative=0.10, minimum=0.1),
        "avwap_multiplier_2": ParameterPerturbationRule(relative=0.10, minimum=0.1),
        "avwap_multiplier_3": ParameterPerturbationRule(relative=0.10, minimum=0.1),
        "rigid_stop_loss_pct": ParameterPerturbationRule(relative=0.20, minimum=0.01),
        "position_notional_pct": ParameterPerturbationRule(
            relative=0.20, minimum=0.01
        ),
        "max_entry_deviation_pct": ParameterPerturbationRule(
            relative=0.20, minimum=0.0
        ),
        "trailing_activation_threshold_pct": ParameterPerturbationRule(relative=0.20, minimum=0.0),
        "trailing_gap_pct": ParameterPerturbationRule(relative=0.20, minimum=0.0),
        "maker_fee_pct": ParameterPerturbationRule(relative=0.50, minimum=0.0),
        "taker_fee_pct": ParameterPerturbationRule(relative=0.50, minimum=0.0),
        "entry_slippage_pct": ParameterPerturbationRule(relative=0.50, absolute=0.0001, minimum=0.0),
        "exit_slippage_pct": ParameterPerturbationRule(relative=0.50, absolute=0.0001, minimum=0.0),
    }


def generate_parameter_perturbation_candidates(
    *,
    base_parameters: Mapping[str, Any],
    rules: Mapping[str, ParameterPerturbationRule],
    samples: int,
    seed: int | None = None,
    include_base: bool = True,
) -> List[ParameterCandidate]:
    if samples < 0:
        raise ValueError("samples must be non-negative")
    rng = random.Random(seed)
    candidates: List[ParameterCandidate] = []
    if include_base:
        candidates.append(ParameterCandidate("base", dict(base_parameters)))
    for idx in range(samples):
        params = dict(base_parameters)
        for key, rule in rules.items():
            if key in params:
                params[key] = rule.perturb(params[key], rng)
        candidates.append(ParameterCandidate(f"perturbation_{idx + 1}", params))
    return candidates


def run_parameter_perturbation(
    *,
    build_backtester: BacktesterFactory,
    run_config: BacktestRunConfig,
    base_parameters: Mapping[str, Any],
    rules: Mapping[str, ParameterPerturbationRule],
    samples: int = 100,
    seed: int | None = None,
    score_metric: str = "net_profit_pct",
) -> Dict[str, Any]:
    candidates = generate_parameter_perturbation_candidates(
        base_parameters=base_parameters,
        rules=rules,
        samples=samples,
        seed=seed,
        include_base=True,
    )
    reports = [
        _run_scored_candidate(
            build_backtester=build_backtester,
            candidate=candidate,
            run_config=run_config,
            period=EvaluationPeriod("parameter_perturbation", run_config.start, run_config.end),
            initial_capital=run_config.initial_capital,
            selection_metric=score_metric,
            scorer=None,
        )
        for candidate in candidates
    ]
    scores = [item.score for item in reports if math.isfinite(item.score)]
    return {
        "score_metric": score_metric,
        "sample_count": samples,
        "candidate_count": len(candidates),
        "best": max(reports, key=lambda item: item.score).as_dict(include_report=False) if reports else None,
        "worst": min(reports, key=lambda item: item.score).as_dict(include_report=False) if reports else None,
        "score_p05": _quantile(scores, 0.05) if scores else 0.0,
        "score_p50": _quantile(scores, 0.50) if scores else 0.0,
        "score_p95": _quantile(scores, 0.95) if scores else 0.0,
        "reports": [item.as_dict(include_report=False) for item in reports],
    }


def score_report(report: BacktestReport, metric: str) -> float:
    stats = report.statistics.as_dict()
    value: Any = stats
    for part in metric.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return float("-inf")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(result):
        return float("-inf")
    return result


def _run_scored_candidate(
    *,
    build_backtester: BacktesterFactory,
    candidate: ParameterCandidate,
    run_config: BacktestRunConfig,
    period: EvaluationPeriod,
    initial_capital: float,
    selection_metric: str,
    scorer: ScoreFunction | None,
) -> CandidateReport:
    report = build_backtester(candidate.parameters, initial_capital).run(
        _run_config_for_period(
            run_config=run_config,
            period=period,
            initial_capital=initial_capital,
        )
    )
    score = scorer(report) if scorer else score_report(report, selection_metric)
    return CandidateReport(candidate=candidate, score=score, report=report)


def _run_config_for_period(
    *,
    run_config: BacktestRunConfig,
    period: EvaluationPeriod,
    initial_capital: float,
) -> BacktestRunConfig:
    return replace(
        run_config,
        start=period.start,
        end=period.end,
        initial_capital=initial_capital,
    )


def _simulate_pnl_path(pnls: Sequence[float], initial_capital: float) -> Dict[str, float]:
    equity = initial_capital
    peak = initial_capital
    max_drawdown_pct = 0.0
    min_equity = initial_capital
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        min_equity = min(min_equity, equity)
        drawdown_pct = ((peak - equity) / peak) * 100.0 if peak > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    return {
        "final_equity": equity,
        "net_profit": equity - initial_capital,
        "net_profit_pct": ((equity - initial_capital) / initial_capital) * 100.0,
        "max_drawdown_pct": max_drawdown_pct,
        "min_equity": min_equity,
    }


def _summarize_monte_carlo_paths(
    paths: Sequence[Mapping[str, float]],
    config: MonteCarloConfig,
) -> Dict[str, Any]:
    if not paths:
        return _empty_monte_carlo_summary(config)
    final_equities = [path["final_equity"] for path in paths]
    net_profit_pct = [path["net_profit_pct"] for path in paths]
    drawdowns = [path["max_drawdown_pct"] for path in paths]
    return {
        "iterations": config.iterations,
        "horizon_trades": config.horizon_trades,
        "final_equity_p05": _quantile(final_equities, 0.05),
        "final_equity_p50": _quantile(final_equities, 0.50),
        "final_equity_p95": _quantile(final_equities, 0.95),
        "net_profit_pct_p05": _quantile(net_profit_pct, 0.05),
        "net_profit_pct_p50": _quantile(net_profit_pct, 0.50),
        "net_profit_pct_p95": _quantile(net_profit_pct, 0.95),
        "max_drawdown_pct_p50": _quantile(drawdowns, 0.50),
        "max_drawdown_pct_p95": _quantile(drawdowns, 0.95),
        "max_drawdown_pct_p99": _quantile(drawdowns, 0.99),
        "probability_of_loss": sum(1 for value in final_equities if value < config.initial_capital) / len(paths),
        "probability_of_drawdown_ge_threshold": sum(
            1 for value in drawdowns if value >= config.drawdown_threshold_pct
        ) / len(paths),
        "drawdown_threshold_pct": config.drawdown_threshold_pct,
        "probability_of_ruin": sum(1 for value in final_equities if value <= 0) / len(paths),
    }


def _empty_monte_carlo_summary(config: MonteCarloConfig) -> Dict[str, Any]:
    return {
        "iterations": config.iterations,
        "horizon_trades": config.horizon_trades,
        "note": "no trades available",
    }


def _apply_random_execution_costs(
    trade: TradePerformance,
    rng: random.Random,
    config: MonteCarloConfig,
) -> float:
    pnl = trade.pnl
    metadata = trade.metadata or {}
    fee_drag = _metadata_float(metadata, "entry_fee") + _metadata_float(metadata, "exit_fee")
    fee_multiplier = rng.choice(config.fee_multipliers)
    pnl -= fee_drag * max(fee_multiplier - 1.0, 0.0)

    lo, hi = config.extra_spread_slippage_pct_range
    extra_spread_pct = rng.uniform(lo, hi)
    qty = abs(_metadata_float(metadata, "qty"))
    entry_price = _first_metadata_float(metadata, "entry_price", "entry_raw_price")
    exit_price = _first_metadata_float(metadata, "exit_price", "exit_raw_price")
    pnl -= (entry_price + exit_price) * qty * extra_spread_pct

    miss_fill = rng.random() < config.missed_fill_probability
    if miss_fill and (not config.missed_fill_winner_only or trade.pnl > 0):
        pnl = 0.0
    return pnl


def _normalize_regime_weights(
    grouped: Mapping[str, Sequence[float]],
    target_weights: Mapping[str, float] | None,
) -> Dict[str, float]:
    if target_weights:
        usable = {
            label: max(float(weight), 0.0)
            for label, weight in target_weights.items()
            if label in grouped
        }
    else:
        total = sum(len(values) for values in grouped.values())
        usable = {label: len(values) / total for label, values in grouped.items()}
    total_weight = sum(usable.values())
    if total_weight <= 0:
        raise ValueError("regime weights must contain at least one positive weight")
    return {label: weight / total_weight for label, weight in usable.items()}


def _cumulative_weights(weights: Sequence[float]) -> List[float]:
    total = 0.0
    cumulative: List[float] = []
    for weight in weights:
        total += weight
        cumulative.append(total)
    if cumulative:
        cumulative[-1] = 1.0
    return cumulative


def _weighted_index(draw: float, cumulative_weights: Sequence[float]) -> int:
    for idx, cutoff in enumerate(cumulative_weights):
        if draw <= cutoff:
            return idx
    return len(cumulative_weights) - 1


def _build_regime_segment(
    *,
    candles: Sequence[Candle],
    start_idx: int,
    end_idx: int,
    trend: str,
    volatility: str,
    observations: int,
) -> RegimeSegment:
    return RegimeSegment(
        name=f"{trend}_{volatility}",
        start=_ensure_utc(candles[start_idx].open_time),
        end=_ensure_utc(candles[end_idx].close_time),
        trend=trend,
        volatility=volatility,
        observations=observations,
    )


def _candidate_list(candidates: Sequence[ParameterCandidate] | None) -> List[ParameterCandidate]:
    if not candidates:
        return [ParameterCandidate("fixed", {})]
    return list(candidates)


def _final_equity_from_report(report: BacktestReport, fallback: float) -> float:
    curve = report.statistics.equity_curve
    if curve:
        return curve[-1]
    return fallback + report.statistics.net_profit


def _ordered_trades(trades: Sequence[TradePerformance]) -> List[TradePerformance]:
    return sorted(
        trades,
        key=lambda trade: (_ensure_utc(trade.exit_time), _ensure_utc(trade.entry_time)),
    )


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _population_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(mean((value - avg) ** 2 for value in values))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


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


__all__ = [
    "BacktesterFactory",
    "CandidateReport",
    "EvaluationPeriod",
    "MarketCase",
    "MonteCarloConfig",
    "OutOfSampleEvaluationResult",
    "OutOfSamplePlan",
    "ParameterCandidate",
    "ParameterPerturbationRule",
    "RegimeSegment",
    "UniverseCaseResult",
    "WalkForwardConfig",
    "WalkForwardFoldResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "build_three_way_oos_plan",
    "build_walk_forward_windows",
    "classify_regime_segments",
    "default_ema_avwap_parameter_perturbation_rules",
    "generate_parameter_perturbation_candidates",
    "label_trades_by_regime",
    "run_block_bootstrap_monte_carlo",
    "run_market_universe",
    "run_monte_carlo_suite",
    "run_out_of_sample_evaluation",
    "run_parameter_perturbation",
    "run_regime_monte_carlo",
    "run_remove_best_trades_analysis",
    "run_slippage_monte_carlo",
    "run_trade_reshuffling_monte_carlo",
    "run_walk_forward",
    "score_report",
    "summarize_trades_by_regime",
]
