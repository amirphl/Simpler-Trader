from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from candle_downloader.binance import MAX_BATCH
from candle_downloader.downloader import CandleDownloader, DownloadRequest
from candle_downloader.models import Candle
from candle_downloader.storage import CandleStore

CandleMatrix = Dict[str, Dict[str, List[Candle]]]
StrategyRunResult = Tuple[Sequence["TradePerformance"], Mapping[str, Any] | None]


@dataclass(frozen=True)
class BacktestRunConfig:
    """Shared configuration for all backtest executions."""

    start: datetime
    end: datetime
    initial_capital: float = 100.0
    override_download: bool = False
    max_batch: int = MAX_BATCH
    risk_free_rate: float = 0.0
    warmup_days: int = 30

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_batch <= 0 or self.max_batch > MAX_BATCH:
            raise ValueError(f"max_batch must live in 1..{MAX_BATCH}")
        if self.warmup_days < 0:
            raise ValueError("warmup_days must be non-negative")


@dataclass(frozen=True)
class TradePerformance:
    """Atomic performance outcome produced by the strategy."""

    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    notes: str | None = None
    metadata: Mapping[str, str | float | int | None] | None = None

    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


@dataclass(frozen=True)
class BacktestContext:
    """Container holding immutable references passed to strategies."""

    config: BacktestRunConfig
    data: CandleMatrix
    ignore_candles: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class BacktestStatistics:
    """Aggregated statistics derived from strategy trade results."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    average_return_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    cagr_pct: float = 0.0
    average_trade_duration_sec: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_profit": self.net_profit,
            "net_profit_pct": self.net_profit_pct,
            "average_return_pct": self.average_return_pct,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "cagr_pct": self.cagr_pct,
            "average_trade_duration_sec": self.average_trade_duration_sec,
            "equity_curve": self.equity_curve,
        }
        payload.update(self.extra)
        return payload


@dataclass
class BacktestReport:
    """Final artifact returned by the base backtester."""

    strategy_name: str
    config: BacktestRunConfig
    statistics: BacktestStatistics
    trades: Sequence[TradePerformance]

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy_name,
            "config": serialize_run_config(self.config),
            "statistics": self.statistics.as_dict(),
            "trades": [
                {
                    "index": idx,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct,
                    "notes": trade.notes,
                    "metadata": dict(trade.metadata or {}),
                }
                for idx, trade in enumerate(self.trades)
            ],
        }


class BacktestStrategy(ABC):
    """Contract every strategy must fulfill to run inside BaseBacktester."""

    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def symbols(self) -> Sequence[str]:
        """Return the tradeable instruments that this strategy needs."""

    @abstractmethod
    def timeframes(self) -> Sequence[str]:
        """Return the Binance-compatible intervals needed by the strategy."""

    @abstractmethod
    def run(self, context: BacktestContext) -> StrategyRunResult:
        """Execute backtest logic and return trade outcomes and optional extra stats."""


class BaseBacktester:
    """Coordinates candle acquisition and invokes the concrete strategy."""

    def __init__(
        self,
        *,
        strategy: BacktestStrategy,
        downloader: CandleDownloader,
        store: CandleStore,
        logger: logging.Logger | None = None,
    ) -> None:
        self._strategy = strategy
        self._downloader = downloader
        self._store = store
        self._log = logger or logging.getLogger(__name__)

    def run(self, config: BacktestRunConfig) -> BacktestReport:
        symbols = _require_non_empty(self._strategy.symbols(), "symbol")
        timeframes = _require_non_empty(self._strategy.timeframes(), "timeframe")
        normalized_config = normalize_run_config(config)

        data = self._download_data(
            symbols=symbols,
            timeframes=timeframes,
            start=compute_warmup_start(normalized_config),
            end=normalized_config.end,
            config=normalized_config,
        )
        context = self._build_context(normalized_config, data)
        trades, custom_stats = self._parse_strategy_result(self._strategy.run(context))
        statistics = self._build_statistics(trades, normalized_config)
        if custom_stats:
            statistics.extra.update(dict(custom_stats))
        return BacktestReport(
            strategy_name=self._strategy.name(),
            config=normalized_config,
            statistics=statistics,
            trades=trades,
        )

    def _build_context(
        self, config: BacktestRunConfig, data: CandleMatrix
    ) -> BacktestContext:
        return BacktestContext(
            config=config,
            data=data,
            ignore_candles=build_ignore_candles(data, config.start),
        )

    def _parse_strategy_result(
        self, result: StrategyRunResult
    ) -> Tuple[Tuple[TradePerformance, ...], Mapping[str, Any] | None]:
        raw_trades, extra_stats = result
        trades = tuple(validate_trade_sequence(raw_trades))
        return trades, extra_stats

    def _download_data(
        self,
        *,
        symbols: Sequence[str],
        timeframes: Sequence[str],
        start: datetime,
        end: datetime,
        config: BacktestRunConfig,
    ) -> CandleMatrix:
        dataset: CandleMatrix = {}
        for symbol in symbols:
            dataset[symbol] = {}
            for timeframe in timeframes:
                self._log.info(
                    "Syncing candles",
                    extra={
                        "symbol": symbol,
                        "interval": timeframe,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "override": config.override_download,
                    },
                )
                request = DownloadRequest(
                    symbol=symbol,
                    interval=timeframe,
                    start=start,
                    end=end,
                    override=config.override_download,
                    max_batch=config.max_batch,
                )
                self._downloader.sync(request)
                dataset[symbol][timeframe] = self._store.load(
                    symbol, timeframe, start, end
                )
        return dataset

    def _build_statistics(
        self,
        trades: Sequence[TradePerformance],
        config: BacktestRunConfig,
    ) -> BacktestStatistics:
        initial_capital = config.initial_capital
        stats = BacktestStatistics(
            total_trades=len(trades),
            equity_curve=[initial_capital],
        )
        if not trades:
            return stats

        stats.winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        stats.losing_trades = sum(1 for trade in trades if trade.pnl < 0)
        stats.win_rate = stats.winning_trades / stats.total_trades
        stats.gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        stats.gross_loss = sum(trade.pnl for trade in trades if trade.pnl < 0)
        stats.net_profit = stats.gross_profit + stats.gross_loss
        stats.net_profit_pct = (stats.net_profit / initial_capital) * 100
        stats.average_return_pct = mean(trade.return_pct for trade in trades)
        stats.expectancy = stats.net_profit / stats.total_trades
        stats.average_trade_duration_sec = mean(
            trade.duration_seconds() for trade in trades
        )
        stats.profit_factor = compute_profit_factor(
            stats.gross_profit, stats.gross_loss
        )
        stats.equity_curve, stats.max_drawdown_pct = build_equity_curve(
            initial_capital=initial_capital,
            trades=trades,
        )
        stats.cagr_pct = compute_cagr_pct(
            initial_capital=initial_capital,
            final_equity=stats.equity_curve[-1],
            start=config.start,
            end=config.end,
        )
        stats.sharpe_ratio = compute_sharpe_ratio(
            trades=trades,
            risk_free_rate=getattr(config, "risk_free_rate", 0.0),
            average_trade_duration_sec=stats.average_trade_duration_sec,
        )
        return stats


def normalize_run_config(config: BacktestRunConfig) -> BacktestRunConfig:
    return replace(
        config,
        start=_ensure_utc(
            config.start.replace(hour=0, minute=0, second=0, microsecond=0)
        ),
        end=_ensure_utc(
            config.end.replace(hour=23, minute=59, second=59, microsecond=999999)
        ),
    )


def serialize_run_config(config: BacktestRunConfig) -> Dict[str, object]:
    return {
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "initial_capital": config.initial_capital,
        "override_download": config.override_download,
        "max_batch": config.max_batch,
        "risk_free_rate": config.risk_free_rate,
        "warmup_days": config.warmup_days,
    }


def compute_warmup_start(config: BacktestRunConfig) -> datetime:
    if config.warmup_days <= 0:
        return config.start
    return config.start - timedelta(days=config.warmup_days)


def build_ignore_candles(
    data: CandleMatrix, trading_start: datetime
) -> Dict[str, Dict[str, int]]:
    return {
        symbol: {
            timeframe: sum(1 for candle in candles if candle.close_time < trading_start)
            for timeframe, candles in timeframe_map.items()
        }
        for symbol, timeframe_map in data.items()
    }


def validate_trade_sequence(
    trades: Sequence[TradePerformance],
) -> Sequence[TradePerformance]:
    validated: List[TradePerformance] = []
    for trade in trades:
        entry_time = _ensure_utc(trade.entry_time)
        exit_time = _ensure_utc(trade.exit_time)
        if exit_time < entry_time:
            raise ValueError(
                "trade exit_time must be greater than or equal to entry_time"
            )
        if not math.isfinite(trade.pnl):
            raise ValueError("trade pnl must be finite")
        if not math.isfinite(trade.return_pct):
            raise ValueError("trade return_pct must be finite")
        validated.append(replace(trade, entry_time=entry_time, exit_time=exit_time))
    validated.sort(key=lambda trade: (trade.exit_time, trade.entry_time))
    return validated


def build_equity_curve(
    *, initial_capital: float, trades: Sequence[TradePerformance]
) -> Tuple[List[float], float]:
    equity_curve = [initial_capital]
    equity = initial_capital
    peak = initial_capital
    max_drawdown = 0.0

    for trade in trades:
        equity += trade.pnl
        equity_curve.append(equity)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    return equity_curve, max_drawdown * 100


def compute_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss < 0:
        return gross_profit / abs(gross_loss) if gross_profit > 0 else 0.0
    if gross_profit > 0:
        return float("inf")
    return 0.0


def compute_cagr_pct(
    *,
    initial_capital: float,
    final_equity: float,
    start: datetime,
    end: datetime,
) -> float:
    duration_years = (end - start).total_seconds() / (365.25 * 24 * 3600)
    if duration_years <= 0 or final_equity <= 0:
        return 0.0
    return ((final_equity / initial_capital) ** (1 / duration_years) - 1) * 100


def compute_sharpe_ratio(
    *,
    trades: Sequence[TradePerformance],
    risk_free_rate: float | None = None,
    average_trade_duration_sec: float,
) -> float:
    returns = [trade.return_pct / 100 for trade in trades]
    if len(returns) <= 1:
        return 0.0

    avg_return = mean(returns)
    variance = mean((value - avg_return) ** 2 for value in returns)
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        return 0.0

    avg_trade_years = average_trade_duration_sec / (365.25 * 24 * 3600)
    trades_per_year = (1.0 / avg_trade_years) if avg_trade_years > 0 else len(trades)
    annual_risk_free_rate = risk_free_rate or 0.0
    rf_per_trade = (
        annual_risk_free_rate * avg_trade_years if avg_trade_years > 0 else 0.0
    )
    excess_return = avg_return - rf_per_trade
    return (excess_return / std_dev) * math.sqrt(trades_per_year)


def _require_non_empty(values: Sequence[str], label: str) -> List[str]:
    items = [value.strip() for value in values if value and value.strip()]
    if not items:
        raise ValueError(f"strategy must provide at least one {label}")
    return items


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


__all__ = [
    "BacktestContext",
    "BacktestReport",
    "BacktestRunConfig",
    "BacktestStatistics",
    "BacktestStrategy",
    "BaseBacktester",
    "CandleMatrix",
    "TradePerformance",
    "build_equity_curve",
    "build_ignore_candles",
    "compute_cagr_pct",
    "compute_profit_factor",
    "compute_sharpe_ratio",
    "compute_warmup_start",
    "normalize_run_config",
    "serialize_run_config",
]
