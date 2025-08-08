from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Dict, List, Mapping, Sequence

from candle_downloader.binance import MAX_BATCH
from candle_downloader.downloader import CandleDownloader, DownloadRequest
from candle_downloader.models import Candle
from candle_downloader.storage import CandleStore

CandleMatrix = Dict[str, Dict[str, List[Candle]]]


@dataclass(frozen=True)
class BacktestRunConfig:
    """Shared configuration for all backtest executions."""

    start: datetime
    end: datetime
    initial_capital: float = 10_000.0
    override_download: bool = False
    max_batch: int = MAX_BATCH
    risk_free_rate: float = 0.0  # expressed as decimal annual rate, e.g. 0.02 = 2%

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_batch <= 0 or self.max_batch > MAX_BATCH:
            raise ValueError(f"max_batch must live in 1..{MAX_BATCH}")


@dataclass(frozen=True)
class TradePerformance:
    """Atomic performance outcome produced by the strategy."""

    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    notes: str | None = None
    metadata: Mapping[str, str | float | int] | None = None

    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


@dataclass(frozen=True)
class BacktestContext:
    """Container holding immutable references passed to strategies."""

    config: BacktestRunConfig
    data: CandleMatrix


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

    def as_dict(self) -> Dict[str, float | int | List[float]]:
        return {
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
            "config": {
                "start": self.config.start.isoformat(),
                "end": self.config.end.isoformat(),
                "initial_capital": self.config.initial_capital,
                "override_download": self.config.override_download,
                "max_batch": self.config.max_batch,
                "risk_free_rate": self.config.risk_free_rate,
            },
            "statistics": self.statistics.as_dict(),
            "trades": [
                {
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct,
                    "notes": trade.notes,
                    "metadata": dict(trade.metadata or {}),
                }
                for trade in self.trades
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
    def run(self, context: BacktestContext) -> Sequence[TradePerformance]:
        """Execute backtest logic and return immutable trade outcomes."""


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
        strategy_symbols = list(self._strategy.symbols())
        timeframes = list(self._strategy.timeframes())
        if not strategy_symbols:
            raise ValueError("strategy must provide at least one symbol")
        if not timeframes:
            raise ValueError("strategy must provide at least one timeframe")

        start = _ensure_utc(config.start)
        end = _ensure_utc(config.end)
        data = self._download_data(strategy_symbols, timeframes, start, end, config)
        context = BacktestContext(config=config, data=data)
        trades = tuple(self._strategy.run(context))
        statistics = self._build_statistics(trades, config)
        return BacktestReport(
            strategy_name=self._strategy.name(),
            config=config,
            statistics=statistics,
            trades=trades,
        )

    def _download_data(
        self,
        symbols: Sequence[str],
        timeframes: Sequence[str],
        start: datetime,
        end: datetime,
        config: BacktestRunConfig,
    ) -> CandleMatrix:
        dataset: CandleMatrix = {}
        for symbol in symbols:
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
                candles = self._store.load(symbol, timeframe, start, end)
                dataset.setdefault(symbol, {})[timeframe] = candles
        return dataset

    def _build_statistics(
        self,
        trades: Sequence[TradePerformance],
        config: BacktestRunConfig,
    ) -> BacktestStatistics:
        stats = BacktestStatistics()
        stats.total_trades = len(trades)
        if not trades:
            stats.equity_curve = [config.initial_capital]
            stats.cagr_pct = 0.0
            stats.net_profit = 0.0
            stats.net_profit_pct = 0.0
            return stats

        stats.winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        stats.losing_trades = sum(1 for trade in trades if trade.pnl < 0)
        stats.win_rate = stats.winning_trades / stats.total_trades
        stats.gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        stats.gross_loss = sum(trade.pnl for trade in trades if trade.pnl < 0)
        stats.net_profit = stats.gross_profit + stats.gross_loss
        stats.net_profit_pct = (stats.net_profit / config.initial_capital) * 100

        stats.average_return_pct = mean(trade.return_pct for trade in trades)
        stats.expectancy = stats.net_profit / stats.total_trades
        stats.average_trade_duration_sec = mean(trade.duration_seconds() for trade in trades)

        if stats.gross_loss < 0:
            stats.profit_factor = stats.gross_profit / abs(stats.gross_loss) if stats.gross_profit > 0 else 0.0
        elif stats.gross_profit > 0:
            stats.profit_factor = float("inf")
        else:
            stats.profit_factor = 0.0

        equity = config.initial_capital
        peak = equity
        stats.equity_curve = [equity]
        max_drawdown = 0.0
        for trade in trades:
            equity += trade.pnl
            stats.equity_curve.append(equity)
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        stats.max_drawdown_pct = max_drawdown * 100

        final_equity = stats.equity_curve[-1]
        duration_years = (config.end - config.start).total_seconds() / (365.25 * 24 * 3600)
        if duration_years > 0 and final_equity > 0:
            stats.cagr_pct = ((final_equity / config.initial_capital) ** (1 / duration_years) - 1) * 100
        else:
            stats.cagr_pct = 0.0

        returns = [trade.return_pct / 100 for trade in trades]
        if len(returns) > 1:
            avg_return = mean(returns)
            variance = mean((r - avg_return) ** 2 for r in returns)
            std_dev = math.sqrt(variance)
            rf = config.risk_free_rate
            stats.sharpe_ratio = (avg_return - rf) / std_dev if std_dev else 0.0
        else:
            stats.sharpe_ratio = 0.0

        return stats


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)

