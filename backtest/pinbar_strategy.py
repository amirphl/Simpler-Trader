from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Mapping, Sequence, Tuple

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .engulfing_strategy import StopLossMode

_EPSILON = 1e-9


@dataclass(frozen=True)
class PinbarStrategyConfig:
    """Configuration for the long-only bullish pinbar strategy."""

    symbol: str
    timeframe: str
    leverage: float
    take_profit_pct: float

    stop_loss_mode: StopLossMode = StopLossMode.PERCENT
    stop_loss_pct: float = 0.005
    exchange_fee_pct: float = 0.0004

    min_shadow_body_ratio: float = 0.5
    shadow_dominance_ratio: float = 2.0
    starting_capital: float = 100.0
    close_open_position_on_finish: bool = True

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        timeframe = self.timeframe.strip()

        if not symbol:
            raise ValueError("symbol must not be empty")
        if not timeframe:
            raise ValueError("timeframe must not be empty")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.stop_loss_mode is StopLossMode.PERCENT and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive for percent mode")
        if self.exchange_fee_pct < 0:
            raise ValueError("exchange_fee_pct must be non-negative")
        if self.min_shadow_body_ratio <= 0:
            raise ValueError("min_shadow_body_ratio must be positive")
        if self.shadow_dominance_ratio <= 0:
            raise ValueError("shadow_dominance_ratio must be positive")
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)


@dataclass(frozen=True)
class PinbarSignal:
    """Normalized bullish pinbar signal derived from a completed candle."""

    index: int
    candle_time: datetime
    body: float
    lower_shadow: float
    upper_shadow: float


@dataclass
class Position:
    """Represents an open long position."""

    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    size: float
    metadata: Mapping[str, float | int | str | None] = field(default_factory=dict)


class PinbarStrategy(BacktestStrategy):
    """Long-only strategy that enters on bullish pinbar reversals."""

    def __init__(self, config: PinbarStrategyConfig) -> None:
        self._config = config
        self._log = logging.getLogger(self.__class__.__name__)

    def name(self) -> str:
        return "PinbarStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        cfg = self._config
        candles = context.data.get(cfg.symbol, {}).get(cfg.timeframe, [])
        ignore_count = context.ignore_candles.get(cfg.symbol, {}).get(cfg.timeframe, 0)
        start_index = max(ignore_count, 2)

        self._log.info(
            "Running pinbar strategy",
            extra={
                "symbol": cfg.symbol,
                "timeframe": cfg.timeframe,
                "length": len(candles),
                "ignore_count": ignore_count,
                "start_index": start_index,
            },
        )

        if len(candles) <= start_index:
            return [], {"note": "insufficient_data", "candles": len(candles)}

        trades: List[TradePerformance] = []
        position: Position | None = None
        current_capital = cfg.starting_capital
        stats: dict[str, int | float | str | bool] = {
            "signals_detected": 0,
            "entries_opened": 0,
            "entries_skipped_invalid_stop": 0,
            "entries_skipped_non_positive_entry": 0,
            "entries_closed_same_bar": 0,
            "open_positions_force_closed": 0,
            "starting_capital": cfg.starting_capital,
            "ending_capital": cfg.starting_capital,
            "close_open_position_on_finish": cfg.close_open_position_on_finish,
        }

        for idx in range(start_index, len(candles)):
            current_candle = candles[idx]

            if position is not None:
                exit_price, exit_reason = self._check_exit(current_candle, position)
                if exit_price is not None and exit_reason is not None:
                    pnl = self._record_exit(
                        candle=current_candle,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        position=position,
                        trades=trades,
                    )
                    current_capital += pnl
                    position = None

            if position is not None:
                continue

            signal = self._build_bullish_pinbar_signal(candles, idx - 1)
            if signal is None:
                continue

            stats["signals_detected"] += 1
            entry_price = float(current_candle.open)
            if entry_price <= 0:
                stats["entries_skipped_non_positive_entry"] += 1
                continue

            signal_candle = candles[signal.index]
            stop_loss = self._compute_stop_loss(entry_price, signal_candle)
            if stop_loss >= entry_price:
                stats["entries_skipped_invalid_stop"] += 1
                continue

            take_profit = self._compute_take_profit(entry_price)
            position = Position(
                entry_time=current_candle.open_time,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=cfg.leverage,
                size=current_capital,
                metadata={
                    "signal_index": signal.index,
                    "signal_open_time": signal.candle_time.isoformat(),
                    "signal_body": signal.body,
                    "signal_lower_shadow": signal.lower_shadow,
                    "signal_upper_shadow": signal.upper_shadow,
                },
            )
            stats["entries_opened"] += 1

            exit_price, exit_reason = self._check_exit(current_candle, position)
            if exit_price is not None and exit_reason is not None:
                pnl = self._record_exit(
                    candle=current_candle,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    position=position,
                    trades=trades,
                )
                current_capital += pnl
                position = None
                stats["entries_closed_same_bar"] += 1

        if position is not None and candles and cfg.close_open_position_on_finish:
            last_candle = candles[-1]
            pnl = self._record_exit(
                candle=last_candle,
                exit_price=last_candle.close,
                exit_reason="End of backtest",
                position=position,
                trades=trades,
                exit_time=last_candle.close_time,
                note_override="End of backtest",
            )
            current_capital += pnl
            stats["open_positions_force_closed"] += 1

        stats["ending_capital"] = current_capital
        return trades, stats

    def _build_bullish_pinbar_signal(
        self, candles: Sequence[Candle], index: int
    ) -> PinbarSignal | None:
        if index <= 0 or index >= len(candles):
            return None

        candle = candles[index]
        previous = candles[index - 1]
        body = abs(candle.close - candle.open)
        if body <= _EPSILON:
            return None
        if candle.close <= candle.open:
            return None

        previous_body = abs(previous.close - previous.open)
        if previous.close >= previous.open or previous_body <= body:
            return None

        lower_shadow, upper_shadow = self._shadow_sizes(candle)
        if lower_shadow <= self._config.min_shadow_body_ratio * body:
            return None
        if lower_shadow <= self._config.shadow_dominance_ratio * max(
            upper_shadow, _EPSILON
        ):
            return None

        return PinbarSignal(
            index=index,
            candle_time=candle.open_time,
            body=body,
            lower_shadow=lower_shadow,
            upper_shadow=upper_shadow,
        )

    def _shadow_sizes(self, candle: Candle) -> tuple[float, float]:
        body_low = min(candle.open, candle.close)
        body_high = max(candle.open, candle.close)
        lower_shadow = max(body_low - candle.low, 0.0)
        upper_shadow = max(candle.high - body_high, 0.0)
        return lower_shadow, upper_shadow

    def _compute_take_profit(self, entry_price: float) -> float:
        return entry_price * (1.0 + self._config.take_profit_pct)

    def _check_exit(
        self, candle: Candle, position: Position
    ) -> tuple[float | None, str | None]:
        if candle.low <= position.stop_loss:
            return position.stop_loss, "Stop Loss"
        if candle.high >= position.take_profit:
            return position.take_profit, "Take Profit"
        return None, None

    def _record_exit(
        self,
        *,
        candle: Candle,
        exit_price: float,
        exit_reason: str,
        position: Position,
        trades: List[TradePerformance],
        exit_time: datetime | None = None,
        note_override: str | None = None,
    ) -> float:
        pnl, fees_paid, return_pct = self._compute_trade_financials(
            exit_price, position
        )
        note = note_override or f"{exit_reason} at {exit_price:.2f}"
        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=exit_time or candle.close_time,
                pnl=pnl,
                return_pct=return_pct,
                notes=note,
                metadata={
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "leverage": position.leverage,
                    "fees": fees_paid,
                    **position.metadata,
                },
            )
        )
        return pnl

    def _compute_trade_financials(
        self, exit_price: float, position: Position
    ) -> tuple[float, float, float]:
        price_change = exit_price - position.entry_price
        gross_pnl = (
            (price_change / position.entry_price) * position.size * position.leverage
        )
        fees_paid = self._calculate_fees(exit_price, position)
        pnl = gross_pnl - fees_paid
        return_pct = (pnl / position.size) * 100.0 if position.size else 0.0
        return pnl, fees_paid, return_pct

    def _calculate_fees(self, exit_price: float, position: Position) -> float:
        fee_pct = self._config.exchange_fee_pct
        if fee_pct <= 0:
            return 0.0
        notional_entry = position.size * position.leverage
        if notional_entry <= 0 or position.entry_price <= 0:
            return 0.0
        quantity = notional_entry / position.entry_price
        entry_fee = notional_entry * fee_pct
        exit_fee = quantity * exit_price * fee_pct
        return entry_fee + exit_fee

    def _compute_stop_loss(self, entry_price: float, signal_candle: Candle) -> float:
        mode = self._config.stop_loss_mode
        if mode is StopLossMode.PERCENT:
            return entry_price * (1.0 - self._config.stop_loss_pct)
        if mode is StopLossMode.CLOSE:
            return signal_candle.close
        if mode is StopLossMode.LOW:
            return signal_candle.low
        if mode is StopLossMode.OPEN:
            return signal_candle.open
        if mode is StopLossMode.BODY:
            body = signal_candle.close - signal_candle.open
            if body <= 0:
                return signal_candle.open
            return signal_candle.close - (self._config.stop_loss_pct * body)
        raise ValueError(f"Unsupported stop-loss mode: {mode}")
