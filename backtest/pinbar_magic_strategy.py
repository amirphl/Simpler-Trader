from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Sequence

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .scalping_FVG_strategy import atr as calc_atr
from .scalping_FVG_strategy import ema as calc_ema
from .scalping_FVG_strategy import sma as calc_sma


@dataclass(frozen=True)
class PinBarMagicStrategyConfig:
    """Configuration for the Pin Bar Magic v1 strategy."""

    symbol: str
    timeframe: str

    equity_risk_pct: float = 3.0
    leverage: float = 1.0
    atr_multiple: float = 0.5
    trail_points: float = 1.0
    trail_offset: float = 1.0

    slow_sma_period: int = 50
    medium_ema_period: int = 18
    fast_ema_period: int = 6
    atr_period: int = 14
    entry_cancel_bars: int = 3

    def __post_init__(self) -> None:
        if self.equity_risk_pct <= 0:
            raise ValueError("equity_risk_pct must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive")
        if self.trail_points <= 0 or self.trail_offset < 0:
            raise ValueError("trail parameters must be non-negative and trail_points > 0")
        if min(
            self.slow_sma_period,
            self.medium_ema_period,
            self.fast_ema_period,
            self.atr_period,
            self.entry_cancel_bars,
        ) <= 0:
            raise ValueError("period-based configuration values must be positive")


@dataclass
class PendingOrder:
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    qty: float
    risk_amount: float
    created_index: int
    expires_after: int  # bars to live
    activate_index: int  # first bar allowed to trigger


@dataclass
class PinBarMagicPosition:
    direction: Literal["long", "short"]
    entry_time: datetime
    entry_index: int
    entry_price: float
    qty: float
    stop_price: float
    highest_price: float
    lowest_price: float
    risk_amount: float


class PinBarMagicStrategy(BacktestStrategy):
    """Implementation of the Pin Bar Magic v1 strategy."""

    def __init__(self, config: PinBarMagicStrategyConfig) -> None:
        self._config = config
        self._log = logging.getLogger(self.__class__.__name__)

    def name(self) -> str:
        return "PinBarMagicStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(self, context: BacktestContext) -> Sequence[TradePerformance]:
        symbol = self._config.symbol
        timeframe = self._config.timeframe
        candles = context.data.get(symbol, {}).get(timeframe, [])
        if len(candles) < max(self._config.slow_sma_period, self._config.atr_period) + 5:
            return []

        closes = [c.close for c in candles]
        fast_ema = calc_ema(closes, self._config.fast_ema_period)
        med_ema = calc_ema(closes, self._config.medium_ema_period)
        slow_sma = calc_sma(closes, self._config.slow_sma_period)
        atr_values = calc_atr(candles, self._config.atr_period)

        trades: List[TradePerformance] = []
        pending_orders: List[PendingOrder] = []
        position: Optional[PinBarMagicPosition] = None
        current_capital = context.config.initial_capital

        start_index = max(
            self._config.slow_sma_period,
            self._config.medium_ema_period,
            self._config.fast_ema_period,
            self._config.atr_period + 1,
        )

        for idx in range(start_index, len(candles)):
            candle = candles[idx]
            fast_prev = fast_ema[idx - 1]
            med_prev = med_ema[idx - 1]
            fast_curr = fast_ema[idx]
            med_curr = med_ema[idx]

            if position:
                exit_price, exit_reason = self._maybe_exit_position(
                    candle, position, fast_prev, fast_curr, med_prev, med_curr
                )
                if exit_price is not None:
                    pnl = self._close_position(position, exit_price, exit_reason or "Exit", candle.close_time, trades)
                    current_capital += pnl
                    position = None

            if position:
                self._update_trailing(position, candle)

            if position is None:
                filled = self._try_fill_orders(pending_orders, idx, candle)
                if filled:
                    position = filled

            self._expire_orders(pending_orders, idx)

            signal_idx = idx
            if position is None:
                self._maybe_create_orders(
                    signal_idx,
                    candles,
                    fast_ema,
                    med_ema,
                    slow_sma,
                    atr_values,
                    pending_orders,
                    current_capital,
                )

        return trades

    # --- Signal helpers -------------------------------------------------

    def _maybe_create_orders(
        self,
        idx: int,
        candles: Sequence[Candle],
        fast_ema: Sequence[Optional[float]],
        med_ema: Sequence[Optional[float]],
        slow_sma: Sequence[Optional[float]],
        atr_values: Sequence[Optional[float]],
        pending_orders: List[PendingOrder],
        current_capital: float,
    ) -> None:
        if idx == 0:
            return
        candle = candles[idx]
        fast = fast_ema[idx]
        med = med_ema[idx]
        slow = slow_sma[idx]
        atr_val = atr_values[idx]
        if any(v is None for v in (fast, med, slow, atr_val)):
            return

        fast = float(fast)
        med = float(med)
        slow = float(slow)
        atr_val = float(atr_val)

        bull_pin = self._is_bullish_pinbar(candle)
        bear_pin = self._is_bearish_pinbar(candle)
        fan_up = fast > med > slow
        fan_down = fast < med < slow
        bull_pierce = self._bull_pierce(candle, fast, med, slow)
        bear_pierce = self._bear_pierce(candle, fast, med, slow)

        risk_amount = (self._config.equity_risk_pct / 100.0) * current_capital

        if fan_up and bull_pin and bull_pierce:
            stop = candle.low - atr_val * self._config.atr_multiple
            entry = candle.high
            qty = self._compute_quantity(entry, stop, risk_amount)
            if qty > 0 and stop < entry:
                pending_orders.append(
                    PendingOrder(
                        direction="long",
                        entry_price=entry,
                        stop_price=stop,
                        qty=qty,
                        risk_amount=risk_amount,
                        created_index=idx,
                        expires_after=self._config.entry_cancel_bars,
                        activate_index=idx + 1,
                    )
                )

        if fan_down and bear_pin and bear_pierce:
            stop = candle.high + atr_val * self._config.atr_multiple
            entry = candle.low
            qty = self._compute_quantity(stop, entry, risk_amount)
            if qty > 0 and stop > entry:
                pending_orders.append(
                    PendingOrder(
                        direction="short",
                        entry_price=entry,
                        stop_price=stop,
                        qty=qty,
                        risk_amount=risk_amount,
                        created_index=idx,
                        expires_after=self._config.entry_cancel_bars,
                        activate_index=idx + 1,
                    )
                )

    def _compute_quantity(self, entry: float, stop: float, risk_amount: float) -> float:
        distance = abs(entry - stop)
        if distance <= 0:
            return 0.0
        return (risk_amount * self._config.leverage) / distance

    def _is_bullish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        cond1 = candle.close > candle.open and (candle.open - candle.low) > 0.66 * rng
        cond2 = candle.close < candle.open and (candle.close - candle.low) > 0.66 * rng
        return cond1 or cond2

    def _is_bearish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        cond1 = candle.close > candle.open and (candle.high - candle.close) > 0.66 * rng
        cond2 = candle.close < candle.open and (candle.high - candle.open) > 0.66 * rng
        return cond1 or cond2

    def _bull_pierce(self, candle: Candle, fast: float, med: float, slow: float) -> bool:
        return any(
            (
                candle.low < ema and candle.open > ema and candle.close > ema
            )
            for ema in (fast, med, slow)
        )

    def _bear_pierce(self, candle: Candle, fast: float, med: float, slow: float) -> bool:
        return any(
            (
                candle.high > ema and candle.open < ema and candle.close < ema
            )
            for ema in (fast, med, slow)
        )

    # --- Pending orders --------------------------------------------------

    def _expire_orders(self, pending_orders: List[PendingOrder], current_index: int) -> None:
        for order in list(pending_orders):
            if current_index - order.created_index > self._config.entry_cancel_bars:
                pending_orders.remove(order)

    def _try_fill_orders(
        self,
        pending_orders: List[PendingOrder],
        idx: int,
        candle: Candle,
    ) -> Optional[PinBarMagicPosition]:
        for order in list(pending_orders):
            if idx < order.activate_index:
                continue
            fill_price = None
            if order.direction == "long":
                if candle.open >= order.entry_price:
                    fill_price = candle.open
                elif candle.high >= order.entry_price:
                    fill_price = order.entry_price
            else:
                if candle.open <= order.entry_price:
                    fill_price = candle.open
                elif candle.low <= order.entry_price:
                    fill_price = order.entry_price

            if fill_price is not None:
                pending_orders.remove(order)
                return PinBarMagicPosition(
                    direction=order.direction,
                    entry_time=candle.open_time,
                    entry_index=idx,
                    entry_price=fill_price,
                    qty=order.qty,
                    stop_price=order.stop_price,
                    highest_price=fill_price,
                    lowest_price=fill_price,
                    risk_amount=order.risk_amount,
                )
        return None

    # --- Exits ------------------------------------------------------------

    def _maybe_exit_position(
        self,
        candle: Candle,
        position: PinBarMagicPosition,
        fast_prev: Optional[float],
        fast_curr: Optional[float],
        med_prev: Optional[float],
        med_curr: Optional[float],
    ) -> tuple[Optional[float], Optional[str]]:
        if self._is_market_close(candle.open_time):
            return candle.open, "Market close"

        if (
            position.direction == "long"
            and fast_prev is not None
            and fast_curr is not None
            and med_prev is not None
            and med_curr is not None
            and fast_prev >= med_prev
            and fast_curr < med_curr
        ):
            return candle.open, "EMA crossunder"

        if (
            position.direction == "short"
            and fast_prev is not None
            and fast_curr is not None
            and med_prev is not None
            and med_curr is not None
            and fast_prev <= med_prev
            and fast_curr > med_curr
        ):
            return candle.open, "EMA crossover"

        # Stop / trailing
        if position.direction == "long":
            if candle.low <= position.stop_price:
                exit_price = position.stop_price
                if candle.open < position.stop_price:
                    exit_price = candle.open
                return exit_price, "Stop loss"
        else:
            if candle.high >= position.stop_price:
                exit_price = position.stop_price
                if candle.open > position.stop_price:
                    exit_price = candle.open
                return exit_price, "Stop loss"

        return None, None

    def _update_trailing(self, position: PinBarMagicPosition, candle: Candle) -> None:
        trail_total = self._config.trail_points + self._config.trail_offset
        if position.direction == "long":
            position.highest_price = max(position.highest_price, candle.high)
            candidate = position.highest_price - trail_total
            if candidate > position.stop_price:
                position.stop_price = candidate
        else:
            position.lowest_price = min(position.lowest_price, candle.low)
            candidate = position.lowest_price + trail_total
            if candidate < position.stop_price:
                position.stop_price = candidate

    def _close_position(
        self,
        position: PinBarMagicPosition,
        exit_price: float,
        reason: str,
        exit_time: datetime,
        trades: List[TradePerformance],
    ) -> float:
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.qty
        else:
            pnl = (position.entry_price - exit_price) * position.qty

        return_pct = 0.0
        if position.risk_amount > 0:
            return_pct = (pnl / position.risk_amount) * 100.0

        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=exit_time,
                pnl=pnl,
                return_pct=return_pct,
                notes=reason,
                metadata={
                    "direction": position.direction,
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "qty": position.qty,
                    "stop_loss": position.stop_price,
                    "risk_amount": position.risk_amount,
                },
            )
        )
        return pnl

    # --- Utilities -------------------------------------------------------

    def _is_market_close(self, moment: datetime) -> bool:
        return moment.weekday() == 4 and moment.hour == 16

