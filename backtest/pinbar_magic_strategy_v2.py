from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Sequence, Tuple, Mapping, Any

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .scalping_FVG_strategy import atr as calc_atr
from .scalping_FVG_strategy import ema as calc_ema
from .scalping_FVG_strategy import sma as calc_sma


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinBarMagicStrategyConfigV2:
    """Configuration for the Pin Bar Magic v1 strategy (Pine-equivalent)."""

    symbol: str
    timeframe: str

    # === Inputs from Pine ===
    equity_risk_pct: float = 3.0           # usr_risk
    atr_multiple: float = 0.5              # atr_mult
    trail_points: float = 1.0              # slPoints (used as price units)
    trail_offset: float = 1.0              # slOffset (used as price units)

    slow_sma_period: int = 50              # sma_slow
    medium_ema_period: int = 18            # ema_medm
    fast_ema_period: int = 6               # ema_fast
    atr_period: int = 14                   # atr_valu
    entry_cancel_bars: int = 3             # ent_canc

    # extra: leverage multiplier (Pine had no leverage; keep = 1.0 for parity)
    leverage: float = 1.0

    def __post_init__(self) -> None:
        if self.equity_risk_pct <= 0:
            raise ValueError("equity_risk_pct must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive")
        if self.trail_points <= 0 or self.trail_offset < 0:
            raise ValueError("trail_points must be > 0 and trail_offset >= 0")
        if min(
            self.slow_sma_period,
            self.medium_ema_period,
            self.fast_ema_period,
            self.atr_period,
            self.entry_cancel_bars,
        ) <= 0:
            raise ValueError("period-based configuration values must be positive")


# ---------------------------------------------------------------------------
# Internal state models
# ---------------------------------------------------------------------------


@dataclass
class PendingOrder:
    direction: Literal["long", "short"]
    entry_price: float
    qty: float
    risk_amount: float
    created_index: int       # bar index where order was created
    activate_index: int      # first bar index where it can be filled (next bar)


@dataclass
class PinBarMagicPosition:
    direction: Literal["long", "short"]
    entry_time: datetime
    entry_index: int
    entry_price: float
    qty: float
    risk_amount: float

    # trailing stop state (strategy.exit with trail_points & trail_offset)
    trailing_active: bool = False
    trailing_stop: Optional[float] = None
    extreme_since_activation: Optional[float] = None  # high (long) / low (short)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class PinBarMagicStrategyV2(BacktestStrategy):
    """Implementation of TradingView 'Pin Bar Magic v1' strategy in Python."""

    def __init__(self, config: PinBarMagicStrategyConfigV2) -> None:
        self._config = config
        self._log = logging.getLogger(self.__class__.__name__)

    def name(self) -> str:
        return "PinBarMagicStrategyV2"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    # ------------------------------------------------------------------ run

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        symbol = self._config.symbol
        timeframe = self._config.timeframe
        candles: Sequence[Candle] = context.data.get(symbol, {}).get(timeframe, [])
        if len(candles) < max(self._config.slow_sma_period, self._config.atr_period) + 5:
            return [], None

        closes = [c.close for c in candles]
        fast_ema = calc_ema(closes, self._config.fast_ema_period)
        med_ema = calc_ema(closes, self._config.medium_ema_period)
        slow_sma = calc_sma(closes, self._config.slow_sma_period)
        atr_values = calc_atr(candles, self._config.atr_period)

        trades: List[TradePerformance] = []
        position: Optional[PinBarMagicPosition] = None

        # One pending "long" and one pending "short" – mirrors strategy.entry ids "long"/"short"
        pending_long: Optional[PendingOrder] = None
        pending_short: Optional[PendingOrder] = None

        # For barssince(longEntry) / barssince(shortEntry)
        last_long_signal_index: Optional[int] = None
        last_short_signal_index: Optional[int] = None

        equity = context.config.initial_capital  # realized-equity; same as strategy.equity when flat

        # We need one extra bar because we reference [1] in ATR/high/low
        start_index = max(
            self._config.slow_sma_period,
            self._config.medium_ema_period,
            self._config.fast_ema_period,
            self._config.atr_period,
        ) + 1

        for idx in range(start_index, len(candles)):
            candle = candles[idx]
            prev_candle = candles[idx - 1]

            fast = fast_ema[idx]
            med = med_ema[idx]
            slow = slow_sma[idx]
            atr_prev = atr_values[idx - 1]  # Pine: atr[1] inside enterlong/entershort

            if (
                fast is None
                or med is None
                or slow is None
                or atr_prev is None
            ):
                continue

            fast = float(fast)
            med = float(med)
            slow = float(slow)
            atr_prev = float(atr_prev)

            fast_prev = fast_ema[idx - 1]
            med_prev = med_ema[idx - 1]

            # ------------------------------------------------------------------
            # 0) Trailing stop exits (strategy.exit with trail_points & trail_offset)
            #    These are STOP orders, so we treat them as same-bar exits when
            #    high/low crosses the trailing level.
            # ------------------------------------------------------------------
            if position is not None:
                self._update_trailing(position, candle)
                exit_price, exit_reason = self._check_trailing_exit(position, candle)
                if exit_price is not None:
                    reason = exit_reason or "Trailing stop"
                    pnl = self._close_position(
                        position,
                        exit_price=exit_price,
                        reason=reason,
                        exit_time=candle.close_time,
                        trades=trades,
                    )
                    equity += pnl
                    position = None

            # ------------------------------------------------------------------
            # 1) strategy.close_all conditions (Friday close, EMA cross)
            #    These are MARKET orders that execute on the same bar close.
            # ------------------------------------------------------------------
            if position is not None:
                close_reason: Optional[str] = None
                if self._is_market_close(candle.close_time):
                    close_reason = "Market close"
                else:
                    if fast_prev is not None and med_prev is not None:
                        fast_prev_f = float(fast_prev)
                        med_prev_f = float(med_prev)
                        crossunder = fast_prev_f > med_prev_f and fast <= med
                        crossover = fast_prev_f < med_prev_f and fast >= med
                        if crossunder:
                            close_reason = "EMA crossunder"
                        elif crossover:
                            close_reason = "EMA crossover"

                if close_reason:
                    pnl = self._close_position(
                        position,
                        exit_price=candle.close,
                        reason=close_reason,
                        exit_time=candle.close_time,
                        trades=trades,
                    )
                    equity += pnl
                    position = None

            # ------------------------------------------------------------------
            # 3) Try to fill pending STOP-entry orders (strategy.entry with stop=)
            #    Orders are active from activate_index (next bar after creation).
            #    We fill at the stop price when crossed by high/low. Orders can
            #    reverse the current position, mirroring strategy.entry behavior.
            # ------------------------------------------------------------------
            if pending_long is not None and idx >= pending_long.activate_index:
                if candle.high >= pending_long.entry_price:
                    fill_price = pending_long.entry_price
                    position, equity = self._enter_stop_position(
                        current=position,
                        direction="long",
                        fill_price=fill_price,
                        order=pending_long,
                        candle=candle,
                        idx=idx,
                        trades=trades,
                        equity=equity,
                    )
                    pending_long = None

            if pending_short is not None and idx >= pending_short.activate_index:
                if candle.low <= pending_short.entry_price:
                    fill_price = pending_short.entry_price
                    position, equity = self._enter_stop_position(
                        current=position,
                        direction="short",
                        fill_price=fill_price,
                        order=pending_short,
                        candle=candle,
                        idx=idx,
                        trades=trades,
                        equity=equity,
                    )
                    pending_short = None

            # ------------------------------------------------------------------
            # 4) Cancel stale orders (strategy.cancel("id", barssince(...) > ent_canc))
            #    barssince(cond) returns 0 on signal bar, then 1,2,... afterwards.
            # ------------------------------------------------------------------
            if pending_long is not None and last_long_signal_index is not None:
                if idx - last_long_signal_index > self._config.entry_cancel_bars:
                    pending_long = None

            if pending_short is not None and last_short_signal_index is not None:
                if idx - last_short_signal_index > self._config.entry_cancel_bars:
                    pending_short = None

            # ------------------------------------------------------------------
            # 5) Generate new entry signals and create Pine-equivalent STOP orders
            #    Only when FLAT (pyramiding default 1 => no new entries while in a trade).
            # ------------------------------------------------------------------
            if position is None:
                bull_pin = self._is_bullish_pinbar(candle)
                bear_pin = self._is_bearish_pinbar(candle)
                fan_up = fast > med > slow
                fan_down = fast < med < slow
                bull_pierce = self._bull_pierce(candle, fast, med, slow)
                bear_pierce = self._bear_pierce(candle, fast, med, slow)

                long_entry = fan_up and bull_pin and bull_pierce
                short_entry = fan_down and bear_pin and bear_pierce

                # risk = usr_risk * 0.01 * strategy.equity
                risk_amount = (self._config.equity_risk_pct / 100.0) * equity

                if long_entry:
                    last_long_signal_index = idx
                    # Pine: uses low[1], high[1], atr[1] for stop sizing & entry
                    entry_price = prev_candle.high
                    stop_for_risk = prev_candle.low - atr_prev * self._config.atr_multiple
                    if entry_price > stop_for_risk:
                        qty = self._compute_quantity(entry_price, stop_for_risk, risk_amount)
                        if qty > 0:
                            pending_long = PendingOrder(
                                direction="long",
                                entry_price=entry_price,
                                qty=qty,
                                risk_amount=risk_amount,
                                created_index=idx,
                                activate_index=idx + 1,  # next bar (no same-bar fill)
                            )

                if short_entry:
                    last_short_signal_index = idx
                    entry_price = prev_candle.low
                    stop_for_risk = prev_candle.high + atr_prev * self._config.atr_multiple
                    if stop_for_risk > entry_price:
                        qty = self._compute_quantity(stop_for_risk, entry_price, risk_amount)
                        if qty > 0:
                            pending_short = PendingOrder(
                                direction="short",
                                entry_price=entry_price,
                                qty=qty,
                                risk_amount=risk_amount,
                                created_index=idx,
                                activate_index=idx + 1,
                            )

        return trades, None

    # ------------------------------------------------------------------ helpers

    def _compute_quantity(self, entry: float, stop: float, risk_amount: float) -> float:
        distance = abs(entry - stop)
        if distance <= 0:
            return 0.0
        # leverage is extra; set leverage=1.0 to match Pine
        return (risk_amount * self._config.leverage) / distance

    # --- Pattern logic (matches Pine boolean expressions) ---------------------

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
        # ((low < MA) and (open > MA) and (close > MA)) for any MA
        return any(
            (candle.low < ma and candle.open > ma and candle.close > ma)
            for ma in (fast, med, slow)
        )

    def _bear_pierce(self, candle: Candle, fast: float, med: float, slow: float) -> bool:
        # ((high > MA) and (open < MA) and (close < MA)) for any MA
        return any(
            (candle.high > ma and candle.open < ma and candle.close < ma)
            for ma in (fast, med, slow)
        )

    # --- Trailing stop emulation of strategy.exit(..., trail_points, trail_offset)

    def _update_trailing(self, position: PinBarMagicPosition, candle: Candle) -> None:
        cfg = self._config
        if position.direction == "long":
            activation = position.entry_price + cfg.trail_points
            if not position.trailing_active:
                # Activate trailing when price moves trail_points in profit
                if candle.high >= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = candle.high
                    position.trailing_stop = candle.high - cfg.trail_offset
            else:
                assert position.extreme_since_activation is not None
                position.extreme_since_activation = max(position.extreme_since_activation, candle.high)
                new_stop = position.extreme_since_activation - cfg.trail_offset
                if position.trailing_stop is None or new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop
        else:
            activation = position.entry_price - cfg.trail_points
            if not position.trailing_active:
                if candle.low <= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = candle.low
                    position.trailing_stop = candle.low + cfg.trail_offset
            else:
                assert position.extreme_since_activation is not None
                position.extreme_since_activation = min(position.extreme_since_activation, candle.low)
                new_stop = position.extreme_since_activation + cfg.trail_offset
                if position.trailing_stop is None or new_stop < position.trailing_stop:
                    position.trailing_stop = new_stop

    def _check_trailing_exit(
        self, position: PinBarMagicPosition, candle: Candle
    ) -> tuple[Optional[float], Optional[str]]:
        if not position.trailing_active or position.trailing_stop is None:
            return None, None

        if position.direction == "long":
            if candle.low <= position.trailing_stop:
                return position.trailing_stop, "Trailing stop"
        else:
            if candle.high >= position.trailing_stop:
                return position.trailing_stop, "Trailing stop"

        return None, None

    # --- Close position & bookkeeping ----------------------------------------

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

        ret_pct = (pnl / position.risk_amount) * 100.0 if position.risk_amount > 0 else 0.0

        metadata: dict[str, float | int | str] = {
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "qty": position.qty,
            "risk_amount": position.risk_amount,
        }
        if position.trailing_stop is not None:
            metadata["trailing_stop"] = position.trailing_stop

        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=exit_time,
                pnl=pnl,
                return_pct=ret_pct,
                notes=reason,
                metadata=metadata,
            )
        )
        return pnl

    def _enter_stop_position(
        self,
        *,
        current: Optional[PinBarMagicPosition],
        direction: Literal["long", "short"],
        fill_price: float,
        order: PendingOrder,
        candle: Candle,
        idx: int,
        trades: List[TradePerformance],
        equity: float,
    ) -> tuple[Optional[PinBarMagicPosition], float]:
        """Handle STOP-order fills, respecting pyramiding=1 semantics."""
        position = current
        updated_equity = equity

        if position is not None:
            if position.direction == direction:
                return position, updated_equity
            pnl = self._close_position(
                position,
                exit_price=fill_price,
                reason="Reversal",
                exit_time=candle.close_time,
                trades=trades,
            )
            updated_equity += pnl
            position = None

        new_position = PinBarMagicPosition(
            direction=direction,
            entry_time=candle.close_time,
            entry_index=idx,
            entry_price=fill_price,
            qty=order.qty,
            risk_amount=order.risk_amount,
        )
        return new_position, updated_equity

    # --- Time helpers --------------------------------------------------------

    def _is_market_close(self, moment: datetime) -> bool:
        # Pine: hour == 16 and dayofweek == Friday (dayofweek.friday)
        # Python datetime.weekday(): Monday=0 ... Sunday=6 -> Friday=4
        return moment.weekday() == 4 and moment.hour == 16
