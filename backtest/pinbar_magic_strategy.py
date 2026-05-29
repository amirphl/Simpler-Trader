"""Pin Bar Magic backtest strategy – merged v2/v3 implementation.

Bug-fix log (relative to the original two-file split):
  1. _compute_quantity no longer multiplies by leverage.  Pine's sizing formula
     is units = risk / distance; leverage determines exchange margin, not units.
  2. gross_loss_long / gross_loss_short now accumulate abs(pnl) so they are
     positive magnitudes as the name implies.
  3. return_pct is now the percentage price return:
     (exit − entry) / entry × 100 for longs,
     (entry − exit) / entry × 100 for shorts.
  4. Same-bar re-entry after close_all is suppressed.  When an EMA cross or
     Friday-close fires in step 3, any signal from step 5 of the same bar is
     not filled in same_bar activation mode.  Pending orders are also
     immediately cancelled when close_all fires, matching Pine semantics.
  5. _fill_pending_entries_for_bar breaks after the first fill so both pending
     orders cannot execute in the same bar (Pine pyramiding=0 semantics).
  6. _compute_quantity now takes explicit (entry_price, stop_price, risk_amount)
     arguments, removing the confusing positional numerator/denominator API.
  7. symbol_mintick is stored in the config as a tick size; conversion to
     absolute price happens at use-time via trail_points_price / trail_offset_price
     properties, consistent with the live coordinator.  The original V3 approach
     of baking the multiplication into a replace()-d frozen dataclass is removed.
  8. trail_offset validator now requires > 0.  A zero offset makes the trailing
     stop equal to the peak price and causes an immediate exit on every tick back.
  9. initial_equity is now a configurable field (default 100.0).
 10. avg_return_pct added to summary stats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .indicators import atr as calc_atr
from .indicators import ema as calc_ema
from .indicators import sma as calc_sma


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinBarMagicStrategyConfig:
    """Unified, fully-configurable Pin Bar Magic backtest configuration.

    Trail parameters
    ----------------
    trail_points and trail_offset are expressed in TICKS, matching Pine Script's
    slPoints / slOffset inputs (integer pip counts).  The strategy multiplies them
    by symbol_mintick at use-time to obtain absolute price distances.

    Example – ETHUSDT, mintick = 0.01 USDT:
        trail_points = 100  →  100 × 0.01 = 1.00 USDT activation gap
        trail_offset  =  50  →   50 × 0.01 = 0.50 USDT trailing gap

    To keep values in raw price units (e.g. configs written before mintick
    support), set symbol_mintick = 1.0 (the default).
    """

    symbol: str
    timeframe: str

    # --- Equity / risk ---
    initial_equity: float = 100.0
    equity_risk_pct: float = 3.0
    # leverage is recorded for reporting only; it does NOT multiply position size.
    leverage: float = 1.0

    # --- Core Pine-compatible inputs ---
    atr_multiple: float = 0.5
    trail_points: float = 1.0  # ticks; must be > 0
    trail_offset: float = 1.0  # ticks; must be > 0
    symbol_mintick: float = 1.0  # absolute price per tick (syminfo.mintick in Pine)

    # --- Indicator periods ---
    slow_sma_period: int = 50
    medium_ema_period: int = 18
    fast_ema_period: int = 6
    atr_period: int = 14
    entry_cancel_bars: int = 3

    # --- Entry / exit extensions ---
    entry_activation_mode: Literal["next_bar", "same_bar"] = "next_bar"
    use_stop_fill_open_gap: bool = True

    enable_friday_close: bool = True
    friday_close_hour_utc: int = 16
    enable_ema_cross_close: bool = True

    # --- Risk equity options ---
    risk_equity_include_unrealized: bool = True
    risk_equity_mark_source: Literal["close", "open", "hl2", "ohlc4"] = "close"

    # --- Tick-emulation trailing ---
    trailing_tick_timeframe: str = "15m"
    use_trailing_tick_emulation: bool = False

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.equity_risk_pct <= 0:
            raise ValueError("equity_risk_pct must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive")
        if self.trail_points <= 0:
            raise ValueError("trail_points must be positive")
        if self.trail_offset <= 0:
            # An offset of 0 sets the trailing stop equal to the peak price and
            # exits the position on the very first adverse tick — never intended.
            raise ValueError("trail_offset must be positive (> 0)")
        if self.symbol_mintick <= 0:
            raise ValueError("symbol_mintick must be positive")
        if (
            min(
                self.slow_sma_period,
                self.medium_ema_period,
                self.fast_ema_period,
                self.atr_period,
                self.entry_cancel_bars,
            )
            <= 0
        ):
            raise ValueError("all period-based values must be positive integers")
        if not (0 <= self.friday_close_hour_utc <= 23):
            raise ValueError("friday_close_hour_utc must be in 0..23")
        if self.risk_equity_mark_source not in {"close", "open", "hl2", "ohlc4"}:
            raise ValueError(
                "risk_equity_mark_source must be one of: close, open, hl2, ohlc4"
            )
        if self.entry_activation_mode not in {"next_bar", "same_bar"}:
            raise ValueError("entry_activation_mode must be one of: next_bar, same_bar")
        interval_to_milliseconds(self.timeframe)
        if self.use_trailing_tick_emulation:
            interval_to_milliseconds(self.trailing_tick_timeframe)

    # ------------------------------------------------------------------
    # Derived trail distances in absolute price units
    # ------------------------------------------------------------------

    @property
    def trail_points_price(self) -> float:
        """Trailing activation distance in absolute price (ticks × mintick)."""
        return self.trail_points * self.symbol_mintick

    @property
    def trail_offset_price(self) -> float:
        """Trailing gap behind the peak in absolute price (ticks × mintick)."""
        return self.trail_offset * self.symbol_mintick


# ---------------------------------------------------------------------------
# Internal state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PendingOrder:
    direction: Literal["long", "short"]
    entry_price: float
    qty: float
    risk_amount: float
    created_index: int
    activate_index: int


@dataclass
class PositionState:
    direction: Literal["long", "short"]
    entry_time: datetime
    entry_index: int
    entry_price: float
    qty: float
    risk_amount: float

    trailing_active: bool = False
    trailing_stop: Optional[float] = None
    extreme_since_activation: Optional[float] = None


@dataclass(frozen=True)
class PriceTick:
    time: datetime
    price: float


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class PinBarMagicStrategy(BacktestStrategy):
    """TradingView Pin Bar Magic backtest strategy (merged v2 / v3).

    All behaviour is controlled by PinBarMagicStrategyConfig.
    """

    def __init__(self, config: PinBarMagicStrategyConfig) -> None:
        self._config = config
        self._log = logging.getLogger(self.__class__.__name__)

    def name(self) -> str:
        return "PinBarMagicStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        if (
            self._config.use_trailing_tick_emulation
            and self._config.trailing_tick_timeframe != self._config.timeframe
        ):
            return [self._config.timeframe, self._config.trailing_tick_timeframe]
        return [self._config.timeframe]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        cfg = self._config
        symbol = cfg.symbol
        timeframe = cfg.timeframe

        candles: Sequence[Candle] = context.data.get(symbol, {}).get(timeframe, [])
        min_history = (
            max(
                cfg.slow_sma_period,
                cfg.medium_ema_period,
                cfg.fast_ema_period,
                cfg.atr_period,
            )
            + 2
        )
        if len(candles) < min_history:
            return [], {"note": "insufficient_data", "candles": len(candles)}

        tick_buckets: Optional[List[List[PriceTick]]] = None
        if cfg.use_trailing_tick_emulation:
            tick_candles = self._resolve_tick_candles(context, candles)
            tick_buckets = self._build_tick_buckets(candles, tick_candles)

        closes = [c.close for c in candles]
        fast_ema = calc_ema(closes, cfg.fast_ema_period)
        med_ema = calc_ema(closes, cfg.medium_ema_period)
        slow_sma = calc_sma(closes, cfg.slow_sma_period)
        atr_values = calc_atr(candles, cfg.atr_period)

        trades: List[TradePerformance] = []
        position: Optional[PositionState] = None
        pending_long: Optional[PendingOrder] = None
        pending_short: Optional[PendingOrder] = None
        last_long_signal_index: Optional[int] = None
        last_short_signal_index: Optional[int] = None

        realized_equity = cfg.initial_equity
        stats: Dict[str, Any] = {
            "initial_equity": cfg.initial_equity,
            "leverage": cfg.leverage,
            "symbol_mintick": cfg.symbol_mintick,
            "trail_points_ticks": cfg.trail_points,
            "trail_offset_ticks": cfg.trail_offset,
            "trail_points_price": cfg.trail_points_price,
            "trail_offset_price": cfg.trail_offset_price,
            "signals_long": 0,
            "signals_short": 0,
            "orders_placed_long": 0,
            "orders_placed_short": 0,
            "orders_replaced_long": 0,
            "orders_replaced_short": 0,
            "orders_cancelled_long": 0,
            "orders_cancelled_short": 0,
            "orders_filled_long": 0,
            "orders_filled_short": 0,
            "signals_skipped_same_direction_position": 0,
            "entries_long": 0,
            "entries_short": 0,
            "reversal_entries": 0,
            "trailing_activations": 0,
            "trailing_updates": 0,
            "trailing_exits": 0,
            "ema_cross_exits": 0,
            "friday_close_exits": 0,
            "total_ticks_processed": 0,
            "tick_source": (
                cfg.trailing_tick_timeframe
                if cfg.use_trailing_tick_emulation
                else cfg.timeframe
            ),
            "tick_emulation_enabled": cfg.use_trailing_tick_emulation,
            "trailing_engine": (
                "close_tick_emulation"
                if cfg.use_trailing_tick_emulation
                else "pine_bar_emulation"
            ),
            "entry_activation_mode": cfg.entry_activation_mode,
            "risk_equity_include_unrealized": cfg.risk_equity_include_unrealized,
            "risk_equity_mark_source": cfg.risk_equity_mark_source,
            "exit_reason_counts": {},
        }

        start_index = (
            max(
                cfg.slow_sma_period,
                cfg.medium_ema_period,
                cfg.fast_ema_period,
                cfg.atr_period,
            )
            + 1
        )

        for idx in range(start_index, len(candles)):
            candle = candles[idx]
            prev_candle = candles[idx - 1]

            fast = fast_ema[idx]
            med = med_ema[idx]
            slow = slow_sma[idx]
            atr_prev = atr_values[idx - 1]
            fast_prev = fast_ema[idx - 1]
            med_prev = med_ema[idx - 1]

            if fast is None or med is None or slow is None or atr_prev is None:
                continue

            fast = float(fast)
            med = float(med)
            slow = float(slow)
            atr_prev = float(atr_prev)

            # ----------------------------------------------------------
            # Step 1 – Trailing stop processing on the current bar.
            #
            # Trailing fires before entry fills so that a position stopped
            # out on bar N does not block an entry signal that also fires
            # on bar N.
            # ----------------------------------------------------------
            if position is not None:
                if cfg.use_trailing_tick_emulation:
                    assert tick_buckets is not None
                    exit_price, exit_time = self._process_trailing_ticks(
                        position=position,
                        ticks=tick_buckets[idx],
                        stats=stats,
                    )
                else:
                    exit_price, exit_time = self._process_trailing_bar(
                        position=position,
                        candle=candle,
                        stats=stats,
                    )
                if exit_price is not None:
                    assert exit_time is not None
                    pnl = self._close_position(
                        position=position,
                        exit_price=exit_price,
                        exit_time=exit_time,
                        exit_index=idx,
                        reason="Trailing stop",
                        trades=trades,
                        stats=stats,
                    )
                    realized_equity += pnl
                    position = None

            # ----------------------------------------------------------
            # Step 2 – Fill pending stop entries (next_bar activation).
            # ----------------------------------------------------------
            pending_long, pending_short, position, realized_equity = (
                self._fill_pending_entries_for_bar(
                    idx=idx,
                    candle=candle,
                    pending_long=pending_long,
                    pending_short=pending_short,
                    position=position,
                    realized_equity=realized_equity,
                    trades=trades,
                    stats=stats,
                )
            )

            # ----------------------------------------------------------
            # Step 3 – Bar-close exit rules (strategy.close_all).
            #
            # bar_close_fired is set to True when either EMA cross or
            # Friday-close fires.  Steps 5 and 6 use this flag to suppress
            # same-bar re-entry, matching Pine's implicit strategy.cancel on
            # strategy.close_all.
            # ----------------------------------------------------------
            bar_close_fired = False
            if position is not None:
                close_reason: Optional[str] = None
                if self._is_friday_close(candle.close_time):
                    close_reason = "Market close"
                elif (
                    cfg.enable_ema_cross_close
                    and fast_prev is not None
                    and med_prev is not None
                ):
                    fp = float(fast_prev)
                    mp = float(med_prev)
                    if fp > mp and fast < med:
                        close_reason = "EMA crossunder"
                    elif fp < mp and fast > med:
                        close_reason = "EMA crossover"

                if close_reason is not None:
                    pnl = self._close_position(
                        position=position,
                        exit_price=candle.close,
                        exit_time=candle.close_time,
                        exit_index=idx,
                        reason=close_reason,
                        trades=trades,
                        stats=stats,
                    )
                    realized_equity += pnl
                    if close_reason in {"EMA crossunder", "EMA crossover"}:
                        stats["ema_cross_exits"] += 1
                    else:
                        stats["friday_close_exits"] += 1
                    position = None
                    bar_close_fired = True

                    # Cancel all pending entries immediately — Pine's close_all
                    # implicitly cancels pending stop orders on the same bar.
                    if pending_long is not None:
                        pending_long = None
                        stats["orders_cancelled_long"] += 1
                    if pending_short is not None:
                        pending_short = None
                        stats["orders_cancelled_short"] += 1

            # ----------------------------------------------------------
            # Step 4 – Cancel stale pending orders.
            # ----------------------------------------------------------
            if pending_long is not None and last_long_signal_index is not None:
                if idx - last_long_signal_index > cfg.entry_cancel_bars:
                    pending_long = None
                    stats["orders_cancelled_long"] += 1
            if pending_short is not None and last_short_signal_index is not None:
                if idx - last_short_signal_index > cfg.entry_cancel_bars:
                    pending_short = None
                    stats["orders_cancelled_short"] += 1

            # ----------------------------------------------------------
            # Step 5 – Generate signals and (re)place stop orders.
            #
            # Signals are generated even when bar_close_fired=True because the
            # entry will activate on bar N+1, not this bar.  Suppression of
            # same-bar fills happens in Step 6.
            # ----------------------------------------------------------
            bull_pin = self._is_bullish_pinbar(candle)
            bear_pin = self._is_bearish_pinbar(candle)
            fan_up = fast > med > slow
            fan_down = fast < med < slow
            bull_pierce = self._bull_pierce(candle, fast, med, slow)
            bear_pierce = self._bear_pierce(candle, fast, med, slow)

            long_entry = fan_up and bull_pin and bull_pierce
            short_entry = fan_down and bear_pin and bear_pierce

            if long_entry:
                stats["signals_long"] += 1
                last_long_signal_index = idx
                if position is not None and position.direction == "long":
                    stats["signals_skipped_same_direction_position"] += 1
                else:
                    risk_equity = self._risk_equity_for_signal(
                        realized_equity=realized_equity,
                        position=position,
                        signal_candle=candle,
                    )
                    risk_amount = (cfg.equity_risk_pct / 100.0) * risk_equity
                    entry_price = float(prev_candle.high)
                    stop_for_risk = float(prev_candle.low) - atr_prev * cfg.atr_multiple
                    if entry_price > stop_for_risk:
                        qty = self._compute_quantity(
                            entry_price=entry_price,
                            stop_price=stop_for_risk,
                            risk_amount=risk_amount,
                        )
                        if qty > 0:
                            if pending_long is not None:
                                stats["orders_replaced_long"] += 1
                            pending_long = PendingOrder(
                                direction="long",
                                entry_price=entry_price,
                                qty=qty,
                                risk_amount=risk_amount,
                                created_index=idx,
                                activate_index=self._entry_activate_index(idx),
                            )
                            stats["orders_placed_long"] += 1

            if short_entry:
                stats["signals_short"] += 1
                last_short_signal_index = idx
                if position is not None and position.direction == "short":
                    stats["signals_skipped_same_direction_position"] += 1
                else:
                    risk_equity = self._risk_equity_for_signal(
                        realized_equity=realized_equity,
                        position=position,
                        signal_candle=candle,
                    )
                    risk_amount = (cfg.equity_risk_pct / 100.0) * risk_equity
                    entry_price = float(prev_candle.low)
                    stop_for_risk = (
                        float(prev_candle.high) + atr_prev * cfg.atr_multiple
                    )
                    if stop_for_risk > entry_price:
                        qty = self._compute_quantity(
                            entry_price=entry_price,
                            stop_price=stop_for_risk,
                            risk_amount=risk_amount,
                        )
                        if qty > 0:
                            if pending_short is not None:
                                stats["orders_replaced_short"] += 1
                            pending_short = PendingOrder(
                                direction="short",
                                entry_price=entry_price,
                                qty=qty,
                                risk_amount=risk_amount,
                                created_index=idx,
                                activate_index=self._entry_activate_index(idx),
                            )
                            stats["orders_placed_short"] += 1

            # ----------------------------------------------------------
            # Step 6 – Same-bar activation (optional).
            #
            # Suppressed when bar_close_fired=True: Pine's strategy.close_all
            # does not allow same-bar re-entry from a signal generated on the
            # same bar.
            # ----------------------------------------------------------
            if cfg.entry_activation_mode == "same_bar" and not bar_close_fired:
                pending_long, pending_short, position, realized_equity = (
                    self._fill_pending_entries_for_bar(
                        idx=idx,
                        candle=candle,
                        pending_long=pending_long,
                        pending_short=pending_short,
                        position=position,
                        realized_equity=realized_equity,
                        trades=trades,
                        stats=stats,
                    )
                )

        stats["pending_orders_at_end"] = int(pending_long is not None) + int(
            pending_short is not None
        )
        stats["open_position_at_end"] = position.direction if position else None
        stats["final_equity"] = realized_equity
        stats.update(self._summarize_trade_stats(trades))

        return trades, stats

    # ------------------------------------------------------------------
    # Signal / price logic helpers
    # ------------------------------------------------------------------

    def _compute_quantity(
        self,
        entry_price: float,
        stop_price: float,
        risk_amount: float,
    ) -> float:
        """Compute position size matching Pine's unit formula.

        Pine:  units = risk / abs(entryPrice − stopLoss)

        risk_amount  – maximum loss in quote currency (e.g. USDT).
        distance     – per-unit loss at the hard stop, also in quote currency.

        Leverage is NOT applied here.  It is a margin parameter that tells the
        exchange how much collateral to require; it does not change the number
        of base-asset units for a given dollar risk.
        """
        distance = abs(entry_price - stop_price)
        if distance <= 0 or risk_amount <= 0:
            return 0.0
        return risk_amount / distance

    def _is_bullish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        return (
            candle.close > candle.open and (candle.open - candle.low) > 0.66 * rng
        ) or (candle.close < candle.open and (candle.close - candle.low) > 0.66 * rng)

    def _is_bearish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        return (
            candle.close > candle.open and (candle.high - candle.close) > 0.66 * rng
        ) or (candle.close < candle.open and (candle.high - candle.open) > 0.66 * rng)

    def _bull_pierce(
        self, candle: Candle, fast: float, med: float, slow: float
    ) -> bool:
        return any(
            candle.low < ma and candle.open > ma and candle.close > ma
            for ma in (fast, med, slow)
        )

    def _bear_pierce(
        self, candle: Candle, fast: float, med: float, slow: float
    ) -> bool:
        return any(
            candle.high > ma and candle.open < ma and candle.close < ma
            for ma in (fast, med, slow)
        )

    # ------------------------------------------------------------------
    # Tick-emulation trailing
    # ------------------------------------------------------------------

    def _resolve_tick_candles(
        self, context: BacktestContext, primary: Sequence[Candle]
    ) -> Sequence[Candle]:
        cfg = self._config
        if (
            not cfg.use_trailing_tick_emulation
            or cfg.trailing_tick_timeframe == cfg.timeframe
        ):
            return primary
        ticks = context.data.get(cfg.symbol, {}).get(cfg.trailing_tick_timeframe, [])
        if ticks:
            return ticks
        self._log.warning(
            "Trailing tick timeframe %s not available for %s; falling back to %s candles",
            cfg.trailing_tick_timeframe,
            cfg.symbol,
            cfg.timeframe,
        )
        return primary

    def _build_tick_buckets(
        self, primary: Sequence[Candle], ticks: Sequence[Candle]
    ) -> List[List[PriceTick]]:
        sorted_ticks = sorted(ticks, key=lambda c: c.close_time)
        buckets: List[List[PriceTick]] = []
        tick_idx = 0

        for idx, bar in enumerate(primary):
            lower_bound = primary[idx - 1].close_time if idx > 0 else bar.open_time
            upper_bound = bar.close_time
            bucket: List[PriceTick] = []

            while (
                tick_idx < len(sorted_ticks)
                and sorted_ticks[tick_idx].close_time <= upper_bound
            ):
                tick = sorted_ticks[tick_idx]
                if tick.close_time > lower_bound:
                    bucket.append(PriceTick(time=tick.close_time, price=tick.close))
                tick_idx += 1

            if not bucket:
                bucket = [PriceTick(time=bar.close_time, price=bar.close)]
            buckets.append(bucket)

        return buckets

    def _process_trailing_ticks(
        self,
        *,
        position: PositionState,
        ticks: Sequence[PriceTick],
        stats: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[datetime]]:
        """Tick-resolution trailing stop emulation.

        Trail distances are derived from config properties (ticks × mintick) so
        they are always in absolute price units regardless of symbol_mintick.
        """
        trail_points = self._config.trail_points_price
        trail_offset = self._config.trail_offset_price

        for tick in ticks:
            stats["total_ticks_processed"] += 1
            price = tick.price

            if position.direction == "long":
                activation = position.entry_price + trail_points
                if not position.trailing_active and price >= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = price
                    position.trailing_stop = price - trail_offset
                    stats["trailing_activations"] += 1
                elif position.trailing_active:
                    assert position.extreme_since_activation is not None
                    prev_stop = position.trailing_stop
                    if price > position.extreme_since_activation:
                        position.extreme_since_activation = price
                        position.trailing_stop = price - trail_offset
                    if position.trailing_stop != prev_stop:
                        stats["trailing_updates"] += 1

                if (
                    position.trailing_active
                    and position.trailing_stop is not None
                    and price <= position.trailing_stop
                ):
                    stats["trailing_exits"] += 1
                    return price, tick.time

            else:  # short
                activation = position.entry_price - trail_points
                if not position.trailing_active and price <= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = price
                    position.trailing_stop = price + trail_offset
                    stats["trailing_activations"] += 1
                elif position.trailing_active:
                    assert position.extreme_since_activation is not None
                    prev_stop = position.trailing_stop
                    if price < position.extreme_since_activation:
                        position.extreme_since_activation = price
                        position.trailing_stop = price + trail_offset
                    if position.trailing_stop != prev_stop:
                        stats["trailing_updates"] += 1

                if (
                    position.trailing_active
                    and position.trailing_stop is not None
                    and price >= position.trailing_stop
                ):
                    stats["trailing_exits"] += 1
                    return price, tick.time

        return None, None

    def _process_trailing_bar(
        self,
        *,
        position: PositionState,
        candle: Candle,
        stats: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[datetime]]:
        """Bar-resolution trailing stop (TradingView broker-emulator heuristic).

        Approximates the intra-bar price path as:
            open → (extreme nearest to open) → (other extreme) → close

        Trail distances are derived from config properties (ticks × mintick) so
        they are always in absolute price units regardless of symbol_mintick.
        """
        trail_points = self._config.trail_points_price
        trail_offset = self._config.trail_offset_price

        # Visit the wick closest to the open price first.
        if abs(candle.open - candle.high) < abs(candle.open - candle.low):
            sequence = ("high", "low", "close")
        else:
            sequence = ("low", "high", "close")

        start_price = candle.open

        for node in sequence:
            end_price = getattr(candle, node)
            seg_high = max(start_price, end_price)
            seg_low = min(start_price, end_price)

            if position.direction == "long":
                activation = position.entry_price + trail_points
                if not position.trailing_active and seg_high >= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = seg_high
                    position.trailing_stop = seg_high - trail_offset
                    stats["trailing_activations"] += 1

                if position.trailing_active:
                    prev_stop = position.trailing_stop
                    if position.extreme_since_activation is None:
                        position.extreme_since_activation = seg_high
                    elif seg_high > position.extreme_since_activation:
                        position.extreme_since_activation = seg_high
                    desired_stop = position.extreme_since_activation - trail_offset
                    if (
                        position.trailing_stop is None
                        or desired_stop > position.trailing_stop
                    ):
                        position.trailing_stop = desired_stop
                    if position.trailing_stop != prev_stop:
                        stats["trailing_updates"] += 1

                    if seg_low <= position.trailing_stop:
                        stats["trailing_exits"] += 1
                        return position.trailing_stop, candle.close_time

            else:  # short
                activation = position.entry_price - trail_points
                if not position.trailing_active and seg_low <= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = seg_low
                    position.trailing_stop = seg_low + trail_offset
                    stats["trailing_activations"] += 1

                if position.trailing_active:
                    prev_stop = position.trailing_stop
                    if position.extreme_since_activation is None:
                        position.extreme_since_activation = seg_low
                    elif seg_low < position.extreme_since_activation:
                        position.extreme_since_activation = seg_low
                    desired_stop = position.extreme_since_activation + trail_offset
                    if (
                        position.trailing_stop is None
                        or desired_stop < position.trailing_stop
                    ):
                        position.trailing_stop = desired_stop
                    if position.trailing_stop != prev_stop:
                        stats["trailing_updates"] += 1

                    if seg_high >= position.trailing_stop:
                        stats["trailing_exits"] += 1
                        return position.trailing_stop, candle.close_time

            start_price = end_price

        return None, None

    # ------------------------------------------------------------------
    # Order handling
    # ------------------------------------------------------------------

    def _entry_activate_index(self, signal_index: int) -> int:
        if self._config.entry_activation_mode == "same_bar":
            return signal_index
        return signal_index + 1

    def _fill_pending_entries_for_bar(
        self,
        *,
        idx: int,
        candle: Candle,
        pending_long: Optional[PendingOrder],
        pending_short: Optional[PendingOrder],
        position: Optional[PositionState],
        realized_equity: float,
        trades: List[TradePerformance],
        stats: Dict[str, Any],
    ) -> Tuple[
        Optional[PendingOrder], Optional[PendingOrder], Optional[PositionState], float
    ]:
        """Attempt to fill pending stop-entry orders for this bar.

        Only ONE fill is allowed per call (Pine pyramiding=0 semantics).
        Long is checked first; if it fills, the short check is skipped for
        this bar so that both opposing stops cannot execute in the same candle.
        """
        if pending_long is not None and idx >= pending_long.activate_index:
            fill_price = self._stop_fill_price(pending_long, candle)
            if fill_price is not None:
                position, realized_equity = self._enter_stop_position(
                    current=position,
                    direction="long",
                    fill_price=fill_price,
                    order=pending_long,
                    candle=candle,
                    idx=idx,
                    trades=trades,
                    equity=realized_equity,
                    stats=stats,
                )
                pending_long = None
                # One fill per bar — do not check the short order this bar.
                return pending_long, pending_short, position, realized_equity

        if pending_short is not None and idx >= pending_short.activate_index:
            fill_price = self._stop_fill_price(pending_short, candle)
            if fill_price is not None:
                position, realized_equity = self._enter_stop_position(
                    current=position,
                    direction="short",
                    fill_price=fill_price,
                    order=pending_short,
                    candle=candle,
                    idx=idx,
                    trades=trades,
                    equity=realized_equity,
                    stats=stats,
                )
                pending_short = None

        return pending_long, pending_short, position, realized_equity

    def _stop_fill_price(self, order: PendingOrder, candle: Candle) -> Optional[float]:
        """Return the fill price for a stop order, or None if not triggered this bar."""
        if order.direction == "long":
            # Gap-up open: if the candle opens at or above the trigger, fill at open.
            if self._config.use_stop_fill_open_gap and candle.open >= order.entry_price:
                return candle.open
            if candle.high >= order.entry_price:
                return order.entry_price
            return None

        # short
        if self._config.use_stop_fill_open_gap and candle.open <= order.entry_price:
            return candle.open
        if candle.low <= order.entry_price:
            return order.entry_price
        return None

    def _enter_stop_position(
        self,
        *,
        current: Optional[PositionState],
        direction: Literal["long", "short"],
        fill_price: float,
        order: PendingOrder,
        candle: Candle,
        idx: int,
        trades: List[TradePerformance],
        equity: float,
        stats: Dict[str, Any],
    ) -> Tuple[Optional[PositionState], float]:
        position = current
        updated_equity = equity

        if direction == "long":
            stats["orders_filled_long"] += 1
        else:
            stats["orders_filled_short"] += 1

        if position is not None:
            if position.direction == direction:
                # Same direction already open; pyramiding disabled — no-op.
                return position, updated_equity

            # Reversal: close existing position at the fill price.
            pnl = self._close_position(
                position=position,
                exit_price=fill_price,
                exit_time=candle.open_time,
                exit_index=idx,
                reason="Reversal",
                trades=trades,
                stats=stats,
            )
            updated_equity += pnl
            stats["reversal_entries"] += 1
            position = None

        new_position = PositionState(
            direction=direction,
            entry_time=candle.open_time,
            entry_index=idx,
            entry_price=fill_price,
            qty=order.qty,
            risk_amount=order.risk_amount,
        )
        if direction == "long":
            stats["entries_long"] += 1
        else:
            stats["entries_short"] += 1
        return new_position, updated_equity

    # ------------------------------------------------------------------
    # PnL / stat helpers
    # ------------------------------------------------------------------

    def _close_position(
        self,
        *,
        position: PositionState,
        exit_price: float,
        exit_time: datetime,
        exit_index: int,
        reason: str,
        trades: List[TradePerformance],
        stats: Dict[str, Any],
    ) -> float:
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.qty
            return_pct = (
                (exit_price - position.entry_price) / position.entry_price * 100.0
            )
        else:
            pnl = (position.entry_price - exit_price) * position.qty
            return_pct = (
                (position.entry_price - exit_price) / position.entry_price * 100.0
            )

        r_multiple = pnl / position.risk_amount if position.risk_amount > 0 else 0.0
        holding_bars = max(exit_index - position.entry_index, 0)

        metadata: Dict[str, float | int | str | bool | None] = {
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "qty": position.qty,
            "risk_amount": position.risk_amount,
            "r_multiple": r_multiple,
            "holding_bars": holding_bars,
            "trailing_active": position.trailing_active,
            "trailing_stop": position.trailing_stop,
            "reason": reason,
        }

        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=exit_time,
                pnl=pnl,
                return_pct=return_pct,
                notes=reason,
                metadata=metadata,
            )
        )

        reason_counts = stats["exit_reason_counts"]
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        return pnl

    def _equity_with_unrealized(
        self,
        realized_equity: float,
        position: Optional[PositionState],
        mark_price: float,
    ) -> float:
        if position is None:
            return realized_equity
        if position.direction == "long":
            unrealized = (mark_price - position.entry_price) * position.qty
        else:
            unrealized = (position.entry_price - mark_price) * position.qty
        return realized_equity + unrealized

    def _risk_equity_for_signal(
        self,
        *,
        realized_equity: float,
        position: Optional[PositionState],
        signal_candle: Candle,
    ) -> float:
        cfg = self._config
        if not cfg.risk_equity_include_unrealized or position is None:
            return realized_equity
        mark_price = self._mark_price_for_equity(signal_candle)
        return self._equity_with_unrealized(realized_equity, position, mark_price)

    def _mark_price_for_equity(self, candle: Candle) -> float:
        source = self._config.risk_equity_mark_source
        if source == "open":
            return candle.open
        if source == "hl2":
            return (candle.high + candle.low) / 2.0
        if source == "ohlc4":
            return (candle.open + candle.high + candle.low + candle.close) / 4.0
        return candle.close

    def _summarize_trade_stats(
        self, trades: Sequence[TradePerformance]
    ) -> Dict[str, Any]:
        if not trades:
            return {
                "trade_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_r_multiple": 0.0,
                "avg_return_pct": 0.0,
                "avg_holding_bars": 0.0,
                "long_trade_count": 0,
                "short_trade_count": 0,
                "long_win_rate": 0.0,
                "short_win_rate": 0.0,
                "gross_profit_long": 0.0,
                "gross_profit_short": 0.0,
                "gross_loss_long": 0.0,
                "gross_loss_short": 0.0,
            }

        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl < 0)

        r_values: List[float] = []
        ret_pcts: List[float] = []
        hold_bars: List[int] = []
        long_trades: List[TradePerformance] = []
        short_trades: List[TradePerformance] = []
        gross_profit_long = 0.0
        gross_profit_short = 0.0
        # gross_loss values are positive magnitudes (absolute values of losses).
        gross_loss_long = 0.0
        gross_loss_short = 0.0

        for trade in trades:
            md = dict(trade.metadata or {})
            direction = str(md.get("direction", ""))
            r_values.append(float(md.get("r_multiple", 0.0)))
            ret_pcts.append(float(trade.return_pct or 0.0))
            hold_bars.append(int(md.get("holding_bars", 0)))

            if direction == "long":
                long_trades.append(trade)
                if trade.pnl >= 0:
                    gross_profit_long += trade.pnl
                else:
                    gross_loss_long += abs(trade.pnl)
            elif direction == "short":
                short_trades.append(trade)
                if trade.pnl >= 0:
                    gross_profit_short += trade.pnl
                else:
                    gross_loss_short += abs(trade.pnl)

        long_wins = sum(1 for t in long_trades if t.pnl > 0)
        short_wins = sum(1 for t in short_trades if t.pnl > 0)

        return {
            "trade_count": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trades),
            "avg_r_multiple": mean(r_values) if r_values else 0.0,
            "avg_return_pct": mean(ret_pcts) if ret_pcts else 0.0,
            "avg_holding_bars": mean(hold_bars) if hold_bars else 0.0,
            "best_trade_pnl": max(t.pnl for t in trades),
            "worst_trade_pnl": min(t.pnl for t in trades),
            "long_trade_count": len(long_trades),
            "short_trade_count": len(short_trades),
            "long_win_rate": long_wins / len(long_trades) if long_trades else 0.0,
            "short_win_rate": short_wins / len(short_trades) if short_trades else 0.0,
            "gross_profit_long": gross_profit_long,
            "gross_profit_short": gross_profit_short,
            "gross_loss_long": gross_loss_long,
            "gross_loss_short": gross_loss_short,
        }

    def _is_friday_close(self, moment: datetime) -> bool:
        if not self._config.enable_friday_close:
            return False
        return (
            moment.weekday() == 4 and moment.hour == self._config.friday_close_hour_utc
        )
