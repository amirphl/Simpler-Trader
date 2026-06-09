"""Backtest-aligned AVWAP, sizing, trailing, and utility helpers."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List, Literal, Optional, Sequence, Tuple

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle

from ..exchange import Position, PositionSide
from ..models import PendingEntryRecord, PositionRecord
from .config import Direction
from ._mixin_typing import EmaAvwapMixinTyping
from .state import _AvwapSnapshot, _CrossDecision, _PositionRuntime, _SetupState, _SizingDecision


class EmaAvwapCalculationMixin(EmaAvwapMixinTyping):
    def _build_avwap_prefixes(
        self, candles: Sequence[Candle]
    ) -> Tuple[List[float], List[float], List[float]]:
        tpv_prefix = [0.0]
        vol_prefix = [0.0]
        tpv2_prefix = [0.0]
        for candle in candles:
            typical_price = (candle.high + candle.low + candle.close) / 3.0
            tpv_prefix.append(tpv_prefix[-1] + typical_price * candle.volume)
            vol_prefix.append(vol_prefix[-1] + candle.volume)
            tpv2_prefix.append(tpv2_prefix[-1] + (typical_price**2) * candle.volume)
        return tpv_prefix, vol_prefix, tpv2_prefix

    def _build_avwap_snapshot(
        self,
        *,
        candles: Sequence[Candle],
        anchor_index: int,
        candle_index: int,
        tpv_prefix: Sequence[float],
        vol_prefix: Sequence[float],
        tpv2_prefix: Sequence[float],
    ) -> _AvwapSnapshot:
        weighted_sum = tpv_prefix[candle_index + 1] - tpv_prefix[anchor_index]
        volume_sum = vol_prefix[candle_index + 1] - vol_prefix[anchor_index]
        weighted_sq_sum = tpv2_prefix[candle_index + 1] - tpv2_prefix[anchor_index]
        if volume_sum <= 0:
            raise ValueError("AVWAP requires positive cumulative volume")
        vwap = weighted_sum / volume_sum
        variance = max((weighted_sq_sum / volume_sum) - (vwap**2), 0.0)
        stdev = math.sqrt(variance)
        return _AvwapSnapshot(
            anchor_index=anchor_index,
            anchor_time=candles[anchor_index].open_time,
            candle_index=candle_index,
            vwap=vwap,
            stdev=stdev,
            upper1=vwap + self._cfg.avwap_multiplier_1 * stdev,
            lower1=vwap - self._cfg.avwap_multiplier_1 * stdev,
            upper2=vwap + self._cfg.avwap_multiplier_2 * stdev,
            lower2=vwap - self._cfg.avwap_multiplier_2 * stdev,
            upper3=vwap + self._cfg.avwap_multiplier_3 * stdev,
            lower3=vwap - self._cfg.avwap_multiplier_3 * stdev,
        )

    def _detect_level_cross(
        self,
        *,
        candle: Candle,
        prev_close: float,
        level: float,
        direction: Literal["up", "down"],
    ) -> _CrossDecision:
        if self._cfg.use_gap_cross_detection:
            if direction == "down" and prev_close >= level >= candle.open:
                return _CrossDecision(True, "gap")
            if direction == "up" and prev_close <= level <= candle.open:
                return _CrossDecision(True, "gap")
        start_price = candle.open
        for end_price in self._price_path(candle):
            if direction == "down" and start_price >= level >= end_price:
                return _CrossDecision(True, "intrabar")
            if direction == "up" and start_price <= level <= end_price:
                return _CrossDecision(True, "intrabar")
            start_price = end_price
        return _CrossDecision(False)

    def _price_path(self, candle: Candle) -> Tuple[float, float, float]:
        if abs(candle.open - candle.high) < abs(candle.open - candle.low):
            return candle.high, candle.low, candle.close
        return candle.low, candle.high, candle.close

    def _trailing_activation_level(
        self, direction: Direction, avwap: _AvwapSnapshot
    ) -> float:
        threshold = self._cfg.trailing_activation_threshold_pct / 100.0
        if direction == "long":
            return avwap.upper1 * (1.0 + threshold)
        return avwap.lower1 * (1.0 - threshold)

    def _rigid_stop_level(
        self, direction: Direction, entry_price: float
    ) -> float | None:
        pct = self._cfg.rigid_stop_loss_pct / 100.0
        if pct <= 0:
            return None
        if direction == "long":
            return entry_price * (1.0 - pct)
        return entry_price * (1.0 + pct)

    def _build_sizing_decision(
        self,
        *,
        direction: Direction,
        raw_entry_price: float,
        stop_level: float,
        risk_amount: float,
    ) -> _SizingDecision | None:
        distance = (
            raw_entry_price - stop_level
            if direction == "long"
            else stop_level - raw_entry_price
        )
        if distance <= 0 or raw_entry_price <= 0:
            return None
        entry_price = self._apply_entry_slippage(direction, raw_entry_price)
        estimated_exit_price = self._apply_exit_slippage(direction, raw_entry_price)
        entry_slippage_per_unit = abs(entry_price - raw_entry_price)
        exit_slippage_per_unit = abs(estimated_exit_price - raw_entry_price)
        entry_fee_per_unit = entry_price * self._cfg.maker_fee_pct
        exit_fee_per_unit = estimated_exit_price * self._cfg.taker_fee_pct
        total_cost_per_unit = (
            entry_slippage_per_unit
            + exit_slippage_per_unit
            + entry_fee_per_unit
            + exit_fee_per_unit
        )
        if self._cfg.position_sizing_mode == "risk_amount_per_price":
            base_qty_before_costs = risk_amount / raw_entry_price
            effective_price_for_sizing = raw_entry_price + total_cost_per_unit
            qty = risk_amount / effective_price_for_sizing
            risk_amount_interpretation = "position_notional_budget"
        else:
            qty = risk_amount / distance
            base_qty_before_costs = qty
            effective_price_for_sizing = distance
            risk_amount_interpretation = "stop_loss_risk"
        return _SizingDecision(
            qty=qty,
            distance=distance,
            entry_price=entry_price,
            estimated_exit_price=estimated_exit_price,
            risk_amount_interpretation=risk_amount_interpretation,
            base_qty_before_costs=base_qty_before_costs,
            qty_reduction_from_costs=max(base_qty_before_costs - qty, 0.0),
            sizing_reference_price=raw_entry_price,
            effective_price_for_sizing=effective_price_for_sizing,
            entry_slippage_per_unit=entry_slippage_per_unit,
            exit_slippage_per_unit=exit_slippage_per_unit,
            entry_fee_per_unit=entry_fee_per_unit,
            exit_fee_per_unit=exit_fee_per_unit,
            total_cost_per_unit=total_cost_per_unit,
        )

    def _apply_entry_slippage(self, direction: Direction, price: float) -> float:
        if direction == "long":
            return price * (1.0 + self._cfg.entry_slippage_pct)
        return price * (1.0 - self._cfg.entry_slippage_pct)

    def _apply_exit_slippage(self, direction: Direction, price: float) -> float:
        if direction == "long":
            return price * (1.0 - self._cfg.exit_slippage_pct)
        return price * (1.0 + self._cfg.exit_slippage_pct)

    # ------------------------------------------------------------------
    # Exit path geometry helpers
    # ------------------------------------------------------------------

    def _check_long_gap_exit(
        self,
        *,
        prev_close: float,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
        trailing_stop: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if trailing_stop is not None and prev_close >= trailing_stop >= open_price:
            candidates.append(("Trailing stop", trailing_stop))
        if rigid_stop_level is not None and prev_close >= rigid_stop_level >= open_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if prev_close >= stop_level >= open_price:
            candidates.append(("Stop loss", stop_level))
        return max(candidates, key=lambda item: item[1]) if candidates else None

    def _check_long_open_exit(
        self,
        *,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
        trailing_stop: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if trailing_stop is not None and open_price <= trailing_stop:
            candidates.append(("Trailing stop", trailing_stop))
        if rigid_stop_level is not None and open_price <= rigid_stop_level:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if open_price <= stop_level:
            candidates.append(("Stop loss", stop_level))
        return max(candidates, key=lambda item: item[1]) if candidates else None

    def _check_short_gap_exit(
        self,
        *,
        prev_close: float,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
        trailing_stop: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if trailing_stop is not None and prev_close <= trailing_stop <= open_price:
            candidates.append(("Trailing stop", trailing_stop))
        if rigid_stop_level is not None and prev_close <= rigid_stop_level <= open_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if prev_close <= stop_level <= open_price:
            candidates.append(("Stop loss", stop_level))
        return min(candidates, key=lambda item: item[1]) if candidates else None

    def _check_short_open_exit(
        self,
        *,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
        trailing_stop: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if trailing_stop is not None and open_price >= trailing_stop:
            candidates.append(("Trailing stop", trailing_stop))
        if rigid_stop_level is not None and open_price >= rigid_stop_level:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if open_price >= stop_level:
            candidates.append(("Stop loss", stop_level))
        return min(candidates, key=lambda item: item[1]) if candidates else None

    def _first_long_downside_exit(
        self,
        *,
        start_price: float,
        end_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
        trailing_stop: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if trailing_stop is not None and start_price >= trailing_stop >= end_price:
            candidates.append(("Trailing stop", trailing_stop))
        if rigid_stop_level is not None and start_price >= rigid_stop_level >= end_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if start_price >= stop_level >= end_price:
            candidates.append(("Stop loss", stop_level))
        return max(candidates, key=lambda item: item[1]) if candidates else None

    def _first_short_upside_exit(
        self,
        *,
        start_price: float,
        end_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
        trailing_stop: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if trailing_stop is not None and start_price <= trailing_stop <= end_price:
            candidates.append(("Trailing stop", trailing_stop))
        if rigid_stop_level is not None and start_price <= rigid_stop_level <= end_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if start_price <= stop_level <= end_price:
            candidates.append(("Stop loss", stop_level))
        return min(candidates, key=lambda item: item[1]) if candidates else None

    def _activate_long_trailing(
        self, runtime: _PositionRuntime, extreme_price: float
    ) -> None:
        runtime.trailing_active = True
        runtime.extreme_price = extreme_price
        runtime.trailing_stop = self._constrain_trailing_stop(
            runtime, extreme_price * (1.0 - self._cfg.trailing_gap_pct / 100.0)
        )

    def _activate_short_trailing(
        self, runtime: _PositionRuntime, extreme_price: float
    ) -> None:
        runtime.trailing_active = True
        runtime.extreme_price = extreme_price
        runtime.trailing_stop = self._constrain_trailing_stop(
            runtime, extreme_price * (1.0 + self._cfg.trailing_gap_pct / 100.0)
        )

    def _update_long_trailing(
        self, runtime: _PositionRuntime, extreme_price: float
    ) -> None:
        if not runtime.trailing_active:
            return
        current_extreme = (
            runtime.extreme_price if runtime.extreme_price is not None else extreme_price
        )
        if extreme_price <= current_extreme:
            return
        runtime.extreme_price = extreme_price
        runtime.trailing_stop = self._constrain_trailing_stop(
            runtime, extreme_price * (1.0 - self._cfg.trailing_gap_pct / 100.0)
        )

    def _update_short_trailing(
        self, runtime: _PositionRuntime, extreme_price: float
    ) -> None:
        if not runtime.trailing_active:
            return
        current_extreme = (
            runtime.extreme_price if runtime.extreme_price is not None else extreme_price
        )
        if extreme_price >= current_extreme:
            return
        runtime.extreme_price = extreme_price
        runtime.trailing_stop = self._constrain_trailing_stop(
            runtime, extreme_price * (1.0 + self._cfg.trailing_gap_pct / 100.0)
        )

    def _constrain_trailing_stop(
        self, runtime: _PositionRuntime, trailing_stop: float
    ) -> float:
        rigid_stop = runtime.rigid_stop_level
        if rigid_stop is None:
            return trailing_stop
        if runtime.direction == "long":
            return max(trailing_stop, rigid_stop)
        return min(trailing_stop, rigid_stop)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _compute_risk_amount(self, symbol: str) -> float:
        balance = self._safe_get_balance()
        if balance is None:
            return 0.0
        if balance <= self._cfg.minimum_balance_usdt:
            self._log.warning(
                "EmaAvwapPullback: balance %.8f is at/below minimum %.8f; "
                "skipping entry for %s",
                balance,
                self._cfg.minimum_balance_usdt,
                symbol,
            )
            return 0.0
        return balance * (self._cfg.equity_risk_pct / 100.0)

    def _safe_get_balance(self) -> Optional[float]:
        try:
            return float(
                self._retry(self._exchange.get_account_balance, "get_account_balance")
            )
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: get_account_balance failed; entry sizing disabled: %s",
                exc,
            )
            return None

    def _safe_fetch_price(self, symbol: str) -> Optional[float]:
        try:
            price = self._retry(lambda: self._exchange.fetch_price(symbol), "fetch_price")
            if price is None:
                return None
            return float(price)
        except Exception:
            return None

    def _safe_get_position(self, symbol: str) -> Optional[Position]:
        try:
            return self._exchange.get_position(symbol)
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: get_position failed for %s: %s", symbol, exc
            )
            return None

    def _latest_price_for_trailing(self, symbol: str) -> Optional[float]:
        if self._cfg.use_trailing_tick_emulation:
            try:
                rows = self._exchange.get_klines(
                    symbol=symbol,
                    interval=self._cfg.trailing_tick_timeframe,
                    limit=2,
                )
                if rows and len(rows) >= 2:
                    candle = Candle.from_binance(
                        symbol, self._cfg.trailing_tick_timeframe, rows[-2]
                    )
                    return float(candle.close)
            except Exception:
                pass
        return self._safe_fetch_price(symbol)

    def _retry(self, fn, label: str):
        last_exc: Exception | None = None
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if self._is_non_retriable_order_error(exc):
                    raise
                self._log.warning(
                    "EmaAvwapPullback: %s failed (attempt %d/%d): %s",
                    label,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{label} failed without exception")

    def _find_matching_pending(
        self, symbol: str, side: PositionSide
    ) -> Optional[PendingEntryRecord]:
        return self._state.pending_entries.get(self._pending_key(symbol, side))

    def _pending_key(self, symbol: str, side: PositionSide) -> str:
        return f"{symbol}:{side.value}"

    def _setup_key(self, symbol: str, direction: Direction) -> tuple[str, Direction]:
        return (symbol, direction)

    def _remove_setup(self, symbol: str, direction: Direction) -> None:
        key = self._setup_key(symbol, direction)
        self._active_setups.pop(key, None)
        self._last_price_by_setup_key.pop(key, None)
        saver = getattr(self, "_save_state", None)
        if callable(saver):
            saver()

    def _has_pending_for_symbol(self, symbol: str) -> bool:
        return any(pending.symbol == symbol for pending in self._state.pending_entries.values())

    def _clear_setups_for_symbol(self, symbol: str) -> None:
        self._remove_setup(symbol, "long")
        self._remove_setup(symbol, "short")

    def _find_anchor_index(
        self, candles: Sequence[Candle], setup: _SetupState
    ) -> int:
        return self._find_anchor_index_by_time(candles, setup.anchor_time)

    def _find_anchor_index_by_time(
        self, candles: Sequence[Candle], anchor_time: datetime
    ) -> int:
        for idx, candle in enumerate(candles):
            if candle.open_time == anchor_time:
                return idx
        raise ValueError(f"anchor candle {anchor_time.isoformat()} not found")

    def _dynamic_stop_from_avwap(
        self, direction: Direction, avwap: _AvwapSnapshot
    ) -> float:
        return avwap.lower2 if direction == "long" else avwap.upper2

    def _protective_stop_price(
        self,
        *,
        direction: Direction,
        dynamic_stop: float,
        rigid_stop: float | None,
        trailing_stop: float | None,
    ) -> float:
        candidates = [dynamic_stop]
        if rigid_stop is not None:
            candidates.append(rigid_stop)
        if trailing_stop is not None:
            candidates.append(trailing_stop)
        if direction == "long":
            return max(candidates)
        return min(candidates)

    def _is_stop_breached_by_price(
        self, *, direction: Direction, price: float, stop_price: float
    ) -> bool:
        if direction == "long":
            return price <= stop_price
        return price >= stop_price

    def _is_less_protective(
        self, direction: Direction, candidate: float, previous: float
    ) -> bool:
        if direction == "long":
            return candidate < previous
        return candidate > previous

    def _sync_runtime_to_record(
        self, record: PositionRecord, runtime: _PositionRuntime
    ) -> None:
        record.trailing_active = runtime.trailing_active
        record.trailing_stop = runtime.trailing_stop
        record.extreme_since_activation = runtime.extreme_price

    def _side_from_direction(self, direction: Direction) -> PositionSide:
        return PositionSide.LONG if direction == "long" else PositionSide.SHORT

    def _direction_from_side(self, side: PositionSide) -> Direction:
        return "long" if side == PositionSide.LONG else "short"

    def _interval_to_minutes(self, interval: str) -> int:
        ms = interval_to_milliseconds(interval)
        return max(int(ms // 60000), 1)

    def _timeframe_seconds(self, interval: str) -> int:
        return self._interval_to_minutes(interval) * 60

    @staticmethod
    def _ensure_aware(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment

    @staticmethod
    def _is_insufficient_balance_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "insufficient balance",
            "insufficient margin",
            "insufficient available",
            "not enough balance",
            "not enough margin",
            "balance not enough",
            "margin not enough",
            "available balance",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_non_retriable_order_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "below mintradevolume",
            "invalid order quantity",
            "invalid order price",
            "normalized order price is invalid",
            "normalized stop loss is invalid",
            "parameter error",
        )
        return any(marker in text for marker in markers)
