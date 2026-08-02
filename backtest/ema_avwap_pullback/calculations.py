"""AVWAP, price-path, and position-sizing calculations."""

from __future__ import annotations

import math
from typing import List, Literal, Sequence, Tuple

from candle_downloader.models import Candle

from .config import Direction, ExitBand
from .models import _AvwapSnapshot, _CrossDecision, _SizingDecision


class EmaAvwapCalculationsMixin:
    """Calculations that depend only on the strategy configuration."""

    def _detect_level_cross(
        self,
        *,
        candle: Candle,
        prev_close: float,
        level: float,
        direction: Literal["up", "down"],
    ) -> _CrossDecision:
        if self._config.use_gap_cross_detection:
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

    def _detect_live_middle_cross(
        self,
        *,
        direction: Direction,
        entry_price: float,
        observed_path: Sequence[float],
    ) -> _CrossDecision:
        if len(observed_path) < 2:
            return _CrossDecision(False)
        for index, (previous_price, current_price) in enumerate(
            zip(observed_path, observed_path[1:])
        ):
            crossed = self._price_crossed_entry_line(
                direction=direction,
                previous_price=previous_price,
                previous_entry_price=entry_price,
                current_price=current_price,
                current_entry_price=entry_price,
            )
            if not crossed:
                continue
            # A candle-boundary transition is still a live tick crossing; only
            # the historical fill model distinguishes the source of the tick.
            return _CrossDecision(True, "live_tick", tuple(observed_path[index + 1 :]))
        return _CrossDecision(False)

    def _price_is_past_entry_line(
        self, *, direction: Direction, price: float, entry_price: float
    ) -> bool:
        if direction == "long":
            return price <= entry_price
        return price >= entry_price

    def _price_crossed_entry_line(
        self,
        *,
        direction: Direction,
        previous_price: float,
        previous_entry_price: float,
        current_price: float,
        current_entry_price: float,
    ) -> bool:
        previous_distance = previous_price - previous_entry_price
        current_distance = current_price - current_entry_price
        if direction == "long":
            return previous_distance > 0 and current_distance <= 0
        return previous_distance < 0 and current_distance >= 0

    def _price_path(self, candle: Candle) -> Tuple[float, float, float]:
        if abs(candle.open - candle.high) < abs(candle.open - candle.low):
            return candle.high, candle.low, candle.close
        return candle.low, candle.high, candle.close

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
        cfg = self._config
        return _AvwapSnapshot(
            anchor_index=anchor_index,
            anchor_time=candles[anchor_index].open_time,
            candle_index=candle_index,
            vwap=vwap,
            stdev=stdev,
            upper1=vwap + cfg.avwap_multiplier_1 * stdev,
            lower1=vwap - cfg.avwap_multiplier_1 * stdev,
            upper2=vwap + cfg.avwap_multiplier_2 * stdev,
            lower2=vwap - cfg.avwap_multiplier_2 * stdev,
            upper3=vwap + cfg.avwap_multiplier_3 * stdev,
            lower3=vwap - cfg.avwap_multiplier_3 * stdev,
        )

    def _trailing_activation_level(self, direction: Direction, avwap: _AvwapSnapshot) -> float:
        threshold = self._config.trailing_activation_threshold_pct / 100.0
        if direction == "long":
            return avwap.upper1 * (1.0 + threshold)
        return avwap.lower1 * (1.0 - threshold)

    def _target_band_level(
        self,
        direction: Direction,
        avwap: _AvwapSnapshot,
        exit_band: ExitBand | None = None,
    ) -> float:
        band = self._config.exit_band if exit_band is None else exit_band
        if band.number == 2:
            return avwap.upper2 if direction == "long" else avwap.lower2
        return avwap.upper1 if direction == "long" else avwap.lower1

    def _rigid_stop_level(self, direction: Direction, entry_price: float) -> float | None:
        pct = self._config.rigid_stop_loss_pct / 100.0
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
        position_notional_budget: float,
    ) -> _SizingDecision | None:
        if raw_entry_price <= 0 or position_notional_budget <= 0:
            return None

        entry_price = self._apply_entry_slippage(direction, raw_entry_price)
        estimated_exit_price = self._apply_exit_slippage(direction, raw_entry_price)
        entry_slippage_per_unit = abs(entry_price - raw_entry_price)
        exit_slippage_per_unit = abs(estimated_exit_price - raw_entry_price)
        # A live EMA/AVWAP entry is a marketable limit order.  The live
        # coordinator reserves taker fees for it, so the backtest must too.
        entry_fee_per_unit = entry_price * self._config.taker_fee_pct
        exit_fee_per_unit = estimated_exit_price * self._config.taker_fee_pct
        total_cost_per_unit = (
            entry_slippage_per_unit
            + exit_slippage_per_unit
            + entry_fee_per_unit
            + exit_fee_per_unit
        )

        base_qty_before_costs = position_notional_budget / entry_price
        effective_price_for_sizing = raw_entry_price + total_cost_per_unit
        qty = position_notional_budget / effective_price_for_sizing

        return _SizingDecision(
            qty=qty,
            distance=0.0,
            entry_price=entry_price,
            estimated_exit_price=estimated_exit_price,
            position_notional_budget=position_notional_budget,
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
        slip = self._config.entry_slippage_pct
        if direction == "long":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _apply_exit_slippage(self, direction: Direction, price: float) -> float:
        slip = self._config.exit_slippage_pct
        if direction == "long":
            return price * (1.0 - slip)
        return price * (1.0 + slip)
