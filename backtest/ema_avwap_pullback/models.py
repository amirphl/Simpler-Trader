"""Internal state objects shared by the EMA/AVWAP backtest modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

from .config import (
    Direction,
    EntryMode,
    ExitBand,
    ExitMode,
    PositionSizingMode,
)


@dataclass(frozen=True)
class _AvwapSnapshot:
    anchor_index: int
    anchor_time: datetime
    candle_index: int
    vwap: float
    stdev: float
    upper1: float
    lower1: float
    upper2: float
    lower2: float
    upper3: float
    lower3: float


@dataclass(frozen=True)
class _SetupState:
    direction: Direction
    anchor_index: int
    detected_index: int
    detected_time: datetime
    consecutive_count: int
    is_waiting_for_cross: bool = False
    # Persisted live strategies can carry an observation pair across a restart.
    # The backtest retains it so that its deterministic boundary proxy follows
    # the same resumed-state transition.
    last_observed_price: float | None = None
    last_observed_middle: float | None = None


@dataclass
class _PositionState:
    direction: Direction
    anchor_index: int
    setup_detected_index: int
    setup_detected_time: datetime
    entry_time: datetime
    entry_index: int
    raw_entry_price: float
    entry_price: float
    qty: float
    position_notional_budget: float
    entry_fee: float
    stop_level_at_entry: float
    rigid_stop_level_at_entry: float | None
    trailing_activation_level_at_entry: float
    entry_trigger_mode: str
    position_sizing_mode: PositionSizingMode
    entry_mode: EntryMode = EntryMode.LIVE
    exit_mode: ExitMode = ExitMode.LIVE
    exit_band: ExitBand = ExitBand.BAND_1
    decision_price: float | None = None
    ema_value_at_entry: float | None = None
    entry_path_remaining: Tuple[float, ...] = ()
    trailing_active: bool = False
    trailing_stop: float | None = None
    extreme_price: float | None = None
    # Live trailing uses the most recently closed AVWAP snapshot.  Target
    # evaluation uses a forming-candle AVWAP independently.
    trailing_avwap: _AvwapSnapshot | None = None


@dataclass(frozen=True)
class _SizingDecision:
    qty: float
    distance: float
    entry_price: float
    estimated_exit_price: float
    position_notional_budget: float
    base_qty_before_costs: float
    qty_reduction_from_costs: float
    sizing_reference_price: float
    effective_price_for_sizing: float
    entry_slippage_per_unit: float
    exit_slippage_per_unit: float
    entry_fee_per_unit: float
    exit_fee_per_unit: float
    total_cost_per_unit: float


@dataclass(frozen=True)
class _CrossDecision:
    crossed: bool
    mode: str | None = None
    remaining_path: Tuple[float, ...] = ()


@dataclass(frozen=True)
class _ExitDecision:
    reason: str
    raw_exit_price: float
    stop_level: float
    activation_level: float
    target_level: float | None = None
