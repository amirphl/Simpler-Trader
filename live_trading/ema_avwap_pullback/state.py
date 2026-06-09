"""Internal state objects for the EMA + AVWAP live strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from candle_downloader.models import Candle

from ..exchange import PositionSide
from .config import Direction, PositionSizingMode

class _InsufficientBalanceError(RuntimeError):
    """Raised when the exchange rejects an action due to balance or margin."""


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
    symbol: str
    direction: Direction
    anchor_time: datetime
    detected_time: datetime
    consecutive_count: int
    detected_avwap: _AvwapSnapshot | None = None
    is_waiting_for_cross: bool = False


@dataclass(frozen=True)
class _SizingDecision:
    qty: float
    distance: float
    entry_price: float
    estimated_exit_price: float
    risk_amount_interpretation: str
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


@dataclass(frozen=True)
class _ExitDecision:
    reason: str
    raw_exit_price: float
    stop_level: float
    activation_level: float


@dataclass(frozen=True)
class _SymbolSnapshot:
    symbol: str
    timeframe: str
    timeframe_minutes: int
    candles: Sequence[Candle]
    candle_index: int
    candle: Candle
    previous_candle: Candle
    ema_value: float
    tpv_prefix: Sequence[float]
    vol_prefix: Sequence[float]
    tpv2_prefix: Sequence[float]


@dataclass(frozen=True)
class _EntryCandidate:
    symbol: str
    side: PositionSide
    direction: Direction
    signal_time: datetime
    anchor_time: datetime
    setup_detected_time: datetime
    candle_index: int
    raw_entry_price: float
    order_price: float
    stop_for_risk: float
    dynamic_stop_at_entry: float
    rigid_stop_at_entry: float | None
    trailing_activation_at_entry: float
    quantity: float
    risk_amount: float
    risk_amount_interpretation: str
    entry_trigger_mode: str
    sizing: _SizingDecision
    avwap: _AvwapSnapshot


@dataclass
class _PositionRuntime:
    direction: Direction
    anchor_time: datetime
    setup_detected_time: datetime
    entry_signal_time: datetime
    raw_entry_price: float
    dynamic_stop_at_entry: float
    rigid_stop_level: float | None
    trailing_activation_at_entry: float
    entry_trigger_mode: str
    risk_amount_interpretation: str
    position_sizing_mode: PositionSizingMode
    last_avwap: _AvwapSnapshot | None = None
    trailing_active: bool = False
    trailing_stop: float | None = None
    extreme_price: float | None = None


@dataclass(frozen=True)
class _PendingEntryMeta:
    candidate: _EntryCandidate
