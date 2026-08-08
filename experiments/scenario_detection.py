"""
experiments/scenario_detection.py

Scenario detection built on top of liquidity-zone and pivot detection.

The detector consumes liquidity-zone *line hunts*.  Once a line is hunted, that
zone is marked consumed immediately and is not eligible for another scenario.

Bullish scenario:
  1. An UPWARD/bullish liquidity-zone line is hunted from above.
  2. Select the relevant bullish pivot and bearish pivot as of the zone hunt.
  3. If the bullish pivot is hunted before the bearish pivot, cancel.
  4. If the bearish pivot is hunted before the bullish pivot, approve.

Bearish scenario mirrors the same design:
  1. A DOWNWARD/bearish liquidity-zone line is hunted from below.
  2. A bearish-pivot hunt cancels; a bullish-pivot hunt confirms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Sequence

from candle_downloader.models import Candle
from experiments.liquidity_zone_detection import (
    Direction,
    LiquidityZone,
    LiquidityZoneConfig,
    LiquidityZoneResult,
    detect_liquidity_zones,
)
from experiments.pivot_detection import Pivot

PenetrationSide = Literal["from_below", "from_above"]
PivotType = Literal["bullish", "bearish"]
PivotSelectionMode = Literal["non_hunted_only", "ignore_hunted_status"]
ScenarioDirection = Literal["BULLISH", "BEARISH"]
ZoneSelection = Literal["all", "level_1", "level_2"]


@dataclass(slots=True)
class ScenarioDetectionConfig:
    """Settings for scenario detection.

    Pivot-selection settings are intentionally separate from liquidity-zone
    pivot-detection settings.  They decide which already-detected pivot is
    relevant when a zone line is hunted.
    """

    liquidity_zone_config: LiquidityZoneConfig = field(
        default_factory=LiquidityZoneConfig
    )
    epsilon: float = 1e-9
    zone_selection: ZoneSelection = "all"
    bullish_pivot_selection: PivotSelectionMode = "non_hunted_only"
    bearish_pivot_selection: PivotSelectionMode = "non_hunted_only"
    mark_zone_hunted: bool = True
    deduplicate_by_zone_and_current: bool = True


@dataclass(slots=True)
class LiquidityZoneLinePenetration:
    """A liquidity-zone line hunt event.

    The historical name is kept for compatibility with existing callers and UI
    payloads.  New scenario logic uses only the first hunt per zone.
    """

    zone_key: str
    zone_id: str
    zone_level: int
    zone_direction: Direction
    candle_index: int
    previous_candle_index: int | None
    side: PenetrationSide
    line_price: float
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    time: datetime | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PivotSnapshot:
    pivot_index: int
    pivot_type: PivotType
    price: float
    activation_index: int
    first_hunt_index: int | None
    hunted_at_zone_hunt: bool
    selection_mode: PivotSelectionMode
    time: datetime | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PivotHuntEvent:
    pivot_index: int
    pivot_type: PivotType
    candle_index: int
    side: PenetrationSide
    price: float
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    time: datetime | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioRecord:
    id: str
    direction: ScenarioDirection
    zone_key: str
    zone: LiquidityZone
    zone_hunt: LiquidityZoneLinePenetration
    confirming_pivot_hunt: PivotHuntEvent
    relevant_bullish_pivot: PivotSnapshot | None
    relevant_bearish_pivot: PivotSnapshot | None
    cancelling_pivot_hunt: PivotHuntEvent | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioDetectionResult:
    liquidity_zone_result: LiquidityZoneResult
    zones: List[LiquidityZone]
    penetrations_by_zone: Dict[str, List[LiquidityZoneLinePenetration]]
    scenarios: List[ScenarioRecord]
    current_index: int | None = None


@dataclass
class ScenarioDetector:
    """Small stateful wrapper for the "on new candle" workflow."""

    config: ScenarioDetectionConfig = field(default_factory=ScenarioDetectionConfig)
    candles: List[Candle] = field(default_factory=list)
    latest_result: ScenarioDetectionResult | None = None
    _seen_scenario_keys: set[str] = field(default_factory=set, init=False)

    def on_new_candle(self, candle: Candle) -> List[ScenarioRecord]:
        self.candles.append(candle)
        if not self.candles:
            return []
        self.latest_result = detect_scenarios(
            self.candles,
            self.config,
            current_index=len(self.candles) - 1,
        )
        return self._new_scenarios(self.latest_result.scenarios)

    def update(self, candles: Sequence[Candle]) -> List[ScenarioRecord]:
        self.candles = list(candles)
        if not self.candles:
            self.latest_result = None
            return []
        self.latest_result = detect_scenarios(
            self.candles,
            self.config,
            current_index=len(self.candles) - 1,
        )
        return self._new_scenarios(self.latest_result.scenarios)

    def _new_scenarios(self, scenarios: Sequence[ScenarioRecord]) -> List[ScenarioRecord]:
        new_records: List[ScenarioRecord] = []
        for scenario in scenarios:
            key = str(scenario.metadata.get("scenario_key", scenario.id))
            if key in self._seen_scenario_keys:
                continue
            self._seen_scenario_keys.add(key)
            new_records.append(scenario)
        return new_records


def detect_scenarios(
    candles: Sequence[Candle],
    config: ScenarioDetectionConfig | None = None,
    *,
    liquidity_zone_result: LiquidityZoneResult | None = None,
    current_index: int | None = None,
) -> ScenarioDetectionResult:
    """Detect bullish/bearish scenarios from zone hunts and pivot hunt order.

    When current_index is provided, only scenarios confirmed on that candle are
    returned.  Zone hunt/consumption metadata is still computed up to that
    candle so live callers can debug why a scenario did or did not appear.
    """
    if not candles:
        raise ValueError("candles cannot be empty")

    cfg = config or ScenarioDetectionConfig()
    _validate_config(cfg)

    if current_index is None:
        scan_end = len(candles) - 1
    else:
        if current_index < 0 or current_index >= len(candles):
            raise ValueError("current_index must point to an existing candle")
        scan_end = current_index

    liquidity_result = liquidity_zone_result or detect_liquidity_zones(
        candles,
        cfg.liquidity_zone_config,
    )
    zones = _flatten_zones(liquidity_result, cfg.zone_selection)
    penetrations_by_zone = {
        _zone_key(zone): _collect_zone_hunts(candles, zone, scan_end, cfg)
        for zone in zones
    }

    scenarios: List[ScenarioRecord] = []
    seen: set[tuple[str, ScenarioDirection, int]] = set()

    for zone in zones:
        zone_key = _zone_key(zone)
        zone_hunts = penetrations_by_zone[zone_key]
        if not zone_hunts:
            continue

        zone_hunt = zone_hunts[0]
        scenario_direction = _scenario_direction_from_zone(zone)
        if scenario_direction is None:
            continue

        evaluation = _evaluate_zone_hunt(
            candles=candles,
            pivots=liquidity_result.pivots,
            zone=zone,
            zone_key=zone_key,
            zone_hunt=zone_hunt,
            direction=scenario_direction,
            config=cfg,
        )

        if cfg.mark_zone_hunted:
            _mark_zone_hunted(zone, zone_hunt, evaluation)

        scenario = evaluation.get("scenario")
        if not isinstance(scenario, ScenarioRecord):
            continue

        if current_index is not None and (
            scenario.confirming_pivot_hunt.candle_index != current_index
        ):
            continue

        if cfg.deduplicate_by_zone_and_current:
            dedupe_key = (
                zone_key,
                scenario.direction,
                scenario.confirming_pivot_hunt.candle_index,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

        scenarios.append(scenario)

    scenarios.sort(
        key=lambda item: (
            item.confirming_pivot_hunt.candle_index,
            item.zone.level,
            item.zone.id,
            item.direction,
        )
    )
    return ScenarioDetectionResult(
        liquidity_zone_result=liquidity_result,
        zones=zones,
        penetrations_by_zone=penetrations_by_zone,
        scenarios=scenarios,
        current_index=current_index,
    )


def _validate_config(config: ScenarioDetectionConfig) -> None:
    if config.epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if config.zone_selection not in ("all", "level_1", "level_2"):
        raise ValueError("zone_selection must be 'all', 'level_1', or 'level_2'")
    selection_modes = ("non_hunted_only", "ignore_hunted_status")
    if config.bullish_pivot_selection not in selection_modes:
        raise ValueError(
            "bullish_pivot_selection must be 'non_hunted_only' or "
            "'ignore_hunted_status'"
        )
    if config.bearish_pivot_selection not in selection_modes:
        raise ValueError(
            "bearish_pivot_selection must be 'non_hunted_only' or "
            "'ignore_hunted_status'"
        )


def _flatten_zones(
    liquidity_result: LiquidityZoneResult,
    zone_selection: ZoneSelection,
) -> List[LiquidityZone]:
    zones: List[LiquidityZone] = []
    for level in sorted(liquidity_result.zones_by_level):
        if zone_selection == "level_1" and level != 1:
            continue
        if zone_selection == "level_2" and level != 2:
            continue
        grouped = liquidity_result.zones_by_level[level]
        for direction in ("UPWARD", "DOWNWARD"):
            zones.extend(grouped[direction])
    return sorted(
        zones,
        key=lambda zone: (
            _zone_activation_index(zone),
            zone.start_index,
            zone.end_index,
            zone.level,
            zone.direction,
            zone.id,
        ),
    )


def _zone_key(zone: LiquidityZone) -> str:
    return (
        f"{zone.level}:{zone.direction}:{zone.id}:"
        f"{zone.start_index}:{zone.end_index}:{zone.line_price:.12g}"
    )


def _zone_activation_index(zone: LiquidityZone) -> int:
    indexes = [
        zone.start_index,
        zone.end_index,
        _safe_index(getattr(zone.left_pivot, "trigger_index", None)),
        _safe_index(getattr(zone.right_pivot, "trigger_index", None)),
    ]
    return max(index for index in indexes if index is not None)


def _safe_index(value: object) -> int | None:
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _collect_zone_hunts(
    candles: Sequence[Candle],
    zone: LiquidityZone,
    scan_end: int,
    config: ScenarioDetectionConfig,
) -> List[LiquidityZoneLinePenetration]:
    activation_index = _zone_activation_index(zone)
    start_index = max(activation_index + 1, 0)
    if start_index > scan_end:
        return []

    events: List[LiquidityZoneLinePenetration] = []
    zone_key = _zone_key(zone)
    for candle_index in range(start_index, scan_end + 1):
        side = _zone_hunt_side(candles[candle_index], zone, config.epsilon)
        if side is None:
            continue
        candle = candles[candle_index]
        events.append(
            LiquidityZoneLinePenetration(
                zone_key=zone_key,
                zone_id=zone.id,
                zone_level=zone.level,
                zone_direction=zone.direction,
                candle_index=candle_index,
                previous_candle_index=candle_index - 1 if candle_index > 0 else None,
                side=side,
                line_price=zone.line_price,
                candle_open=candle.open,
                candle_high=candle.high,
                candle_low=candle.low,
                candle_close=candle.close,
                time=getattr(candle, "open_time", None),
                metadata={
                    "event": "liquidity_zone_line_hunt",
                    "zone_start_index": zone.start_index,
                    "zone_end_index": zone.end_index,
                    "zone_activation_index": activation_index,
                    "line_method": zone.line_method,
                    "price_low": zone.price_range.low,
                    "price_high": zone.price_range.high,
                    "consumes_zone": True,
                },
            )
        )
        break
    return events


def _zone_hunt_side(
    candle: Candle,
    zone: LiquidityZone,
    epsilon: float,
) -> PenetrationSide | None:
    if zone.direction == "UPWARD" and candle.low <= zone.line_price + epsilon:
        return "from_above"
    if zone.direction == "DOWNWARD" and candle.high >= zone.line_price - epsilon:
        return "from_below"
    return None


def _scenario_direction_from_zone(zone: LiquidityZone) -> ScenarioDirection | None:
    if zone.direction == "UPWARD":
        return "BULLISH"
    if zone.direction == "DOWNWARD":
        return "BEARISH"
    return None


def _evaluate_zone_hunt(
    *,
    candles: Sequence[Candle],
    pivots: Sequence[Pivot],
    zone: LiquidityZone,
    zone_key: str,
    zone_hunt: LiquidityZoneLinePenetration,
    direction: ScenarioDirection,
    config: ScenarioDetectionConfig,
) -> Dict[str, object]:
    bullish = _select_relevant_pivot(
        candles,
        pivots,
        "bullish",
        zone_hunt.candle_index,
        config.bullish_pivot_selection,
        config.epsilon,
    )
    bearish = _select_relevant_pivot(
        candles,
        pivots,
        "bearish",
        zone_hunt.candle_index,
        config.bearish_pivot_selection,
        config.epsilon,
    )

    cancel_snapshot = bullish if direction == "BULLISH" else bearish
    confirm_snapshot = bearish if direction == "BULLISH" else bullish

    cancel_event = _first_pivot_hunt_event_from_snapshot(
        candles,
        cancel_snapshot,
        zone_hunt.candle_index + 1,
        config.epsilon,
    )
    confirm_event = _first_pivot_hunt_event_from_snapshot(
        candles,
        confirm_snapshot,
        zone_hunt.candle_index + 1,
        config.epsilon,
    )

    decision = _scenario_decision(cancel_event, confirm_event)
    metadata = _decision_metadata(
        zone=zone,
        zone_key=zone_key,
        zone_hunt=zone_hunt,
        direction=direction,
        bullish=bullish,
        bearish=bearish,
        cancel_event=cancel_event,
        confirm_event=confirm_event,
        decision=decision,
        config=config,
    )

    scenario: ScenarioRecord | None = None
    if decision == "approved" and confirm_event is not None:
        scenario = _build_scenario(
            zone=zone,
            zone_key=zone_key,
            direction=direction,
            zone_hunt=zone_hunt,
            confirming_pivot_hunt=confirm_event,
            cancelling_pivot_hunt=cancel_event,
            relevant_bullish_pivot=bullish,
            relevant_bearish_pivot=bearish,
            metadata=metadata,
        )

    return {
        "decision": decision,
        "scenario": scenario,
        "metadata": metadata,
        "relevant_bullish_pivot": bullish,
        "relevant_bearish_pivot": bearish,
        "cancel_event": cancel_event,
        "confirm_event": confirm_event,
    }


def _select_relevant_pivot(
    candles: Sequence[Candle],
    pivots: Sequence[Pivot],
    pivot_type: PivotType,
    zone_hunt_index: int,
    selection_mode: PivotSelectionMode,
    epsilon: float,
) -> PivotSnapshot | None:
    candidates: List[tuple[Pivot, int | None]] = []
    for pivot in pivots:
        if _pivot_type(pivot) != pivot_type:
            continue
        activation_index = _pivot_activation_index(pivot)
        if activation_index > zone_hunt_index:
            continue

        first_hunt_index = _first_pivot_hunt_index(
            candles,
            pivot,
            start_index=pivot.index + 1,
            epsilon=epsilon,
        )
        hunted_at_zone_hunt = (
            first_hunt_index is not None and first_hunt_index <= zone_hunt_index
        )
        if selection_mode == "non_hunted_only" and hunted_at_zone_hunt:
            continue
        candidates.append((pivot, first_hunt_index))

    if not candidates:
        return None

    pivot, first_hunt_index = max(
        candidates,
        key=lambda item: (_pivot_activation_index(item[0]), item[0].index),
    )
    hunted_at_zone_hunt = (
        first_hunt_index is not None and first_hunt_index <= zone_hunt_index
    )
    return PivotSnapshot(
        pivot_index=pivot.index,
        pivot_type=pivot_type,
        price=_pivot_hunt_price(pivot),
        activation_index=_pivot_activation_index(pivot),
        first_hunt_index=first_hunt_index,
        hunted_at_zone_hunt=hunted_at_zone_hunt,
        selection_mode=selection_mode,
        time=getattr(candles[pivot.index], "open_time", None)
        if 0 <= pivot.index < len(candles)
        else None,
        metadata={
            "reference_index": getattr(pivot, "reference_index", None),
            "trigger_index": getattr(pivot, "trigger_index", None),
            "invalidation_index": getattr(pivot, "invalidation_index", None),
            "high": getattr(pivot, "high", None),
            "low": getattr(pivot, "low", None),
        },
    )


def _pivot_activation_index(pivot: Pivot) -> int:
    indexes = [
        _safe_index(getattr(pivot, "index", None)),
        _safe_index(getattr(pivot, "reference_index", None)),
        _safe_index(getattr(pivot, "trigger_index", None)),
    ]
    return max(index for index in indexes if index is not None)


def _pivot_type(pivot: Pivot) -> PivotType | str:
    return str(getattr(pivot, "type", "")).lower()


def _pivot_hunt_price(pivot: Pivot | PivotSnapshot) -> float:
    pivot_type = _pivot_type(pivot) if isinstance(pivot, Pivot) else pivot.pivot_type
    if pivot_type == "bullish":
        return float(getattr(pivot, "low", getattr(pivot, "price", 0.0)))
    return float(getattr(pivot, "high", getattr(pivot, "price", 0.0)))


def _first_pivot_hunt_event_from_snapshot(
    candles: Sequence[Candle],
    pivot: PivotSnapshot | None,
    start_index: int,
    epsilon: float,
) -> PivotHuntEvent | None:
    if pivot is None:
        return None
    hunt_index = _first_pivot_hunt_index_from_snapshot(
        candles,
        pivot,
        start_index=start_index,
        epsilon=epsilon,
    )
    if hunt_index is None:
        return None
    candle = candles[hunt_index]
    return PivotHuntEvent(
        pivot_index=pivot.pivot_index,
        pivot_type=pivot.pivot_type,
        candle_index=hunt_index,
        side="from_above" if pivot.pivot_type == "bullish" else "from_below",
        price=pivot.price,
        candle_open=candle.open,
        candle_high=candle.high,
        candle_low=candle.low,
        candle_close=candle.close,
        time=getattr(candle, "open_time", None),
        metadata={
            "pivot_activation_index": pivot.activation_index,
            "pivot_hunted_at_zone_hunt": pivot.hunted_at_zone_hunt,
            "pivot_selection_mode": pivot.selection_mode,
        },
    )


def _first_pivot_hunt_index_from_snapshot(
    candles: Sequence[Candle],
    pivot: PivotSnapshot,
    *,
    start_index: int,
    epsilon: float,
) -> int | None:
    for candle_index in range(max(start_index, 0), len(candles)):
        candle = candles[candle_index]
        if pivot.pivot_type == "bullish":
            if candle.low < pivot.price - epsilon:
                return candle_index
        elif candle.high > pivot.price + epsilon:
            return candle_index
    return None


def _first_pivot_hunt_index(
    candles: Sequence[Candle],
    pivot: Pivot,
    *,
    start_index: int,
    epsilon: float,
) -> int | None:
    snapshot = PivotSnapshot(
        pivot_index=pivot.index,
        pivot_type="bullish" if _pivot_type(pivot) == "bullish" else "bearish",
        price=_pivot_hunt_price(pivot),
        activation_index=_pivot_activation_index(pivot),
        first_hunt_index=None,
        hunted_at_zone_hunt=False,
        selection_mode="ignore_hunted_status",
    )
    return _first_pivot_hunt_index_from_snapshot(
        candles,
        snapshot,
        start_index=start_index,
        epsilon=epsilon,
    )


def _scenario_decision(
    cancel_event: PivotHuntEvent | None,
    confirm_event: PivotHuntEvent | None,
) -> str:
    if confirm_event is None and cancel_event is None:
        return "waiting_for_pivot_hunt"
    if cancel_event is not None and (
        confirm_event is None or cancel_event.candle_index <= confirm_event.candle_index
    ):
        return "cancelled_by_relevant_pivot"
    if confirm_event is not None:
        return "approved"
    return "waiting_for_pivot_hunt"


def _decision_metadata(
    *,
    zone: LiquidityZone,
    zone_key: str,
    zone_hunt: LiquidityZoneLinePenetration,
    direction: ScenarioDirection,
    bullish: PivotSnapshot | None,
    bearish: PivotSnapshot | None,
    cancel_event: PivotHuntEvent | None,
    confirm_event: PivotHuntEvent | None,
    decision: str,
    config: ScenarioDetectionConfig,
) -> Dict[str, object]:
    return {
        "scenario_detection_version": "zone_hunt_pivot_order_v2",
        "decision": decision,
        "zone_key": zone_key,
        "zone_id": zone.id,
        "zone_direction": zone.direction,
        "zone_level": zone.level,
        "zone_line_price": zone.line_price,
        "zone_line_method": zone.line_method,
        "zone_price_low": zone.price_range.low,
        "zone_price_high": zone.price_range.high,
        "zone_start_index": zone.start_index,
        "zone_end_index": zone.end_index,
        "zone_hunt_index": zone_hunt.candle_index,
        "zone_hunt_side": zone_hunt.side,
        "direction": direction,
        "cancel_pivot_type": "bullish" if direction == "BULLISH" else "bearish",
        "confirm_pivot_type": "bearish" if direction == "BULLISH" else "bullish",
        "bullish_pivot": _snapshot_metadata(bullish),
        "bearish_pivot": _snapshot_metadata(bearish),
        "cancel_pivot_hunt": _pivot_hunt_metadata(cancel_event),
        "confirm_pivot_hunt": _pivot_hunt_metadata(confirm_event),
        "config": {
            "epsilon": config.epsilon,
            "zone_selection": config.zone_selection,
            "bullish_pivot_selection": config.bullish_pivot_selection,
            "bearish_pivot_selection": config.bearish_pivot_selection,
            "mark_zone_hunted": config.mark_zone_hunted,
            "deduplicate_by_zone_and_current": config.deduplicate_by_zone_and_current,
        },
    }


def _snapshot_metadata(snapshot: PivotSnapshot | None) -> Dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "pivot_index": snapshot.pivot_index,
        "pivot_type": snapshot.pivot_type,
        "price": snapshot.price,
        "activation_index": snapshot.activation_index,
        "first_hunt_index": snapshot.first_hunt_index,
        "hunted_at_zone_hunt": snapshot.hunted_at_zone_hunt,
        "selection_mode": snapshot.selection_mode,
        "metadata": dict(snapshot.metadata),
    }


def _pivot_hunt_metadata(event: PivotHuntEvent | None) -> Dict[str, object] | None:
    if event is None:
        return None
    return {
        "pivot_index": event.pivot_index,
        "pivot_type": event.pivot_type,
        "candle_index": event.candle_index,
        "side": event.side,
        "price": event.price,
        "candle_open": event.candle_open,
        "candle_high": event.candle_high,
        "candle_low": event.candle_low,
        "candle_close": event.candle_close,
        "metadata": dict(event.metadata),
    }


def _build_scenario(
    *,
    zone: LiquidityZone,
    zone_key: str,
    direction: ScenarioDirection,
    zone_hunt: LiquidityZoneLinePenetration,
    confirming_pivot_hunt: PivotHuntEvent,
    cancelling_pivot_hunt: PivotHuntEvent | None,
    relevant_bullish_pivot: PivotSnapshot | None,
    relevant_bearish_pivot: PivotSnapshot | None,
    metadata: Dict[str, object],
) -> ScenarioRecord:
    scenario_id = (
        f"{direction}_{zone.id}_"
        f"{zone_hunt.candle_index}_{confirming_pivot_hunt.candle_index}"
    )
    scenario_metadata = {
        **metadata,
        "scenario_key": scenario_id,
        "confirming_pivot_index": confirming_pivot_hunt.pivot_index,
        "confirming_pivot_type": confirming_pivot_hunt.pivot_type,
        "confirming_pivot_hunt_index": confirming_pivot_hunt.candle_index,
    }
    if cancelling_pivot_hunt is not None:
        scenario_metadata["cancelling_pivot_hunt_index"] = (
            cancelling_pivot_hunt.candle_index
        )
    return ScenarioRecord(
        id=scenario_id,
        direction=direction,
        zone_key=zone_key,
        zone=zone,
        zone_hunt=zone_hunt,
        confirming_pivot_hunt=confirming_pivot_hunt,
        cancelling_pivot_hunt=cancelling_pivot_hunt,
        relevant_bullish_pivot=relevant_bullish_pivot,
        relevant_bearish_pivot=relevant_bearish_pivot,
        metadata=scenario_metadata,
    )


def _mark_zone_hunted(
    zone: LiquidityZone,
    zone_hunt: LiquidityZoneLinePenetration,
    evaluation: Dict[str, object],
) -> None:
    zone.is_hunted = True
    zone.metadata["scenario_hunted"] = True
    zone.metadata["scenario_hunted_index"] = zone_hunt.candle_index
    zone.metadata["scenario_hunted_side"] = zone_hunt.side
    zone.metadata["scenario_hunted_time"] = zone_hunt.time
    zone.metadata["scenario_decision"] = evaluation.get("decision")
    zone.metadata["scenario_detection_metadata"] = evaluation.get("metadata")
    scenario = evaluation.get("scenario")
    if isinstance(scenario, ScenarioRecord):
        zone.metadata["scenario_hunted_by"] = scenario.id
        zone.metadata["scenario_hunted_direction"] = scenario.direction
    hunted_indices = zone.metadata.setdefault("scenario_hunted_indices", [])
    if isinstance(hunted_indices, list) and zone_hunt.candle_index not in hunted_indices:
        hunted_indices.append(zone_hunt.candle_index)
