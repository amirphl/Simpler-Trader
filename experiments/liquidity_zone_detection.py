from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Literal, Sequence

from candle_downloader.models import Candle, to_milliseconds
from experiments.bos_choch_detection import (
    BOSCHoCHResult,
    CHoCHUpdate,
    DetectionConfig as BOSCHoCHConfig,
    DetectionEvent,
    detect_bos_choch,
    get_candles,
    infer_initial_direction,
)
from experiments.pivot_detection import Pivot, detect_pivots

Direction = Literal["UPWARD", "DOWNWARD"]
PivotFilter = Literal["BULLISH", "BEARISH", "ALL"]
PairScanOrder = Literal["newest_to_oldest", "oldest_to_newest"]
PivotGroupingMode = Literal["combined", "separate_by_type"]
RepresentativeMode = Literal["choch", "latest_eligible"]
HuntMode = Literal["wick", "close"]


@dataclass(slots=True)
class PriceRange:
    low: float
    high: float

    @property
    def size(self) -> float:
        return self.high - self.low


@dataclass(slots=True)
class DirectionSegment:
    index: int
    direction: Direction
    start_index: int
    end_index: int
    pivots: List[Pivot] = field(default_factory=list)
    representative_pivot: Pivot | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class LiquidityZone:
    id: str
    direction: Direction
    level: int
    left_pivot: Pivot  # older / left-side pivot on the chart
    right_pivot: Pivot  # newer / right-side pivot on the chart
    price_range: PriceRange
    start_index: int
    end_index: int
    is_hunted: bool
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class LiquidityZoneConfig:
    # Upstream detection settings
    scan_length: int = 500
    direction_window: int = 3
    hunt_mode: HuntMode = "wick"
    include_bos_in_choch_range: bool = False
    include_hunt_candle_in_choch_range: bool = True

    # Pivot selection for level-1 liquidity zones.
    # Defaults implement the requested behaviour:
    #   UPWARD   -> bullish, non-hunted pivots
    #   DOWNWARD -> bearish, non-hunted pivots
    up_pivot_filter: PivotFilter = "BULLISH"
    down_pivot_filter: PivotFilter = "BEARISH"
    include_hunted_pivots: bool = False

    # If pivot_filter="ALL":
    #   combined         -> bullish and bearish pivots may pair with each other
    #   separate_by_type -> bullish pairs only with bullish, bearish only with bearish
    pivot_grouping: PivotGroupingMode = "combined"

    # Pair scan policy.  Default is right-to-left/newest-to-oldest as requested.
    pair_scan_order: PairScanOrder = "newest_to_oldest"
    allow_reuse: bool = False

    # Match thresholds
    maximum_pivot_distance: int | None = None
    minimum_overlap: float = 0.0
    minimum_overlap_ratio: float = 0.0
    relaxed_slope: bool = False
    slope_epsilon: float = 0.0
    epsilon: float = 1e-9

    # Representative pivot policy for level-2 zones.
    # "choch" selects the pivot nearest to the CHoCH/reversal source.
    # If a segment has no CHoCH source, allow_representative_fallback controls
    # whether the latest eligible pivot is used.
    representative_mode: RepresentativeMode = "choch"
    representative_include_hunted: bool = False
    allow_representative_fallback: bool = True

    # Zone hunted status.  This is independent from pivot hunted status.
    zone_hunt_mode: HuntMode = "wick"


@dataclass(slots=True)
class LiquidityZoneResult:
    pivots: List[Pivot]
    bos_choch_result: BOSCHoCHResult
    direction_segments: List[DirectionSegment]
    zones_by_level: Dict[int, Dict[Direction, List[LiquidityZone]]]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def _parse_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _opposite_direction(direction: Direction) -> Direction:
    return "DOWNWARD" if direction == "UPWARD" else "UPWARD"


def _validate_config(config: LiquidityZoneConfig) -> None:
    if config.scan_length <= 0:
        raise ValueError("scan_length must be > 0")
    if config.direction_window <= 0:
        raise ValueError("direction_window must be > 0")
    if config.maximum_pivot_distance is not None and config.maximum_pivot_distance <= 0:
        raise ValueError("maximum_pivot_distance must be > 0 when provided")
    if config.minimum_overlap < 0:
        raise ValueError("minimum_overlap must be >= 0")
    if config.minimum_overlap_ratio < 0:
        raise ValueError("minimum_overlap_ratio must be >= 0")
    if config.slope_epsilon < 0:
        raise ValueError("slope_epsilon must be >= 0")
    if config.epsilon < 0:
        raise ValueError("epsilon must be >= 0")


def _pivot_type(pivot: Pivot) -> str:
    return str(getattr(pivot, "type", "")).lower()


def _pivot_is_hunted(pivot: Pivot) -> bool:
    """Return pivot hunted status while staying compatible with older code.

    Some existing pivot code uses the misspelled field name `haunted`; newer
    code should ideally expose `hunted`.  This detector accepts both.
    """
    if hasattr(pivot, "hunted"):
        return bool(getattr(pivot, "hunted"))
    if hasattr(pivot, "haunted"):
        return bool(getattr(pivot, "haunted"))
    return False


def _pivot_filter_for_direction(
    direction: Direction, config: LiquidityZoneConfig
) -> PivotFilter:
    return config.up_pivot_filter if direction == "UPWARD" else config.down_pivot_filter


def _pivot_matches_filter(
    pivot: Pivot,
    filter_type: PivotFilter,
    *,
    include_hunted: bool,
) -> bool:
    if not include_hunted and _pivot_is_hunted(pivot):
        return False

    if filter_type == "ALL":
        return True

    pivot_type = _pivot_type(pivot)
    if filter_type == "BULLISH":
        return pivot_type == "bullish"
    return pivot_type == "bearish"


def _filter_pivots(
    pivots: Sequence[Pivot],
    *,
    direction: Direction,
    include_hunted: bool,
    config: LiquidityZoneConfig,
) -> List[Pivot]:
    filter_type = _pivot_filter_for_direction(direction, config)
    return [
        pivot
        for pivot in pivots
        if _pivot_matches_filter(pivot, filter_type, include_hunted=include_hunted)
    ]


def _body_range(candle: Candle) -> PriceRange:
    return PriceRange(
        low=min(candle.open, candle.close), high=max(candle.open, candle.close)
    )


def _pivot_reference_price(pivot: Pivot, candle: Candle) -> float:
    pivot_type = _pivot_type(pivot)
    if pivot_type == "bullish":
        return candle.low
    if pivot_type == "bearish":
        return candle.high
    return candle.close


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


def _overlap_range(
    older_candle: Candle,
    newer_candle: Candle,
    *,
    epsilon: float,
) -> PriceRange | None:
    """Return the intersection of two candle bodies, or None if they do not intersect."""
    older_body = _body_range(older_candle)
    newer_body = _body_range(newer_candle)
    low = max(older_body.low, newer_body.low)
    high = min(older_body.high, newer_body.high)

    # Touching bodies count as intersection.  Use minimum_overlap > 0 if this
    # should require a non-zero area.
    if high + epsilon < low:
        return None
    return PriceRange(low=low, high=high)


def _passes_slope(
    direction: Direction,
    older_close: float,
    newer_close: float,
    *,
    relaxed: bool,
    epsilon: float,
) -> bool:
    """UPWARD needs ascending closes; DOWNWARD is the exact reverse."""
    if direction == "UPWARD":
        if relaxed:
            return older_close <= newer_close + epsilon
        return older_close < newer_close - epsilon

    if relaxed:
        return older_close >= newer_close - epsilon
    return older_close > newer_close + epsilon


def _passes_overlap_thresholds(
    overlap: PriceRange,
    older_candle: Candle,
    newer_candle: Candle,
    config: LiquidityZoneConfig,
) -> bool:
    overlap_size = overlap.size
    if overlap_size + config.epsilon < config.minimum_overlap:
        return False

    if config.minimum_overlap_ratio <= 0:
        return True

    older_body = _body_range(older_candle)
    newer_body = _body_range(newer_candle)
    reference = min(
        max(older_body.size, config.epsilon),
        max(newer_body.size, config.epsilon),
    )
    return (overlap_size / reference) + config.epsilon >= config.minimum_overlap_ratio


def _pivots_too_far_apart(
    older: Pivot, newer: Pivot, config: LiquidityZoneConfig
) -> bool:
    if config.maximum_pivot_distance is None:
        return False
    return (newer.index - older.index) > config.maximum_pivot_distance


def _is_valid_match(
    *,
    direction: Direction,
    older: Pivot,
    newer: Pivot,
    candles: Sequence[Candle],
    config: LiquidityZoneConfig,
) -> PriceRange | None:
    if older.index < 0 or newer.index >= len(candles) or older.index >= newer.index:
        return None
    if _pivots_too_far_apart(older, newer, config):
        return None

    older_candle = candles[older.index]
    newer_candle = candles[newer.index]
    overlap = _overlap_range(older_candle, newer_candle, epsilon=config.epsilon)
    if overlap is None:
        return None
    if not _passes_overlap_thresholds(overlap, older_candle, newer_candle, config):
        return None
    if not _passes_slope(
        direction,
        older_candle.close,
        newer_candle.close,
        relaxed=config.relaxed_slope,
        epsilon=max(config.epsilon, config.slope_epsilon),
    ):
        return None
    return overlap


def _zone_hunted(
    candles: Sequence[Candle], zone: LiquidityZone, config: LiquidityZoneConfig
) -> bool:
    """Return True once later price sweeps the liquidity zone.

    UPWARD zones are treated as support-side liquidity: hunted if later price
    moves through the lower edge.  DOWNWARD zones are resistance-side liquidity:
    hunted if later price moves through the upper edge.
    """
    if zone.end_index + 1 >= len(candles):
        return False

    low = zone.price_range.low
    high = zone.price_range.high
    eps = config.epsilon

    for candle in candles[zone.end_index + 1 :]:
        if zone.direction == "UPWARD":
            value = candle.low if config.zone_hunt_mode == "wick" else candle.close
            if value <= low + eps:
                return True
        else:
            value = candle.high if config.zone_hunt_mode == "wick" else candle.close
            if value >= high - eps:
                return True
    return False


def _build_zone(
    *,
    zone_id: str,
    level: int,
    direction: Direction,
    older: Pivot,
    newer: Pivot,
    overlap: PriceRange,
    candles: Sequence[Candle],
    config: LiquidityZoneConfig,
    metadata: Dict[str, object] | None = None,
) -> LiquidityZone:
    zone = LiquidityZone(
        id=zone_id,
        direction=direction,
        level=level,
        left_pivot=older,
        right_pivot=newer,
        price_range=overlap,
        start_index=older.index,
        end_index=newer.index,
        is_hunted=False,
        metadata=metadata or {},
    )
    zone.is_hunted = _zone_hunted(candles, zone, config)
    zone.metadata.setdefault("overlap_size", overlap.size)
    zone.metadata.setdefault("pivot_distance", newer.index - older.index)
    zone.metadata.setdefault(
        "close_delta", candles[newer.index].close - candles[older.index].close
    )
    zone.metadata.setdefault("left_pivot_hunted", _pivot_is_hunted(older))
    zone.metadata.setdefault("right_pivot_hunted", _pivot_is_hunted(newer))
    return zone


def _split_for_pairing(
    pivots: Sequence[Pivot], config: LiquidityZoneConfig
) -> List[tuple[str, List[Pivot]]]:
    if config.pivot_grouping == "combined":
        return [("", list(pivots))]

    bullish = [pivot for pivot in pivots if _pivot_type(pivot) == "bullish"]
    bearish = [pivot for pivot in pivots if _pivot_type(pivot) == "bearish"]
    groups: List[tuple[str, List[Pivot]]] = []
    if bullish:
        groups.append(("bullish", bullish))
    if bearish:
        groups.append(("bearish", bearish))
    return groups


def _pair_zones(
    *,
    pivots: Sequence[Pivot],
    direction: Direction,
    level: int,
    candles: Sequence[Candle],
    config: LiquidityZoneConfig,
    zone_prefix: str,
    metadata_builder: Callable[[Pivot, Pivot], Dict[str, object]] | None = None,
) -> List[LiquidityZone]:
    """Pair pivots into liquidity zones.

    The default scan is newest->oldest.  For ordered pivots [p1, p2, p3, p4]:
      - if p1+p2 match, produce a zone and continue with p3+p4;
      - if p1+p2 fail, shift by one and try p2+p3.
    With allow_reuse=True, a matched p2 can still be reused in p2+p3.
    """
    zones: List[LiquidityZone] = []
    reverse = config.pair_scan_order == "newest_to_oldest"

    for group_name, group_pivots in _split_for_pairing(pivots, config):
        ordered = sorted(group_pivots, key=lambda pivot: pivot.index, reverse=reverse)
        if len(ordered) < 2:
            continue

        i = 0
        while i < len(ordered) - 1:
            first = ordered[i]
            second = ordered[i + 1]
            older, newer = (
                (first, second) if first.index < second.index else (second, first)
            )

            overlap = _is_valid_match(
                direction=direction,
                older=older,
                newer=newer,
                candles=candles,
                config=config,
            )
            if overlap is None:
                i += 1
                continue

            metadata = metadata_builder(older, newer) if metadata_builder else {}
            if group_name:
                metadata.setdefault("pivot_pair_group", group_name)

            zones.append(
                _build_zone(
                    zone_id=f"{zone_prefix}_{len(zones)}",
                    level=level,
                    direction=direction,
                    older=older,
                    newer=newer,
                    overlap=overlap,
                    candles=candles,
                    config=config,
                    metadata=metadata,
                )
            )
            i += 1 if config.allow_reuse else 2

    return zones


# ---------------------------------------------------------------------------
# Direction segmentation and CHoCH representative selection
# ---------------------------------------------------------------------------


def _bos_direction_by_id(bos_choch_result: BOSCHoCHResult) -> Dict[int, Direction]:
    return {bos.index: bos.direction for bos in bos_choch_result.bos_records}


def _choch_update_for_reversal(
    reversal: DetectionEvent,
    bos_choch_result: BOSCHoCHResult,
) -> CHoCHUpdate | None:
    """Find the CHoCH update that produced a direction reversal.

    In the BOS/CHoCH detector, a reversal happens after the newest active BOS's
    CHoCH is hunted.  The detector records one or more CHOCH continuous updates
    on the reversal candle.  We choose the newest BOS from the previous
    direction; this is the CHoCH source of the new segment.
    """
    bos_direction = _bos_direction_by_id(bos_choch_result)
    previous_direction = _opposite_direction(reversal.direction)

    same_candle_candidates = [
        update
        for update in bos_choch_result.choch_updates
        if update.candle_index == reversal.candle_index
        and update.reason == "continuous"
        and bos_direction.get(update.bos_index) == previous_direction
    ]
    if same_candle_candidates:
        return max(same_candle_candidates, key=lambda update: update.bos_index)

    # Defensive fallback for older BOS/CHoCH implementations that may not have
    # reason="continuous" or exact same-candle metadata.
    fallback_candidates = [
        update
        for update in bos_choch_result.choch_updates
        if update.candle_index <= reversal.candle_index
        and bos_direction.get(update.bos_index) == previous_direction
    ]
    if not fallback_candidates:
        return None
    return max(
        fallback_candidates, key=lambda update: (update.candle_index, update.bos_index)
    )


def _latest_eligible_representative(
    segment: DirectionSegment,
    config: LiquidityZoneConfig,
) -> Pivot | None:
    eligible = _filter_pivots(
        segment.pivots,
        direction=segment.direction,
        include_hunted=config.representative_include_hunted,
        config=config,
    )
    if not eligible:
        eligible = [
            pivot
            for pivot in segment.pivots
            if config.representative_include_hunted or not _pivot_is_hunted(pivot)
        ]
    if not eligible:
        return None
    return max(eligible, key=lambda pivot: pivot.index)


def _choch_representative(
    segment: DirectionSegment,
    candles: Sequence[Candle],
    config: LiquidityZoneConfig,
) -> Pivot | None:
    source_index = segment.metadata.get("representative_source_index")
    source_level = segment.metadata.get("representative_source_level")

    if source_index is None:
        return (
            _latest_eligible_representative(segment, config)
            if config.allow_representative_fallback
            else None
        )

    eligible = _filter_pivots(
        segment.pivots,
        direction=segment.direction,
        include_hunted=config.representative_include_hunted,
        config=config,
    )
    if not eligible and config.allow_representative_fallback:
        eligible = [
            pivot
            for pivot in segment.pivots
            if config.representative_include_hunted or not _pivot_is_hunted(pivot)
        ]
    if not eligible:
        return None

    src_idx = int(source_index)
    src_level = float(source_level) if source_level is not None else None

    def key(pivot: Pivot) -> tuple[float, float, int]:
        index_distance = abs(pivot.index - src_idx)
        if src_level is None or pivot.index < 0 or pivot.index >= len(candles):
            price_distance = 0.0
        else:
            price_distance = abs(
                _pivot_reference_price(pivot, candles[pivot.index]) - src_level
            )
        # Final tie-breaker: prefer the newest pivot.
        return (index_distance, price_distance, -pivot.index)

    return min(eligible, key=key)


def _segment_representative(
    segment: DirectionSegment,
    candles: Sequence[Candle],
    config: LiquidityZoneConfig,
) -> Pivot | None:
    if config.representative_mode == "latest_eligible":
        return _latest_eligible_representative(segment, config)
    return _choch_representative(segment, candles, config)


def _build_direction_segments(
    candles: Sequence[Candle],
    pivots: Sequence[Pivot],
    bos_choch_result: BOSCHoCHResult,
    config: LiquidityZoneConfig,
) -> List[DirectionSegment]:
    if not candles:
        return []

    initial_direction = infer_initial_direction(
        candles, config.direction_window
    ).direction
    reversals = sorted(
        (
            event
            for event in bos_choch_result.events
            if event.event == "DIRECTION_REVERSED"
        ),
        key=lambda event: event.candle_index,
    )

    segments: List[DirectionSegment] = []
    current_direction = initial_direction
    current_start = 0
    current_metadata: Dict[str, object] = {
        "reversal_candle_index": None,
        "representative_source": None,
    }

    for reversal in reversals:
        end_index = reversal.candle_index - 1
        if end_index >= current_start:
            segment_pivots = [
                pivot for pivot in pivots if current_start <= pivot.index <= end_index
            ]
            metadata = dict(current_metadata)
            metadata["next_reversal_candle_index"] = reversal.candle_index
            segment = DirectionSegment(
                index=len(segments),
                direction=current_direction,
                start_index=current_start,
                end_index=end_index,
                pivots=segment_pivots,
                metadata=metadata,
            )
            segment.representative_pivot = _segment_representative(
                segment, candles, config
            )
            segments.append(segment)

        choch_update = _choch_update_for_reversal(reversal, bos_choch_result)
        current_direction = reversal.direction
        current_start = reversal.candle_index
        current_metadata = {
            "reversal_candle_index": reversal.candle_index,
            "representative_source": "choch",
            "representative_source_bos_index": choch_update.bos_index
            if choch_update
            else None,
            "representative_source_index": choch_update.candle_index
            if choch_update
            else reversal.candle_index,
            "representative_source_level": choch_update.level if choch_update else None,
        }

    tail_pivots = [
        pivot for pivot in pivots if current_start <= pivot.index < len(candles)
    ]
    tail = DirectionSegment(
        index=len(segments),
        direction=current_direction,
        start_index=current_start,
        end_index=len(candles) - 1,
        pivots=tail_pivots,
        metadata={**current_metadata, "next_reversal_candle_index": None},
    )
    tail.representative_pivot = _segment_representative(tail, candles, config)
    segments.append(tail)
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_liquidity_zones(
    candles: Sequence[Candle],
    config: LiquidityZoneConfig | None = None,
) -> LiquidityZoneResult:
    if not candles:
        raise ValueError("candles cannot be empty")

    cfg = config or LiquidityZoneConfig()
    _validate_config(cfg)

    pivots = detect_pivots(candles, cfg.scan_length)
    bos_choch_result = detect_bos_choch(
        candles,
        BOSCHoCHConfig(
            direction_window=cfg.direction_window,
            hunt_mode=cfg.hunt_mode,
            include_bos_in_choch_range=cfg.include_bos_in_choch_range,
            include_hunt_candle_in_choch_range=cfg.include_hunt_candle_in_choch_range,
        ),
    )
    direction_segments = _build_direction_segments(
        candles, pivots, bos_choch_result, cfg
    )

    zones_by_level: Dict[int, Dict[Direction, List[LiquidityZone]]] = {
        1: {"UPWARD": [], "DOWNWARD": []},
        2: {"UPWARD": [], "DOWNWARD": []},
    }

    # Level 1: zones inside each direction segment.
    for segment in direction_segments:
        filtered = _filter_pivots(
            segment.pivots,
            direction=segment.direction,
            include_hunted=cfg.include_hunted_pivots,
            config=cfg,
        )
        zones = _pair_zones(
            pivots=filtered,
            direction=segment.direction,
            level=1,
            candles=candles,
            config=cfg,
            zone_prefix=f"L1_{segment.direction}_{segment.index}",
            metadata_builder=lambda older, newer, segment=segment: {
                "segment_index": segment.index,
                "segment_start_index": segment.start_index,
                "segment_end_index": segment.end_index,
                "older_pivot_type": _pivot_type(older),
                "newer_pivot_type": _pivot_type(newer),
            },
        )
        zones_by_level[1][segment.direction].extend(zones)

    # Level 2: zones between CHoCH representatives of same-direction segments.
    representatives_by_direction: Dict[Direction, List[Pivot]] = {
        "UPWARD": [],
        "DOWNWARD": [],
    }
    representative_segment_ids: Dict[Direction, Dict[int, int]] = {
        "UPWARD": {},
        "DOWNWARD": {},
    }
    for segment in direction_segments:
        representative = segment.representative_pivot
        if representative is None:
            continue
        representatives_by_direction[segment.direction].append(representative)
        representative_segment_ids[segment.direction][representative.index] = (
            segment.index
        )

    for direction in ("UPWARD", "DOWNWARD"):
        zones = _pair_zones(
            pivots=representatives_by_direction[direction],
            direction=direction,
            level=2,
            candles=candles,
            config=cfg,
            zone_prefix=f"L2_{direction}",
            metadata_builder=lambda older, newer, direction=direction: {
                "older_segment_index": representative_segment_ids[direction].get(
                    older.index
                ),
                "newer_segment_index": representative_segment_ids[direction].get(
                    newer.index
                ),
                "older_pivot_type": _pivot_type(older),
                "newer_pivot_type": _pivot_type(newer),
            },
        )
        zones_by_level[2][direction].extend(zones)

    return LiquidityZoneResult(
        pivots=pivots,
        bos_choch_result=bos_choch_result,
        direction_segments=direction_segments,
        zones_by_level=zones_by_level,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_liquidity_zone_plotly_figure(
    candles: Sequence[Candle], result: LiquidityZoneResult
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly is required for plotting. Install with: pip install plotly"
        ) from exc

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=[c.open_time for c in candles],
                open=[c.open for c in candles],
                high=[c.high for c in candles],
                low=[c.low for c in candles],
                close=[c.close for c in candles],
                name="candles",
            )
        ]
    )

    valid_pivots = [pivot for pivot in result.pivots if 0 <= pivot.index < len(candles)]
    pivot_x = [candles[pivot.index].open_time for pivot in valid_pivots]
    pivot_y = [
        candles[pivot.index].low
        if _pivot_type(pivot) == "bullish"
        else candles[pivot.index].high
        for pivot in valid_pivots
    ]
    pivot_text = [
        f"{_pivot_type(pivot)} pivot idx={pivot.index} hunted={_pivot_is_hunted(pivot)}"
        for pivot in valid_pivots
    ]
    fig.add_trace(
        go.Scatter(
            x=pivot_x,
            y=pivot_y,
            mode="markers",
            marker={
                "size": 7,
                "color": [
                    "#22c55e" if _pivot_type(pivot) == "bullish" else "#ef4444"
                    for pivot in valid_pivots
                ],
                "symbol": [
                    "triangle-up"
                    if _pivot_type(pivot) == "bullish"
                    else "triangle-down"
                    for pivot in valid_pivots
                ],
            },
            name="pivots",
            text=pivot_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    representative_segments = [
        segment
        for segment in result.direction_segments
        if segment.representative_pivot is not None
    ]
    fig.add_trace(
        go.Scatter(
            x=[
                candles[segment.representative_pivot.index].open_time
                for segment in representative_segments
            ],
            y=[
                candles[segment.representative_pivot.index].close
                for segment in representative_segments
            ],
            mode="markers+text",
            marker={"size": 10, "color": "#f59e0b", "symbol": "diamond"},
            text=[f"SEG_{segment.index}" for segment in representative_segments],
            textposition="top center",
            name="representatives",
            hovertemplate=(
                "segment=%{text}<br>"
                "direction=%{customdata[0]}<br>"
                "pivot_index=%{customdata[1]}<br>"
                "source=%{customdata[2]}<extra></extra>"
            ),
            customdata=[
                [
                    segment.direction,
                    segment.representative_pivot.index,
                    segment.metadata.get("representative_source"),
                ]
                for segment in representative_segments
            ],
        )
    )

    zone_styles = {
        (1, "UPWARD"): ("rgba(34, 197, 94, 0.18)", "#22c55e"),
        (1, "DOWNWARD"): ("rgba(239, 68, 68, 0.18)", "#ef4444"),
        (2, "UPWARD"): ("rgba(59, 130, 246, 0.18)", "#3b82f6"),
        (2, "DOWNWARD"): ("rgba(234, 179, 8, 0.18)", "#eab308"),
    }

    for level, zones_by_direction in result.zones_by_level.items():
        for direction, zones in zones_by_direction.items():
            fillcolor, line_color = zone_styles[(level, direction)]
            for zone in zones:
                x0 = candles[zone.start_index].open_time
                x1 = candles[zone.end_index].open_time
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=zone.price_range.low,
                    y1=zone.price_range.high,
                    line={"width": 1, "color": line_color},
                    fillcolor=fillcolor,
                    opacity=0.22,
                )
                fig.add_annotation(
                    x=x0,
                    y=zone.price_range.high,
                    text=zone.id,
                    showarrow=False,
                    font={"color": line_color, "size": 10},
                    xanchor="left",
                    yanchor="bottom",
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[zone.price_range.high, zone.price_range.high],
                        mode="lines",
                        line={"width": 2, "color": line_color},
                        name=zone.id,
                        hovertemplate=(
                            f"{zone.id}<br>"
                            f"direction={zone.direction}<br>"
                            f"level={zone.level}<br>"
                            f"range={zone.price_range.low:.6f}-{zone.price_range.high:.6f}<br>"
                            f"hunted={zone.is_hunted}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title="Liquidity Zone Detection",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run liquidity-zone detection on candle data."
    )
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    parser.add_argument("--timeframe", required=True, help="Timeframe, e.g. 1h")
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime in ISO format, e.g. 2026-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime in ISO format, e.g. 2026-01-10T00:00:00Z",
    )
    parser.add_argument(
        "--scan-length", type=int, default=500, help="Pivot scan length"
    )
    parser.add_argument("--direction-window", type=int, default=3)
    parser.add_argument("--hunt-mode", choices=["wick", "close"], default="wick")
    parser.add_argument(
        "--include-bos-in-choch-range",
        action="store_true",
        help="Include BOS completion candle in initial CHoCH range",
    )
    parser.add_argument(
        "--exclude-hunt-candle-in-choch-range",
        action="store_true",
        help="Exclude BOS hunt candle from initial CHoCH range",
    )
    parser.add_argument(
        "--up-pivot-filter", choices=["BULLISH", "BEARISH", "ALL"], default="BULLISH"
    )
    parser.add_argument(
        "--down-pivot-filter", choices=["BULLISH", "BEARISH", "ALL"], default="BEARISH"
    )
    parser.add_argument(
        "--include-hunted-pivots",
        action="store_true",
        help="Include hunted pivots during level-1 zone detection",
    )
    parser.add_argument(
        "--pivot-grouping", choices=["combined", "separate_by_type"], default="combined"
    )
    parser.add_argument(
        "--pair-scan-order",
        choices=["newest_to_oldest", "oldest_to_newest"],
        default="newest_to_oldest",
    )
    parser.add_argument("--allow-reuse", action="store_true")
    parser.add_argument("--maximum-pivot-distance", type=int, default=None)
    parser.add_argument("--minimum-overlap", type=float, default=0.0)
    parser.add_argument("--minimum-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--relaxed-slope", action="store_true")
    parser.add_argument("--slope-epsilon", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument(
        "--representative-mode", choices=["choch", "latest_eligible"], default="choch"
    )
    parser.add_argument(
        "--representative-include-hunted",
        action="store_true",
        help="Allow hunted pivots to be selected as segment representatives",
    )
    parser.add_argument(
        "--disable-representative-fallback",
        action="store_true",
        help="Do not use latest eligible pivot when a segment has no CHoCH source",
    )
    parser.add_argument("--zone-hunt-mode", choices=["wick", "close"], default="wick")
    parser.add_argument(
        "--source", choices=["binance", "csv"], default="binance", help="Candle source"
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Required when --source csv; path to candle CSV file",
    )
    parser.add_argument("--http-proxy", default=None, help="Optional HTTP proxy URL")
    parser.add_argument("--https-proxy", default=None, help="Optional HTTPS proxy URL")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render Plotly chart with pivots, segment representatives, and liquidity zones",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional .html output path for Plotly figure",
    )
    args = parser.parse_args()

    proxies = {}
    if args.http_proxy:
        proxies["http"] = args.http_proxy
    if args.https_proxy:
        proxies["https"] = args.https_proxy

    candles = get_candles(
        source=args.source,
        symbol=args.symbol.strip().upper(),
        interval=args.timeframe.strip(),
        start_ms=to_milliseconds(_parse_datetime(args.start)),
        end_ms=to_milliseconds(_parse_datetime(args.end)),
        csv_path=args.csv_path,
        proxies=proxies or None,
    )

    result = detect_liquidity_zones(
        candles,
        LiquidityZoneConfig(
            scan_length=args.scan_length,
            direction_window=args.direction_window,
            hunt_mode=args.hunt_mode,
            include_bos_in_choch_range=args.include_bos_in_choch_range,
            include_hunt_candle_in_choch_range=not args.exclude_hunt_candle_in_choch_range,
            up_pivot_filter=args.up_pivot_filter,
            down_pivot_filter=args.down_pivot_filter,
            include_hunted_pivots=args.include_hunted_pivots,
            pivot_grouping=args.pivot_grouping,
            pair_scan_order=args.pair_scan_order,
            allow_reuse=args.allow_reuse,
            maximum_pivot_distance=args.maximum_pivot_distance,
            minimum_overlap=args.minimum_overlap,
            minimum_overlap_ratio=args.minimum_overlap_ratio,
            relaxed_slope=args.relaxed_slope,
            slope_epsilon=args.slope_epsilon,
            epsilon=args.epsilon,
            representative_mode=args.representative_mode,
            representative_include_hunted=args.representative_include_hunted,
            allow_representative_fallback=not args.disable_representative_fallback,
            zone_hunt_mode=args.zone_hunt_mode,
        ),
    )

    print(f"Candles: {len(candles)}")
    print(f"Pivots: {len(result.pivots)}")
    print(f"Direction segments: {len(result.direction_segments)}")
    for segment in result.direction_segments:
        representative = (
            segment.representative_pivot.index if segment.representative_pivot else None
        )
        print(
            f"SEG_{segment.index} dir={segment.direction} start={segment.start_index} end={segment.end_index} "
            f"pivots={len(segment.pivots)} representative={representative} "
            f"source={segment.metadata.get('representative_source')}"
        )

    for level in (1, 2):
        for direction in ("UPWARD", "DOWNWARD"):
            zones = result.zones_by_level[level][direction]
            print(f"Level {level} {direction}: {len(zones)}")
            for zone in zones:
                print(
                    f"{zone.id} left={zone.left_pivot.index} right={zone.right_pivot.index} "
                    f"range=({zone.price_range.low}, {zone.price_range.high}) hunted={zone.is_hunted}"
                )

    if args.plot:
        fig = build_liquidity_zone_plotly_figure(candles, result)
        if args.plot_output:
            fig.write_html(args.plot_output)
            print(f"Plot written to {args.plot_output}")
        else:
            fig.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
