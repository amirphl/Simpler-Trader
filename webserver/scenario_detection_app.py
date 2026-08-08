from __future__ import annotations

import logging
import os
from typing import Sequence

from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]

from candle_downloader.models import Candle, to_milliseconds
from experiments.bos_choch_detection import BOSRecord
from experiments.liquidity_zone_detection import LiquidityZone, LiquidityZoneConfig
from experiments.pivot_detection import Pivot
from experiments.scenario_detection import (
    LiquidityZoneLinePenetration,
    PivotHuntEvent,
    PivotSnapshot,
    ScenarioDetectionConfig,
    ScenarioRecord,
    detect_scenarios,
)
from .common import (
    UI_DIR,
    bool_env,
    build_proxy_map,
    build_trusted_hosts,
    candle_payloads as build_candle_payloads,
    configure_standard_app,
    list_env,
)
from .models import (
    BOSCHoCHMarker,
    DirectionReversalEvent,
    LiquidityDirectionSegment,
    LiquidityZonePayload,
    LiquidityZonePivot,
    ScenarioBOSPayload,
    ScenarioDetectionRequest,
    ScenarioDetectionResponse,
    ScenarioPayload,
    ScenarioPenetrationPayload,
    ScenarioPivotHuntPayload,
    ScenarioPivotSnapshotPayload,
)


logger = logging.getLogger("scenariodetectionserver")
app = FastAPI(title="Scenario Detection Experiment", version="1.0.0")

DEFAULT_TRUSTED_HOSTS = [
    "localhost",
    "127.0.0.1",
]

TRUSTED_HOSTS = build_trusted_hosts(DEFAULT_TRUSTED_HOSTS)
ALLOWED_ORIGINS = list_env(
    "WEB_ALLOWED_ORIGINS",
    "http://localhost:9096,http://127.0.0.1:9096",
)
FORCE_HTTPS = bool_env("WEB_FORCE_HTTPS", False)
SCENARIO_CANDLE_SOURCE = (
    os.getenv("SCENARIO_CANDLE_SOURCE", "binance").strip().lower()
)
SCENARIO_CANDLE_CSV_PATH = os.getenv("SCENARIO_CANDLE_CSV_PATH")

configure_standard_app(
    app,
    trusted_hosts=TRUSTED_HOSTS,
    allowed_origins=ALLOWED_ORIGINS,
    force_https=FORCE_HTTPS,
)


@app.get("/", response_class=FileResponse)
async def scenario_detection_page() -> FileResponse:
    return FileResponse(UI_DIR / "scenario_detection.html")


@app.post("/api/scenarios", response_model=ScenarioDetectionResponse)
async def compute_scenarios(
    payload: ScenarioDetectionRequest,
) -> ScenarioDetectionResponse:
    proxies = build_proxy_map(
        http_proxy=payload.http_proxy,
        https_proxy=payload.https_proxy,
        include_standard_env=True,
    )

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    source = (payload.source or SCENARIO_CANDLE_SOURCE or "binance").strip().lower()
    csv_path = payload.csv_path or SCENARIO_CANDLE_CSV_PATH
    symbol = payload.symbol.upper()
    try:
        from experiments.pivot_detection import get_candles

        candles = get_candles(
            source=source,
            symbol=symbol,
            interval=payload.timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            csv_path=csv_path,
            proxies=proxies or None,
            logger=logger.getChild(source),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Candle fetch failed: %s", exc)
        raise HTTPException(
            status_code=502, detail=f"Candle source error: {exc}"
        ) from exc

    if not candles:
        raise HTTPException(
            status_code=422, detail="No candles returned for the given parameters"
        )

    current_index = _current_index(payload, len(candles))
    try:
        result = detect_scenarios(
            candles,
            ScenarioDetectionConfig(
                liquidity_zone_config=_liquidity_config(payload),
                epsilon=payload.scenario_epsilon,
                zone_selection=payload.zone_selection,
                bullish_pivot_selection=payload.bullish_pivot_selection,
                bearish_pivot_selection=payload.bearish_pivot_selection,
                mark_zone_hunted=payload.mark_zone_hunted,
                deduplicate_by_zone_and_current=payload.deduplicate_by_zone_and_current,
            ),
            current_index=current_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    liquidity_result = result.liquidity_zone_result
    response_candles = build_candle_payloads(candles)
    pivots = [_pivot_payload(pivot, candles, payload) for pivot in liquidity_result.pivots]
    segments = [
        _segment_payload(segment, candles, payload)
        for segment in liquidity_result.direction_segments
    ]
    zones = [_zone_payload(zone, candles, payload) for zone in result.zones]
    zones.sort(key=lambda zone: (zone.start_index, zone.level, zone.id))
    penetrations = [
        _penetration_payload(penetration)
        for zone_penetrations in result.penetrations_by_zone.values()
        for penetration in zone_penetrations
    ]
    penetrations.sort(
        key=lambda item: (item.candle_index, item.zone_level, item.zone_id, item.side)
    )
    scenarios = [
        _scenario_payload(scenario)
        for scenario in result.scenarios
    ]
    markers = _marker_payloads(candles, liquidity_result.bos_choch_result, payload)
    direction_reversals = [
        DirectionReversalEvent(
            candle_index=event.candle_index,
            time=candles[event.candle_index].open_time,
            direction=event.direction,
            details=event.details,
        )
        for event in liquidity_result.bos_choch_result.events
        if event.event == "DIRECTION_REVERSED"
        and 0 <= event.candle_index < len(candles)
    ]

    return ScenarioDetectionResponse(
        candles=response_candles,
        pivots=pivots,
        segments=segments,
        zones=zones,
        markers=markers,
        direction_reversals=direction_reversals,
        penetrations=penetrations,
        scenarios=scenarios,
        current_index=result.current_index,
    )


def _current_index(payload: ScenarioDetectionRequest, candle_count: int) -> int | None:
    if payload.current_index is not None:
        if payload.current_index >= candle_count:
            raise HTTPException(
                status_code=400,
                detail="current_index must point to a fetched candle",
            )
        return payload.current_index
    if payload.current_candle_only:
        return candle_count - 1
    return None


def _liquidity_config(payload: ScenarioDetectionRequest) -> LiquidityZoneConfig:
    return LiquidityZoneConfig(
        scan_length=payload.scan_length,
        pivot_min_swing_pct=payload.pivot_min_swing_pct,
        direction_window=payload.direction_window,
        hunt_mode=payload.hunt_mode,
        include_hunt_candle_in_choch_range=payload.include_hunt_candle_in_choch_range,
        min_swing_pct=payload.min_swing_pct,
        include_pullback_in_bos_level=payload.include_pullback_in_bos_level,
        up_pivot_filter=payload.up_pivot_filter,
        down_pivot_filter=payload.down_pivot_filter,
        include_hunted_pivots=payload.include_hunted_pivots,
        pair_scan_order=payload.pair_scan_order,
        representative_include_hunted=payload.representative_include_hunted,
        representative_mode=payload.representative_mode,
        allow_representative_fallback=payload.allow_representative_fallback,
        maximum_pivot_distance=payload.maximum_pivot_distance,
        minimum_overlap=payload.minimum_overlap,
        minimum_overlap_ratio=payload.minimum_overlap_ratio,
        allow_reuse=payload.allow_reuse,
        relaxed_slope=payload.relaxed_slope,
        slope_epsilon=payload.slope_epsilon,
        epsilon=payload.epsilon,
        intersection_method=payload.intersection_method,
        slope_attribute=payload.slope_attribute,
    )


def _pivot_payload(
    pivot: Pivot,
    candles: Sequence[Candle],
    payload: ScenarioDetectionRequest,
) -> LiquidityZonePivot:
    return LiquidityZonePivot(
        index=pivot.index,
        type=pivot.type,  # type: ignore[arg-type]
        high=pivot.high,
        low=pivot.low,
        haunted=_pivot_is_hunted(pivot),
        time=candles[pivot.index].open_time
        if 0 <= pivot.index < len(candles)
        else payload.start,
    )


def _pivot_is_hunted(pivot: object) -> bool:
    if hasattr(pivot, "hunted"):
        return bool(getattr(pivot, "hunted"))
    if hasattr(pivot, "haunted"):
        return bool(getattr(pivot, "haunted"))
    return False


def _segment_payload(
    segment: object,
    candles: Sequence[Candle],
    payload: ScenarioDetectionRequest,
) -> LiquidityDirectionSegment:
    representative = getattr(segment, "representative_pivot", None)
    representative_index = representative.index if representative is not None else None
    return LiquidityDirectionSegment(
        index=getattr(segment, "index"),
        direction=getattr(segment, "direction"),
        start_index=getattr(segment, "start_index"),
        end_index=getattr(segment, "end_index"),
        start_time=candles[getattr(segment, "start_index")].open_time
        if 0 <= getattr(segment, "start_index") < len(candles)
        else payload.start,
        end_time=candles[getattr(segment, "end_index")].open_time
        if 0 <= getattr(segment, "end_index") < len(candles)
        else payload.end,
        pivot_count=len(getattr(segment, "pivots")),
        representative_pivot_index=representative_index,
        representative_pivot_time=(
            candles[representative_index].open_time
            if representative_index is not None
            and 0 <= representative_index < len(candles)
            else None
        ),
    )


def _zone_payload(
    zone: LiquidityZone,
    candles: Sequence[Candle],
    payload: ScenarioDetectionRequest,
) -> LiquidityZonePayload:
    return LiquidityZonePayload(
        id=zone.id,
        direction=zone.direction,
        level=zone.level,  # type: ignore[arg-type]
        start_index=zone.start_index,
        end_index=zone.end_index,
        start_time=candles[zone.start_index].open_time
        if 0 <= zone.start_index < len(candles)
        else payload.start,
        end_time=candles[zone.end_index].open_time
        if 0 <= zone.end_index < len(candles)
        else payload.end,
        price_low=zone.price_range.low,
        price_high=zone.price_range.high,
        line_price=zone.line_price,
        line_method=zone.line_method,
        is_hunted=zone.is_hunted,
        left_pivot_index=zone.left_pivot.index,
        right_pivot_index=zone.right_pivot.index,
        left_pivot_time=candles[zone.left_pivot.index].open_time
        if 0 <= zone.left_pivot.index < len(candles)
        else payload.start,
        right_pivot_time=candles[zone.right_pivot.index].open_time
        if 0 <= zone.right_pivot.index < len(candles)
        else payload.end,
        left_pivot_type=zone.left_pivot.type,  # type: ignore[arg-type]
        right_pivot_type=zone.right_pivot.type,  # type: ignore[arg-type]
        metadata=zone.metadata,
    )


def _penetration_payload(
    penetration: LiquidityZoneLinePenetration,
) -> ScenarioPenetrationPayload:
    return ScenarioPenetrationPayload(
        zone_key=penetration.zone_key,
        zone_id=penetration.zone_id,
        zone_level=penetration.zone_level,  # type: ignore[arg-type]
        zone_direction=penetration.zone_direction,
        candle_index=penetration.candle_index,
        previous_candle_index=penetration.previous_candle_index,
        side=penetration.side,
        line_price=penetration.line_price,
        candle_open=penetration.candle_open,
        candle_high=penetration.candle_high,
        candle_low=penetration.candle_low,
        candle_close=penetration.candle_close,
        time=penetration.time,
        metadata=penetration.metadata,
    )


def _scenario_payload(
    scenario: ScenarioRecord,
) -> ScenarioPayload:
    return ScenarioPayload(
        id=scenario.id,
        direction=scenario.direction,
        zone_key=scenario.zone_key,
        zone_id=scenario.zone.id,
        zone_direction=scenario.zone.direction,
        zone_level=scenario.zone.level,  # type: ignore[arg-type]
        line_price=scenario.zone.line_price,
        zone_hunt=_penetration_payload(scenario.zone_hunt),
        confirming_pivot_hunt=_pivot_hunt_payload(scenario.confirming_pivot_hunt),
        cancelling_pivot_hunt=_pivot_hunt_payload(scenario.cancelling_pivot_hunt)
        if scenario.cancelling_pivot_hunt is not None
        else None,
        relevant_bullish_pivot=_pivot_snapshot_payload(
            scenario.relevant_bullish_pivot
        ),
        relevant_bearish_pivot=_pivot_snapshot_payload(
            scenario.relevant_bearish_pivot
        ),
        metadata=scenario.metadata,
    )


def _pivot_snapshot_payload(
    pivot: PivotSnapshot | None,
) -> ScenarioPivotSnapshotPayload | None:
    if pivot is None:
        return None
    return ScenarioPivotSnapshotPayload(
        pivot_index=pivot.pivot_index,
        pivot_type=pivot.pivot_type,
        price=pivot.price,
        activation_index=pivot.activation_index,
        first_hunt_index=pivot.first_hunt_index,
        hunted_at_zone_hunt=pivot.hunted_at_zone_hunt,
        selection_mode=pivot.selection_mode,
        time=pivot.time,
        metadata=pivot.metadata,
    )


def _pivot_hunt_payload(event: PivotHuntEvent) -> ScenarioPivotHuntPayload:
    return ScenarioPivotHuntPayload(
        pivot_index=event.pivot_index,
        pivot_type=event.pivot_type,
        candle_index=event.candle_index,
        side=event.side,
        price=event.price,
        candle_open=event.candle_open,
        candle_high=event.candle_high,
        candle_low=event.candle_low,
        candle_close=event.candle_close,
        time=event.time,
        metadata=event.metadata,
    )


def _bos_payload(bos: BOSRecord, candles: Sequence[Candle]) -> ScenarioBOSPayload:
    return ScenarioBOSPayload(
        index=bos.index,
        direction=bos.direction,
        hunt_index=bos.hunt_index,
        hunt_time=candles[bos.hunt_index].open_time
        if 0 <= bos.hunt_index < len(candles)
        else None,
        level=bos.level,
        label=bos.label,
    )


def _marker_payloads(
    candles: Sequence[Candle],
    bos_choch_result: object,
    payload: ScenarioDetectionRequest,
) -> list[BOSCHoCHMarker]:
    markers: list[BOSCHoCHMarker] = []

    for bos in bos_choch_result.bos_records:
        if not 0 <= bos.hunt_index < len(candles):
            continue
        candle = candles[bos.hunt_index]
        line_start_index = max(0, bos.start_index)
        markers.append(
            BOSCHoCHMarker(
                type="BOS",
                index=bos.index,
                direction=bos.direction,
                candle_index=bos.hunt_index,
                time=candle.open_time,
                price=bos.level,
                high=candle.high,
                low=candle.low,
                label=f"BOS {bos.index}",
                line_start_time=candles[line_start_index].open_time,
                line_end_time=candle.open_time,
            )
        )

    bos_by_index = {bos.index: bos for bos in bos_choch_result.bos_records}
    for choch_index, choch in enumerate(bos_choch_result.choch_records_by_bos.values()):
        if not 0 <= choch.candle_index < len(candles):
            continue
        bos = bos_by_index.get(choch.bos_index)
        if bos is None:
            continue
        candle = candles[choch.candle_index]
        line_end_index = min(choch.candle_index + 3, len(candles) - 1)
        markers.append(
            BOSCHoCHMarker(
                type="CHoCH",
                index=choch_index,
                direction=bos.direction,
                candle_index=choch.candle_index,
                time=candle.open_time,
                price=choch.level,
                high=candle.high,
                low=candle.low,
                label=f"CHoCH {choch_index}",
                line_start_time=candle.open_time,
                line_end_time=candles[line_end_index].open_time,
                bos_index=choch.bos_index,
            )
        )

    markers.sort(key=lambda marker: (marker.candle_index, marker.type, marker.index))
    return markers
