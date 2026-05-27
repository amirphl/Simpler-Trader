from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]

from candle_downloader.models import to_milliseconds
from experiments.bos_choch_detection import (
    CHoCHUpdate,
    DetectionConfig,
    detect_bos_choch,
    get_candles,
)
from experiments.pivot_detection import PivotDetectionConfig, detect_pivots
from experiments.pivot_detection_v2 import PivotConfigV2, detect_pivots_v2
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
    BOSCHoCHDirectionState,
    BOSCHoCHMarker,
    BOSCHoCHRequest,
    BOSCHoCHResponse,
    CHoCHUpdatePayload,
    DetectionEventPayload,
    DirectionReversalEvent,
    PivotPoint,
    PivotV2EntryPayload,
)


logger = logging.getLogger("boschochserver")
app = FastAPI(title="BOS CHoCH Experiment", version="1.0.0")

DEFAULT_TRUSTED_HOSTS = [
    "188.121.124.28",
    "188.121.124.28:9094",
    "localhost",
    "127.0.0.1",
]

TRUSTED_HOSTS = build_trusted_hosts(DEFAULT_TRUSTED_HOSTS)
ALLOWED_ORIGINS = list_env(
    "WEB_ALLOWED_ORIGINS",
    "http://188.121.124.28:9094,http://localhost:9094,http://127.0.0.1:9094",
)
FORCE_HTTPS = bool_env("WEB_FORCE_HTTPS", False)
BOS_CHOCH_CANDLE_SOURCE = (
    os.getenv("BOS_CHOCH_CANDLE_SOURCE", "binance").strip().lower()
)
BOS_CHOCH_CANDLE_CSV_PATH = os.getenv("BOS_CHOCH_CANDLE_CSV_PATH")

configure_standard_app(
    app,
    trusted_hosts=TRUSTED_HOSTS,
    allowed_origins=ALLOWED_ORIGINS,
    force_https=FORCE_HTTPS,
)


@app.get("/", response_class=FileResponse)
async def bos_choch_page() -> FileResponse:
    return FileResponse(UI_DIR / "bos_choch.html")


@app.post("/api/bos-choch", response_model=BOSCHoCHResponse)
async def compute_bos_choch(payload: BOSCHoCHRequest) -> BOSCHoCHResponse:
    proxies = build_proxy_map(
        http_proxy=payload.http_proxy,
        https_proxy=payload.https_proxy,
        include_standard_env=True,
    )

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    source = (payload.source or BOS_CHOCH_CANDLE_SOURCE or "binance").strip().lower()
    csv_path = payload.csv_path or BOS_CHOCH_CANDLE_CSV_PATH
    symbol = payload.symbol.upper()
    try:
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

    try:
        result = detect_bos_choch(
            candles,
            DetectionConfig(
                direction_window=payload.direction_window,
                hunt_mode=payload.hunt_mode,
                include_hunt_candle_in_choch_range=payload.include_hunt_candle_in_choch_range,
                min_swing_pct=payload.min_swing_pct,
                include_pullback_in_bos_level=payload.include_pullback_in_bos_level,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.pivot_version == "v2":
        pivot_results = []
        pivot_v2_results = detect_pivots_v2(
            candles,
            payload.scan_length,
            PivotConfigV2(min_swing_pct=payload.pivot_min_swing_pct),
        )
    else:
        pdc = PivotDetectionConfig(
            scan_length=payload.scan_length,
            min_swing_pct=payload.pivot_min_swing_pct,
            restart_on_invalidation=payload.restart_on_invalidation,
            use_structural_left_bound=payload.use_structural_left_bound,
            include_reference_candle=payload.include_reference_candle,
        )
        pivot_results = detect_pivots(candles, payload.scan_length, pdc.as_v1_config())
        pivot_v2_results = []

    markers: list[BOSCHoCHMarker] = []
    for bos in result.bos_records:
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

    # Filter choch_updates to first, last, or all per BOS.
    choch_mode = payload.choch_display_mode
    if choch_mode == "first":
        seen_bos: set[int] = set()
        filtered_updates = []
        for update in result.choch_updates:
            if update.bos_index not in seen_bos:
                filtered_updates.append(update)
                seen_bos.add(update.bos_index)
    elif choch_mode == "last":
        last_per_bos: dict[int, CHoCHUpdate] = {}
        for update in result.choch_updates:
            last_per_bos[update.bos_index] = update
        filtered_updates = list(last_per_bos.values())
    else:
        filtered_updates = list(result.choch_updates)

    # Deduplicate: if multiple BOSes produced the same (candle_index, level),
    # keep only the first occurrence so overlapping CHoCHs appear once.
    seen_choch: set[tuple[int, float]] = set()
    deduped_updates = []
    for update in filtered_updates:
        key = (update.candle_index, update.level)
        if key not in seen_choch:
            seen_choch.add(key)
            deduped_updates.append(update)

    bos_by_index = {bos.index: bos for bos in result.bos_records}
    for choch_index, update in enumerate(deduped_updates):
        if not 0 <= update.candle_index < len(candles):
            continue
        bos = bos_by_index.get(update.bos_index)
        if bos is None:
            continue
        candle = candles[update.candle_index]
        line_end_index = min(update.candle_index + 3, len(candles) - 1)
        choch_direction = bos.direction
        markers.append(
            BOSCHoCHMarker(
                type="CHoCH",
                index=choch_index,
                direction=choch_direction,
                candle_index=update.candle_index,
                time=candle.open_time,
                price=update.level,
                high=candle.high,
                low=candle.low,
                label=f"CHoCH {choch_index}",
                line_start_time=candle.open_time,
                line_end_time=candles[line_end_index].open_time,
                bos_index=update.bos_index,
            )
        )

    for ichoch_index, ichoch in enumerate(result.independent_chochs):
        if not 0 <= ichoch.candle_index < len(candles):
            continue
        candle = candles[ichoch.candle_index]
        line_end_index = min(ichoch.candle_index + 3, len(candles) - 1)
        markers.append(
            BOSCHoCHMarker(
                type="ICHOCH",
                index=ichoch_index,
                direction=ichoch.direction,
                candle_index=ichoch.candle_index,
                time=candle.open_time,
                price=ichoch.level,
                high=candle.high,
                low=candle.low,
                label=f"ICHOCH {ichoch_index}",
                line_start_time=candle.open_time,
                line_end_time=candles[line_end_index].open_time,
                bos_index=None,
            )
        )

    markers.sort(key=lambda marker: (marker.candle_index, marker.type, marker.index))
    response_candles = build_candle_payloads(candles)
    n = len(candles)
    pivot_points: list[PivotPoint] = [
        PivotPoint(
            index=p.index,
            type=p.type,  # type: ignore[arg-type]
            high=p.high,
            low=p.low,
            haunted=p.haunted,
            invalidation_level=p.invalidation_level,
            time=candles[p.index].open_time if 0 <= p.index < n else payload.start,
            reference_index=p.reference_index,
            reference_time=candles[p.reference_index].open_time if 0 <= p.reference_index < n else payload.start,
            trigger_index=p.trigger_index,
            trigger_time=candles[p.trigger_index].open_time if 0 <= p.trigger_index < n else payload.start,
            invalidation_index=p.invalidation_index,
            invalidation_time=(
                candles[p.invalidation_index].open_time
                if p.invalidation_index is not None and 0 <= p.invalidation_index < n
                else None
            ),
            next_bullish_index=p.next_bullish_index,
            next_bullish_time=(
                candles[p.next_bullish_index].open_time
                if p.next_bullish_index is not None and 0 <= p.next_bullish_index < n
                else None
            ),
            next_bearish_index=p.next_bearish_index,
            next_bearish_time=(
                candles[p.next_bearish_index].open_time
                if p.next_bearish_index is not None and 0 <= p.next_bearish_index < n
                else None
            ),
            previous_bullish_index=p.previous_bullish_index,
            previous_bullish_time=(
                candles[p.previous_bullish_index].open_time
                if p.previous_bullish_index is not None and 0 <= p.previous_bullish_index < n
                else None
            ),
            previous_bearish_index=p.previous_bearish_index,
            previous_bearish_time=(
                candles[p.previous_bearish_index].open_time
                if p.previous_bearish_index is not None and 0 <= p.previous_bearish_index < n
                else None
            ),
        )
        for p in pivot_results
    ]
    pivot_v2_entry_payloads: list[PivotV2EntryPayload] = [
        PivotV2EntryPayload(
            candle_index=e.candle_index,
            pivot_index=e.pivot_index,
            pivot_type=e.pivot_type,  # type: ignore[arg-type]
            hunt_index=e.hunt_index,
            candle_time=candles[e.candle_index].open_time if 0 <= e.candle_index < n else payload.start,
            pivot_time=(
                candles[e.pivot_index].open_time
                if e.pivot_index is not None and 0 <= e.pivot_index < n
                else None
            ),
            hunt_time=(
                candles[e.hunt_index].open_time
                if e.hunt_index is not None and 0 <= e.hunt_index < n
                else None
            ),
        )
        for e in pivot_v2_results
    ]

    reversal_events = [
        DirectionReversalEvent(
            candle_index=ev.candle_index,
            time=candles[ev.candle_index].open_time,
            direction=ev.direction,
            details=ev.details,
        )
        for ev in result.events
        if ev.event == "DIRECTION_REVERSED" and 0 <= ev.candle_index < len(candles)
    ]

    choch_update_payloads = [
        CHoCHUpdatePayload(
            bos_index=u.bos_index,
            candle_index=u.candle_index,
            time=candles[u.candle_index].open_time,
            level=u.level,
            reason=u.reason,
            direction=bos_by_index[u.bos_index].direction
            if u.bos_index in bos_by_index
            else "UPWARD",
        )
        for u in result.choch_updates
        if 0 <= u.candle_index < len(candles)
    ]

    detection_event_payloads = [
        DetectionEventPayload(
            candle_index=ev.candle_index,
            time=candles[ev.candle_index].open_time,
            event=ev.event,
            direction=ev.direction,
            details=ev.details,
        )
        for ev in result.events
        if 0 <= ev.candle_index < len(candles)
    ]

    return BOSCHoCHResponse(
        candles=response_candles,
        markers=markers,
        direction_state=BOSCHoCHDirectionState(
            direction=result.direction_state.direction,
            since_index=result.direction_state.since_index,
        ),
        pivots=pivot_points,
        pivot_v2_entries=pivot_v2_entry_payloads,
        direction_reversals=reversal_events,
        choch_updates=choch_update_payloads,
        detection_events=detection_event_payloads,
    )
