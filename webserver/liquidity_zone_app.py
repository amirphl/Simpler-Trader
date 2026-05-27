from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]

from candle_downloader.models import to_milliseconds
from experiments.liquidity_zone_detection import (
    LiquidityZoneConfig,
    detect_liquidity_zones,
)
from experiments.pivot_detection import get_candles
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
    LiquidityZoneRequest,
    LiquidityZoneResponse,
)


logger = logging.getLogger("liquidityzoneserver")
app = FastAPI(title="Liquidity Zone Experiment", version="1.0.0")

DEFAULT_TRUSTED_HOSTS = [
    "localhost",
    "127.0.0.1",
]

TRUSTED_HOSTS = build_trusted_hosts(DEFAULT_TRUSTED_HOSTS)
ALLOWED_ORIGINS = list_env(
    "WEB_ALLOWED_ORIGINS",
    "http://localhost:9095,http://127.0.0.1:9095",
)
FORCE_HTTPS = bool_env("WEB_FORCE_HTTPS", False)
LIQUIDITY_ZONE_CANDLE_SOURCE = (
    os.getenv("LIQUIDITY_ZONE_CANDLE_SOURCE", "binance").strip().lower()
)
LIQUIDITY_ZONE_CANDLE_CSV_PATH = os.getenv("LIQUIDITY_ZONE_CANDLE_CSV_PATH")


def _pivot_is_hunted(pivot: object) -> bool:
    if hasattr(pivot, "hunted"):
        return bool(getattr(pivot, "hunted"))
    if hasattr(pivot, "haunted"):
        return bool(getattr(pivot, "haunted"))
    return False


configure_standard_app(
    app,
    trusted_hosts=TRUSTED_HOSTS,
    allowed_origins=ALLOWED_ORIGINS,
    force_https=FORCE_HTTPS,
)


@app.get("/", response_class=FileResponse)
async def liquidity_zone_page() -> FileResponse:
    return FileResponse(UI_DIR / "liquidity_zones.html")


@app.post("/api/liquidity-zones", response_model=LiquidityZoneResponse)
async def compute_liquidity_zones(
    payload: LiquidityZoneRequest,
) -> LiquidityZoneResponse:
    proxies = build_proxy_map(
        http_proxy=payload.http_proxy,
        https_proxy=payload.https_proxy,
        include_standard_env=True,
    )

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    source = (
        (payload.source or LIQUIDITY_ZONE_CANDLE_SOURCE or "binance").strip().lower()
    )
    csv_path = payload.csv_path or LIQUIDITY_ZONE_CANDLE_CSV_PATH
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
    result = detect_liquidity_zones(
        candles,
        LiquidityZoneConfig(
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
            pivot_grouping=payload.pivot_grouping,
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
            zone_hunt_mode=payload.zone_hunt_mode,
            intersection_method=payload.intersection_method,
            slope_attribute=payload.slope_attribute,
        ),
    )

    response_candles = build_candle_payloads(candles)
    pivots = [
        LiquidityZonePivot(
            index=p.index,
            type=p.type,  # type: ignore[arg-type]
            high=p.high,
            low=p.low,
            haunted=_pivot_is_hunted(p),
            time=candles[p.index].open_time
            if 0 <= p.index < len(candles)
            else payload.start,
        )
        for p in result.pivots
    ]
    segments = [
        LiquidityDirectionSegment(
            index=segment.index,
            direction=segment.direction,
            start_index=segment.start_index,
            end_index=segment.end_index,
            start_time=candles[segment.start_index].open_time
            if 0 <= segment.start_index < len(candles)
            else payload.start,
            end_time=candles[segment.end_index].open_time
            if 0 <= segment.end_index < len(candles)
            else payload.end,
            pivot_count=len(segment.pivots),
            representative_pivot_index=(
                segment.representative_pivot.index
                if segment.representative_pivot is not None
                else None
            ),
            representative_pivot_time=(
                candles[segment.representative_pivot.index].open_time
                if segment.representative_pivot is not None
                and 0 <= segment.representative_pivot.index < len(candles)
                else None
            ),
        )
        for segment in result.direction_segments
    ]

    zones: list[LiquidityZonePayload] = []
    for level, grouped in result.zones_by_level.items():
        for direction in ("UPWARD", "DOWNWARD"):
            for zone in grouped[direction]:
                zones.append(
                    LiquidityZonePayload(
                        id=zone.id,
                        direction=zone.direction,
                        level=level,  # type: ignore[arg-type]
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
                )

    zones.sort(key=lambda zone: (zone.start_index, zone.level, zone.id))

    # Build BOS/CHoCH markers from the result already computed inside detect_liquidity_zones
    bc = result.bos_choch_result
    choch_mode = payload.choch_display_mode

    markers: list[BOSCHoCHMarker] = []
    for bos in bc.bos_records:
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

    if choch_mode == "first":
        seen_bos_ids: set[int] = set()
        filtered_updates = []
        for update in bc.choch_updates:
            if update.bos_index not in seen_bos_ids:
                filtered_updates.append(update)
                seen_bos_ids.add(update.bos_index)
    elif choch_mode == "last":
        last_per_bos: dict[int, object] = {}
        for update in bc.choch_updates:
            last_per_bos[update.bos_index] = update
        filtered_updates = list(last_per_bos.values())
    else:
        filtered_updates = list(bc.choch_updates)

    seen_choch: set[tuple[int, float]] = set()
    deduped_updates = []
    for update in filtered_updates:
        key = (update.candle_index, update.level)
        if key not in seen_choch:
            seen_choch.add(key)
            deduped_updates.append(update)

    bos_by_index = {bos.index: bos for bos in bc.bos_records}
    for choch_index, update in enumerate(deduped_updates):
        if not 0 <= update.candle_index < len(candles):
            continue
        bos = bos_by_index.get(update.bos_index)
        if bos is None:
            continue
        candle = candles[update.candle_index]
        line_end_index = min(update.candle_index + 3, len(candles) - 1)
        markers.append(
            BOSCHoCHMarker(
                type="CHoCH",
                index=choch_index,
                direction=bos.direction,
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

    markers.sort(key=lambda m: (m.candle_index, m.type, m.index))

    direction_reversals = [
        DirectionReversalEvent(
            candle_index=ev.candle_index,
            time=candles[ev.candle_index].open_time,
            direction=ev.direction,
            details=ev.details,
        )
        for ev in bc.events
        if ev.event == "DIRECTION_REVERSED" and 0 <= ev.candle_index < len(candles)
    ]

    return LiquidityZoneResponse(
        candles=response_candles,
        pivots=pivots,
        segments=segments,
        zones=zones,
        markers=markers,
        direction_reversals=direction_reversals,
    )
