from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]

from candle_downloader.models import to_milliseconds
from experiments.pivot_detection import PivotConfig, detect_pivots, get_candles
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
    PivotPoint,
    PivotRequest,
    PivotResponse,
    PivotV2EntryPayload,
    PivotV2Request,
    PivotV2Response,
)


logger = logging.getLogger("pivotserver")
app = FastAPI(title="Pivot Experiment", version="1.0.0")

DEFAULT_TRUSTED_HOSTS = [
    "188.121.124.28",
    "188.121.124.28:9093",
    "localhost",
    "127.0.0.1",
]

TRUSTED_HOSTS = build_trusted_hosts(DEFAULT_TRUSTED_HOSTS)
ALLOWED_ORIGINS = list_env(
    "WEB_ALLOWED_ORIGINS",
    "http://188.121.124.28:9093,http://localhost:9093,http://127.0.0.1:9093",
)
FORCE_HTTPS = bool_env("WEB_FORCE_HTTPS", False)
PIVOT_CANDLE_SOURCE = os.getenv("PIVOT_CANDLE_SOURCE", "binance").strip().lower()
PIVOT_CANDLE_CSV_PATH = os.getenv("PIVOT_CANDLE_CSV_PATH")

configure_standard_app(
    app,
    trusted_hosts=TRUSTED_HOSTS,
    allowed_origins=ALLOWED_ORIGINS,
    force_https=FORCE_HTTPS,
)


@app.get("/", response_class=FileResponse)
async def pivots_page() -> FileResponse:
    return FileResponse(UI_DIR / "pivots.html")


@app.get("/v2", response_class=FileResponse)
async def pivots_v2_page() -> FileResponse:
    return FileResponse(UI_DIR / "pivots_v2.html")


@app.post("/api/pivots", response_model=PivotResponse)
async def compute_pivots(payload: PivotRequest) -> PivotResponse:
    proxies = build_proxy_map(
        http_proxy=payload.http_proxy,
        https_proxy=payload.https_proxy,
        include_standard_env=True,
    )

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    source = (payload.source or PIVOT_CANDLE_SOURCE or "binance").strip().lower()
    csv_path = payload.csv_path or PIVOT_CANDLE_CSV_PATH
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
        pivots = detect_pivots(
            candles,
            payload.scan_length,
            PivotConfig(
                restart_on_invalidation=payload.restart_on_invalidation,
                min_swing_pct=payload.min_swing_pct,
                use_structural_left_bound=payload.use_structural_left_bound,
                include_reference_candle=payload.include_reference_candle,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    n = len(candles)
    pivot_points = [
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
        for p in pivots
    ]
    response_candles = build_candle_payloads(candles)
    return PivotResponse(pivots=pivot_points, candles=response_candles)


@app.post("/api/pivots/v2", response_model=PivotV2Response)
async def compute_pivots_v2(payload: PivotV2Request) -> PivotV2Response:
    proxies = build_proxy_map(
        http_proxy=payload.http_proxy,
        https_proxy=payload.https_proxy,
        include_standard_env=True,
    )

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    source = (payload.source or PIVOT_CANDLE_SOURCE or "binance").strip().lower()
    csv_path = payload.csv_path or PIVOT_CANDLE_CSV_PATH
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
        raise HTTPException(status_code=502, detail=f"Candle source error: {exc}") from exc

    if not candles:
        raise HTTPException(status_code=422, detail="No candles returned for the given parameters")

    try:
        entries = detect_pivots_v2(
            candles,
            payload.scan_length,
            PivotConfigV2(min_swing_pct=payload.min_swing_pct),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    n = len(candles)
    pivot_entries = [
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
        for e in entries
    ]
    response_candles = build_candle_payloads(candles)
    return PivotV2Response(pivot_entries=pivot_entries, candles=response_candles)
