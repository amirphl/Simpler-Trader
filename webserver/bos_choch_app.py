from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]
from fastapi.staticfiles import StaticFiles  # type: ignore[import-not-found]

from candle_downloader.models import to_milliseconds
from experiments.bos_choch_detection import (
    CHoCHUpdate,
    DetectionConfig,
    detect_bos_choch,
    get_candles,
)
from experiments.pivot_detection import PivotConfig, detect_pivots
from .models import (
    BOSCHoCHDirectionState,
    BOSCHoCHMarker,
    BOSCHoCHRequest,
    BOSCHoCHResponse,
    CandleForPivot,
    CHoCHUpdatePayload,
    DetectionEventPayload,
    DirectionReversalEvent,
    PivotPoint,
)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: str) -> list[str]:
    return [
        item.strip() for item in os.getenv(name, default).split(",") if item.strip()
    ]


logger = logging.getLogger("boschochserver")
app = FastAPI(title="BOS CHoCH Experiment", version="1.0.0")

ROOT_DIR = Path(__file__).resolve().parents[1]
UI_DIR = ROOT_DIR / "web" / "ui"
UI_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

DEFAULT_TRUSTED_HOSTS = [
    "localhost",
    "127.0.0.1",
]


def _build_trusted_hosts() -> list[str]:
    configured = _list_env("WEB_TRUSTED_HOSTS", "")
    if not configured:
        return DEFAULT_TRUSTED_HOSTS.copy()

    merged: list[str] = []
    seen: set[str] = set()
    for candidate in configured + DEFAULT_TRUSTED_HOSTS:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        merged.append(candidate)
        seen.add(candidate)
    return merged


TRUSTED_HOSTS = _build_trusted_hosts()
ALLOWED_ORIGINS = _list_env(
    "WEB_ALLOWED_ORIGINS",
    "http://localhost:9094,http://127.0.0.1:9094",
)
FORCE_HTTPS = _bool_env("WEB_FORCE_HTTPS", False)
PROXY_FALLBACK = os.getenv("WEB_CANDLE_PROXY")
HTTP_PROXY_FALLBACK = os.getenv("WEB_CANDLE_HTTP_PROXY")
HTTPS_PROXY_FALLBACK = os.getenv("WEB_CANDLE_HTTPS_PROXY")
STD_HTTP_PROXY = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
STD_HTTPS_PROXY = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
BOS_CHOCH_CANDLE_SOURCE = (
    os.getenv("BOS_CHOCH_CANDLE_SOURCE", "binance").strip().lower()
)
BOS_CHOCH_CANDLE_CSV_PATH = os.getenv("BOS_CHOCH_CANDLE_CSV_PATH")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

if FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)


@app.middleware("http")
async def security_headers(request: Request, call_next: Callable[[Request], Awaitable]):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/", response_class=FileResponse)
async def bos_choch_page() -> FileResponse:
    return FileResponse(UI_DIR / "bos_choch.html")


@app.post("/api/bos-choch", response_model=BOSCHoCHResponse)
async def compute_bos_choch(payload: BOSCHoCHRequest) -> BOSCHoCHResponse:
    proxies = {}
    http_proxy = payload.http_proxy or HTTP_PROXY_FALLBACK or PROXY_FALLBACK or STD_HTTP_PROXY
    https_proxy = payload.https_proxy or HTTPS_PROXY_FALLBACK or PROXY_FALLBACK or STD_HTTPS_PROXY
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

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
                include_bos_in_choch_range=payload.include_bos_in_choch_range,
                include_hunt_candle_in_choch_range=payload.include_hunt_candle_in_choch_range,
                min_swing_pct=payload.min_swing_pct,
                include_pullback_in_bos_level=payload.include_pullback_in_bos_level,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        pivots = detect_pivots(
            candles,
            payload.scan_length,
            PivotConfig(
                restart_on_invalidation=payload.restart_on_invalidation,
                min_swing_pct=payload.pivot_min_swing_pct,
                use_structural_left_bound=payload.use_structural_left_bound,
                include_reference_candle=payload.include_reference_candle,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        line_end_index = min(update.candle_index + 10, len(candles) - 1)
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

    markers.sort(key=lambda marker: (marker.candle_index, marker.type, marker.index))
    candle_payloads = [
        CandleForPivot(
            open_time=c.open_time,
            close_time=c.close_time,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
        )
        for c in candles
    ]
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
        candles=candle_payloads,
        markers=markers,
        direction_state=BOSCHoCHDirectionState(
            direction=result.direction_state.direction,
            since_index=result.direction_state.since_index,
        ),
        pivots=pivot_points,
        direction_reversals=reversal_events,
        choch_updates=choch_update_payloads,
        detection_events=detection_event_payloads,
    )
