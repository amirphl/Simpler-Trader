from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, Request  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]
from fastapi.staticfiles import StaticFiles  # type: ignore[import-not-found]

from candle_downloader.models import to_milliseconds
from experiments.bos_choch_detection import DetectionConfig, detect_bos_choch, get_candles
from .models import (
    BOSCHoCHDirectionState,
    BOSCHoCHMarker,
    BOSCHoCHRequest,
    BOSCHoCHResponse,
    CandleForPivot,
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
BOS_CHOCH_CANDLE_SOURCE = os.getenv("BOS_CHOCH_CANDLE_SOURCE", "binance").strip().lower()
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
    http_proxy = payload.http_proxy or HTTP_PROXY_FALLBACK or PROXY_FALLBACK
    https_proxy = payload.https_proxy or HTTPS_PROXY_FALLBACK or PROXY_FALLBACK
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    source = (payload.source or BOS_CHOCH_CANDLE_SOURCE or "binance").strip().lower()
    csv_path = payload.csv_path or BOS_CHOCH_CANDLE_CSV_PATH
    candles = get_candles(
        source=source,
        symbol=payload.symbol,
        interval=payload.timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        csv_path=csv_path,
        proxies=proxies or None,
        logger=logger.getChild("binance"),
    )
    result = detect_bos_choch(
        candles,
        DetectionConfig(
            direction_window=payload.direction_window,
            hunt_mode=payload.hunt_mode,
            include_bos_in_choch_range=payload.include_bos_in_choch_range,
            include_hunt_candle_in_choch_range=payload.include_hunt_candle_in_choch_range,
        ),
    )

    markers: list[BOSCHoCHMarker] = []
    for bos in result.bos_records:
        if not 0 <= bos.hunt_index < len(candles):
            continue
        candle = candles[bos.hunt_index]
        markers.append(
            BOSCHoCHMarker(
                type="BOS",
                index=bos.index,
                direction=bos.direction,
                candle_index=bos.hunt_index,
                time=candle.close_time,
                price=bos.level,
                high=candle.high,
                low=candle.low,
                label=f"BOS {bos.index}",
            )
        )

    bos_by_index = {bos.index: bos for bos in result.bos_records}
    for choch_index, update in enumerate(result.choch_updates):
        if not 0 <= update.candle_index < len(candles):
            continue
        bos = bos_by_index.get(update.bos_index)
        if bos is None:
            continue
        candle = candles[update.candle_index]
        markers.append(
            BOSCHoCHMarker(
                type="CHoCH",
                index=choch_index,
                direction=bos.direction,
                candle_index=update.candle_index,
                time=candle.close_time,
                price=update.level,
                high=candle.high,
                low=candle.low,
                label=f"CHoCH {choch_index}",
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

    return BOSCHoCHResponse(
        candles=candle_payloads,
        markers=markers,
        direction_state=BOSCHoCHDirectionState(
            direction=result.direction_state.direction,
            since_index=result.direction_state.since_index,
        ),
    )
