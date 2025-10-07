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
from experiments.pivot_detection import detect_pivots, download_candles
from .models import CandleForPivot, PivotPoint, PivotRequest, PivotResponse


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: str) -> list[str]:
    return [
        item.strip() for item in os.getenv(name, default).split(",") if item.strip()
    ]


logger = logging.getLogger("pivotserver")
app = FastAPI(title="Pivot Experiment", version="1.0.0")

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
    "http://localhost:9093,http://127.0.0.1:9093",
)
FORCE_HTTPS = _bool_env("WEB_FORCE_HTTPS", False)
PROXY_FALLBACK = os.getenv("WEB_CANDLE_PROXY")
HTTP_PROXY_FALLBACK = os.getenv("WEB_CANDLE_HTTP_PROXY")
HTTPS_PROXY_FALLBACK = os.getenv("WEB_CANDLE_HTTPS_PROXY")

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
async def pivots_page() -> FileResponse:
    return FileResponse(UI_DIR / "pivots.html")


@app.post("/api/pivots", response_model=PivotResponse)
async def compute_pivots(payload: PivotRequest) -> PivotResponse:
    proxies = {}
    if payload.http_proxy:
        proxies["http"] = payload.http_proxy
    if payload.https_proxy:
        proxies["https"] = payload.https_proxy

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    candles = download_candles(
        symbol=payload.symbol,
        interval=payload.timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        proxies=proxies or None,
        logger=logger.getChild("pivots.binance"),
    )
    pivots = detect_pivots(candles, payload.scan_length)

    pivot_points = [
        PivotPoint(
            index=p.index,
            type=p.type,  # type: ignore[arg-type]
            high=p.high,
            low=p.low,
            haunted=p.haunted,
            time=candles[p.index].close_time
            if 0 <= p.index < len(candles)
            else payload.start,
            reference_index=p.reference_index,
            reference_time=candles[p.reference_index].close_time
            if 0 <= p.reference_index < len(candles)
            else payload.start,
            trigger_index=p.trigger_index,
            trigger_time=candles[p.trigger_index].close_time
            if 0 <= p.trigger_index < len(candles)
            else payload.start,
            invalidation_index=p.invalidation_index,
            invalidation_time=(
                candles[p.invalidation_index].close_time
                if p.invalidation_index is not None
                and 0 <= p.invalidation_index < len(candles)
                else None
            ),
            next_bullish_index=p.next_bullish_index,
            next_bullish_time=(
                candles[p.next_bullish_index].close_time
                if p.next_bullish_index is not None
                and 0 <= p.next_bullish_index < len(candles)
                else None
            ),
            next_bearish_index=p.next_bearish_index,
            next_bearish_time=(
                candles[p.next_bearish_index].close_time
                if p.next_bearish_index is not None
                and 0 <= p.next_bearish_index < len(candles)
                else None
            ),
            previous_bullish_index=p.previous_bullish_index,
            previous_bullish_time=(
                candles[p.previous_bullish_index].close_time
                if p.previous_bullish_index is not None
                and 0 <= p.previous_bullish_index < len(candles)
                else None
            ),
            previous_bearish_index=p.previous_bearish_index,
            previous_bearish_time=(
                candles[p.previous_bearish_index].close_time
                if p.previous_bearish_index is not None
                and 0 <= p.previous_bearish_index < len(candles)
                else None
            ),
        )
        for p in pivots
    ]
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
    return PivotResponse(pivots=pivot_points, candles=candle_payloads)
