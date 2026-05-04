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
from experiments.liquidity_zone_detection import LiquidityZoneConfig, detect_liquidity_zones
from experiments.pivot_detection import get_candles
from .models import (
    CandleForPivot,
    LiquidityDirectionSegment,
    LiquidityZonePayload,
    LiquidityZonePivot,
    LiquidityZoneRequest,
    LiquidityZoneResponse,
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


logger = logging.getLogger("liquidityzoneserver")
app = FastAPI(title="Liquidity Zone Experiment", version="1.0.0")

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
    "http://localhost:9095,http://127.0.0.1:9095",
)
FORCE_HTTPS = _bool_env("WEB_FORCE_HTTPS", False)
PROXY_FALLBACK = os.getenv("WEB_CANDLE_PROXY")
HTTP_PROXY_FALLBACK = os.getenv("WEB_CANDLE_HTTP_PROXY")
HTTPS_PROXY_FALLBACK = os.getenv("WEB_CANDLE_HTTPS_PROXY")
LIQUIDITY_ZONE_CANDLE_SOURCE = os.getenv("LIQUIDITY_ZONE_CANDLE_SOURCE", "binance").strip().lower()
LIQUIDITY_ZONE_CANDLE_CSV_PATH = os.getenv("LIQUIDITY_ZONE_CANDLE_CSV_PATH")

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
async def liquidity_zone_page() -> FileResponse:
    return FileResponse(UI_DIR / "liquidity_zones.html")


@app.post("/api/liquidity-zones", response_model=LiquidityZoneResponse)
async def compute_liquidity_zones(payload: LiquidityZoneRequest) -> LiquidityZoneResponse:
    proxies = {}
    http_proxy = payload.http_proxy or HTTP_PROXY_FALLBACK or PROXY_FALLBACK
    https_proxy = payload.https_proxy or HTTPS_PROXY_FALLBACK or PROXY_FALLBACK
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

    start_ms = to_milliseconds(payload.start)
    end_ms = to_milliseconds(payload.end)
    source = (payload.source or LIQUIDITY_ZONE_CANDLE_SOURCE or "binance").strip().lower()
    csv_path = payload.csv_path or LIQUIDITY_ZONE_CANDLE_CSV_PATH
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
    result = detect_liquidity_zones(
        candles,
        LiquidityZoneConfig(
            scan_length=payload.scan_length,
            direction_window=payload.direction_window,
            hunt_mode=payload.hunt_mode,
            include_bos_in_choch_range=payload.include_bos_in_choch_range,
            include_hunt_candle_in_choch_range=payload.include_hunt_candle_in_choch_range,
            up_pivot_filter=payload.up_pivot_filter,
            down_pivot_filter=payload.down_pivot_filter,
            include_hunted_pivots=payload.include_hunted_pivots,
            representative_include_hunted=payload.representative_include_hunted,
            maximum_pivot_distance=payload.maximum_pivot_distance,
            minimum_overlap=payload.minimum_overlap,
            minimum_overlap_ratio=payload.minimum_overlap_ratio,
            allow_reuse=payload.allow_reuse,
            relaxed_slope=payload.relaxed_slope,
            slope_epsilon=payload.slope_epsilon,
            epsilon=payload.epsilon,
        ),
    )

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
    pivots = [
        LiquidityZonePivot(
            index=p.index,
            type=p.type,  # type: ignore[arg-type]
            high=p.high,
            low=p.low,
            haunted=p.haunted,
            time=candles[p.index].close_time if 0 <= p.index < len(candles) else payload.start,
        )
        for p in result.pivots
    ]
    segments = [
        LiquidityDirectionSegment(
            index=segment.index,
            direction=segment.direction,
            start_index=segment.start_index,
            end_index=segment.end_index,
            start_time=candles[segment.start_index].close_time if 0 <= segment.start_index < len(candles) else payload.start,
            end_time=candles[segment.end_index].close_time if 0 <= segment.end_index < len(candles) else payload.end,
            pivot_count=len(segment.pivots),
            representative_pivot_index=(
                segment.representative_pivot.index if segment.representative_pivot is not None else None
            ),
            representative_pivot_time=(
                candles[segment.representative_pivot.index].close_time
                if segment.representative_pivot is not None and 0 <= segment.representative_pivot.index < len(candles)
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
                        start_time=candles[zone.start_index].close_time if 0 <= zone.start_index < len(candles) else payload.start,
                        end_time=candles[zone.end_index].close_time if 0 <= zone.end_index < len(candles) else payload.end,
                        price_low=zone.price_range.low,
                        price_high=zone.price_range.high,
                        is_hunted=zone.is_hunted,
                        left_pivot_index=zone.left_pivot.index,
                        right_pivot_index=zone.right_pivot.index,
                        left_pivot_time=candles[zone.left_pivot.index].close_time if 0 <= zone.left_pivot.index < len(candles) else payload.start,
                        right_pivot_time=candles[zone.right_pivot.index].close_time if 0 <= zone.right_pivot.index < len(candles) else payload.end,
                        left_pivot_type=zone.left_pivot.type,  # type: ignore[arg-type]
                        right_pivot_type=zone.right_pivot.type,  # type: ignore[arg-type]
                        metadata=zone.metadata,
                    )
                )

    zones.sort(key=lambda zone: (zone.start_index, zone.level, zone.id))
    return LiquidityZoneResponse(
        candles=candle_payloads,
        pivots=pivots,
        segments=segments,
        zones=zones,
    )
