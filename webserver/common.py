from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

from fastapi import FastAPI, Request  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # type: ignore[import-not-found]
from fastapi.staticfiles import StaticFiles  # type: ignore[import-not-found]

from .models import CandleForPivot


ROOT_DIR = Path(__file__).resolve().parents[1]
UI_DIR = ROOT_DIR / "web" / "ui"


def bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def list_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def build_trusted_hosts(default_hosts: Iterable[str]) -> list[str]:
    configured = list_env("WEB_TRUSTED_HOSTS", "")
    if not configured:
        return list(default_hosts)

    merged: list[str] = []
    seen: set[str] = set()
    for candidate in [*configured, *default_hosts]:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        merged.append(candidate)
        seen.add(candidate)
    return merged


async def security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Any]]
):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


def configure_standard_app(
    app: FastAPI,
    *,
    trusted_hosts: Sequence[str],
    allowed_origins: Sequence[str],
    force_https: bool,
) -> None:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(trusted_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    if force_https:
        app.add_middleware(HTTPSRedirectMiddleware)
    app.middleware("http")(security_headers)


def resolve_proxy_values(
    *,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
    include_standard_env: bool = False,
) -> tuple[str | None, str | None]:
    fallback = os.getenv("WEB_CANDLE_PROXY")
    resolved_http = http_proxy or os.getenv("WEB_CANDLE_HTTP_PROXY") or fallback
    resolved_https = https_proxy or os.getenv("WEB_CANDLE_HTTPS_PROXY") or fallback

    if include_standard_env:
        resolved_http = resolved_http or os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
        resolved_https = (
            resolved_https or os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        )
    return resolved_http, resolved_https


def build_proxy_map(
    *,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
    include_standard_env: bool = False,
) -> dict[str, str]:
    resolved_http, resolved_https = resolve_proxy_values(
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        include_standard_env=include_standard_env,
    )
    proxies: dict[str, str] = {}
    if resolved_http:
        proxies["http"] = resolved_http
    if resolved_https:
        proxies["https"] = resolved_https
    return proxies


def candle_payload(candle: object) -> CandleForPivot:
    return CandleForPivot(
        open_time=getattr(candle, "open_time"),
        close_time=getattr(candle, "close_time"),
        open=getattr(candle, "open"),
        high=getattr(candle, "high"),
        low=getattr(candle, "low"),
        close=getattr(candle, "close"),
        volume=getattr(candle, "volume"),
    )


def candle_payloads(candles: Iterable[object]) -> list[CandleForPivot]:
    return [candle_payload(candle) for candle in candles]
