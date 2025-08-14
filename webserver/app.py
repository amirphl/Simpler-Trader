from __future__ import annotations

import os
from pathlib import Path

import logging
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware  # type: ignore[import-not-found]
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # type: ignore[import-not-found]
from fastapi.responses import FileResponse, HTMLResponse  # type: ignore[import-not-found]
from fastapi.staticfiles import StaticFiles  # type: ignore[import-not-found]

from .manager import BacktestJobManager
from .models import BacktestResultPayload, BacktestSubmission, JobResponse


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


logger = logging.getLogger("webbacktest")
app = FastAPI(title="Backtest Control Panel", version="1.0.0")
# manager = BacktestJobManager(logger=logger.getChild("jobs"))
manager = BacktestJobManager()    

UI_DIR = Path("web/ui")
UI_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

DEFAULT_TRUSTED_HOSTS = [
    "balut.jaazebeh.ir",
    "balut.jaazebeh.ir:9091",
    ".jaazebeh.ir",
    "jaazebeh.ir",
    "localhost",
    "127.0.0.1",
]


def _build_trusted_hosts() -> list[str]:
    configured = _list_env("WEB_TRUSTED_HOSTS", "")
    if not configured:
        return DEFAULT_TRUSTED_HOSTS.copy()

    # Always keep defaults available to avoid accidentally blocking
    # legitimate upstream hosts (e.g., when the env omits the port).
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
    (
        "https://balut.jaazebeh.ir:9091,"
        "http://balut.jaazebeh.ir:9091,"
        "http://localhost:9092,"
        "http://127.0.0.1:9092"
    ),
)
FORCE_HTTPS = _bool_env("WEB_FORCE_HTTPS", True)
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


def _inject_proxy_defaults(submission: BacktestSubmission) -> BacktestSubmission:
    http_proxy = submission.params.http_proxy or HTTP_PROXY_FALLBACK or PROXY_FALLBACK
    https_proxy = submission.params.https_proxy or HTTPS_PROXY_FALLBACK or PROXY_FALLBACK
    if http_proxy == submission.params.http_proxy and https_proxy == submission.params.https_proxy:
        return submission
    params = submission.params.model_copy(update={"http_proxy": http_proxy, "https_proxy": https_proxy})
    return submission.model_copy(update={"params": params})


@app.middleware("http")
async def security_headers(request: Request, call_next: Callable[[Request], Awaitable]):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.post("/api/backtests", response_model=JobResponse)
async def create_backtest(submission: BacktestSubmission) -> JobResponse:
    submission = _inject_proxy_defaults(submission)
    logger.info(
        "Submitting backtest job",
        extra={"strategy": submission.strategy, "symbol": submission.params.symbol, "timeframe": submission.params.timeframe},
    )
    state = await manager.submit_job(submission)
    return state.to_response()


@app.get("/api/backtests/{job_id}", response_model=JobResponse)
async def get_backtest(job_id: str) -> JobResponse:
    state = manager.get_job(job_id)
    if not state:
        logger.warning("Backtest status requested for unknown job", extra={"job_id": job_id})
        raise HTTPException(status_code=404, detail="Backtest not found")
    return state.to_response()


@app.get("/api/backtests/{job_id}/result", response_model=BacktestResultPayload)
async def get_backtest_result(job_id: str) -> BacktestResultPayload:
    state = manager.get_job(job_id)
    if not state:
        logger.warning("Backtest result requested for unknown job", extra={"job_id": job_id})
        raise HTTPException(status_code=404, detail="Backtest not found")
    if not state.result:
        logger.info("Backtest result requested before completion", extra={"job_id": job_id, "status": state.status})
        raise HTTPException(status_code=404, detail="Result not ready")
    return state.to_result_payload()


@app.websocket("/ws/backtests/{job_id}")
async def backtest_updates(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()

    queue = await manager.add_subscriber(job_id)
    if queue is None:
        state = manager.get_job(job_id)
        if state and state.result:
            await websocket.send_json(
                {"event": "result", "job_id": job_id, "status": state.status, "result": state.result}
            )
        else:
            await websocket.send_json({"event": "error", "job_id": job_id, "error": "Unknown job"})
        await websocket.close()
        return

    snapshot = manager.snapshot(job_id)
    if snapshot:
        await websocket.send_json(snapshot)

    state = manager.get_job(job_id)
    if state and state.status in {"completed", "failed"}:
        if state.result:
            await websocket.send_json({"event": "result", "job_id": job_id, "status": state.status, "result": state.result})
        elif state.error:
            await websocket.send_json({"event": "error", "job_id": job_id, "error": state.error, "status": state.status})
        await manager.remove_subscriber(job_id, queue)
        await websocket.close()
        return

    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", extra={"job_id": job_id, "client": websocket.client.host if websocket.client else "unknown"})
    finally:
        await manager.remove_subscriber(job_id, queue)

