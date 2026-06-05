from __future__ import annotations

import asyncio

import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # type: ignore[import-not-found]
from fastapi.responses import FileResponse  # type: ignore[import-not-found]

from .common import (
    UI_DIR,
    bool_env,
    build_trusted_hosts,
    configure_standard_app,
    list_env,
    resolve_proxy_values,
)
from .manager import BacktestJobManager
from .models import BacktestResultPayload, BacktestSubmission, JobResponse


logger = logging.getLogger("webbacktest")
app = FastAPI(title="Backtest Control Panel", version="1.0.0")
manager = BacktestJobManager(logger=logger.getChild("jobs"))

DEFAULT_TRUSTED_HOSTS = [
    "balut.jaazebeh.ir",
    "balut.jaazebeh.ir:9091",
    ".jaazebeh.ir",
    "jaazebeh.ir",
    "localhost",
    "127.0.0.1",
]

TRUSTED_HOSTS = build_trusted_hosts(DEFAULT_TRUSTED_HOSTS)
ALLOWED_ORIGINS = list_env(
    "WEB_ALLOWED_ORIGINS",
    (
        "https://balut.jaazebeh.ir:9091,"
        "http://balut.jaazebeh.ir:9091,"
        "http://localhost:9092,"
        "http://127.0.0.1:9092"
    ),
)
# Default to plain HTTP for local development. Production deployments can
# explicitly enable redirects with WEB_FORCE_HTTPS=true.
FORCE_HTTPS = bool_env("WEB_FORCE_HTTPS", False)
configure_standard_app(
    app,
    trusted_hosts=TRUSTED_HOSTS,
    allowed_origins=ALLOWED_ORIGINS,
    force_https=FORCE_HTTPS,
)


def _inject_proxy_defaults(submission: BacktestSubmission) -> BacktestSubmission:
    http_proxy, https_proxy = resolve_proxy_values(
        http_proxy=submission.params.http_proxy,
        https_proxy=submission.params.https_proxy,
    )
    if (
        http_proxy == submission.params.http_proxy
        and https_proxy == submission.params.https_proxy
    ):
        return submission
    params = submission.params.model_copy(
        update={"http_proxy": http_proxy, "https_proxy": https_proxy}
    )
    return submission.model_copy(update={"params": params})


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.post("/api/backtests", response_model=JobResponse)
async def create_backtest(submission: BacktestSubmission) -> JobResponse:
    submission = _inject_proxy_defaults(submission)
    params = submission.params
    symbol = getattr(params, "symbol", None)
    if symbol is None and hasattr(params, "symbols"):
        symbols = getattr(params, "symbols")
        if symbols and isinstance(symbols, (list, tuple)):
            symbol = symbols[0]
    timeframe = getattr(params, "timeframe", None) or getattr(
        params, "base_timeframe", None
    )
    logger.info(
        "Submitting backtest job",
        extra={
            "strategy": submission.strategy,
            "symbol": symbol,
            "timeframe": timeframe,
        },
    )
    state = await manager.submit_job(submission)
    return state.to_response()


@app.get("/api/backtests/{job_id}", response_model=JobResponse)
async def get_backtest(job_id: str) -> JobResponse:
    state = manager.get_job(job_id)
    if not state:
        logger.warning(
            "Backtest status requested for unknown job", extra={"job_id": job_id}
        )
        raise HTTPException(status_code=404, detail="Backtest not found")
    return state.to_response()


@app.get("/api/backtests/{job_id}/result", response_model=BacktestResultPayload)
async def get_backtest_result(job_id: str) -> BacktestResultPayload:
    state = manager.get_job(job_id)
    if not state:
        logger.warning(
            "Backtest result requested for unknown job", extra={"job_id": job_id}
        )
        raise HTTPException(status_code=404, detail="Backtest not found")
    if not state.result:
        logger.info(
            "Backtest result requested before completion",
            extra={"job_id": job_id, "status": state.status},
        )
        raise HTTPException(status_code=404, detail="Result not ready")
    return state.to_result_payload()


@app.on_event("shutdown")
async def shutdown_backtest_manager() -> None:
    await manager.shutdown()


@app.websocket("/ws/backtests/{job_id}")
async def backtest_updates(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()

    queue = await manager.add_subscriber(job_id)
    if queue is None:
        state = manager.get_job(job_id)
        if state and state.result:
            await websocket.send_json(
                {
                    "event": "result",
                    "job_id": job_id,
                    "status": state.status,
                    "result": state.result,
                }
            )
        else:
            await websocket.send_json(
                {"event": "error", "job_id": job_id, "error": "Unknown job"}
            )
        await websocket.close()
        return

    snapshot = manager.snapshot(job_id)
    if snapshot:
        await websocket.send_json(snapshot)

    state = manager.get_job(job_id)
    if state and state.status in {"completed", "failed"}:
        if state.result:
            await websocket.send_json(
                {
                    "event": "result",
                    "job_id": job_id,
                    "status": state.status,
                    "result": state.result,
                }
            )
        elif state.error:
            await websocket.send_json(
                {
                    "event": "error",
                    "job_id": job_id,
                    "error": state.error,
                    "status": state.status,
                }
            )
        await manager.remove_subscriber(job_id, queue)
        await websocket.close()
        return

    try:
        while True:
            message = await queue.get()
            if message.get("event") == "shutdown":
                break
            await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected",
            extra={
                "job_id": job_id,
                "client": websocket.client.host if websocket.client else "unknown",
            },
        )
    except asyncio.CancelledError:
        logger.info("WebSocket update task cancelled", extra={"job_id": job_id})
    finally:
        await manager.remove_subscriber(job_id, queue)
