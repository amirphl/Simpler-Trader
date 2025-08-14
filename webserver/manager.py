from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, MutableMapping, Set
from uuid import uuid4

from .models import BacktestResultPayload, BacktestSubmission, JobResponse
from .runner import run_backtest_job
from .storage import ResultStore, StoredJob


class JobState:
    __slots__ = (
        "job_id",
        "status",
        "submitted_at",
        "completed_at",
        "error",
        "request",
        "result",
        "result_path",
    )

    def __init__(
        self,
        job_id: str,
        *,
        status: str = "queued",
        submitted_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
        request: Dict[str, Any] | None = None,
        result: Dict[str, Any] | None = None,
        result_path: Path | None = None,
    ) -> None:
        self.job_id = job_id
        self.status = status
        self.submitted_at = submitted_at or datetime.now(timezone.utc)
        self.completed_at = completed_at
        self.error = error
        self.request = dict(request) if request else {}
        self.result = result
        self.result_path = result_path

    def to_response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            submitted_at=self.submitted_at,
            completed_at=self.completed_at,
            error=self.error,
        )

    def to_result_payload(self) -> BacktestResultPayload:
        return BacktestResultPayload(
            job_id=self.job_id,
            status=self.status,
            submitted_at=self.submitted_at,
            completed_at=self.completed_at,
            error=self.error,
            result=self.result,
            request=self.request,
        )


class BacktestJobManager:
    """Coordinates background execution of web-submitted backtests."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._jobs: MutableMapping[str, JobState] = {}
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._results = ResultStore(Path("results/web_backtests"))
        self._cache_dir = Path("data/web_backtests")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Jobs persisted on disk are lazily loaded on demand.
        self._log = logger or logging.getLogger("webbacktest.jobs")

    async def submit_job(self, submission: BacktestSubmission) -> JobState:
        job_id = uuid4().hex
        state = JobState(job_id=job_id, request=submission.model_dump(mode="json"))
        self._jobs[job_id] = state
        self._log.info(
            "Job queued",
            extra={
                "job_id": job_id,
                "strategy": submission.strategy,
                "symbol": submission.params.symbol,
                "timeframe": submission.params.timeframe,
            },
        )
        await self._broadcast(job_id, {"event": "status", "job_id": job_id, "status": state.status})
        asyncio.create_task(self._run_job(job_id, submission))
        return state

    def get_job(self, job_id: str) -> JobState | None:
        state = self._jobs.get(job_id)
        if state:
            return state
        stored = self._results.load(job_id)
        if stored:
            state = JobState(
                job_id=stored.job_id,
                status=stored.status,
                submitted_at=stored.submitted_at,
                completed_at=stored.completed_at,
                error=stored.error,
                request=stored.request,
                result=stored.result,
                result_path=self._results.path_for(job_id),
            )
            self._jobs[job_id] = state
            return state
        return None

    async def add_subscriber(self, job_id: str) -> asyncio.Queue | None:
        if job_id not in self._jobs:
            state = self.get_job(job_id)
            if not state:
                return None
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].add(queue)
        return queue

    async def remove_subscriber(self, job_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(job_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers and job_id in self._subscribers:
            self._subscribers.pop(job_id, None)

    def snapshot(self, job_id: str) -> Dict[str, Any] | None:
        state = self._jobs.get(job_id)
        if not state:
            return None
        payload = {
            "event": "status",
            "job_id": job_id,
            "status": state.status,
            "submitted_at": state.submitted_at.isoformat(),
        }
        if state.completed_at:
            payload["completed_at"] = state.completed_at.isoformat()
        if state.error:
            payload["error"] = state.error
        return payload

    async def _run_job(self, job_id: str, submission: BacktestSubmission) -> None:
        state = self._jobs[job_id]
        state.status = "running"
        self._log.info("Job started", extra={"job_id": job_id})
        await self._broadcast(job_id, {"event": "status", "job_id": job_id, "status": "running"})

        store_path = self._cache_dir / "shared_web_candles.db"
        try:
            result = await asyncio.to_thread(
                run_backtest_job,
                job_id,
                submission,
                cache_dir=self._cache_dir,
                store_path=store_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            state.status = "failed"
            state.error = str(exc)
            state.completed_at = datetime.now(timezone.utc)
            stored = StoredJob(
                job_id=job_id,
                status=state.status,
                submitted_at=state.submitted_at,
                completed_at=state.completed_at,
                error=state.error,
                request=state.request,
                result=None,
            )
            state.result_path = self._results.save(stored)
            self._log.exception("Job failed", extra={"job_id": job_id})
            await self._broadcast(
                job_id,
                {
                    "event": "error",
                    "job_id": job_id,
                    "status": state.status,
                    "error": state.error,
                },
            )
            return

        state.status = "completed"
        state.completed_at = datetime.now(timezone.utc)
        state.result = result

        stored = StoredJob(
            job_id=job_id,
            status=state.status,
            submitted_at=state.submitted_at,
            completed_at=state.completed_at,
            error=None,
            request=state.request,
            result=result,
        )
        state.result_path = self._results.save(stored)

        self._log.info("Job completed", extra={"job_id": job_id})
        await self._broadcast(
            job_id,
            {
                "event": "result",
                "job_id": job_id,
                "status": state.status,
                "result": result,
            },
        )

    async def _broadcast(self, job_id: str, payload: Dict[str, Any]) -> None:
        subscribers = self._subscribers.get(job_id)
        if not subscribers:
            return
        for queue in list(subscribers):
            await queue.put(payload)

