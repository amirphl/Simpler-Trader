from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class StoredJob:
    job_id: str
    status: str
    submitted_at: datetime
    completed_at: datetime | None
    error: str | None
    request: Dict[str, Any]
    result: Dict[str, Any] | None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "StoredJob":
        submitted = datetime.fromisoformat(payload["submitted_at"])
        completed_raw = payload.get("completed_at")
        completed = datetime.fromisoformat(completed_raw) if completed_raw else None
        return cls(
            job_id=payload["job_id"],
            status=payload["status"],
            submitted_at=submitted,
            completed_at=completed,
            error=payload.get("error"),
            request=payload.get("request") or {},
            result=payload.get("result"),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": self.submitted_at.astimezone(timezone.utc).isoformat(),
            "completed_at": self.completed_at.astimezone(timezone.utc).isoformat()
            if self.completed_at
            else None,
            "error": self.error,
            "request": self.request,
            "result": self.result,
        }


class ResultStore:
    """File-backed persistence for completed backtest jobs."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self._root / f"{job_id}.json"

    def save(self, job: StoredJob) -> Path:
        payload = job.to_payload()
        path = self.path_for(job.job_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def load(self, job_id: str) -> StoredJob | None:
        path = self.path_for(job_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return StoredJob.from_payload(payload)

