from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from dataevol.storage.sqlite import connect, init_db


def initialize_training_jobs(db_path: str | Path) -> list[dict[str, Any]]:
    init_db(db_path)
    recovered: list[dict[str, Any]] = []
    now = time.time()
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT job_id, status, job_json FROM local_model_training_jobs ORDER BY created_at"
        ).fetchall()
        for row in rows:
            job = json.loads(row["job_json"])
            if row["status"] in {"queued", "running"}:
                job.update({
                    "status": "failed",
                    "ok": False,
                    "recoverable": True,
                    "completed_at": now,
                    "error": "training job was interrupted by service restart",
                })
                _upsert(conn, job)
            recovered.append(_runtime_job(job, db_path))
    return recovered


def persist_training_job(job: dict[str, Any]) -> None:
    db_path = job.get("_db_path")
    if not db_path:
        return
    init_db(db_path)
    with connect(db_path) as conn:
        _upsert(conn, job)


def jobs_for_db(jobs: Iterable[dict[str, Any]], db_path: str | Path) -> list[dict[str, Any]]:
    target = str(Path(db_path).resolve())
    return [job for job in jobs if str(Path(str(job.get("_db_path") or "")).resolve()) == target]


def _upsert(conn, job: dict[str, Any]) -> None:  # noqa: ANN001
    serialized = _serialized_job(job)
    conn.execute(
        """
        INSERT INTO local_model_training_jobs(
          job_id, job_type, status, recoverable, normalized_payload, job_json,
          created_at, updated_at, completed_at, retry_job_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
          job_type=excluded.job_type,
          status=excluded.status,
          recoverable=excluded.recoverable,
          normalized_payload=excluded.normalized_payload,
          job_json=excluded.job_json,
          updated_at=excluded.updated_at,
          completed_at=excluded.completed_at,
          retry_job_id=excluded.retry_job_id
        """,
        (
            str(job["job_id"]),
            str(job.get("job_type") or "local_model_training"),
            str(job.get("status") or "unknown"),
            int(bool(job.get("recoverable"))),
            json.dumps(job.get("normalized_payload"), sort_keys=True) if job.get("normalized_payload") is not None else None,
            json.dumps(serialized, sort_keys=True),
            float(job.get("created_at") or time.time()),
            time.time(),
            float(job["completed_at"]) if job.get("completed_at") is not None else None,
            job.get("retry_job_id"),
        ),
    )


def _serialized_job(job: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in job.items() if not key.startswith("_") and key != "logs"}
    result["logs"] = list(job.get("logs") or [])
    return result


def _runtime_job(job: dict[str, Any], db_path: str | Path) -> dict[str, Any]:
    result = dict(job)
    result["logs"] = deque(result.get("logs") or [], maxlen=160)
    result["_db_path"] = str(Path(db_path).resolve())
    return result
