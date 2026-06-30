from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from dataevol.dedupe.exact import canonical_hash, normalized_text
from dataevol.privacy.redaction import privacy_status_for_mode, redact_value
from dataevol.schemas.traces import CanonicalTrace, TraceValidationError, validate_trace
from dataevol.storage.sqlite import connect, init_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IngestReport:
    run_id: int
    accepted: int = 0
    rejected: int = 0
    malformed: int = 0
    duplicates: int = 0
    trace_ids: list[int] = field(default_factory=list)
    duplicate_hashes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "malformed": self.malformed,
            "duplicates": self.duplicates,
            "trace_ids": self.trace_ids,
            "duplicate_hashes": self.duplicate_hashes,
            "errors": self.errors,
        }


def _write_raw(raw_root: Path, run_id: int, line_number: int, payload: dict[str, Any]) -> Path:
    run_dir = raw_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / f"{line_number:06d}_{uuid4().hex}.json"
    raw_path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return raw_path


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _insert_run(conn, source_system: str, privacy_mode: str, external_run_id: str | None, objective: str | None) -> int:
    cursor = conn.execute(
        """
        INSERT INTO runs (external_run_id, source_system, objective, status, privacy_mode, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (external_run_id, source_system, objective, "ingested", privacy_mode, _now()),
    )
    return int(cursor.lastrowid)


def _insert_trace(
    conn,
    run_id: int,
    trace: CanonicalTrace,
    raw_path: Path,
    content_hash: str,
    cluster_id: int | None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO traces (
          run_id, trace_type, task_id, agent_id, provider, model, prompt, response,
          tool_calls, files_changed, tests_run, raw_path, normalized_text, content_hash,
          duplicate_cluster_id, privacy_status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            trace.trace_type,
            trace.task_id,
            trace.agent_id,
            trace.provider,
            trace.model,
            trace.prompt,
            trace.response,
            _json_dumps(trace.tool_calls),
            _json_dumps(trace.files_changed),
            _json_dumps(trace.tests_run),
            str(raw_path),
            normalized_text(trace),
            content_hash,
            cluster_id,
            privacy_status_for_mode(trace.privacy_mode),
            _now(),
        ),
    )
    return int(cursor.lastrowid)


def _ensure_cluster(conn, content_hash: str, canonical_trace_id: int | None = None) -> int:
    row = conn.execute("SELECT id FROM duplicate_clusters WHERE content_hash = ?", (content_hash,)).fetchone()
    if row:
        return int(row["id"])
    cursor = conn.execute(
        """
        INSERT INTO duplicate_clusters (
          content_hash, canonical_trace_id, duplicate_count, first_seen_at, last_seen_at
        )
        VALUES (?, ?, 0, ?, ?)
        """,
        (content_hash, canonical_trace_id, _now(), _now()),
    )
    return int(cursor.lastrowid)


def _mark_duplicate(conn, run_id: int, content_hash: str, raw_path: Path) -> None:
    cluster_id = _ensure_cluster(conn, content_hash)
    conn.execute(
        """
        UPDATE duplicate_clusters
        SET duplicate_count = duplicate_count + 1, last_seen_at = ?
        WHERE id = ?
        """,
        (_now(), cluster_id),
    )
    conn.execute(
        "INSERT INTO duplicate_events (cluster_id, run_id, raw_path, seen_at) VALUES (?, ?, ?, ?)",
        (cluster_id, run_id, str(raw_path), _now()),
    )


def _set_cluster_canonical(conn, content_hash: str, trace_id: int) -> int:
    cluster_id = _ensure_cluster(conn, content_hash, canonical_trace_id=trace_id)
    conn.execute(
        "UPDATE duplicate_clusters SET canonical_trace_id = COALESCE(canonical_trace_id, ?) WHERE id = ?",
        (trace_id, cluster_id),
    )
    conn.execute("UPDATE traces SET duplicate_cluster_id = ? WHERE id = ?", (cluster_id, trace_id))
    return cluster_id


def ingest_jsonl(
    jsonl_path: str | Path,
    db_path: str | Path,
    *,
    source_system: str,
    privacy_mode: str = "private-local-only",
    raw_root: str | Path = ".dataevol/raw",
    external_run_id: str | None = None,
    objective: str | None = None,
) -> IngestReport:
    init_db(db_path)
    jsonl_path = Path(jsonl_path)
    raw_root = Path(raw_root)

    with connect(db_path) as conn:
        run_id = _insert_run(conn, source_system, privacy_mode, external_run_id, objective)
        report = IngestReport(run_id=run_id)

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    trace = validate_trace(payload, default_privacy_mode=privacy_mode)
                    redacted_payload = redact_value(trace.to_record())
                    redacted_trace = validate_trace(redacted_payload, default_privacy_mode=privacy_mode)
                    raw_path = _write_raw(raw_root, run_id, line_number, redacted_payload)
                    content_hash = canonical_hash(redacted_trace)
                    existing = conn.execute(
                        "SELECT id FROM traces WHERE content_hash = ?", (content_hash,)
                    ).fetchone()
                    if existing:
                        _mark_duplicate(conn, run_id, content_hash, raw_path)
                        report.duplicates += 1
                        report.duplicate_hashes.append(content_hash)
                        continue
                    trace_id = _insert_trace(conn, run_id, redacted_trace, raw_path, content_hash, None)
                    _set_cluster_canonical(conn, content_hash, trace_id)
                    report.accepted += 1
                    report.trace_ids.append(trace_id)
                except json.JSONDecodeError as exc:
                    report.malformed += 1
                    report.errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                except TraceValidationError as exc:
                    report.rejected += 1
                    report.errors.append(f"line {line_number}: {exc}")

        return report
