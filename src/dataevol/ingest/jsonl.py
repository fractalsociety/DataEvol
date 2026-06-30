from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from dataevol.dedupe.exact import canonical_hash, normalized_text
from dataevol.dedupe.similarity import near_duplicate_score
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


def _trace_mapping(trace: CanonicalTrace) -> dict[str, Any]:
    return {
        "trace_id": trace.task_id,
        "task_id": trace.task_id,
        "trace_type": trace.trace_type,
        "prompt": trace.prompt,
        "response": trace.response,
        "provider": trace.provider,
        "model": trace.model,
    }


def _find_near_duplicate(conn, trace: CanonicalTrace, *, threshold: float) -> tuple[int, int, float] | None:
    current = _trace_mapping(trace)
    rows = conn.execute(
        """
        SELECT id, duplicate_cluster_id, task_id, trace_type, prompt, response, provider, model
        FROM traces
        ORDER BY id DESC
        LIMIT 500
        """
    ).fetchall()
    best: tuple[int, int, float] | None = None
    for row in rows:
        existing = dict(row)
        if current.get("task_id") and existing.get("task_id") and current["task_id"] != existing["task_id"]:
            continue
        score = near_duplicate_score(current, existing)
        if score >= threshold and (best is None or score > best[2]):
            cluster_id = int(existing.get("duplicate_cluster_id") or _ensure_cluster(conn, f"near_duplicate_{existing['id']}", int(existing["id"])))
            best = (int(existing["id"]), cluster_id, score)
    return best


def _mark_near_duplicate(conn, run_id: int, cluster_id: int, raw_path: Path) -> None:
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


def ingest_jsonl(
    jsonl_path: str | Path,
    db_path: str | Path,
    *,
    source_system: str,
    privacy_mode: str = "private-local-only",
    raw_root: str | Path = ".dataevol/raw",
    external_run_id: str | None = None,
    objective: str | None = None,
    near_duplicate_threshold: float | None = 0.82,
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
                    if near_duplicate_threshold is not None:
                        near = _find_near_duplicate(conn, redacted_trace, threshold=float(near_duplicate_threshold))
                        if near is not None:
                            existing_id, cluster_id, score = near
                            _mark_near_duplicate(conn, run_id, cluster_id, raw_path)
                            report.duplicates += 1
                            report.duplicate_hashes.append(f"near:{existing_id}:{score:.4f}")
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
