from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from dataevol.storage.sqlite import connect
from .interfaces import LocalModelLabeler, load_human_overrides


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def label_trace(trace: dict[str, Any]) -> tuple[str, float, str]:
    text = " ".join(
        str(trace.get(key) or "")
        for key in ("trace_type", "prompt", "response", "normalized_text")
    ).lower()
    tests = _load_json(trace.get("tests_run"), [])

    if "unsafe" in text or "policy blocked" in text:
        return "unsafe_or_policy_blocked", 0.9, "matched safety or policy language"
    if any(test.get("status") in {"failed", "fail", "error"} for test in tests):
        return "failed_tests", 0.95, "one or more tests failed"
    if "hallucinat" in text or "fabricated" in text:
        return "hallucinated", 0.85, "matched hallucination/fabrication language"
    if "verification failed" in text or "failed verification" in text:
        return "failed_verification", 0.9, "matched verification failure language"
    if "too expensive" in text or "over budget" in text:
        return "too_expensive", 0.8, "matched cost failure language"
    if "too slow" in text or "timeout" in text:
        return "too_slow", 0.8, "matched latency failure language"
    if trace.get("trace_type") in {"failure_trace", "correction_trace"}:
        return "useful_negative_example", 0.75, "failure/correction traces are useful negatives by default"
    if any(test.get("status") in {"passed", "pass", "ok"} for test in tests):
        return "accepted", 0.9, "tests passed"
    if "accepted" in text or "verified" in text:
        return "accepted", 0.8, "matched accepted/verified language"
    return "inconclusive", 0.5, "no strong rule matched"


def label_run(
    db_path: str | Path,
    run_id: int,
    *,
    source: str = "rule_based",
    local_model: LocalModelLabeler | None = None,
    override_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    overrides = load_human_overrides(override_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM traces WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        for row in rows:
            trace = dict(row)
            override = overrides.get(str(row["id"])) or overrides.get(str(row["task_id"]))
            if override:
                label, confidence, notes, label_source = override, 1.0, "human review override", "human_review"
            else:
                label, confidence, notes = label_trace(trace)
                label_source = source
                if label == "inconclusive" and local_model is not None:
                    label, confidence, notes = local_model.label(trace)
                    label_source = "local_model"
            conn.execute(
                """
                INSERT INTO labels (trace_id, label, confidence, source, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row["id"], label, confidence, label_source, notes, _now()),
            )
            labels.append(
                {
                    "trace_id": row["id"],
                    "label": label,
                    "confidence": confidence,
                    "source": label_source,
                    "notes": notes,
                }
            )
    return labels
