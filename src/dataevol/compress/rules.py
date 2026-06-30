from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataevol.storage.sqlite import connect
from .interfaces import LocalCompressionModel, key_fact_retention


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


FAILURE_TYPE_BY_LABEL = {
    "failed_tests": "failed_code_tests",
    "failed_verification": "weak_evidence",
    "hallucinated": "hallucinated_citation",
    "unsafe_or_policy_blocked": "unsafe_output",
    "too_expensive": "overused_frontier_model",
    "too_slow": "format_failure",
}


def _first_sentence(text: str | None, limit: int = 240) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    stop = normalized.find(". ")
    candidate = normalized[: stop + 1] if stop > 0 else normalized
    return candidate[:limit].rstrip()


def compress_trace(trace: dict[str, Any], label: str | None, *, model: LocalCompressionModel | None = None) -> dict[str, Any]:
    prompt = _first_sentence(trace.get("prompt"), 180)
    response = _first_sentence(trace.get("response"), 220)
    summary_parts = [part for part in [f"Prompt: {prompt}" if prompt else "", f"Response: {response}" if response else ""] if part]
    summary = model.summarize(trace, label) if model else " ".join(summary_parts)
    summary = summary or _first_sentence(trace.get("normalized_text"), 260) or "Trace contained structured tool/test data."
    if trace.get("task_id") and str(trace["task_id"]).lower() not in summary.lower():
        summary = f"Task {trace['task_id']}: {summary}"
    original_tokens = max(1, len(str(trace.get("normalized_text") or "").split()))
    compressed_tokens = max(1, len(summary.split()))
    failure_type = FAILURE_TYPE_BY_LABEL.get(label)
    why_useful = (
        "Useful as a negative/evaluator example with preserved failure signal."
        if failure_type
        else "Useful as a compact accepted/inconclusive trace summary."
    )
    return {
        "summary": summary,
        "failure_type": failure_type,
        "why_useful": why_useful,
        "corrected_trace_id": trace.get("corrected_trace_id") or _extract_corrected_trace_id(trace),
        "key_fact_retention": key_fact_retention(summary, trace),
        "token_reduction_ratio": max(0.0, 1.0 - (compressed_tokens / original_tokens)),
    }


def _extract_corrected_trace_id(trace: dict[str, Any]) -> int | None:
    raw = trace.get("metadata")
    if isinstance(raw, dict):
        value = raw.get("corrected_trace_id")
        return int(value) if value is not None else None
    return None


def compress_run(db_path: str | Path, run_id: int, *, model: LocalCompressionModel | None = None) -> list[dict[str, Any]]:
    compressed: list[dict[str, Any]] = []
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT t.*, l.label
            FROM traces t
            LEFT JOIN labels l ON l.trace_id = t.id
            WHERE t.run_id = ?
            GROUP BY t.id
            ORDER BY t.id
            """,
            (run_id,),
        ).fetchall()
        for row in rows:
            item = compress_trace(dict(row), row["label"], model=model)
            conn.execute(
                """
                INSERT INTO compressed_traces (
                  trace_id, summary, failure_type, why_useful, corrected_trace_id, token_reduction_ratio, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                  summary=excluded.summary,
                  failure_type=excluded.failure_type,
                  why_useful=excluded.why_useful,
                  corrected_trace_id=excluded.corrected_trace_id,
                  token_reduction_ratio=excluded.token_reduction_ratio,
                  created_at=excluded.created_at
                """,
                (
                    row["id"],
                    item["summary"],
                    item["failure_type"],
                    item["why_useful"],
                    item["corrected_trace_id"],
                    item["token_reduction_ratio"],
                    _now(),
                ),
            )
            compressed.append({"trace_id": row["id"], **item})
    return compressed
