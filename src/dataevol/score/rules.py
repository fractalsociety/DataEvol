from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from dataevol.storage.sqlite import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_trace(trace: dict[str, Any], label: str | None) -> dict[str, float]:
    metrics = {}
    raw_path = trace.get("raw_path")
    if raw_path:
        try:
            metrics = json.loads(Path(raw_path).read_text(encoding="utf-8")).get("metrics", {})
        except (OSError, json.JSONDecodeError):
            metrics = {}

    tests = _load_json(trace.get("tests_run"), [])
    failed = any(test.get("status") in {"failed", "fail", "error"} for test in tests)
    passed = any(test.get("status") in {"passed", "pass", "ok"} for test in tests)

    correctness = 0.5
    if label in {"accepted", "good_training_candidate"} or passed:
        correctness = 1.0
    if label in {"failed_tests", "failed_verification", "hallucinated", "unsafe_or_policy_blocked"} or failed:
        correctness = 0.0

    latency_ms = float(metrics.get("latency_ms") or 0)
    cost_usd = float(metrics.get("cost_usd") or 0)
    safety = 0.0 if label == "unsafe_or_policy_blocked" else 1.0
    cost_score = _clamp(1.0 - min(cost_usd, 1.0))
    latency_score = _clamp(1.0 - min(latency_ms / 60000.0, 1.0)) if latency_ms else 0.8
    novelty = 0.35 if label == "duplicated" else 0.75
    rescue = 1.0 if label == "rescued_by_stronger_model" or "rescued" in str(trace.get("normalized_text", "")).lower() else 0.0
    quality = _clamp((correctness * 0.6) + (safety * 0.2) + (latency_score * 0.1) + (cost_score * 0.1))
    training_value = _clamp((quality * 0.65) + (0.2 if label == "useful_negative_example" else 0.0) + (novelty * 0.1) + (rescue * 0.05))

    return {
        "quality_score": quality,
        "correctness_score": correctness,
        "cost_score": cost_score,
        "latency_score": latency_score,
        "novelty_score": novelty,
        "escalation_rescue_score": rescue,
        "safety_score": safety,
        "training_value_score": training_value,
    }


def score_run(db_path: str | Path, run_id: int) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
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
            score = score_trace(dict(row), row["label"])
            conn.execute(
                """
                INSERT INTO scores (
                  trace_id, quality_score, correctness_score, cost_score, latency_score,
                  novelty_score, escalation_rescue_score, safety_score, training_value_score, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                  quality_score=excluded.quality_score,
                  correctness_score=excluded.correctness_score,
                  cost_score=excluded.cost_score,
                  latency_score=excluded.latency_score,
                  novelty_score=excluded.novelty_score,
                  escalation_rescue_score=excluded.escalation_rescue_score,
                  safety_score=excluded.safety_score,
                  training_value_score=excluded.training_value_score,
                  created_at=excluded.created_at
                """,
                (
                    row["id"],
                    score["quality_score"],
                    score["correctness_score"],
                    score["cost_score"],
                    score["latency_score"],
                    score["novelty_score"],
                    score["escalation_rescue_score"],
                    score["safety_score"],
                    score["training_value_score"],
                    _now(),
                ),
            )
            scores.append({"trace_id": row["id"], **score})
    return scores
