from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .redaction import can_export_publicly, redact_value


def build_training_candidate(trace: dict[str, Any], label: str, score: float) -> dict[str, Any]:
    return redact_value(
        {
            "id": f"candidate_{trace['id']}",
            "source_run_id": trace.get("run_id"),
            "task": trace.get("task_id") or trace.get("trace_type"),
            "input": trace.get("prompt"),
            "output": trace.get("response"),
            "label": label,
            "score": score,
            "use_for": [trace.get("trace_type", "trace").replace("_trace", "")],
            "provider": trace.get("provider"),
            "model": trace.get("model"),
            "privacy_status": trace.get("privacy_status"),
            "why_good": "Rule-scored training candidate.",
            "failure_notes": None if label == "accepted" else label,
        }
    )


def assert_public_export_allowed(candidate: dict[str, Any]) -> None:
    if not can_export_publicly(str(candidate.get("privacy_status"))):
        raise PermissionError("private or anonymous traces cannot be exported as public benchmarks")


def export_training_candidates(
    traces: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    public: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("public_training_candidates.jsonl" if public else "training_candidates.jsonl")
    candidates: list[dict[str, Any]] = []
    for trace in traces:
        label = str(trace.get("label") or trace.get("outcome") or "inconclusive")
        score = float(trace.get("training_value_score") or trace.get("quality_score") or trace.get("score") or 0.0)
        candidate = build_training_candidate(trace, label, score)
        if public:
            assert_public_export_allowed(candidate)
        candidates.append(candidate)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "candidate_count": len(candidates),
        "public": public,
        "candidates": candidates,
    }
