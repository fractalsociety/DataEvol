from __future__ import annotations

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

