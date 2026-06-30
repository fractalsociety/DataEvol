from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TRACE_TYPES = {
    "planner_trace",
    "router_trace",
    "worker_trace",
    "coding_trace",
    "scientific_trace",
    "verification_trace",
    "critic_trace",
    "tool_trace",
    "failure_trace",
    "correction_trace",
    "promotion_trace",
    "benchmark_trace",
}

OUTCOME_LABELS = {
    "accepted",
    "rejected",
    "failed_tests",
    "failed_verification",
    "hallucinated",
    "duplicated",
    "too_expensive",
    "too_slow",
    "rescued_by_stronger_model",
    "useful_negative_example",
    "good_training_candidate",
    "unsafe_or_policy_blocked",
    "inconclusive",
}

FAILURE_TYPES = {
    "wrong_reasoning",
    "wrong_tool",
    "bad_router_assignment",
    "hallucinated_citation",
    "fabricated_data",
    "failed_code_tests",
    "bad_scientific_claim",
    "weak_evidence",
    "missed_dependency",
    "duplicated_work",
    "overused_frontier_model",
    "underused_local_model",
    "context_loss",
    "format_failure",
    "unsafe_output",
}

PRIVACY_MODES = {
    "private-local-only",
    "shared-anonymous-learning",
    "public-benchmark-contribution",
}

PRIVACY_STATUS_BY_MODE = {
    "private-local-only": "local_only",
    "shared-anonymous-learning": "anonymous_learning",
    "public-benchmark-contribution": "public_benchmark",
}


class TraceValidationError(ValueError):
    """Raised when an input payload cannot become a canonical trace."""


@dataclass(frozen=True)
class CanonicalTrace:
    trace_type: str
    task_id: str | None = None
    agent_id: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt: str | None = None
    response: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    tests_run: list[dict[str, Any]] = field(default_factory=list)
    objective: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    outcome: str | None = None
    failure_type: str | None = None
    privacy_mode: str = "private-local-only"

    def to_record(self) -> dict[str, Any]:
        return {
            "trace_type": self.trace_type,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "files_changed": self.files_changed,
            "tests_run": self.tests_run,
            "objective": self.objective,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "outcome": self.outcome,
            "failure_type": self.failure_type,
            "privacy_mode": self.privacy_mode,
        }


def _string_or_none(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _first_string(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _string_or_none(payload, key)
        if value:
            return value
    return None


def normalize_task_type(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if any(part in text for part in ("doc", "literature", "readme", "guide")):
        return "documentation"
    if any(part in text for part in ("code", "test", "bug", "refactor")):
        return "coding"
    if any(part in text for part in ("science", "protocol", "bio", "verify")):
        return "scientific"
    if "route" in text:
        return "routing"
    return text or "unknown"


def normalize_outcome_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "success": "accepted",
        "passed": "accepted",
        "pass": "accepted",
        "ok": "accepted",
        "failure": "failed_verification",
        "failed": "failed_verification",
        "test_failed": "failed_tests",
        "tests_failed": "failed_tests",
        "duplicate": "duplicated",
        "expensive": "too_expensive",
        "slow": "too_slow",
        "unsafe": "unsafe_or_policy_blocked",
    }
    normalized = aliases.get(text, text)
    if normalized not in OUTCOME_LABELS:
        raise TraceValidationError(f"outcome must be one of {sorted(OUTCOME_LABELS)}")
    return normalized


def _list_of_dicts(value: Any, key: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TraceValidationError(f"{key} must be a list of objects")
    return value


def _list_of_strings(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TraceValidationError(f"{key} must be a list")
    return [item if isinstance(item, str) else str(item) for item in value]


def normalize_trace(payload: dict[str, Any], default_privacy_mode: str = "private-local-only") -> CanonicalTrace:
    if not isinstance(payload, dict):
        raise TraceValidationError("trace payload must be an object")

    trace_type = _first_string(payload, "trace_type", "type")
    if trace_type not in TRACE_TYPES:
        raise TraceValidationError(f"trace_type must be one of {sorted(TRACE_TYPES)}")

    privacy_mode = _string_or_none(payload, "privacy_mode") or default_privacy_mode
    if privacy_mode not in PRIVACY_MODES:
        raise TraceValidationError(f"privacy_mode must be one of {sorted(PRIVACY_MODES)}")

    outcome = normalize_outcome_label(payload.get("outcome") or payload.get("label"))

    failure_type = _string_or_none(payload, "failure_type")
    if failure_type is not None and failure_type not in FAILURE_TYPES:
        raise TraceValidationError(f"failure_type must be one of {sorted(FAILURE_TYPES)}")

    metadata = payload.get("metadata") or {}
    metrics = payload.get("metrics") or {}
    if not isinstance(metadata, dict):
        raise TraceValidationError("metadata must be an object")
    if not isinstance(metrics, dict):
        raise TraceValidationError("metrics must be an object")

    if "task_type" in payload and "task_type" not in metadata:
        metadata = {**metadata, "task_type": normalize_task_type(payload.get("task_type"))}

    return CanonicalTrace(
        trace_type=trace_type,
        task_id=_string_or_none(payload, "task_id"),
        agent_id=_string_or_none(payload, "agent_id"),
        provider=(_string_or_none(payload, "provider") or "").lower() or None,
        model=_string_or_none(payload, "model"),
        prompt=_first_string(payload, "prompt", "input"),
        response=_first_string(payload, "response", "output"),
        tool_calls=_list_of_dicts(payload.get("tool_calls"), "tool_calls"),
        files_changed=_list_of_strings(payload.get("files_changed"), "files_changed"),
        tests_run=_list_of_dicts(payload.get("tests_run"), "tests_run"),
        objective=_string_or_none(payload, "objective"),
        metadata=metadata,
        metrics=metrics,
        outcome=outcome,
        failure_type=failure_type,
        privacy_mode=privacy_mode,
    )


def validate_trace(payload: dict[str, Any], default_privacy_mode: str = "private-local-only") -> CanonicalTrace:
    trace = normalize_trace(payload, default_privacy_mode=default_privacy_mode)
    if not any([trace.prompt, trace.response, trace.tool_calls, trace.objective]):
        raise TraceValidationError("trace must include prompt, response, tool_calls, or objective")
    return trace
