from __future__ import annotations

from typing import Any, Iterable, Mapping


def _synthetic(kind: str, source: Mapping[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "synthetic": True,
        "measured": False,
        "generation_method": kind,
        "source_trace_id": source.get("id") or source.get("trace_id"),
        "provenance": {
            "source_run_id": source.get("run_id") or source.get("external_run_id"),
            "source_trace_type": source.get("trace_type"),
        },
    }


def generate_corrected_failures(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _synthetic(
            "corrected_failure",
            trace,
            {
                "trace_type": "correction_trace",
                "prompt": trace.get("prompt"),
                "response": f"Corrected response for failure type {trace.get('failure_type') or trace.get('label')}.",
                "label": "good_training_candidate",
            },
        )
        for trace in traces
        if trace.get("failure_type") or str(trace.get("label", "")).startswith("failed")
    ]


def generate_alternate_decompositions(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _synthetic(
            "alternate_task_decomposition",
            trace,
            {
                "trace_type": "planner_trace",
                "task": trace.get("task") or trace.get("objective") or trace.get("prompt"),
                "steps": ["clarify objective", "choose worker", "verify result"],
            },
        )
        for trace in traces
    ]


def generate_better_router_decisions(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _synthetic(
            "better_router_decision",
            trace,
            {
                "trace_type": "router_trace",
                "task": trace.get("task") or trace.get("objective") or trace.get("prompt"),
                "provider": "openrouter",
                "model": "free-or-cheap-verified",
                "label": "good_training_candidate",
            },
        )
        for trace in traces
        if trace.get("failure_type") == "bad_router_assignment" or trace.get("label") in {"too_expensive", "rescued_by_stronger_model"}
    ]


def generate_hard_negatives(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_synthetic("hard_negative", trace, {"trace_type": "critic_trace", "label": "useful_negative_example", "failure_notes": "plausible but wrong variant"}) for trace in traces]


def generate_adversarial_verifier_examples(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_synthetic("adversarial_verifier", trace, {"trace_type": "verification_trace", "expected": "reject unsupported claim"}) for trace in traces]


def generate_prompt_variants(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_synthetic("prompt_variant", trace, {"trace_type": "planner_trace", "prompt": f"Be explicit and verifiable: {trace.get('prompt') or trace.get('task') or ''}"}) for trace in traces]


def generate_synthetic_benchmark_tasks(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_synthetic("synthetic_benchmark_task", trace, {"trace_type": "benchmark_trace", "task": trace.get("task") or trace.get("prompt"), "expected": trace.get("label") or "accepted"}) for trace in traces]


def filter_and_score_synthetic(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        if not row.get("synthetic") or row.get("measured") is True:
            continue
        row["synthetic_quality_score"] = 0.8 if row.get("source_trace_id") else 0.6
        filtered.append(row)
    return filtered


def generate_synthetic_data(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(trace) for trace in traces]
    items = []
    items.extend(generate_corrected_failures(rows))
    items.extend(generate_alternate_decompositions(rows))
    items.extend(generate_better_router_decisions(rows))
    items.extend(generate_hard_negatives(rows))
    items.extend(generate_adversarial_verifier_examples(rows))
    items.extend(generate_prompt_variants(rows))
    items.extend(generate_synthetic_benchmark_tasks(rows))
    return filter_and_score_synthetic(items)
