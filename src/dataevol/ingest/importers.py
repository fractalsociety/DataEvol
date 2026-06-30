from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_from_record(record: dict[str, Any], *, default_type: str) -> dict[str, Any]:
    trace = dict(record)
    trace.setdefault("trace_type", trace.pop("type", default_type))
    if "input" in trace and "prompt" not in trace:
        trace["prompt"] = trace.pop("input")
    if "output" in trace and "response" not in trace:
        trace["response"] = trace.pop("output")
    return trace


def import_coordinate_run(path: str | Path) -> list[dict[str, Any]]:
    """Read a Coordinate run JSON file or directory and return canonical-ish traces."""
    root = Path(path)
    payload = _load_json(root) if root.is_file() else _load_json(root / "run.json")
    traces = payload.get("traces") or payload.get("tasks") or []
    return [_trace_from_record(dict(item), default_type="worker_trace") for item in traces]


def import_biolatent_run(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    payload = _load_json(root) if root.is_file() else _load_json(root / "biolatent_run.json")
    traces = payload.get("verification_traces") or payload.get("traces") or []
    return [_trace_from_record(dict(item), default_type="scientific_trace") for item in traces]


def import_fractal_router_decisions(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    payload = _load_json(root) if root.is_file() else _load_json(root / "router_decisions.json")
    decisions = payload.get("decisions") or payload.get("traces") or []
    traces: list[dict[str, Any]] = []
    for decision in decisions:
        trace = _trace_from_record(dict(decision), default_type="router_trace")
        trace.setdefault("metadata", {})
        trace["metadata"]["router_decision"] = {
            "provider": trace.get("provider"),
            "model": trace.get("model"),
            "agent_id": trace.get("agent_id"),
        }
        traces.append(trace)
    return traces


def parse_openrouter_metadata(record: dict[str, Any]) -> dict[str, Any]:
    provider = "openrouter" if "openrouter" in str(record.get("provider", "")).lower() else record.get("provider", "openrouter")
    cost = record.get("cost_usd")
    if cost is None:
        prompt_cost = float(record.get("prompt_cost_usd", 0) or 0)
        completion_cost = float(record.get("completion_cost_usd", 0) or 0)
        cost = prompt_cost + completion_cost
    return {
        "provider": str(provider).lower(),
        "model": record.get("model") or record.get("model_slug"),
        "cost_usd": float(cost or 0),
        "latency_ms": int(record.get("latency_ms") or record.get("duration_ms") or 0),
        "free_or_cheap": float(cost or 0) <= 0.001,
    }
