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


def worker_report_to_trace(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize a structured Coordinate worker report into a worker trace."""
    task = report.get("task") if isinstance(report.get("task"), dict) else {}
    verification = report.get("verification") if isinstance(report.get("verification"), dict) else {}
    changed_files = report.get("changed_files") or report.get("files_changed") or report.get("files")
    tests_run = report.get("tests_run") or verification.get("tests_run") or report.get("tests")
    trace = {
        "trace_type": "worker_trace",
        "task_id": report.get("task_id") or task.get("id") or report.get("id"),
        "agent_id": report.get("agent_id") or report.get("worker_id") or report.get("worker"),
        "provider": str(report.get("provider") or report.get("client") or "").lower() or None,
        "model": report.get("model"),
        "objective": task.get("title") or report.get("title") or report.get("objective"),
        "prompt": report.get("assignment") or task.get("body") or task.get("title") or report.get("summary"),
        "response": report.get("summary") or report.get("result") or report.get("notes"),
        "files_changed": _string_list(changed_files),
        "tests_run": _test_list(tests_run),
        "outcome": report.get("outcome") or report.get("status") or ("accepted" if verification.get("passed") is True else None),
        "failure_type": report.get("failure_type"),
        "privacy_mode": report.get("privacy_mode", "private-local-only"),
        "metadata": {
            "source": "coordinate_worker_report",
            "task": task,
            "verification": verification,
            "risks": report.get("risks") or report.get("residual_risks") or [],
            "files_inspected": report.get("files_inspected") or report.get("inspected_files") or [],
        },
        "metrics": {
            "duration_seconds": report.get("duration_seconds"),
            "token_usage": report.get("token_usage"),
            "changed_file_count": len(_string_list(changed_files)),
            "test_count": len(_test_list(tests_run)),
        },
    }
    return {key: value for key, value in trace.items() if value is not None}


def worker_reports_to_traces(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [worker_report_to_trace(dict(report)) for report in reports]


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


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _test_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        tests: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                tests.append(dict(item))
            else:
                tests.append({"command": str(item)})
        return tests
    if isinstance(value, dict):
        return [dict(value)]
    return [{"command": str(value)}]
