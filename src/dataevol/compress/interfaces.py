from __future__ import annotations

from typing import Any, Protocol


class LocalCompressionModel(Protocol):
    def summarize(self, trace: dict[str, Any], label: str | None) -> str:
        ...


class ExtractiveCompressionModel:
    def summarize(self, trace: dict[str, Any], label: str | None) -> str:
        prompt = " ".join(str(trace.get("prompt") or "").split())[:160]
        response = " ".join(str(trace.get("response") or "").split())[:200]
        return " ".join(part for part in (prompt, response) if part) or str(label or "trace")


def key_fact_retention(summary: str, trace: dict[str, Any]) -> float:
    facts = [
        str(trace.get("task_id") or ""),
        str(trace.get("trace_type") or ""),
        str(trace.get("provider") or ""),
        str(trace.get("model") or ""),
    ]
    facts = [fact.lower() for fact in facts if fact]
    if not facts:
        return 1.0
    summary_lower = summary.lower()
    return sum(1 for fact in facts if fact in summary_lower) / len(facts)
