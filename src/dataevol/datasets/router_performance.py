from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def build_router_performance_dataset(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        rows.append(
            {
                "task": trace.get("task") or trace.get("objective") or trace.get("prompt"),
                "selected_model": trace.get("model"),
                "provider": trace.get("provider"),
                "result": trace.get("label") or trace.get("outcome"),
                "score": trace.get("quality_score") or trace.get("score") or 0.0,
                "cost_usd": (trace.get("metrics") or {}).get("cost_usd", trace.get("cost_usd", 0.0)),
                "latency_ms": (trace.get("metrics") or {}).get("latency_ms", trace.get("latency_ms", 0)),
            }
        )
    return rows


def provider_success_rate(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    for row in rows:
        provider = str(row.get("provider") or "unknown")
        totals[provider] += 1
        if row.get("result") in {"accepted", "good_training_candidate", "success"}:
            successes[provider] += 1
    return {provider: successes[provider] / total for provider, total in totals.items()}


def cost_normalized_quality(row: Mapping[str, Any]) -> float:
    cost = float(row.get("cost_usd") or 0.0)
    quality = float(row.get("score") or 0.0)
    return quality / max(cost, 0.001)


def escalation_rescue_rate(rows: Iterable[Mapping[str, Any]]) -> float:
    data = list(rows)
    if not data:
        return 0.0
    rescued = sum(1 for row in data if row.get("result") == "rescued_by_stronger_model")
    return rescued / len(data)


def generate_candidate_router_policy(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    performance = list(rows)
    provider_rates = provider_success_rate(performance)
    preferred = sorted(provider_rates, key=provider_rates.get, reverse=True)
    return {
        "version": "router_policy_candidate_v1",
        "rule": "prefer lowest-cost provider among providers with acceptable success rate",
        "preferred_providers": preferred,
        "provider_success_rate": provider_rates,
        "min_success_rate": 0.8,
    }
