from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def detect_opportunities(traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(trace) for trace in traces]
    opportunities: list[dict[str, Any]] = []
    opportunities.extend(_router_mistakes(rows))
    opportunities.extend(_repeated_failures(rows))
    opportunities.extend(_expensive_successes(rows))
    opportunities.extend(_duplicate_effort(rows))
    opportunities.extend(_missing_benchmarks(rows))
    opportunities.extend(_prompt_confusion(rows))
    return opportunities


def _router_mistakes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = [r for r in rows if r.get("failure_type") == "bad_router_assignment" or r.get("label") == "rescued_by_stronger_model"]
    if not matches:
        return []
    return [_opportunity("router_mistake", matches, "Router assignments failed or required rescue.", "Tighter router policy can route risky tasks to stronger workers.", "Adjust router policy thresholds and examples.", "verification_pass_rate")]


def _repeated_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(r.get("failure_type") for r in rows if r.get("failure_type"))
    repeated = [name for name, count in counts.items() if count >= 2 and name != "duplicated_work"]
    if not repeated:
        return []
    return [_opportunity("repeated_failure", rows, f"Repeated failures: {', '.join(sorted(repeated))}.", "A focused prompt or evaluator can reduce repeated failure modes.", "Add negative examples and failure-specific checks.", "task_success_rate")]


def _expensive_successes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successes = [r for r in rows if r.get("label") == "accepted" and float((r.get("metrics") or {}).get("cost_usd", r.get("cost_usd", 0)) or 0) > 0.05]
    if not successes:
        return []
    return [_opportunity("expensive_success", successes, "Accepted tasks used comparatively expensive models.", "Cheaper models may preserve verification quality for low-risk tasks.", "Introduce cost-aware routing fallback.", "cost_per_verified_task")]


def _duplicate_effort(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_key = str(row.get("task") or row.get("objective") or "").strip().lower()
        if task_key:
            by_task[task_key].append(row)
    duplicates = [group for group in by_task.values() if len(group) >= 2]
    if not duplicates and not any(r.get("failure_type") == "duplicated_work" for r in rows):
        return []
    return [_opportunity("duplicate_effort", rows, "Similar tasks were attempted repeatedly.", "Duplicate detection can reuse verified work or compact repeats.", "Add task hash/similarity checks before worker assignment.", "duplicate_rate")]


def _prompt_confusion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = [r for r in rows if r.get("failure_type") in {"format_failure", "context_loss"} or "confus" in str(r.get("notes", "")).lower()]
    if not matches:
        return []
    return [_opportunity("prompt_confusion", matches, "Workers showed format, context, or instruction confusion.", "Clearer prompt packs can reduce ambiguity without raising cost.", "Revise prompt pack with explicit output contract.", "format_pass_rate")]


def _missing_benchmarks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [r for r in rows if r.get("failure_type") or str(r.get("label", "")).startswith("failed")]
    if not failures:
        return []
    covered = any(r.get("trace_type") == "benchmark_trace" for r in rows)
    if covered:
        return []
    return [_opportunity("missing_benchmark", failures, "Failures were observed without matching benchmark coverage.", "Freezing representative failures as benchmark cases can prevent regressions.", "Create benchmark tasks from repeated failures.", "regression_rate")]


def _opportunity(category: str, rows: list[dict[str, Any]], observation: str, hypothesis: str, proposed_change: str, metric: str) -> dict[str, Any]:
    return {
        "id": f"opp_{category}",
        "category": category,
        "trace_ids": [str(row.get("id") or row.get("trace_id")) for row in rows if row.get("id") or row.get("trace_id")],
        "observation": observation,
        "hypothesis": hypothesis,
        "proposed_change": proposed_change,
        "expected_metric": metric,
        "risk_level": "medium" if category in {"router_mistake", "prompt_confusion"} else "low",
        "status": "proposed",
    }


def reject_weak_opportunities(opportunities: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    strong: list[dict[str, Any]] = []
    for opportunity in opportunities:
        row = dict(opportunity)
        if not row.get("observation") or not row.get("hypothesis") or not row.get("expected_metric"):
            row["status"] = "NO_IDEA"
        strong.append(row)
    return strong


def save_learning_opportunities(opportunities: Iterable[Mapping[str, Any]], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "learning_opportunities.json"
    rows = reject_weak_opportunities(opportunities)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
