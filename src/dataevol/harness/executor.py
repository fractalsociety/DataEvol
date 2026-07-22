"""Harness execution / evaluation.

A HarnessExecutor runs a genome against a benchmark and returns a
HarnessEvaluation. The default ReferenceExecutor is a fully deterministic,
in-process capability model: it needs no model and no Docker, so the whole
evolution loop runs and is unit-testable offline. A real subprocess/Docker
sandbox can later implement the same Protocol and be swapped in.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .genome import HarnessGenome
from .scoring import HarnessEvaluation, ScoreWeights, composite_score


# --- model cost / latency tiers (deterministic heuristics) -------------------
_EXPENSIVE_HINTS = ("opus", "frontier", "gpt-4", "gpt-5", "gpt5", "sonnet", "claude", "gemini-pro", "reasoning", "ultra")
_CHEAP_HINTS = ("local", "mlx", "7b", "8b", "9b", "1.5b", "3b", "mini", "haiku", "free", "small", "nano")

_COST_TIER = {"expensive": 0.060, "mid": 0.020, "cheap": 0.004}
_LATENCY_TIER = {"expensive": 2500.0, "mid": 1200.0, "cheap": 500.0}


def _model_tier(model: str) -> str:
    m = (model or "").lower()
    if any(h in m for h in _EXPENSIVE_HINTS):
        return "expensive"
    if any(h in m for h in _CHEAP_HINTS):
        return "cheap"
    return "mid"


@dataclass
class _Capabilities:
    base_quality: float
    has_verifier: bool
    independent_verifier: bool
    output_strict: bool
    workflow_depth: int
    agent_count: int
    memory_type: str
    retries: int
    retry_coverage: float
    router_threshold: float
    tools: set[str] = field(default_factory=set)
    per_task_cost: float = 0.0
    per_task_latency: float = 0.0
    prompt_richness: float = 0.0


@dataclass(frozen=True)
class _PreparedBenchmark:
    cases: tuple[Mapping[str, Any], ...]


def _capabilities(genome: HarnessGenome) -> _Capabilities:
    roles = {a.role for a in genome.agents}
    has_verifier = "verifier" in roles or any("verifier" in w.agent_role for w in genome.workflow)
    # An "independent" verifier cannot see prior agents' confidence.
    verifier_agents = [a for a in genome.agents if a.role == "verifier"]
    independent_verifier = any("previous_agent_confidence" in a.cannot_view for a in verifier_agents)
    output_strict = genome.output.validation == "strict"
    tools: set[str] = set()
    for a in genome.agents:
        tools.update(a.tools)

    # Base quality from structural richness.
    base = 0.45
    if output_strict:
        base += 0.10
    base += 0.06 * min(len(genome.agents), 3)
    if len(genome.workflow) >= 3:
        base += 0.08
    if genome.memory.type != "none":
        base += 0.05
    if has_verifier:
        base += 0.07
    refs = [a.prompt_ref for a in genome.agents if a.prompt_ref]
    prompt_richness = min(1.0, (sum(len(r) for r in refs) / max(1, len(refs)) / 40.0)) if refs else 0.0
    base += 0.05 * prompt_richness
    base = max(0.10, min(0.95, base))

    # retry coverage: fraction of impactful failure categories the genome recovers from
    impactful = {"VERIFICATION_FAILURE", "TOOL_ARGUMENT_ERROR", "TOOL_SELECTION", "OUTPUT_FORMAT", "REASONING_FAILURE"}
    covered = {c for c in genome.recovery.retry_on}
    retry_coverage = len(impactful & covered) / len(impactful) if impactful else 0.0

    # per-task cost / latency
    step_models = [genome.router.model] + [a.model for a in genome.agents]
    per_call_cost = sum(_COST_TIER[_model_tier(m)] for m in step_models)
    per_call_latency = sum(_LATENCY_TIER[_model_tier(m)] for m in step_models)
    retry_factor = 1.0 + 0.5 * genome.recovery.max_retries
    per_task_cost = per_call_cost * retry_factor
    backoff_factor = {"fixed": 1.0, "exponential": 1.3, "linear": 1.15}.get(genome.recovery.backoff, 1.0)
    per_task_latency = per_call_latency * retry_factor * backoff_factor

    return _Capabilities(
        base_quality=base,
        has_verifier=has_verifier,
        independent_verifier=independent_verifier,
        output_strict=output_strict,
        workflow_depth=len(genome.workflow),
        agent_count=len(genome.agents),
        memory_type=genome.memory.type,
        retries=genome.recovery.max_retries,
        retry_coverage=retry_coverage,
        router_threshold=genome.router.confidence_threshold,
        tools=tools,
        per_task_cost=per_task_cost,
        per_task_latency=per_task_latency,
        prompt_richness=prompt_richness,
    )


# Earliest-causal failure taxonomy labels by category.
_FAILURE_TAXONOMY = {
    "adversarial": "VERIFICATION_FAILURE",
    "tool_failure": "TOOL_ARGUMENT_ERROR",
    "ambiguous": "BAD_ROUTING",
    "long_context": "MISSING_CONTEXT",
    "edge": "OUTPUT_FORMAT",
    "normal": "REASONING_FAILURE",
    "regression": "REASONING_FAILURE",
    "hidden_holdout": "REASONING_FAILURE",
}


def _case_pass(genome: HarnessGenome, caps: _Capabilities, category: str) -> tuple[bool, str | None]:
    """Return (passed, earliest_causal_failure_label_or_None) for one case."""
    category = (category or "normal").lower()
    fail_label = _FAILURE_TAXONOMY.get(category, "REASONING_FAILURE")
    if category == "adversarial":
        return (caps.has_verifier and caps.retries >= 1, None if caps.has_verifier and caps.retries >= 1 else fail_label)
    if category == "tool_failure":
        ok = caps.retries >= 1 and ("TOOL_ARGUMENT_ERROR" in genome.recovery.retry_on or "TOOL_SELECTION" in genome.recovery.retry_on)
        return (ok, None if ok else fail_label)
    if category == "ambiguous":
        ok = 0.40 <= caps.router_threshold <= 0.85
        return (ok, None if ok else fail_label)
    if category == "long_context":
        ok = caps.memory_type != "none"
        return (ok, None if ok else fail_label)
    if category == "edge":
        ok = caps.output_strict and caps.has_verifier
        return (ok, None if ok else fail_label)
    return (True, None)


def _benchmark_cases(benchmark: Any) -> list[Mapping[str, Any]]:
    """Normalize a benchmark (FrozenBenchmark | Path | list[dict] | jsonl str) to case dicts."""
    if isinstance(benchmark, str):
        if benchmark.strip().startswith("[") or benchmark.strip().startswith("{"):
            import json
            parsed = json.loads(benchmark)
            return parsed if isinstance(parsed, list) else [parsed]
        return _read_jsonl(Path(benchmark))
    if isinstance(benchmark, list):
        return [c if isinstance(c, Mapping) else dict(c) for c in benchmark]
    if isinstance(benchmark, Mapping):
        items = benchmark.get("items") or benchmark.get("cases")
        if items:
            return [dict(c) for c in items]
    path = getattr(benchmark, "benchmark_path", None)
    if path is not None:
        return _read_jsonl(Path(path))
    if isinstance(benchmark, Path):
        return _read_jsonl(benchmark)
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    import json
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


class HarnessExecutor(Protocol):
    def evaluate(
        self,
        genome: HarnessGenome,
        benchmark: Any,
        *,
        seed: int = 17,
        repeated_runs: int = 1,
        weights: ScoreWeights | None = None,
    ) -> HarnessEvaluation: ...


class ReferenceExecutor:
    """Deterministic, model-free evaluator.

    Same (genome, benchmark, seed) always yields an identical evaluation.
    repeated_runs produces per_run_scores (one composite per run) for paired
    bootstrap comparison; runs differ only by small seeded jitter on per-case
    correctness, modeling nondeterminism without changing structural cost/latency.
    """

    def evaluate(
        self,
        genome: HarnessGenome,
        benchmark: Any,
        *,
        seed: int = 17,
        repeated_runs: int = 1,
        weights: ScoreWeights | None = None,
    ) -> HarnessEvaluation:
        weights = weights or ScoreWeights()
        caps = _capabilities(genome)
        cases = benchmark.cases if isinstance(benchmark, _PreparedBenchmark) else _benchmark_cases(benchmark)
        if not cases:
            raise ValueError("benchmark contains no cases")

        run_metrics: list[dict[str, float]] = []
        failure_categories_ordered: list[str] = []
        case_outcomes: list[tuple[str, bool, str | None]] = []
        category_counts: dict[str, int] = {}
        for case in cases:
            category = str(case.get("category") or case.get("benchmark_type") or "normal").lower()
            passed, label = _case_pass(genome, caps, category)
            case_outcomes.append((category, passed, label))
            category_counts[category] = category_counts.get(category, 0) + 1
        per_category_totals = {
            category: {"quality": 0.0, "failed": 0.0} for category in category_counts
        }

        robustness = caps_robustness(caps)
        verifier_agreement = caps_verifier(caps)

        for run_index in range(max(1, repeated_runs)):
            # Common random numbers make incumbent/challenger comparisons truly
            # paired: the same seed and case position receive the same jitter.
            rng = random.Random(seed * 1000 + run_index)
            correctness_sum = 0.0
            failed = 0
            for category, passed, label in case_outcomes:
                # small deterministic jitter so repeated runs vary slightly
                jitter = (rng.random() - 0.5) * 0.04
                if passed:
                    correctness = max(0.0, min(1.0, caps.base_quality + jitter))
                else:
                    correctness = caps.base_quality * 0.4
                    failed += 1
                    if label and label not in failure_categories_ordered:
                        failure_categories_ordered.append(label)
                correctness_sum += correctness
                totals = per_category_totals[category]
                totals["quality"] += correctness
                totals["failed"] += float(not passed)
            n = len(cases)
            run_q = correctness_sum / n
            run_f = failed / n
            run_metrics.append({
                "quality": run_q,
                "robustness": robustness,
                "verifier_agreement": verifier_agreement,
                "cost": caps.per_task_cost,
                "latency": caps.per_task_latency,
                "failure_rate": run_f,
            })

        # aggregate across runs
        def _mean(key: str) -> float:
            return sum(m[key] for m in run_metrics) / len(run_metrics)

        mean_metrics = {k: _mean(k) for k in run_metrics[0]}
        score = composite_score(mean_metrics, weights)
        per_run_scores = tuple(composite_score(m, weights) for m in run_metrics)
        per_category = {
            category: {
                "quality": totals["quality"] / (category_counts[category] * len(run_metrics)),
                "failure_rate": totals["failed"] / (category_counts[category] * len(run_metrics)),
                "count": category_counts[category],
            }
            for category, totals in per_category_totals.items()
        }

        return HarnessEvaluation(
            genome_id=genome.genome_id,
            quality=mean_metrics["quality"],
            robustness=mean_metrics["robustness"],
            verifier_agreement=mean_metrics["verifier_agreement"],
            cost=mean_metrics["cost"],
            latency=mean_metrics["latency"],
            failure_rate=mean_metrics["failure_rate"],
            score=score,
            per_category=per_category,
            failure_categories=tuple(failure_categories_ordered),
            run_count=len(run_metrics),
            per_run_scores=per_run_scores,
            raw={
                "benchmark_cases": len(cases),
                "task_type": genome.task_type,
                "per_run_quality": [m["quality"] for m in run_metrics],
            },
        )


def caps_robustness(caps: _Capabilities) -> float:
    return _clamp01(0.30 + 0.12 * min(caps.retries, 2) + 0.20 * caps.retry_coverage + 0.15 * (1.0 if caps.memory_type != "none" else 0.0) + 0.20 * (1.0 if caps.has_verifier else 0.0))


def caps_verifier(caps: _Capabilities) -> float:
    return _clamp01(0.20 + 0.40 * (1.0 if caps.has_verifier else 0.0) + 0.20 * (1.0 if caps.output_strict else 0.0) + 0.20 * (1.0 if caps.independent_verifier else 0.0))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parallel_evaluate(
    executor: HarnessExecutor,
    genomes: Sequence[HarnessGenome],
    benchmark: Any,
    *,
    seed: int = 17,
    repeated_runs: int = 3,
    weights: ScoreWeights | None = None,
    max_workers: int | None = None,
) -> list[HarnessEvaluation]:
    """Evaluate genomes with matched seeds, using concurrency for external executors."""
    if not genomes:
        return []
    reference_executor = isinstance(executor, ReferenceExecutor)
    if reference_executor:
        benchmark = _PreparedBenchmark(tuple(_benchmark_cases(benchmark)))
    default_workers = 1 if reference_executor else len(genomes)
    workers = max(1, min(max_workers or default_workers, len(genomes)))
    results: list[HarnessEvaluation | None] = [None] * len(genomes)

    def _job(idx: int) -> None:
        results[idx] = executor.evaluate(genomes[idx], benchmark, seed=seed, repeated_runs=repeated_runs, weights=weights)

    if workers == 1:
        for idx in range(len(genomes)):
            _job(idx)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_job, range(len(genomes))))
    return [r for r in results if r is not None]


def earliest_causal_failures(evaluation: HarnessEvaluation) -> tuple[str, ...]:
    return evaluation.failure_categories


# Re-export for convenience.
__all__ = [
    "HarnessExecutor",
    "ReferenceExecutor",
    "parallel_evaluate",
    "earliest_causal_failures",
]
