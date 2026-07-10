"""Compat/CLI/API dispatch for harness operations.

Model-driven operations (design/benchmark/failures/mutate/judge/evolve) require
a live model and surface a clean ``model_not_configured`` status when none is
configured. Read-only operations (lineage/incumbent/training_records) and
``evaluate`` (ReferenceExecutor) need no model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dataevol.config import DataEvolConfig

from . import storage as harness_storage
from .executor import ReferenceExecutor
from .genome import HarnessGenome
from .model_client import ModelClient, ModelClientError, ModelNotConfiguredError, resolve_model_client
from .scoring import HarnessEvaluation, ScoreWeights
from .specialists import (
    BenchmarkBuilder,
    FailureAnalyst,
    FailureAnalysis,
    FailureClassification,
    HarnessArchitect,
    HarnessMutator,
    ExperimentJudge,
    SpecialistError,
    apply_mutation,
    hash_task_spec,
)


def _not_configured(function_name: str, exc: Exception) -> dict[str, Any]:
    return {"ok": False, "status": "model_not_configured", "operation": function_name, "detail": str(exc)}


def _load_jsonish(value: Any, *, label: str, default: Any = None) -> Any:
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{label} is required")
    if isinstance(value, (Mapping, list)):
        return value
    if isinstance(value, (str, Path)):
        text = str(value)
        stripped = text.strip()
        if isinstance(value, str) and (stripped.startswith("{") or stripped.startswith("[")):
            return json.loads(stripped)
        path = Path(value)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return json.loads(text)
    return json.loads(str(value))


def _resolve_task(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        raise ValueError("task specification is required (JSON object, JSON string, or file path)")
    loaded = _load_jsonish(value, label="task")
    if not isinstance(loaded, Mapping):
        raise ValueError("task specification must be a JSON object")
    return dict(loaded)


def _resolve_genome(value: Any) -> HarnessGenome:
    loaded = _load_jsonish(value, label="genome")
    if not isinstance(loaded, Mapping):
        raise ValueError("genome must be a JSON object, JSON string, or file path")
    return HarnessGenome.from_dict(loaded)


def _float_value(data: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _resolve_evaluation(value: Any, *, genome_id: str = "") -> HarnessEvaluation:
    loaded = _load_jsonish(value, label="evaluation", default={})
    if not isinstance(loaded, Mapping):
        raise ValueError("evaluation must be a JSON object, JSON string, or file path")
    data = dict(loaded)
    per_category = data.get("per_category") if isinstance(data.get("per_category"), Mapping) else {}
    raw = data.get("raw") if isinstance(data.get("raw"), Mapping) else {}
    return HarnessEvaluation(
        genome_id=str(data.get("genome_id") or genome_id),
        quality=_float_value(data, "quality"),
        robustness=_float_value(data, "robustness"),
        verifier_agreement=_float_value(data, "verifier_agreement"),
        cost=_float_value(data, "cost"),
        latency=_float_value(data, "latency"),
        failure_rate=_float_value(data, "failure_rate"),
        score=_float_value(data, "score"),
        per_category={str(k): dict(v) for k, v in per_category.items() if isinstance(v, Mapping)},
        failure_categories=tuple(str(v) for v in (data.get("failure_categories") or [])),
        run_count=int(data.get("run_count", 1) or 1),
        per_run_scores=tuple(float(v) for v in (data.get("per_run_scores") or [])),
        raw=dict(raw),
    )


def _resolve_failure_analysis(value: Any) -> FailureAnalysis:
    if value is None:
        return FailureAnalysis(failures=(), summary="")
    loaded = _load_jsonish(value, label="failures", default={})
    if not isinstance(loaded, Mapping):
        raise ValueError("failures must be a JSON object, JSON string, or file path")
    raw_failures = loaded.get("failures") or []
    if isinstance(raw_failures, Mapping):
        raw_failures = [raw_failures]
    failures: list[FailureClassification] = []
    if isinstance(raw_failures, list):
        for item in raw_failures:
            if not isinstance(item, Mapping):
                continue
            failures.append(FailureClassification(
                category=str(item.get("category") or "REASONING_FAILURE"),
                earliest_cause=str(item.get("earliest_cause") or item.get("cause") or ""),
                evidence=str(item.get("evidence") or ""),
            ))
    return FailureAnalysis(failures=tuple(failures), summary=str(loaded.get("summary") or ""))


def _ensure_genome_registered(config: DataEvolConfig, genome: HarnessGenome) -> None:
    task_spec = {
        "task_type": genome.task_type,
        "task_spec_hash": genome.task_spec_hash,
    }
    task_hash = genome.task_spec_hash or hash_task_spec(task_spec)
    task_id = harness_storage.register_task(config.db_path, genome.task_type or "general", task_spec, task_hash)
    harness_storage.register_genome(config.db_path, _with_hash(genome), task_id)


def _client(config: DataEvolConfig) -> ModelClient:
    return resolve_model_client(config)


def run_harness_operation(function_name: str, payload: dict[str, Any], config: DataEvolConfig) -> dict[str, Any]:
    try:
        return _run_harness_operation(function_name, payload, config)
    except ModelClientError as exc:
        return {"ok": False, "status": "model_error", "operation": function_name, "detail": str(exc)}
    except SpecialistError as exc:
        return {"ok": False, "status": "specialist_error", "operation": function_name, "detail": str(exc)}


def _run_harness_operation(function_name: str, payload: dict[str, Any], config: DataEvolConfig) -> dict[str, Any]:
    if function_name == "evolve":
        try:
            model_client = _client(config)
        except ModelNotConfiguredError as exc:
            return _not_configured(function_name, exc)
        from .loop import EvolutionConfig, run_harness_evolution

        task_spec = _resolve_task(payload.get("task"))
        evolution = EvolutionConfig(
            max_generations=int(payload.get("max_generations") or 20),
            number_of_candidates=int(payload.get("number_of_candidates") or 8),
            repeated_runs=int(payload.get("repeated_runs") or 3),
        )
        result = run_harness_evolution(task_spec, config=config, model_client=model_client, evolution=evolution)
        return {
            "ok": True,
            "status": "completed",
            "operation": function_name,
            "task_id": result.task_id,
            "benchmark_id": result.benchmark_id,
            "generations": result.generations,
            "promotions": result.promotions,
            "final_strategy": result.final_strategy,
            "incumbent": result.incumbent.to_dict(),
            "incumbent_evaluation": dict(result.incumbent_evaluation),
            "lineage_count": len(result.lineage),
            "incumbent_rollback_snapshot": result.incumbent_rollback_snapshot,
        }

    if function_name == "design":
        try:
            model_client = _client(config)
        except ModelNotConfiguredError as exc:
            return _not_configured(function_name, exc)
        task_spec = _resolve_task(payload.get("task"))
        genome = HarnessArchitect(model_client, model=getattr(config, "model_name", None)).design(task_spec)
        task_id = harness_storage.register_task(config.db_path, genome.task_type, task_spec, hash_task_spec(task_spec))
        harness_storage.register_genome(config.db_path, _with_hash(genome), task_id)
        return {"ok": True, "status": "completed", "operation": function_name, "genome": genome.to_dict(), "genome_id": genome.genome_id, "task_id": task_id}

    if function_name == "benchmark":
        try:
            model_client = _client(config)
        except ModelNotConfiguredError as exc:
            return _not_configured(function_name, exc)
        from dataevol.benchmarks import build_frozen_benchmark

        task_spec = _resolve_task(payload.get("task"))
        cases = BenchmarkBuilder(model_client, model=getattr(config, "model_name", None)).build(task_spec)
        task_type = str(task_spec.get("task_type") or "general")
        out_dir = Path(payload.get("output") or Path(config.artifacts_path) / "harness" / "benchmarks")
        frozen = build_frozen_benchmark(cases, out_dir, name=f"{task_type}_bench", version="v0", source="harness-benchmark-builder", overwrite=True)
        task_id = harness_storage.register_task(config.db_path, task_type, task_spec, hash_task_spec(task_spec))
        benchmark_id = harness_storage.register_benchmark(
            config.db_path, task_id=task_id, name=f"{task_type}_bench", version="v0", category="combined",
            path=str(frozen.benchmark_path), manifest_path=str(frozen.manifest_path), sha256=frozen.sha256, item_count=frozen.item_count,
        )
        return {"ok": True, "status": "completed", "operation": function_name, "cases": cases, "benchmark_path": str(frozen.benchmark_path), "item_count": frozen.item_count, "benchmark_id": benchmark_id}

    if function_name == "evaluate":
        genome = _resolve_genome(payload.get("genome"))
        benchmark = payload.get("benchmark") or payload.get("cases")
        weights = ScoreWeights.default_for_task(genome.task_type)
        evaluation = ReferenceExecutor().evaluate(
            genome, benchmark, seed=int(payload.get("seed") or 17), repeated_runs=int(payload.get("repeated_runs") or 3), weights=weights,
        )
        _ensure_genome_registered(config, genome)
        eval_id = harness_storage.register_evaluation(config.db_path, evaluation.to_dict(), benchmark_id=payload.get("benchmark_id"))
        return {"ok": True, "status": "completed", "operation": function_name, "evaluation": evaluation.to_dict(), "evaluation_id": eval_id}

    if function_name == "failures":
        try:
            model_client = _client(config)
        except ModelNotConfiguredError as exc:
            return _not_configured(function_name, exc)
        genome = _resolve_genome(payload.get("genome"))
        evaluation = _resolve_evaluation(payload.get("evaluation"), genome_id=genome.genome_id)
        analysis = FailureAnalyst(model_client, model=getattr(config, "model_name", None)).analyze(genome, evaluation)
        return {"ok": True, "status": "completed", "operation": function_name, "analysis": analysis.to_dict()}

    if function_name == "mutate":
        try:
            model_client = _client(config)
        except ModelNotConfiguredError as exc:
            return _not_configured(function_name, exc)
        incumbent = _resolve_genome(payload.get("incumbent"))
        analysis = _resolve_failure_analysis(payload.get("failures"))
        proposals = HarnessMutator(model_client, model=getattr(config, "model_name", None)).propose(
            incumbent, analysis, number_of_candidates=int(payload.get("number_of_candidates") or 8),
        )
        candidates = [apply_mutation(incumbent, p) for p in proposals]
        return {"ok": True, "status": "completed", "operation": function_name, "proposals": [p.to_dict() for p in proposals], "candidates": [c.to_dict() for c in candidates]}

    if function_name == "judge":
        try:
            model_client = _client(config)
        except ModelNotConfiguredError as exc:
            return _not_configured(function_name, exc)
        bootstrap = tuple(payload.get("bootstrap") or (0.0, 0.0, 0.0))
        review = ExperimentJudge(model_client, judge_model=getattr(config, "judge_model_name", None) or getattr(config, "model_name", None)).compare(
            incumbent=_resolve_evaluation(payload.get("incumbent_eval")), challenger=_resolve_evaluation(payload.get("challenger_eval")), bootstrap=bootstrap, mutator_model=getattr(config, "model_name", None),
        )
        return {"ok": True, "status": "completed", "operation": function_name, "review": review.to_dict()}

    if function_name == "lineage":
        task_id = payload.get("task_id")
        rows = harness_storage.load_lineage(config.db_path, int(task_id) if task_id is not None else None)
        return {"ok": True, "status": "completed", "operation": function_name, "lineage": rows, "count": len(rows)}

    if function_name == "incumbent":
        task_id = payload.get("task_id")
        incumbent = harness_storage.load_incumbent(config.db_path, int(task_id) if task_id is not None else None)
        return {"ok": True, "status": "completed", "operation": function_name, "incumbent": incumbent}

    if function_name == "training_records":
        genome_id = payload.get("genome_id")
        rows = harness_storage.load_training_records(config.db_path, genome_id)
        return {"ok": True, "status": "completed", "operation": function_name, "records": rows, "count": len(rows)}

    return {"ok": False, "status": "unknown_operation", "operation": function_name, "detail": f"unknown harness operation: {function_name}"}


def _with_hash(genome: HarnessGenome) -> dict[str, Any]:
    data = genome.to_dict()
    data["content_hash"] = genome.content_hash()
    return data
