"""Compat/CLI/API dispatch for harness operations.

Model-driven operations (design/benchmark/failures/mutate/judge/evolve) require
a live model and surface a clean ``model_not_configured`` status when none is
configured. Read-only operations (lineage/incumbent/training_records) and
``evaluate`` (ReferenceExecutor) need no model.
"""
from __future__ import annotations

import json
import hashlib
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
from .verdicts import issue_harness_verdict
from .compiled import CompiledHarness, DeterministicHarnessRouter, ExternalHarnessCompiler
from .controller import (
    HarnessExecutionController,
    HarnessExecutionState,
    distillation_example,
    execution_metrics,
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
    except (TypeError, ValueError) as exc:
        return {"ok": False, "status": "validation_error", "operation": function_name, "detail": str(exc)}


def _run_harness_operation(function_name: str, payload: dict[str, Any], config: DataEvolConfig) -> dict[str, Any]:
    if function_name == "compile":
        try:
            model_client = _client(config)
        except ModelNotConfiguredError as exc:
            return _not_configured(function_name, exc)
        task = _resolve_task(payload.get("task"))
        rich_harness = payload.get("rich_harness")
        if not isinstance(rich_harness, (Mapping, str)):
            raise ValueError("rich_harness must be an object or string")
        harness = ExternalHarnessCompiler(
            model_client, model=getattr(config, "model_name", None)
        ).compile(
            task=task,
            rich_harness=rich_harness,
            harness_id=str(payload.get("harness_id") or "").strip(),
            version=int(payload.get("version", 1)),
            parent_id=payload.get("parent_id"),
            source_genome_id=payload.get("source_genome_id"),
        )
        harness_storage.register_compiled_harness(config.db_path, harness)
        return {"ok": True, "status": "completed", "operation": function_name, "harness": harness.to_dict()}

    if function_name == "register_compiled":
        loaded = _load_jsonish(payload.get("harness"), label="harness")
        if not isinstance(loaded, Mapping):
            raise ValueError("harness must be an object")
        harness = CompiledHarness.from_dict(loaded)
        harness_storage.register_compiled_harness(config.db_path, harness)
        return {"ok": True, "status": "completed", "operation": function_name, "harness": harness.to_dict()}

    if function_name == "compiled_registry":
        rows = harness_storage.list_compiled_harnesses(
            config.db_path,
            status=str(payload["status"]) if payload.get("status") else None,
            category=str(payload["category"]) if payload.get("category") else None,
        )
        return {"ok": True, "status": "completed", "operation": function_name, "harnesses": rows, "count": len(rows)}

    if function_name == "route_compiled":
        features = payload.get("features") or payload.get("task") or {}
        if not isinstance(features, Mapping):
            raise ValueError("features must be an object")
        harnesses = [
            CompiledHarness.from_dict(row)
            for row in harness_storage.list_compiled_harnesses(config.db_path, status="active")
        ]
        decision = DeterministicHarnessRouter().route(features, harnesses, top_k=int(payload.get("top_k", 3)))
        return {"ok": True, "status": "completed", "operation": function_name, "route": decision.to_dict()}

    if function_name == "start_execution":
        task = payload.get("task") or {}
        features = payload.get("features") or task
        if not isinstance(task, Mapping) or not isinstance(features, Mapping):
            raise ValueError("task and features must be objects")
        harnesses = [
            CompiledHarness.from_dict(row)
            for row in harness_storage.list_compiled_harnesses(config.db_path, status="active")
        ]
        decision = DeterministicHarnessRouter().route(features, harnesses, top_k=int(payload.get("top_k", 3)))
        teacher_selection = str(payload.get("teacher_selected_harness_id") or "").strip()
        if decision.teacher_required and not teacher_selection:
            return {
                "ok": False,
                "status": "teacher_required",
                "operation": function_name,
                "route": decision.to_dict(),
            }
        selected_id = teacher_selection or (decision.candidates[0].harness_id if decision.candidates else "")
        selected = next((item for item in harnesses if item.harness_id == selected_id), None)
        if selected is None:
            raise ValueError("selected compiled harness is not active or does not exist")
        state = HarnessExecutionState.start(task, selected, session_id=payload.get("session_id"))
        harness_storage.create_execution_session(
            config.db_path, state, task_features=features, route_decision=decision.to_dict()
        )
        expected = HarnessExecutionController().expected_action(state, selected)
        return {
            "ok": True,
            "status": "completed",
            "operation": function_name,
            "route": decision.to_dict(),
            "teacher_selected": bool(teacher_selection),
            "execution": state.to_dict(),
            "expected_action": expected,
        }

    if function_name == "execution_action":
        session_id = str(payload.get("session_id") or "").strip()
        session, state, harness = _execution_context(config, session_id)
        proposal = payload.get("proposal") or {}
        if not isinstance(proposal, Mapping):
            raise ValueError("proposal must be an object")
        controller = HarnessExecutionController()
        before = state.to_dict()
        decision = controller.propose(state, harness, proposal)
        harness_storage.update_execution_session(config.db_path, decision.state)
        harness_storage.register_execution_event(config.db_path, {
            "session_id": session_id,
            "kind": "action",
            "state_before": before,
            "proposal": dict(proposal),
            "accepted": decision.accepted,
            "violations": list(decision.violations),
            "expected_action": dict(decision.expected_action),
            "state_after": decision.state.to_dict(),
        })
        correction_result = None
        correction = payload.get("teacher_correction")
        final_state = decision.state
        if not decision.accepted and isinstance(correction, Mapping):
            corrected_state = controller.apply_teacher_correction(decision.state)
            correction_result = controller.propose(corrected_state, harness, correction)
            final_state = correction_result.state
            harness_storage.update_execution_session(config.db_path, final_state)
            harness_storage.register_execution_event(config.db_path, {
                "session_id": session_id,
                "kind": "action",
                "state_before": corrected_state.to_dict(),
                "proposal": dict(correction),
                "accepted": correction_result.accepted,
                "violations": list(correction_result.violations),
                "expected_action": dict(correction_result.expected_action),
                "state_after": correction_result.state.to_dict(),
                "teacher_correction": dict(correction),
            })
        return {
            "ok": decision.accepted or bool(correction_result and correction_result.accepted),
            "status": final_state.status.lower(),
            "operation": function_name,
            "decision": decision.to_dict(),
            "teacher_decision": correction_result.to_dict() if correction_result else None,
            "execution": final_state.to_dict(),
            "route": session["route_decision"],
        }

    if function_name == "execution_observation":
        session_id = str(payload.get("session_id") or "").strip()
        _, state, harness = _execution_context(config, session_id)
        controller = HarnessExecutionController()
        before = state.to_dict()
        evidence = payload.get("evidence") or {}
        verifier = payload.get("verifier") or {}
        if not isinstance(evidence, Mapping) or not isinstance(verifier, Mapping):
            raise ValueError("evidence and verifier must be objects")
        result = controller.observe(
            state,
            harness,
            success=bool(payload.get("success")),
            produced_flags=tuple(str(item) for item in (payload.get("produced_flags") or [])),
            evidence=evidence,
        )
        harness_storage.update_execution_session(config.db_path, result.state)
        harness_storage.register_execution_event(config.db_path, {
            "session_id": session_id,
            "kind": "observation",
            "state_before": before,
            "observation": {
                "success": bool(payload.get("success")),
                "produced_flags": list(payload.get("produced_flags") or []),
                "evidence": dict(evidence),
            },
            "state_after": result.state.to_dict(),
            "expected_action": dict(result.expected_action or {}),
            "verifier": dict(verifier),
        })
        return {"ok": True, "status": result.state.status.lower(), "operation": function_name, **result.to_dict()}

    if function_name == "get_execution":
        session_id = str(payload.get("session_id") or "").strip()
        session = harness_storage.load_execution_session(config.db_path, session_id)
        if session is None:
            return {"ok": False, "status": "not_found", "operation": function_name, "session_id": session_id}
        events = harness_storage.load_execution_events(config.db_path, session_id)
        return {"ok": True, "status": "completed", "operation": function_name, "execution": session, "events": events}

    if function_name == "execution_metrics":
        events = harness_storage.load_execution_events(config.db_path, payload.get("session_id"))
        return {"ok": True, "status": "completed", "operation": function_name, "metrics": execution_metrics(events)}

    if function_name == "export_next_actions":
        events = harness_storage.load_execution_events(config.db_path, payload.get("session_id"))
        examples = _next_action_examples(events)
        output = Path(payload.get("output") or Path(config.artifacts_path) / "harness" / "next_action_distillation.jsonl")
        artifact = _write_next_action_export(output, examples)
        return {"ok": True, "status": "completed", "operation": function_name, **artifact}

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

    if function_name == "verdict":
        verdict = issue_harness_verdict(payload)
        harness_storage.register_verdict(config.db_path, verdict)
        return {
            "ok": True,
            "status": "completed",
            "operation": function_name,
            "verdict": verdict.to_dict(),
        }

    if function_name == "get_verdict":
        verdict_id = str(payload.get("verdict_id") or "").strip()
        if not verdict_id:
            raise ValueError("verdict_id is required")
        verdict = harness_storage.load_verdict(config.db_path, verdict_id)
        if verdict is None:
            return {
                "ok": False,
                "status": "not_found",
                "operation": function_name,
                "verdict_id": verdict_id,
            }
        return {"ok": True, "status": "completed", "operation": function_name, "verdict": verdict}

    return {"ok": False, "status": "unknown_operation", "operation": function_name, "detail": f"unknown harness operation: {function_name}"}


def _execution_context(
    config: DataEvolConfig,
    session_id: str,
) -> tuple[dict[str, Any], HarnessExecutionState, CompiledHarness]:
    if not session_id:
        raise ValueError("session_id is required")
    session = harness_storage.load_execution_session(config.db_path, session_id)
    if session is None:
        raise ValueError(f"execution session not found: {session_id}")
    state = HarnessExecutionState.from_dict(session["state"])
    raw_harness = harness_storage.load_compiled_harness(
        config.db_path, state.harness_id, state.harness_version
    )
    if raw_harness is None:
        raise ValueError("execution session references a missing compiled harness")
    harness = CompiledHarness.from_dict(raw_harness)
    return session, state, harness


def _next_action_examples(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    accepted_example_index: int | None = None
    for event in events:
        if event.get("kind") == "observation":
            if accepted_example_index is not None and event.get("verifier"):
                example = examples[accepted_example_index]
                example["verifier"] = dict(event["verifier"])
                unhashed = {key: value for key, value in example.items() if key != "example_hash"}
                example["example_hash"] = hashlib.sha256(
                    json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                ).hexdigest()
            accepted_example_index = None
            continue
        if event.get("kind") != "action":
            continue
        examples.append(distillation_example(
            state_before=event.get("state_before") or {},
            expected_action=event.get("expected_action") or {},
            proposal=event.get("proposal") or {},
            accepted=bool(event.get("accepted")),
            violations=event.get("violations") or [],
            state_after=event.get("state_after") or {},
            teacher_correction=event.get("teacher_correction") or None,
            verifier=event.get("verifier") or None,
        ))
        if event.get("accepted"):
            accepted_example_index = len(examples) - 1
    return examples


def _write_next_action_export(output: Path, examples: list[dict[str, Any]]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(example, sort_keys=True, ensure_ascii=False) + "\n" for example in examples)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(output)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    manifest = {
        "schema": "dataevol.next_action_dataset_manifest.v1",
        "path": str(output),
        "sha256": digest,
        "example_count": len(examples),
        "includes_negative_examples": any(not example["accepted"] for example in examples),
        "includes_teacher_corrections": any(example["teacher_corrected"] for example in examples),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dataset": manifest, "manifest_path": str(manifest_path)}


def _with_hash(genome: HarnessGenome) -> dict[str, Any]:
    data = genome.to_dict()
    data["content_hash"] = genome.content_hash()
    return data
