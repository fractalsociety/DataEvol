from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from dataevol.config import DataEvolConfig, ensure_project_dirs


def _parse_run_id(value: Any) -> int:
    text = str(value)
    if text.startswith("run_"):
        text = text.removeprefix("run_")
    return int(text)


def _normalize_result(result: Any, operation: str) -> dict[str, Any]:
    if isinstance(result, dict):
        normalized = dict(result)
    elif hasattr(result, "to_dict"):
        normalized = result.to_dict()
    elif is_dataclass(result):
        normalized = asdict(result)
    else:
        normalized = {"result": result}
    normalized.setdefault("ok", True)
    normalized.setdefault("status", "completed")
    normalized.setdefault("operation", operation)
    return normalized


def _write_trace_batch(config: DataEvolConfig, name: str, traces: list[dict[str, Any]]) -> Path:
    ensure_project_dirs(config)
    path = config.artifacts_path / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def _load_experiment_payload(payload: dict[str, Any], config: DataEvolConfig) -> dict[str, Any]:
    if isinstance(payload.get("report"), dict):
        return payload["report"]
    experiment_id = payload.get("experiment") or payload.get("experiment_id")
    if experiment_id:
        path = config.artifacts_path / "experiments" / f"{experiment_id}.report.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return payload


def _call_known_operation(
    module_name: str,
    function_name: str,
    payload: dict[str, Any],
    config: DataEvolConfig,
) -> dict[str, Any] | None:
    if function_name == "ingest" and payload.get("jsonl"):
        from dataevol.ingest import ingest_jsonl

        result = ingest_jsonl(
            payload["jsonl"],
            config.db_path,
            source_system=payload.get("source") or "unknown",
            privacy_mode=config.privacy_mode,
            raw_root=config.raw_path,
        )
        return _normalize_result(result, function_name)

    if function_name == "ingest" and payload.get("path"):
        from dataevol.ingest import import_biolatent_run, import_coordinate_run, import_fractal_router_decisions, ingest_jsonl

        source = payload.get("source") or "unknown"
        if source == "coordinate":
            traces = import_coordinate_run(payload["path"])
        elif source == "biolatent":
            traces = import_biolatent_run(payload["path"])
        elif source in {"fractal-router", "fractal_router", "router"}:
            traces = import_fractal_router_decisions(payload["path"])
        else:
            traces = []
        path = _write_trace_batch(config, f"import_{source}", traces)
        result = ingest_jsonl(path, config.db_path, source_system=source, privacy_mode=config.privacy_mode, raw_root=config.raw_path)
        return _normalize_result(result, function_name)

    if function_name == "ingest_trace":
        from dataevol.ingest import ingest_jsonl

        trace = payload.get("trace")
        if not isinstance(trace, dict):
            return None
        path = _write_trace_batch(config, "api_ingest_trace", [trace])
        result = ingest_jsonl(
            path,
            config.db_path,
            source_system=payload.get("source_system") or "api",
            privacy_mode=config.privacy_mode,
            raw_root=config.raw_path,
        )
        return _normalize_result(result, function_name)

    if function_name == "ingest_run":
        from dataevol.ingest import ingest_jsonl

        run = payload.get("run") or {}
        traces = run.get("traces") if isinstance(run, dict) else None
        if not isinstance(traces, list):
            return None
        path = _write_trace_batch(config, "api_ingest_run", [dict(trace) for trace in traces])
        result = ingest_jsonl(
            path,
            config.db_path,
            source_system=payload.get("source_system") or run.get("source_system") or "api",
            privacy_mode=config.privacy_mode,
            raw_root=config.raw_path,
            external_run_id=run.get("external_run_id"),
            objective=run.get("objective"),
        )
        return _normalize_result(result, function_name)

    if function_name == "label_run":
        from dataevol.label import label_run

        return _normalize_result(label_run(config.db_path, _parse_run_id(payload["run_id"])), function_name)

    if function_name == "score_run":
        from dataevol.score import score_run

        return _normalize_result(score_run(config.db_path, _parse_run_id(payload["run_id"])), function_name)

    if function_name == "compress_run":
        from dataevol.compress import compress_run

        return _normalize_result(compress_run(config.db_path, _parse_run_id(payload["run_id"])), function_name)

    if function_name == "build_dataset":
        from dataevol.datasets import build_dataset, build_router_dataset

        dataset_type = payload.get("type") or payload.get("dataset_type") or "router"
        if dataset_type == "router":
            result = build_router_dataset(payload.get("traces") or [], config.artifacts_path / "datasets", privacy_mode=config.privacy_mode)
        else:
            result = build_dataset(dataset_type, payload.get("traces") or [], config.artifacts_path / "datasets", privacy_mode=config.privacy_mode)
        return _normalize_result(result, function_name)

    if function_name == "build_benchmark":
        from dataevol.benchmarks import build_benchmark, build_frozen_benchmark

        benchmark_type = payload.get("type") or payload.get("benchmark_type")
        if benchmark_type:
            result = build_benchmark(benchmark_type, payload.get("items") or [], config.artifacts_path / "benchmarks", overwrite=True)
        else:
            result = build_frozen_benchmark(payload.get("items") or [], config.artifacts_path / "benchmarks", source=str(payload.get("from_runs") or payload.get("source") or "unspecified"), overwrite=True)
        return _normalize_result(result, function_name)

    if function_name == "reflect":
        from dataevol.evolve import detect_opportunities, save_learning_opportunities
        from dataevol.evolve.context import load_evolution_context

        traces = payload.get("traces")
        if traces is None and payload.get("run_id") is not None:
            traces = load_evolution_context(config.db_path, _parse_run_id(payload["run_id"]))["traces"]
        opportunities = detect_opportunities(traces or [])
        path = save_learning_opportunities(opportunities, config.artifacts_path / "evolution")
        return _normalize_result({"opportunities": opportunities, "path": str(path)}, function_name)

    if function_name == "idea_prd":
        from dataevol.evolve import generate_component_idea_prd, generate_idea_prd, save_idea_prd

        opportunity = payload.get("opportunity")
        if isinstance(opportunity, dict):
            component = str(payload.get("component") or "router")
            prd = generate_component_idea_prd(opportunity, component) if component in {"router", "prompt", "verifier", "local_model", "benchmark"} else generate_idea_prd(opportunity)
            path = save_idea_prd(prd, config.artifacts_path / "evolution" / "ideas", slug=str(opportunity.get("id") or "idea_prd"))
            return {"ok": True, "status": "completed", "operation": function_name, "prd": prd, "path": str(path)}
        return {
            "ok": False,
            "status": "error",
            "operation": function_name,
            "detail": "opportunity must be a JSON object",
        }

    if function_name == "synthetic_generate":
        from dataevol.synthetic import generate_synthetic_data

        return _normalize_result({"items": generate_synthetic_data(payload.get("traces") or [])}, function_name)

    if function_name == "experiment":
        from dataevol.experiments import create_rollback_snapshot, run_measured_router_policy_experiment, run_router_policy_experiment

        rollback = payload.get("rollback_snapshot")
        if not rollback:
            rollback = str(create_rollback_snapshot("router", "control", config.artifacts_path / "rollbacks"))
        metrics = payload.get("fixture_metrics")
        if metrics:
            result = run_router_policy_experiment(metrics, config.artifacts_path / "experiments", rollback_snapshot=str(rollback))
        else:
            run_id = _parse_run_id(payload["run_id"]) if payload.get("run_id") is not None else None
            result = run_measured_router_policy_experiment(
                config.db_path,
                config.artifacts_path / "experiments",
                run_id=run_id,
                rollback_snapshot=str(rollback),
                variant_provider=str(payload.get("variant_provider") or "openrouter"),
            )
        return _normalize_result(result, function_name)

    if function_name == "compare":
        from dataevol.experiments import compare_experiment

        return _normalize_result(compare_experiment(_load_experiment_payload(payload, config), config.artifacts_path / "experiments"), function_name)

    if function_name == "promote":
        from dataevol.promotion import PromotionRejected
        from dataevol.experiments import promote_experiment

        report = _load_experiment_payload(payload, config)
        try:
            return _normalize_result(promote_experiment(report, config.artifacts_path / "promotions"), function_name)
        except PromotionRejected as exc:
            return {
                "ok": False,
                "status": "rejected",
                "operation": function_name,
                "experiment_id": report.get("experiment_id"),
                "detail": str(exc),
            }

    if function_name == "reject":
        from dataevol.experiments import reject_experiment

        return _normalize_result(reject_experiment(_load_experiment_payload(payload, config), config.artifacts_path / "rejections"), function_name)

    if module_name in {"local_models", "local_model"}:
        from dataevol.local_models import (
            EXPERTS,
            evaluate_local_adapter,
            prepare_local_adapter_training,
            promote_local_adapter,
            run_local_adapter_training,
        )

        output = Path(payload.get("output") or config.artifacts_path / "local_models")
        experts_payload = payload.get("experts") or payload.get("expert")
        if isinstance(experts_payload, str):
            experts = (experts_payload,)
        elif experts_payload:
            experts = tuple(str(expert) for expert in experts_payload)
        else:
            experts = EXPERTS

        if function_name == "prepare":
            plan = prepare_local_adapter_training(
                output,
                python_bin=payload.get("python_bin") or "python",
                base_model=str(payload.get("base_model") or "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
                experts=experts,
                count=int(payload.get("count") or 24),
                iters=int(payload.get("iters") or 2),
            )
            return _normalize_result(plan.to_dict(), function_name)
        if function_name == "train":
            result = run_local_adapter_training(
                output,
                python_bin=payload.get("python_bin") or "python",
                base_model=str(payload.get("base_model") or "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
                experts=experts,
                count=int(payload.get("count") or 24),
                iters=int(payload.get("iters") or 2),
                execute=bool(payload.get("execute", False)),
                timeout=int(payload.get("timeout") or 1800),
            )
            return _normalize_result(result, function_name)
        if function_name == "evaluate":
            return _normalize_result(evaluate_local_adapter(payload.get("metrics") or payload), function_name)
        if function_name == "promote":
            evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else payload
            return _normalize_result({"path": str(promote_local_adapter(evaluation, output)), "promoted": True}, function_name)

    if module_name == "reports":
        from dataevol.reports import build_report_payload, export_markdown_report, list_benchmarks, list_datasets, list_experiments, list_runs

        if function_name == "runs":
            return _normalize_result({"runs": list_runs(config.db_path)}, function_name)
        if function_name == "datasets":
            return _normalize_result({"datasets": list_datasets(config.db_path)}, function_name)
        if function_name == "benchmarks":
            return _normalize_result({"benchmarks": list_benchmarks(config.db_path)}, function_name)
        if function_name == "experiments":
            return _normalize_result({"experiments": list_experiments(config.db_path)}, function_name)
        if function_name in {"inbox", "markdown"}:
            report = build_report_payload(config.db_path, config.artifacts_path)
            if function_name == "markdown":
                path = export_markdown_report(report, config.artifacts_path / "reports" / "dataevol_report.md")
                report["markdown_path"] = str(path)
            return _normalize_result(report, function_name)

    return None


def call_core(
    module_name: str,
    function_name: str,
    payload: dict[str, Any] | None = None,
    *,
    config: DataEvolConfig,
) -> dict[str, Any]:
    """Dispatch CLI/API operations to implemented core helpers with structured errors."""
    payload = payload or {}
    try:
        known = _call_known_operation(module_name, function_name, payload, config)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        known = {
            "ok": False,
            "status": "error",
            "operation": function_name,
            "detail": str(exc),
            "payload": payload,
        }
    if known is not None:
        return known
    module_path = f"dataevol.{module_name}"
    try:
        module = importlib.import_module(module_path)
        helper: Callable[..., Any] = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        return {
            "ok": False,
            "status": "not_implemented",
            "operation": function_name,
            "module": module_path,
            "detail": f"Core helper unavailable: {exc}",
            "payload": payload,
        }

    try:
        result = helper(payload, config=config)
    except TypeError:
        result = helper(**payload)
    except Exception as exc:  # pragma: no cover - defensive shell behavior
        return {
            "ok": False,
            "status": "error",
            "operation": function_name,
            "module": module_path,
            "detail": str(exc),
            "payload": payload,
        }
    return _normalize_result(result, function_name)
