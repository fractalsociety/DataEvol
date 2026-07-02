from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .router import _sha256_file


DATA_PIPELINE_SPECIALIST_TYPES = {
    "ingestor",
    "deduper",
    "cleaner",
    "classifier",
    "difficulty-scorer",
    "quality-scorer",
    "trace-compressor",
    "early-failure-detector",
    "verifier",
    "critic",
    "error-taxonomist",
    "synthetic-generator",
    "mutation-evolver",
    "recipe-generator",
    "recipe-verifier",
    "curriculum-builder",
    "dataset-mixer",
    "benchmark-generator",
    "regression-tester",
    "promotion-gatekeeper",
}

ECOSYSTEM_SPECIALIST_TYPES = {
    "router",
    "planner",
    "worker",
    "manager",
    "coordinator",
    "inspector",
    "compressor",
    "duplicate-detector",
    "failure-classifier",
    "prompt-improver",
    "prompt-pack-generator",
    "local-model-trainer",
    "local-evaluator",
    "model-mix-optimizer",
    "scientific-method",
    "coding-agent",
    "tool-trace-analyzer",
    "correction-linker",
    "benchmark-task",
    "privacy-redactor",
    "report-builder",
    "integration-bridge",
}

SPECIALIST_TYPES = DATA_PIPELINE_SPECIALIST_TYPES | ECOSYSTEM_SPECIALIST_TYPES

DATASET_TYPES = {
    "router",
    "critic",
    "verifier",
    "planner",
    "prompt",
    "trace-compression",
    "duplicate-detection",
    "failure-classification",
    "scientific-method",
    "coding-agent",
    "benchmark-task",
    "local-router",
    "local-compressor",
    "local-duplicate-detector",
    "local-evaluator",
    "local-scientific-method",
    "local-coding-agent",
    "local-benchmark-task",
} | SPECIALIST_TYPES | {f"local-{name}" for name in SPECIALIST_TYPES}


@dataclass(frozen=True)
class GenericDatasetResult:
    dataset_path: Path
    manifest_path: Path
    dataset_type: str
    item_count: int
    dataset_hash: str


def build_dataset(
    dataset_type: str,
    traces: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    version: str = "v0",
    privacy_mode: str = "private-local-only",
) -> GenericDatasetResult:
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"unknown dataset type: {dataset_type}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = dataset_type.replace("-", "_")
    dataset_path = out / f"{slug}_dataset.jsonl"
    manifest_path = out / f"{slug}_dataset.manifest.json"
    rows = [_row_for_dataset(dataset_type, dict(trace), index) for index, trace in enumerate(traces, start=1)]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    sha = _sha256_file(dataset_path)
    manifest = {
        "name": f"{slug}_dataset",
        "dataset_type": dataset_type,
        "version": version,
        "item_count": len(rows),
        "sha256": sha,
        "path": str(dataset_path),
        "privacy_mode": privacy_mode,
        "provenance": sorted({str(row.get("source_run_id", "unknown")) for row in rows}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return GenericDatasetResult(dataset_path, manifest_path, dataset_type, len(rows), sha)


def export_local_training_datasets(
    traces: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    opt_in: bool,
) -> dict[str, GenericDatasetResult]:
    if not opt_in:
        raise PermissionError("local training data export requires explicit opt-in")
    return {
        name: build_dataset(name, traces, output_dir)
        for name in sorted(f"local-{dataset_type}" for dataset_type in SPECIALIST_TYPES)
    }


def _row_for_dataset(dataset_type: str, trace: dict[str, Any], index: int) -> dict[str, Any]:
    base = {
        "id": f"{dataset_type.replace('-', '_')}_{index:03d}",
        "source_trace_id": trace.get("id") or trace.get("trace_id"),
        "source_run_id": trace.get("run_id") or trace.get("external_run_id") or "unknown",
        "privacy_status": trace.get("privacy_status", "local_only"),
        "input": trace.get("prompt") or trace.get("objective") or trace.get("task"),
        "output": trace.get("response") or trace.get("label") or trace.get("failure_type"),
        "label": trace.get("label") or trace.get("outcome") or "inconclusive",
        "score": trace.get("score") or trace.get("quality_score") or 0.0,
    }
    if dataset_type == "critic":
        base["target"] = "critique_trace_quality"
    elif dataset_type == "verifier":
        base["target"] = "verify_claim_or_task"
    elif dataset_type.endswith("ingestor"):
        base["target"] = "normalize_and_validate_trace"
    elif dataset_type.endswith("deduper"):
        base["target"] = "detect_duplicate_trace"
    elif dataset_type.endswith("cleaner"):
        base["target"] = "clean_trace_for_training"
    elif dataset_type.endswith("classifier"):
        base["target"] = "classify_trace_outcome"
    elif dataset_type.endswith("difficulty-scorer"):
        base["target"] = "score_task_difficulty"
    elif dataset_type.endswith("quality-scorer"):
        base["target"] = "score_trace_quality"
    elif dataset_type.endswith("trace-compressor"):
        base["target"] = "compress_trace"
    elif dataset_type.endswith("early-failure-detector"):
        base["target"] = "detect_early_failure_signal"
    elif dataset_type.endswith("error-taxonomist"):
        base["target"] = "taxonomy_label_error"
    elif dataset_type.endswith("synthetic-generator"):
        base["target"] = "generate_synthetic_training_case"
    elif dataset_type.endswith("mutation-evolver"):
        base["target"] = "mutate_case_for_improvement"
    elif dataset_type.endswith("recipe-generator"):
        base["target"] = "generate_dataset_recipe"
    elif dataset_type.endswith("recipe-verifier"):
        base["target"] = "verify_dataset_recipe"
    elif dataset_type.endswith("curriculum-builder"):
        base["target"] = "build_training_curriculum"
    elif dataset_type.endswith("dataset-mixer"):
        base["target"] = "mix_dataset_slices"
    elif dataset_type.endswith("benchmark-generator"):
        base["target"] = "generate_benchmark_case"
    elif dataset_type.endswith("regression-tester"):
        base["target"] = "test_candidate_for_regressions"
    elif dataset_type.endswith("promotion-gatekeeper"):
        base["target"] = "apply_promotion_gate"
    elif dataset_type.endswith("router"):
        base["target"] = "optimize_router_assignment"
    elif dataset_type.endswith("worker"):
        base["target"] = "execute_worker_task"
    elif dataset_type.endswith("manager"):
        base["target"] = "manage_agent_workflow"
    elif dataset_type.endswith("coordinator"):
        base["target"] = "coordinate_prd_execution"
    elif dataset_type.endswith("inspector"):
        base["target"] = "inspect_code_or_output"
    elif dataset_type.endswith("compressor"):
        base["target"] = "compress_trace"
    elif dataset_type.endswith("duplicate-detector"):
        base["target"] = "detect_duplicate_trace"
    elif dataset_type.endswith("failure-classifier"):
        base["target"] = "classify_failure"
    elif dataset_type.endswith("prompt-improver"):
        base["target"] = "improve_prompt"
    elif dataset_type.endswith("prompt-pack-generator"):
        base["target"] = "generate_prompt_pack"
    elif dataset_type.endswith("local-model-trainer"):
        base["target"] = "plan_local_adapter_training"
    elif dataset_type.endswith("model-mix-optimizer"):
        base["target"] = "optimize_model_mix"
    elif dataset_type.endswith("tool-trace-analyzer"):
        base["target"] = "analyze_tool_trace"
    elif dataset_type.endswith("correction-linker"):
        base["target"] = "link_failure_to_correction"
    elif dataset_type.endswith("privacy-redactor"):
        base["target"] = "redact_private_training_data"
    elif dataset_type.endswith("report-builder"):
        base["target"] = "build_operator_report"
    elif dataset_type.endswith("integration-bridge"):
        base["target"] = "shape_integration_payload"
    elif dataset_type == "planner":
        base["target"] = "decompose_task"
    elif dataset_type == "prompt":
        base["target"] = "improve_prompt"
    elif dataset_type == "trace-compression":
        base["target"] = "compress_trace"
    elif dataset_type == "duplicate-detection":
        base["target"] = "classify_duplicate"
    elif dataset_type == "failure-classification":
        base["target"] = "classify_failure"
    elif dataset_type in {"scientific-method", "local-scientific-method"}:
        base["target"] = "design_scientific_evaluation"
    elif dataset_type in {"coding-agent", "local-coding-agent"}:
        base["target"] = "improve_coding_agent_workflow"
    elif dataset_type in {"benchmark-task", "local-benchmark-task"}:
        base["target"] = "generate_or_score_benchmark_task"
    elif dataset_type == "local-evaluator":
        base["target"] = "evaluate_local_model_output"
    return base
