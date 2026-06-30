from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .router import _sha256_file


DATASET_TYPES = {
    "router",
    "critic",
    "verifier",
    "planner",
    "prompt",
    "trace-compression",
    "duplicate-detection",
    "failure-classification",
    "local-router",
    "local-compressor",
    "local-duplicate-detector",
    "local-evaluator",
}


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
        for name in ("local-router", "local-compressor", "local-duplicate-detector", "local-evaluator")
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
    return base
