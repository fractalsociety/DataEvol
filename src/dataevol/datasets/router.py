from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_PRIVACY_MODE = "private-local-only"
DEFAULT_PRIVACY_STATUS = "local_only"


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset_path: Path
    manifest_path: Path
    item_count: int
    dataset_hash: str


class RouterDatasetBuilder:
    """Builds local router training candidates from normalized trace mappings."""

    def __init__(self, clock: Any | None = None) -> None:
        self._clock = clock

    def build(
        self,
        traces: Iterable[Mapping[str, Any]],
        output_dir: str | Path,
        *,
        dataset_name: str = "router_dataset",
        version: str = "v0",
        privacy_mode: str = DEFAULT_PRIVACY_MODE,
        source_description: str = "json-fallback",
    ) -> DatasetBuildResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dataset_path = out / f"{dataset_name}.jsonl"
        manifest_path = out / f"{dataset_name}.manifest.json"

        items = [self._candidate_from_trace(trace, idx) for idx, trace in enumerate(traces, start=1)]
        with dataset_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

        dataset_hash = _sha256_file(dataset_path)
        manifest = {
            "name": dataset_name,
            "dataset_type": "router",
            "version": version,
            "created_at": self._now(),
            "item_count": len(items),
            "path": str(dataset_path),
            "sha256": dataset_hash,
            "privacy": {
                "mode": privacy_mode,
                "default_status": DEFAULT_PRIVACY_STATUS,
                "contains_raw_user_data": False,
                "public_export_allowed": privacy_mode == "public-benchmark-contribution",
                "redaction": "assumed-upstream-or-compact-fields-only",
            },
            "provenance": {
                "source": source_description,
                "source_run_ids": sorted({str(item["source_run_id"]) for item in items}),
                "builder": "dataevol.datasets.router.RouterDatasetBuilder",
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return DatasetBuildResult(dataset_path, manifest_path, len(items), dataset_hash)

    def _candidate_from_trace(self, trace: Mapping[str, Any], index: int) -> dict[str, Any]:
        metrics = trace.get("metrics") or {}
        decision = trace.get("decision") or trace.get("router_decision") or {}
        task = trace.get("task") or trace.get("objective") or trace.get("prompt") or "Unspecified router task"
        label = trace.get("label") or trace.get("outcome") or "inconclusive"
        trace_id = str(trace.get("id") or trace.get("trace_id") or f"trace_{index:03d}")
        privacy_status = trace.get("privacy_status") or DEFAULT_PRIVACY_STATUS
        return {
            "id": f"router_candidate_{index:03d}",
            "source_trace_id": trace_id,
            "source_run_id": str(trace.get("run_id") or trace.get("external_run_id") or "unknown_run"),
            "task": task,
            "input": {
                "task_type": trace.get("task_type", "unknown"),
                "constraints": trace.get("constraints", {}),
                "available_workers": trace.get("available_workers", []),
            },
            "output": {
                "worker": decision.get("worker") or trace.get("worker") or trace.get("agent_id"),
                "provider": decision.get("provider") or trace.get("provider"),
                "model": decision.get("model") or trace.get("model"),
                "reason": decision.get("reason") or trace.get("why_good") or trace.get("notes"),
            },
            "label": label,
            "score": float(trace.get("score", metrics.get("quality", 0.0)) or 0.0),
            "use_for": ["router"],
            "provider": decision.get("provider") or trace.get("provider"),
            "model": decision.get("model") or trace.get("model"),
            "cost_usd": float(metrics.get("cost_usd", trace.get("cost_usd", 0.0)) or 0.0),
            "latency_ms": int(metrics.get("latency_ms", trace.get("latency_ms", 0)) or 0),
            "privacy_status": privacy_status,
            "why_good": trace.get("why_good"),
            "failure_notes": trace.get("failure_notes"),
            "provenance": {
                "source_system": trace.get("source_system", "unknown"),
                "trace_type": trace.get("trace_type", "router_trace"),
                "created_at": trace.get("created_at"),
            },
        }

    def _now(self) -> str:
        if self._clock:
            return self._clock().isoformat()
        return datetime.now(timezone.utc).isoformat()


def build_router_dataset(
    traces: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    **kwargs: Any,
) -> DatasetBuildResult:
    return RouterDatasetBuilder().build(traces, output_dir, **kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
