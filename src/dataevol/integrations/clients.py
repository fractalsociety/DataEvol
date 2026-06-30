from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def coordinate_completion_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    return {"source_system": "coordinate", "run": dict(run), "event": "run_completed"}


def router_dataset_pull(dataset_manifest: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    return {"consumer": "fractal-router-api", "dataset": manifest}


def biolatent_verification_payload(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {"source_system": "biolatent", "trace": {**dict(trace), "trace_type": trace.get("trace_type", "verification_trace")}}


OPENROUTER_MODEL_METADATA = {
    "provider": "openrouter",
    "cost_source": "trace metadata",
    "ranking": "cost-normalized quality score",
}

LOCAL_MODEL_METADATA = {
    "provider": "local",
    "fields": ["adapter", "base_model", "quantization", "benchmark_score"],
}
