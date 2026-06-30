from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib import request


def coordinate_completion_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    return {"source_system": "coordinate", "run": dict(run), "event": "run_completed"}


def post_coordinate_completion(endpoint: str, run: Mapping[str, Any], *, token: str | None = None, timeout: int = 30) -> dict[str, Any]:
    payload = json.dumps(coordinate_completion_payload(run)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(endpoint, data=payload, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {"status": "sent"}


def router_dataset_pull(dataset_manifest: str | Path, *, endpoint: str | None = None, token: str | None = None, timeout: int = 30) -> dict[str, Any]:
    if endpoint:
        source = endpoint.rstrip("/") + "/datasets/router"
        req = request.Request(source, headers=_auth_headers(token), method="GET")
        with request.urlopen(req, timeout=timeout) as response:
            manifest = json.loads(response.read().decode("utf-8"))
    else:
        text_source = str(dataset_manifest)
        if text_source.startswith(("http://", "https://")):
            req = request.Request(text_source, headers=_auth_headers(token), method="GET")
            with request.urlopen(req, timeout=timeout) as response:
                manifest = json.loads(response.read().decode("utf-8"))
        else:
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


def _auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}
