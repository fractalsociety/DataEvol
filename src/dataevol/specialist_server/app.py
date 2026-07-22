from __future__ import annotations

import os
import hashlib
import json
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from dataevol import __version__
from dataevol.api.auth import require_token
from dataevol.config import DataEvolConfig, load_config
from dataevol.specialist_server.model_host import ModelHost


class OperationRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


def create_server_app(config: DataEvolConfig | None = None, host: ModelHost | None = None) -> FastAPI:
    cfg = config or load_config()
    protected = Depends(require_token(cfg))
    app = FastAPI(title="DataEvol Specialist Server", version=__version__)
    state_lock = threading.Lock()
    model_host = host or ModelHost()

    def locked():
        timeout = max(1.0, float(os.environ.get("SPECIALIST_SERVER_LOCK_TIMEOUT_S", "120")))
        acquired = state_lock.acquire(timeout=timeout)
        if not acquired:
            raise HTTPException(status_code=409, detail="specialist server is busy")
        return acquired

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "dataevol-specialist-server",
            "version": __version__,
            "schema": "dataevol.specialist_server_status.v1",
            **model_host.status(),
        }

    @app.post("/model/load", dependencies=[protected])
    async def model_load(request: OperationRequest) -> dict[str, Any]:
        locked()
        try:
            base_model = str(request.payload.get("base_model") or request.payload.get("model") or "mlx-community/Qwen3-0.6B-4bit")
            revision = str(request.payload.get("base_model_revision") or "").strip() or None
            load_options: dict[str, Any] = {"max_specialists": int(request.payload.get("max_specialists") or 8)}
            if revision:
                load_options["base_model_revision"] = revision
            return {"ok": True, **model_host.load(base_model, **load_options)}
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            state_lock.release()

    @app.post("/model/unload", dependencies=[protected])
    async def model_unload() -> dict[str, Any]:
        locked()
        try:
            return {"ok": True, **model_host.unload()}
        finally:
            state_lock.release()

    @app.post("/specialists/load", dependencies=[protected])
    async def specialists_load(request: OperationRequest) -> dict[str, Any]:
        if model_host.swapper is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            manifest_path = _specialist_manifest_path(request.payload, cfg)
            record = model_host.swapper.register(manifest_path)
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            return {"ok": True, **record.__dict__, "manifest": manifest}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            state_lock.release()

    @app.get("/specialists", dependencies=[protected])
    def specialists_list() -> list[dict[str, Any]]:
        return model_host.swapper.list() if model_host.swapper else []

    @app.post("/specialists/unload", dependencies=[protected])
    async def specialists_unload(request: OperationRequest) -> dict[str, Any]:
        if model_host.swapper is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            model_host.swapper.unload(str(request.payload.get("name") or ""))
            return {"ok": True, "specialists": model_host.swapper.list()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            state_lock.release()

    @app.post("/routes/activate", dependencies=[protected])
    async def route_activate(request: OperationRequest) -> dict[str, Any]:
        if model_host.swapper is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            name = request.payload.get("name")
            record = _activate_bound_specialist(model_host, request.payload, name=str(name) if name else None)
            binding = model_host.swapper.active_binding if model_host.swapper else None
            return {
                "ok": True,
                "activated": record.name if record else None,
                "fallback": "base" if record is None else None,
                "binding": binding,
                **(binding or {}),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            state_lock.release()

    @app.post("/routes/deactivate", dependencies=[protected])
    async def route_deactivate() -> dict[str, Any]:
        if model_host.swapper is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            model_host.swapper.restore_base()
            return {"ok": True, "active_specialist": None}
        finally:
            state_lock.release()

    @app.post("/generate", dependencies=[protected])
    async def generate(request: OperationRequest) -> dict[str, Any]:
        if model_host.model is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            selected = _select_specialist(model_host, request.payload)
            active = model_host.swapper.registry.get(model_host.swapper.active) if model_host.swapper and model_host.swapper.active else None
            body = model_host.generate(
                str(request.payload.get("prompt") or ""),
                max_tokens=int(request.payload.get("max_tokens") or 128),
                temperature=float(request.payload.get("temperature") or 0.0),
            )
            _post_trait_telemetry(model_host, request.payload, body, active)
            return {
                "ok": True,
                "schema": "dataevol.specialist_generation.v1",
                **body,
                "specialist": active.name if active else selected,
                "layer_index": active.layer_index if active else None,
                "manifest_hash": active.manifest_hash if active else None,
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            state_lock.release()

    @app.post("/v1/chat/completions", dependencies=[protected])
    async def chat_completions(request: dict[str, Any]) -> dict[str, Any]:
        if request.get("stream") is True:
            raise HTTPException(status_code=400, detail="streaming is not supported in phase 2")
        messages = request.get("messages") or []
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages if isinstance(m, dict))
        result = await generate(OperationRequest(payload={
            "prompt": prompt,
            "specialist": request.get("specialist"),
            "max_tokens": request.get("max_tokens") or 128,
            "temperature": request.get("temperature") or 0,
        }))
        return {
            "id": "chatcmpl_layerscope",
            "object": "chat.completion",
            "model": request.get("model") or model_host.base_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
            },
        }

    return app


def _select_specialist(host: ModelHost, payload: dict[str, Any]) -> str:
    if not host.swapper:
        return "base"
    requested_name = str(payload.get("specialist") or "").strip()
    requested_task = str(payload.get("task_type") or "").strip()
    if not host.swapper.active or not host.swapper.active_binding:
        if requested_name or requested_task:
            raise ValueError("specialist route has not been authority-activated")
        return "base"
    record = host.swapper.registry.get(host.swapper.active)
    if record is None:
        raise ValueError("active specialist is not registered")
    if requested_name and requested_name != record.name:
        raise ValueError("requested specialist does not match the authority-bound route")
    if requested_task and requested_task != record.task_type:
        raise ValueError("requested task_type does not match the authority-bound route")
    for field in ("genome_id", "candidate_content_hash", "harness_deployment_id", "verdict_hash"):
        value = payload.get(field)
        if value is not None and str(value) != str(host.swapper.active_binding.get(field) or ""):
            raise ValueError(f"{field} does not match the authority-bound route")
    return record.name


def _activate_bound_specialist(host: ModelHost, payload: dict[str, Any], *, name: str | None = None):
    if not host.swapper:
        raise RuntimeError("model not loaded")
    if name:
        if name not in host.swapper.registry:
            raise KeyError(f"unknown specialist: {name}")
        record = host.swapper.registry[name]
    else:
        task_type = str(payload.get("task_type") or "")
        record = next((item for item in host.swapper.registry.values() if item.task_type == task_type), None)
        if record is None:
            host.swapper.restore_base()
            return None
    expected_genome_id = str(payload.get("genome_id") or "")
    expected_content_hash = str(payload.get("candidate_content_hash") or "").lower()
    deployment_id = str(payload.get("harness_deployment_id") or "").strip()
    verdict_hash = str(payload.get("verdict_hash") or "").lower()
    if not expected_genome_id or not expected_content_hash or not deployment_id or not verdict_hash:
        raise ValueError("genome_id, candidate_content_hash, harness_deployment_id, and verdict_hash are required for specialist activation")
    if len(expected_content_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_content_hash):
        raise ValueError("candidate_content_hash must be a 64-character hexadecimal digest")
    if len(verdict_hash) != 64 or any(ch not in "0123456789abcdef" for ch in verdict_hash):
        raise ValueError("verdict_hash must be a 64-character hexadecimal digest")
    if record.genome_id != expected_genome_id:
        raise ValueError("specialist genome_id mismatch")
    if record.candidate_content_hash != expected_content_hash:
        raise ValueError("specialist candidate_content_hash mismatch")
    activated = host.swapper.activate(record.name)
    host.swapper.active_binding = {
        "genome_id": record.genome_id,
        "candidate_content_hash": record.candidate_content_hash,
        "harness_deployment_id": deployment_id,
        "verdict_hash": verdict_hash,
    }
    return activated


def _guard_manifest_path(raw: str, cfg: DataEvolConfig) -> Path:
    if not raw:
        raise ValueError("manifest_path is required")
    path = Path(raw).expanduser().resolve()
    roots = [Path(".dataevol").resolve(), cfg.artifacts_path.resolve()]
    for extra in os.environ.get("SPECIALIST_ARTIFACT_ROOTS", "").split(":"):
        if extra.strip():
            roots.append(Path(extra).expanduser().resolve())
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("manifest_path is outside allowed artifact roots")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _specialist_manifest_path(payload: dict[str, Any], cfg: DataEvolConfig) -> Path:
    manifest_url = str(payload.get("manifest_url") or "").strip()
    if not manifest_url:
        return _guard_manifest_path(str(payload.get("manifest_path") or ""), cfg)
    return _download_specialist_bundle(manifest_url, payload, cfg)


def _download_specialist_bundle(manifest_url: str, payload: dict[str, Any], cfg: DataEvolConfig) -> Path:
    cache_root = cfg.artifacts_path / "layerscope_remote_specialists" / hashlib.sha256(manifest_url.encode("utf-8")).hexdigest()[:24]
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    _download_url(manifest_url, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_urls = payload.get("tensor_urls") or {}
    if isinstance(tensor_urls, list):
        tensor_urls = {Path(str(urlparse(str(url)).path)).name: str(url) for url in tensor_urls}
    if not isinstance(tensor_urls, dict):
        raise ValueError("tensor_urls must be an object mapping tensor file names to URLs")
    single_tensor_url = str(payload.get("tensor_url") or "").strip()
    tensor_files = [str(item) for item in manifest.get("tensor_files") or []]
    if single_tensor_url and len(tensor_files) == 1 and tensor_files[0] not in tensor_urls:
        tensor_urls[tensor_files[0]] = single_tensor_url
    for tensor_file in tensor_files:
        rel = _safe_relative_tensor_path(tensor_file)
        url = str(tensor_urls.get(tensor_file) or tensor_urls.get(rel.name) or "").strip()
        if not url:
            raise ValueError(f"missing tensor URL for {tensor_file}")
        _download_url(url, cache_root / rel)
    return manifest_path


def _safe_relative_tensor_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe tensor file path: {raw}")
    return path


def _download_url(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http(s) artifact URLs are supported")
    max_bytes = int(os.environ.get("SPECIALIST_REMOTE_ARTIFACT_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))
    request = Request(url, headers={"User-Agent": "dataevol-specialist-server/1"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with urlopen(request, timeout=float(os.environ.get("SPECIALIST_REMOTE_ARTIFACT_TIMEOUT_S", "120"))) as response, destination.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("remote specialist artifact exceeds size limit")
            fh.write(chunk)


def _post_trait_telemetry(model_host: ModelHost, payload: dict[str, Any], generation: dict[str, Any], active: Any | None) -> None:
    if os.environ.get("TELEMETRY_TRAITS") not in {"1", "true", "TRUE", "yes"}:
        return
    endpoint = os.environ.get("TRAIT_EVIDENCE_URL") or os.environ.get("FRACTALWORK_TRAIT_EVIDENCE_URL")
    if not endpoint:
        return
    layer_index = active.layer_index if active else payload.get("layer_index")
    num_layers = max(1, int(getattr(model_host, "num_layers", 1) or 1))
    try:
        layer = int(layer_index) if layer_index is not None else 0
    except (TypeError, ValueError):
        layer = 0
    buckets = max(1, int(os.environ.get("TRAIT_REGION_BUCKETS", "16")))
    depth_bucket = max(0, min(buckets - 1, round((layer / max(1, num_layers - 1)) * (buckets - 1))))
    soul_id = payload.get("soul_id") or payload.get("soulId") or payload.get("agent_id") or payload.get("agentId")
    if not soul_id and active:
        soul_id = active.name
    event = {
        "soulId": str(soul_id or "specialist-server"),
        "genomeHash": str(payload.get("genome_hash") or payload.get("genomeHash") or (active.manifest_hash if active else _hash_text(str(payload.get("model") or model_host.base_model or "base")))),
        "epoch": int(payload.get("epoch") or payload.get("life_epoch") or 0),
        "taskType": str(payload.get("task_type") or payload.get("taskType") or (active.task_type if active else "generation")),
        "depthBucket": depth_bucket,
        "source": "telemetry",
        "value": _telemetry_value(generation),
        "subRegion": None,
    }
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("FRACTALWORK_TRAIT_EVIDENCE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = Request(endpoint, data=json.dumps({"events": [event]}).encode("utf-8"), headers=headers, method="POST")
        with urlopen(req, timeout=float(os.environ.get("TRAIT_EVIDENCE_TIMEOUT_S", "2"))) as response:
            response.read()
    except Exception:
        return


def _telemetry_value(generation: dict[str, Any]) -> float:
    token_count = generation.get("completion_tokens") or generation.get("tokens") or 1
    try:
        return max(0.0, min(1.0, float(token_count) / 256.0))
    except (TypeError, ValueError):
        return 0.01


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
