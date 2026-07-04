from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

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
    def model_load(request: OperationRequest) -> dict[str, Any]:
        locked()
        try:
            base_model = str(request.payload.get("base_model") or request.payload.get("model") or "mlx-community/Qwen3-0.6B-4bit")
            return {"ok": True, **model_host.load(base_model, max_specialists=int(request.payload.get("max_specialists") or 8))}
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            state_lock.release()

    @app.post("/model/unload", dependencies=[protected])
    def model_unload() -> dict[str, Any]:
        locked()
        try:
            return {"ok": True, **model_host.unload()}
        finally:
            state_lock.release()

    @app.post("/specialists/load", dependencies=[protected])
    def specialists_load(request: OperationRequest) -> dict[str, Any]:
        if model_host.swapper is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            manifest_path = _guard_manifest_path(str(request.payload.get("manifest_path") or ""), cfg)
            record = model_host.swapper.register(manifest_path)
            return {"ok": True, **record.__dict__}
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
    def specialists_unload(request: OperationRequest) -> dict[str, Any]:
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
    def route_activate(request: OperationRequest) -> dict[str, Any]:
        if model_host.swapper is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            name = request.payload.get("name")
            record = model_host.swapper.activate(str(name)) if name else model_host.swapper.activate_for_task(str(request.payload.get("task_type") or ""))
            return {"ok": True, "activated": record.name if record else None, "fallback": "base" if record is None else None}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            state_lock.release()

    @app.post("/routes/deactivate", dependencies=[protected])
    def route_deactivate() -> dict[str, Any]:
        if model_host.swapper is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            model_host.swapper.restore_base()
            return {"ok": True, "active_specialist": None}
        finally:
            state_lock.release()

    @app.post("/generate", dependencies=[protected])
    def generate(request: OperationRequest) -> dict[str, Any]:
        if model_host.model is None:
            raise HTTPException(status_code=409, detail="model not loaded")
        locked()
        try:
            selected = _select_specialist(model_host, request.payload)
            body = model_host.generate(
                str(request.payload.get("prompt") or ""),
                max_tokens=int(request.payload.get("max_tokens") or 128),
                temperature=float(request.payload.get("temperature") or 0.0),
            )
            active = model_host.swapper.registry.get(model_host.swapper.active) if model_host.swapper and model_host.swapper.active else None
            if request.payload.get("restore_after") is True and model_host.swapper:
                model_host.swapper.restore_base()
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
    def chat_completions(request: dict[str, Any]) -> dict[str, Any]:
        if request.get("stream") is True:
            raise HTTPException(status_code=400, detail="streaming is not supported in phase 2")
        messages = request.get("messages") or []
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages if isinstance(m, dict))
        result = generate(OperationRequest(payload={
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
    if payload.get("specialist"):
        return host.swapper.activate(str(payload["specialist"])).name
    if payload.get("task_type"):
        record = host.swapper.activate_for_task(str(payload["task_type"]))
        return record.name if record else "base"
    return host.swapper.active or "base"


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
