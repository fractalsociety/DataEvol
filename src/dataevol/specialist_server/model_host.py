from __future__ import annotations

import time
from typing import Any

from dataevol.local_models.layer_specialist import _decoder_layers, model_fingerprint
from dataevol.specialist_server.swapper import MlxLayerSwapper, _ensure_mlx_thread_stream


class ModelHost:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.base_model: str | None = None
        self.base_model_revision: str | None = None
        self.base_model_fingerprint: dict[str, Any] | None = None
        self.num_layers = 0
        self.dequantized_module_count = 0
        self.swapper: MlxLayerSwapper | None = None

    def load(self, base_model: str, *, base_model_revision: str | None = None, max_specialists: int = 8) -> dict[str, Any]:
        if self.model is not None and self.base_model == base_model and self.base_model_revision == base_model_revision:
            return self.status()
        if self.model is not None and self.base_model != base_model:
            raise RuntimeError("different model already loaded; unload first")
        from mlx_lm import load

        fingerprint = model_fingerprint(base_model, base_model_revision=base_model_revision)
        load_kwargs = {"revision": fingerprint["resolved_revision"]} if fingerprint["kind"] == "remote_revision" else {}
        model, tokenizer = load(base_model, **load_kwargs)
        layers = _decoder_layers(model)
        if layers is None:
            raise RuntimeError("loaded model does not expose decoder layers")
        swapper = MlxLayerSwapper(
            model,
            base_model_id=base_model,
            base_model_hash=str(fingerprint["sha256"]),
            base_model_revision=fingerprint.get("resolved_revision"),
            num_layers=len(layers),
            max_specialists=max_specialists,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.dequantized_module_count = 0
        self.num_layers = len(layers)
        self.base_model = base_model
        self.base_model_revision = fingerprint.get("resolved_revision")
        self.base_model_fingerprint = fingerprint
        self.swapper = swapper
        return self.status()

    def unload(self) -> dict[str, Any]:
        if self.swapper:
            self.swapper.restore_base()
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self.base_model_revision = None
        self.base_model_fingerprint = None
        self.num_layers = 0
        self.dequantized_module_count = 0
        self.swapper = None
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "base_model_hash": self.base_model_fingerprint.get("sha256") if self.base_model_fingerprint else None,
            "model_loaded": self.model is not None,
            "num_layers": self.num_layers,
            "dequantized_layers": sorted(self.swapper.dequantized_layers) if self.swapper else [],
            "dequantized_module_count": self.swapper.dequantized_module_count if self.swapper else 0,
            "specialists_loaded": len(self.swapper.registry) if self.swapper else 0,
            "active_specialist": self.swapper.active if self.swapper else None,
            "active_layer_index": self.swapper.registry[self.swapper.active].layer_index if self.swapper and self.swapper.active else None,
            "active_binding": self.swapper.active_binding if self.swapper else None,
        }

    def generate(self, prompt: str, *, max_tokens: int = 128, temperature: float = 0.0) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model not loaded")
        _ensure_mlx_thread_stream()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        started = time.time()
        sampler = make_sampler(temp=max(0.0, float(temperature)))
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max(1, int(max_tokens)),
            sampler=sampler,
            verbose=False,
        )
        elapsed = max(0, int((time.time() - started) * 1000))
        gpu_seconds = max(0.0, (elapsed / 1000.0) * float(__import__("os").environ.get("DATAEVOL_GPU_DEVICE_FACTOR") or 1.0))
        prompt_tokens = len(self.tokenizer.encode(prompt)) if hasattr(self.tokenizer, "encode") else 0
        completion_tokens = len(self.tokenizer.encode(text)) if hasattr(self.tokenizer, "encode") else 0
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": elapsed,
            "gpu_seconds": round(gpu_seconds, 6),
        }
