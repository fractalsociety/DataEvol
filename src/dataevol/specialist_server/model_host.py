from __future__ import annotations

import time
from typing import Any

from dataevol.local_models.layer_specialist import _decoder_layers, _dequantize_quantized_linears, model_hash
from dataevol.specialist_server.swapper import MlxLayerSwapper


class ModelHost:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.base_model: str | None = None
        self.num_layers = 0
        self.dequantized_module_count = 0
        self.swapper: MlxLayerSwapper | None = None

    def load(self, base_model: str, *, max_specialists: int = 8) -> dict[str, Any]:
        if self.model is not None and self.base_model == base_model:
            return self.status()
        if self.model is not None and self.base_model != base_model:
            raise RuntimeError("different model already loaded; unload first")
        from mlx_lm import load

        self.model, self.tokenizer = load(base_model)
        self.dequantized_module_count = _dequantize_quantized_linears(self.model)
        layers = _decoder_layers(self.model)
        if layers is None:
            raise RuntimeError("loaded model does not expose decoder layers")
        self.num_layers = len(layers)
        self.base_model = base_model
        self.swapper = MlxLayerSwapper(
            self.model,
            base_model_id=base_model,
            base_model_hash=model_hash(base_model),
            num_layers=self.num_layers,
            max_specialists=max_specialists,
        )
        return self.status()

    def unload(self) -> dict[str, Any]:
        if self.swapper:
            self.swapper.restore_base()
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self.num_layers = 0
        self.dequantized_module_count = 0
        self.swapper = None
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "model_loaded": self.model is not None,
            "num_layers": self.num_layers,
            "dequantized_layers": list(range(self.num_layers)) if self.model is not None else [],
            "dequantized_module_count": self.dequantized_module_count,
            "specialists_loaded": len(self.swapper.registry) if self.swapper else 0,
            "active_specialist": self.swapper.active if self.swapper else None,
            "active_layer_index": self.swapper.registry[self.swapper.active].layer_index if self.swapper and self.swapper.active else None,
        }

    def generate(self, prompt: str, *, max_tokens: int = 128, temperature: float = 0.0) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model not loaded")
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
