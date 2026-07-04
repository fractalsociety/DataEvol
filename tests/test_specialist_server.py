from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataevol.config import DataEvolConfig
from dataevol.local_models.layer_specialist import FREEZE_STRATEGY, SCHEMA, _flatten_params, _layer_parameters
from dataevol.specialist_server.app import create_server_app
from dataevol.specialist_server.swapper import MlxLayerSwapper, sha256_file


def test_specialist_server_health_and_auth(tmp_path: Path) -> None:
    client = TestClient(create_server_app(_cfg(tmp_path)))
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["model_loaded"] is False

    unauthorized = client.post("/model/load", json={"payload": {"base_model": "x"}})
    assert unauthorized.status_code == 401


def test_specialist_server_lifecycle_generate_and_chat_contract(tmp_path: Path) -> None:
    host = _FakeHost()
    client = TestClient(create_server_app(_cfg(tmp_path), host=host))
    headers = {"Authorization": "Bearer secret"}

    before_load = client.post("/generate", headers=headers, json={"payload": {"prompt": "x"}})
    assert before_load.status_code == 409
    assert "model not loaded" in before_load.text

    loaded = client.post("/model/load", headers=headers, json={"payload": {"base_model": "tiny-qwen"}})
    assert loaded.status_code == 200
    assert loaded.json()["model_loaded"] is True

    generated = client.post("/generate", headers=headers, json={"payload": {"prompt": "hello", "max_tokens": 4}})
    assert generated.status_code == 200
    assert generated.json()["text"] == "generated:hello"
    assert generated.json()["specialist"] == "base"

    chat = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "tiny-qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("generated:")
    assert body["usage"]["total_tokens"] == 3

    streaming = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "tiny-qwen", "stream": True, "messages": []},
    )
    assert streaming.status_code == 400
    assert "streaming is not supported" in streaming.text


def test_specialist_server_loads_manifest_and_tensors_from_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "schema": SCHEMA,
        "name": "tiny__compression__L1",
        "base_model_id": "tiny-qwen",
        "base_model_hash": None,
        "layer_index": 1,
        "task_type": "compression",
        "training_mode": "sft",
        "freeze_strategy": FREEZE_STRATEGY,
        "tensor_files": ["layer_1.safetensors"],
        "sha256": {"layer_1.safetensors": hashlib.sha256(b"tensor-bytes").hexdigest()},
        "trainable_param_names": [],
        "param_shapes": {},
    }
    payloads = {
        "http://artifact.test/manifest.json": json.dumps(manifest).encode("utf-8"),
        "http://artifact.test/layer_1.safetensors": b"tensor-bytes",
    }

    def fake_urlopen(request, timeout):  # noqa: ANN001
        url = request.full_url
        return _FakeResponse(payloads[url])

    monkeypatch.setattr("dataevol.specialist_server.app.urlopen", fake_urlopen)
    host = _FakeHost()
    host.swapper = _RecordingSwapper()
    client = TestClient(create_server_app(_cfg(tmp_path), host=host))
    loaded = client.post(
        "/specialists/load",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {
            "manifest_url": "http://artifact.test/manifest.json",
            "tensor_urls": {"layer_1.safetensors": "http://artifact.test/layer_1.safetensors"},
        }},
    )
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["ok"] is True
    assert body["verified"] is True
    assert body["manifest_path"].endswith("manifest.json")
    registered_path = Path(body["manifest_path"])
    assert registered_path.exists()
    assert (registered_path.parent / "layer_1.safetensors").read_bytes() == b"tensor-bytes"


def test_mlx_swapper_validates_activates_restores_and_switches_layers(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    model = _tiny_model()
    swapper = MlxLayerSwapper(model, base_model_id="tiny-qwen", base_model_hash=None, num_layers=2)
    before = _digests(mx, model)

    l1_manifest = _write_specialist(tmp_path, mx, model, layer_index=1, name="tiny__compression__L1")
    loaded = swapper.register(l1_manifest)
    assert loaded.verified is True
    assert loaded.layer_index == 1

    swapper.activate("tiny__compression__L1")
    active = _digests(mx, model)
    changed = [name for name, digest in active.items() if before[name] != digest]
    assert changed
    assert all(".layers.1." in name for name in changed)

    swapper.restore_base()
    assert _digests(mx, model) == before

    l0_manifest = _write_specialist(tmp_path, mx, model, layer_index=0, name="tiny__compression__L0")
    swapper.register(l0_manifest)
    swapper.activate("tiny__compression__L1")
    swapper.activate("tiny__compression__L0")
    after_switch = _digests(mx, model)
    changed_after_switch = [name for name, digest in after_switch.items() if before[name] != digest]
    assert changed_after_switch
    assert all(".layers.0." in name for name in changed_after_switch)
    assert swapper.active == "tiny__compression__L0"


def test_mlx_swapper_rejects_bad_manifest_and_tensor_shape(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    model = _tiny_model()
    swapper = MlxLayerSwapper(model, base_model_id="tiny-qwen", base_model_hash=None, num_layers=2)

    bad_schema = _write_specialist(tmp_path, mx, model, layer_index=1, name="bad-schema", schema="wrong")
    with pytest.raises(ValueError, match="unsupported schema"):
        swapper.register(bad_schema)

    bad_shape = _write_specialist(tmp_path, mx, model, layer_index=1, name="bad-shape", wrong_shape=True)
    with pytest.raises(ValueError, match="incompatible"):
        swapper.register(bad_shape)


def _tiny_model():
    nn = pytest.importorskip("mlx.nn")

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [nn.Linear(4, 4), nn.Linear(4, 4)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

        def __call__(self, x):
            return self.model(x)

    return TinyModel()


def _write_specialist(
    root: Path,
    mx,
    model,
    *,
    layer_index: int,
    name: str,
    schema: str = SCHEMA,
    wrong_shape: bool = False,
) -> Path:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    params = {
        key: (mx.ones((1,), dtype=value.dtype) if wrong_shape and idx == 0 else value + 0.01)
        for idx, (key, value) in enumerate(_layer_parameters(model, layer_index).items())
    }
    tensor_name = f"layer_{layer_index}.safetensors"
    mx.save_safetensors(str(out / tensor_name), params)
    manifest = {
        "schema": schema,
        "name": name,
        "base_model_id": "tiny-qwen",
        "base_model_hash": None,
        "layer_index": layer_index,
        "task_type": "compression",
        "training_mode": "sft",
        "freeze_strategy": FREEZE_STRATEGY,
        "tensor_files": [tensor_name],
        "sha256": {tensor_name: sha256_file(out / tensor_name)},
        "trainable_param_names": list(params),
        "param_shapes": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in params.items()},
    }
    manifest_path = out / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _digests(mx, model) -> dict[str, str]:
    return {name: hashlib.sha256(bytes(memoryview(value.view(mx.uint8)))).hexdigest() for name, value in _flatten_params(model.parameters())}


class _FakeHost:
    def __init__(self) -> None:
        self.model = None
        self.swapper = None
        self.base_model = None

    def status(self) -> dict[str, object]:
        return {
            "base_model": self.base_model,
            "model_loaded": self.model is not None,
            "num_layers": 2 if self.model else 0,
            "dequantized_layers": [0, 1] if self.model else [],
            "specialists_loaded": 0,
            "active_specialist": None,
            "active_layer_index": None,
        }

    def load(self, base_model: str, *, max_specialists: int = 8) -> dict[str, object]:
        self.model = object()
        self.base_model = base_model
        return self.status()

    def unload(self) -> dict[str, object]:
        self.model = None
        self.base_model = None
        return self.status()

    def generate(self, prompt: str, *, max_tokens: int = 128, temperature: float = 0.0) -> dict[str, object]:
        return {
            "text": f"generated:{prompt}",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "latency_ms": 3,
        }


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._body):
            return b""
        if size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


class _RecordingSwapper:
    def register(self, manifest_path: Path):
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        tensor = path.parent / manifest["tensor_files"][0]
        assert tensor.exists()

        class Loaded:
            def __init__(self) -> None:
                self.name = manifest["name"]
                self.manifest_path = str(path)
                self.manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                self.layer_index = manifest["layer_index"]
                self.task_type = manifest["task_type"]
                self.base_model_id = manifest["base_model_id"]
                self.tensor_bytes = tensor.stat().st_size
                self.verified = True
                self.loaded_at = "2026-07-03T00:00:00+00:00"
                self.active = False
                self.dtype_cast = False

        return Loaded()

    def list(self) -> list[dict[str, object]]:
        return []


def _cfg(tmp_path: Path) -> DataEvolConfig:
    return DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / ".dataevol/dataevol.sqlite3",
        raw_path=tmp_path / ".dataevol/raw",
        artifacts_path=tmp_path / ".dataevol/artifacts",
        api_token="secret",
    )
