from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataevol.api.app import create_app
from dataevol.config import DataEvolConfig
from dataevol.local_models.layer_specialist import (
    _flatten_params,
    _layer_parameters,
    _train_loaded_model,
)


def test_layer_specialist_dry_run_and_validation(tmp_path: Path) -> None:
    client = TestClient(create_app(_cfg(tmp_path)))
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n", encoding="utf-8")
    payload = {
        "base_model": "mlx-community/Qwen3-0.6B-4bit",
        "output": str(tmp_path / "out"),
        "task_type": "compression",
        "training_mode": "sft",
        "layer_index": 14,
        "dataset_uri": str(dataset),
        "execute": False,
    }
    res = client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": payload})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["status"] == "planned"
    assert body["dry_run"] is True
    assert body["schema"] == "dataevol.mlx_layer_specialist.v1"
    assert body["layer_index"] == 14
    assert body["freeze_strategy"] == "mlx_full_layer"
    assert "dataevol.local_models.layer_specialist" in body["planned_command"]
    assert body["acceptance_criteria"]["trainable_layers"] == [14]

    invalid = dict(payload, layer_index=-1)
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, training_mode="bad")
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, dataset_uri=str(tmp_path / "missing.jsonl"))
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, dataset_uri="s3://bucket/train.jsonl")
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, output="../../../etc")
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, contribution=-0.3)
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, contribution=0.2, min_contribution=0.5)
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    assert client.post("/local_model/layerscope/train_layer_specialist", json={"payload": payload}).status_code == 401


def test_layer_specialist_artifact_export_adds_specialists_without_regressing_adapters(tmp_path: Path) -> None:
    client = TestClient(create_app(_cfg(tmp_path)))
    output = tmp_path / "out"
    adapter_dir = output / "adapters" / "ingestor"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_bytes(b"adapter")

    specialist_dir = output / "layerscope" / "layer_14"
    specialist_dir.mkdir(parents=True)
    manifest = {
        "schema": "dataevol.mlx_layer_specialist.v1",
        "name": "qwen3__compression__L14",
        "layer_index": 14,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (specialist_dir / "qwen3__compression__L14.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (specialist_dir / "layer_14.safetensors").write_bytes(b"layer-tensors")

    res = client.post(
        "/local_model/artifacts/export",
        headers=_auth(),
        json={"payload": {"output": str(output), "experts": ["ingestor"]}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["schema"] == "dataevol.local_model_artifact_export.v1"
    assert body["adapters"]["ingestor"]["exists"] is True
    assert body["specialists"]["layer_14"]["exists"] is True
    assert body["specialists"]["layer_14"]["manifest"]["content_base64"]
    assert body["specialists"]["layer_14"]["tensors"][0]["relative_path"] == "layerscope/layer_14/layer_14.safetensors"
    assert body["file_count"] == 3
    assert len(body["payload_hash"]) == 64


def test_layer_specialist_core_trains_only_one_layer_and_exports_mlx_safetensors(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 16 for ch in text][:12] or [1, 2]

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [nn.Linear(4, 4), nn.Linear(4, 4)]

        def __call__(self, x):
            for layer in self.layers:
                x = nn.relu(layer(x))
            return x

    class TinyLayerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.model = Inner()
            self.output = nn.Linear(4, 16)

        def __call__(self, x):
            return self.output(self.model(self.embed(x)))

    rows = [{"prompt": f"prompt {i}", "completion": f"answer {i}"} for i in range(6)]
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model = TinyLayerModel()
    before = {name: _array_digest(mx, value) for name, value in _flatten_params(model.parameters())}

    result = _train_loaded_model(
        model=model,
        tokenizer=Tokenizer(),
        base_model="tiny-mlx-model",
        layer_index=1,
        rows=rows,
        dataset_uri=str(dataset),
        output_dir=tmp_path / "out",
        task_type="compression",
        training_mode="sft",
        learning_rate=0.05,
        batch_size=2,
        max_steps=2,
        max_seq_len=12,
        eval_split=0.2,
    )

    after = {name: value for name, value in _flatten_params(model.parameters())}
    changed = [name for name, value in after.items() if before[name] != _array_digest(mx, value)]
    assert changed
    assert all("model.layers.1." in name for name in changed)

    exported = mx.load(Path(result["manifest_path"]).with_name("layer_1.safetensors"))
    layer_params = _layer_parameters(model, 1)
    assert exported.keys() == layer_params.keys()
    for name, value in layer_params.items():
        assert bool(mx.allclose(exported[name], value))

    manifest = result["manifest"]
    assert manifest["freeze_strategy"] == "mlx_full_layer"
    assert all("model.layers.1." in name for name in manifest["trainable_param_names"])
    total_numel = sum(_numel(value) for value in after.values())
    assert manifest["trainable_param_count"] + manifest["frozen_param_count"] == total_numel
    assert isinstance(manifest["baseline_metric"], float)
    assert isinstance(manifest["eval_metric"], float)

    with pytest.raises(NotImplementedError):
        _train_loaded_model(
            model=TinyLayerModel(),
            tokenizer=Tokenizer(),
            base_model="tiny-mlx-model",
            layer_index=1,
            rows=rows,
            dataset_uri=str(dataset),
            output_dir=tmp_path / "rl",
            task_type="compression",
            training_mode="rl",
        )


def _numel(value) -> int:
    n = 1
    for dim in getattr(value, "shape", ()):
        n *= int(dim)
    return n


def _array_digest(mx, value) -> str:
    return hashlib.sha256(bytes(memoryview(value.view(mx.uint8)))).hexdigest()


def _cfg(tmp_path: Path) -> DataEvolConfig:
    return DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / ".dataevol/dataevol.sqlite3",
        raw_path=tmp_path / ".dataevol/raw",
        artifacts_path=tmp_path / ".dataevol/artifacts",
        api_token="secret",
    )


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer secret"}
