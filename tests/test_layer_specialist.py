from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dataevol.api.app import create_app
from dataevol.config import DataEvolConfig


def test_layer_specialist_dry_run_and_validation(tmp_path: Path) -> None:
    client = TestClient(create_app(_cfg(tmp_path)))
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n", encoding="utf-8")
    payload = {
        "base_model": "mlx-community/Ornith-1.0-9B",
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
        "name": "ornith__compression__L14",
        "layer_index": 14,
    }
    (specialist_dir / "ornith__compression__L14.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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
