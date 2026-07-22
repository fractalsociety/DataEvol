from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataevol.api.app import create_app
from dataevol.config import DataEvolConfig
from dataevol.local_models.layer_specialist import (
    _assert_finite_update,
    _dpo_loss_from_logratios,
    _flatten_params,
    _grouped_preference_split,
    _layer_parameters,
    _prepare_preference_rows,
    _supervised_texts,
    _train_loaded_model,
    build_manifest,
    candidate_content_hash,
    model_fingerprint,
    train_layer_specialist,
)


def test_layer_specialist_dry_run_and_validation(tmp_path: Path) -> None:
    client = TestClient(create_app(_cfg(tmp_path)))
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n", encoding="utf-8")
    payload = {
        "base_model": "mlx-community/Qwen3-0.6B-4bit",
        "base_model_revision": "a" * 40,
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
    alias = dict(payload, max_seq_length=384)
    alias_body = client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": alias}).json()
    max_seq_pos = alias_body["planned_command"].index("--max-seq-length")
    assert alias_body["planned_command"][max_seq_pos + 1] == "384"

    invalid = dict(payload, layer_index=-1)
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, training_mode="bad")
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, training_mode="rl")
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, training_mode="rl", rl_algorithm="grpo", execute=True)
    assert client.post("/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": invalid}).status_code == 422
    invalid = dict(payload, rl_algorithm="dpo")
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


def test_dpo_api_contract_threads_objective_and_verified_sft_parent(tmp_path: Path) -> None:
    client = TestClient(create_app(_cfg(tmp_path)))
    dataset = tmp_path / "preferences.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"prompt": f"p{i}", "chosen": f"good{i}", "rejected": f"bad{i}"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
    )
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    tensor_name = "parent.safetensors"
    (parent_dir / tensor_name).write_bytes(b"preflight-only-tensor")
    parent = build_manifest(
        base_model="org/remote-model",
        base_model_revision="a" * 40,
        layer_index=1,
        task_type="compression",
        training_mode="sft",
        dataset_uri=str(dataset),
        output_dir=parent_dir,
        tensor_files=[tensor_name],
        trainable_param_names=["model.layers.1.weight"],
        trainable_param_count=1,
        frozen_param_count=1,
        param_shapes={"model.layers.1.weight": {"shape": [1], "dtype": "float32"}},
        baseline_metric=1.0,
        eval_metric=0.9,
    )
    parent_path = parent_dir / "parent.manifest.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    payload = {
        "base_model": "org/remote-model",
        "base_model_revision": "a" * 40,
        "output": str(tmp_path / "out"),
        "task_type": "compression",
        "training_mode": "rl",
        "rl_algorithm": "dpo",
        "beta": 0.2,
        "sft_coef": 0.05,
        "initial_specialist_manifest": str(parent_path),
        "layer_index": 1,
        "dataset_uri": str(dataset),
        "execute": False,
    }

    response = client.post(
        "/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": payload}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rl_algorithm"] == "dpo"
    assert body["beta"] == 0.2
    assert body["sft_coef"] == 0.05
    assert body["initial_specialist_candidate_content_hash"] == parent["candidate_content_hash"]
    command = body["planned_command"]
    assert command[command.index("--rl-algorithm") + 1] == "dpo"
    assert command[command.index("--initial-specialist-manifest") + 1] == str(parent_path.resolve())

    tampered = dict(parent, candidate_content_hash="0" * 64)
    parent_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = client.post(
        "/local_model/layerscope/train_layer_specialist", headers=_auth(), json={"payload": payload}
    )
    assert rejected.status_code == 422
    assert "candidate_content_hash mismatch" in rejected.json()["detail"]


def test_layer_specialist_accepts_fractalwork_chat_examples() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):  # noqa: ANN001
            assert tokenize is False
            suffix = "assistant: " if add_generation_prompt else ""
            return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + suffix

    prefix, full = _supervised_texts(
        {"messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Compress this."},
            {"role": "assistant", "content": "Compressed."},
        ]},
        Tokenizer(),
    )
    assert "Compress this." in prefix
    assert prefix.endswith("assistant: ")
    assert full.endswith("assistant: Compressed.")


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


def test_large_layer_tensor_remains_exportable_by_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.import_module("dataevol.api.app"), "MAX_ADAPTER_EXPORT_FILE_BYTES", 4)
    client = TestClient(create_app(_cfg(tmp_path)))
    output = tmp_path / "out"
    specialist_dir = output / "layerscope" / "layer_1"
    specialist_dir.mkdir(parents=True)
    (specialist_dir / "tiny__task__L1.manifest.json").write_text(
        json.dumps({"schema": "dataevol.mlx_layer_specialist.v1"}), encoding="utf-8"
    )
    tensor = specialist_dir / "layer_1.safetensors"
    tensor.write_bytes(b"full-layer-tensor")

    body = client.post(
        "/local_model/artifacts/export", headers=_auth(), json={"payload": {"output": str(output)}}
    ).json()

    record = body["specialists"]["layer_1"]["tensors"][0]
    assert record["content_omitted"] is True
    assert record["reason"] == "use_artifact_file_endpoint"
    assert record["sha256"] == hashlib.sha256(tensor.read_bytes()).hexdigest()
    assert body["file_count"] == 2


def test_dpo_loss_is_stable_and_rewards_policy_margin() -> None:
    mx = pytest.importorskip("mlx.core")
    reference = mx.array(0.25)
    tied = _dpo_loss_from_logratios(mx.array(0.25), reference, 0.2)
    preferred = _dpo_loss_from_logratios(mx.array(2.0), reference, 0.2)
    rejected = _dpo_loss_from_logratios(mx.array(-2.0), reference, 0.2)
    extremes = [
        _dpo_loss_from_logratios(mx.array(1e6), mx.array(0.0), 1.0),
        _dpo_loss_from_logratios(mx.array(-1e6), mx.array(0.0), 1.0),
    ]
    mx.eval(tied, preferred, rejected, *extremes)
    assert float(tied) == pytest.approx(0.69314718, rel=1e-6)
    assert float(preferred) < float(tied) < float(rejected)
    assert all(bool(mx.isfinite(value)) for value in extremes)


def test_non_finite_loss_or_gradient_aborts_before_optimizer_update() -> None:
    mx = pytest.importorskip("mlx.core")

    with pytest.raises(RuntimeError, match="non-finite training loss"):
        _assert_finite_update(
            mx.array(float("nan")),
            {"model": {"layers": [{"weight": mx.ones((1,))}]}},
            objective_name="SFT",
        )
    with pytest.raises(RuntimeError, match="non-finite gradients"):
        _assert_finite_update(
            mx.array(1.0),
            {"model": {"layers": [{"weight": mx.array([float("inf")])}]}},
            objective_name="DPO",
        )


def test_dpo_rows_are_strict_and_split_by_prompt_group() -> None:
    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(char) % 32 for char in text]

    rows = [
        {"pair_id": "p0-a", "prompt": "same prompt", "chosen": "good a", "rejected": "bad a"},
        {"pair_id": "p0-b", "prompt": "same prompt", "chosen": "good b", "rejected": "bad b"},
        {"pair_id": "p1", "prompt": "other prompt", "chosen": "good", "rejected": "bad"},
        {"pair_id": "p2", "prompt": "third prompt", "chosen": "yes", "rejected": "no"},
    ]
    prepared = _prepare_preference_rows(rows, Tokenizer(), 64)
    train, evaluate, provenance = _grouped_preference_split(prepared, 0.34, 19)
    assert {row["prompt_group"] for row in train}.isdisjoint({row["prompt_group"] for row in evaluate})
    assert len(train) + len(evaluate) == len(rows)
    reversed_prepared = _prepare_preference_rows(list(reversed(rows)), Tokenizer(), 64)
    _, _, repeated = _grouped_preference_split(reversed_prepared, 0.34, 19)
    assert provenance["split_hash"] == repeated["split_hash"]

    with pytest.raises(ValueError, match="chosen and rejected"):
        _prepare_preference_rows([{"prompt": "p", "chosen": "same", "rejected": "same"}], Tokenizer(), 64)
    with pytest.raises(ValueError, match="requires non-empty"):
        _prepare_preference_rows([{"prompt": "p", "chosen": "yes"}], Tokenizer(), 64)
    with pytest.raises(ValueError, match="distinct prompt groups"):
        single_prompt = _prepare_preference_rows(rows[:2], Tokenizer(), 64)
        _grouped_preference_split(single_prompt, 0.5, 19)


def test_layer_specialist_core_trains_only_one_layer_and_exports_mlx_safetensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 16 for ch in text][:64] or [1, 2]

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

    rows = [
        {"prompt": f"prompt {i}", "completion": f"answer {i}"}
        if i % 2 == 0
        else {"messages": [
            {"role": "user", "content": f"prompt {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ]}
        for i in range(6)
    ]
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
        max_seq_len=32,
        eval_split=0.2,
        contribution=0.83,
        base_model_revision="a" * 40,
    )

    after = {name: value for name, value in _flatten_params(model.parameters())}
    changed = [name for name, value in after.items() if before[name] != _array_digest(mx, value)]
    assert changed
    assert all("model.layers.1." in name for name in changed)

    tensor_name = result["manifest"]["tensor_files"][0]
    exported = mx.load(Path(result["manifest_path"]).with_name(tensor_name))
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
    assert manifest["contribution"] == 0.83
    assert manifest["base_model_revision"] == "a" * 40
    assert manifest["candidate_content_hash"] == candidate_content_hash(manifest)
    assert manifest["genome_id"].startswith("layerspecialist_")
    assert manifest["optimizer"]["kind"] == "adamw"
    assert manifest["optimizer"]["eps"] == 1e-6
    assert manifest["optimizer"]["bias_correction"] is True
    assert manifest["optimizer"]["gradient_clip_norm"] == 0.5
    assert manifest["optimizer"]["target_dtype"] == "float32"

    with pytest.raises(ValueError, match="requires rl_algorithm='dpo'"):
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
            base_model_revision="a" * 40,
        )

    preference_rows = [
        {"prompt": f"preference prompt {i}", "chosen": f"chosen {i}", "rejected": f"rejected {i}"}
        for i in range(6)
    ]
    preference_dataset = tmp_path / "preferences.jsonl"
    preference_dataset.write_text(
        "\n".join(json.dumps(row) for row in preference_rows) + "\n", encoding="utf-8"
    )
    dpo_model = TinyLayerModel()
    dpo_before = {name: _array_digest(mx, value) for name, value in _flatten_params(dpo_model.parameters())}
    dpo_result = _train_loaded_model(
        model=dpo_model,
        tokenizer=Tokenizer(),
        base_model="tiny-mlx-model",
        layer_index=1,
        rows=preference_rows,
        dataset_uri=str(preference_dataset),
        output_dir=tmp_path / "dpo",
        task_type="compression",
        training_mode="rl",
        rl_algorithm="dpo",
        beta=0.2,
        sft_coef=0.01,
        learning_rate=0.05,
        batch_size=2,
        max_steps=3,
        max_seq_len=48,
        eval_split=0.34,
        seed=23,
        base_model_revision="a" * 40,
        initial_specialist_manifest=result["manifest_path"],
    )
    dpo_after = dict(_flatten_params(dpo_model.parameters()))
    dpo_changed = [
        name for name, value in dpo_after.items() if dpo_before[name] != _array_digest(mx, value)
    ]
    assert dpo_changed
    assert all("model.layers.1." in name for name in dpo_changed)
    dpo_manifest = dpo_result["manifest"]
    assert dpo_manifest["training_mode"] == "rl"
    assert dpo_manifest["rl_algorithm"] == "dpo"
    assert dpo_manifest["beta"] == 0.2
    assert dpo_manifest["sft_coef"] == 0.01
    assert dpo_manifest["parent_candidate_content_hash"] == manifest["candidate_content_hash"]
    assert dpo_manifest["initial_policy"]["kind"] == "sft_layer_specialist"
    assert dpo_manifest["objective"]["schema"] == "dataevol.offline_dpo.v1"
    assert dpo_manifest["objective"]["train_prompt_group_count"] > 0
    assert dpo_manifest["objective"]["eval_prompt_group_count"] > 0
    assert dpo_manifest["objective"]["baseline_metrics"]["dpo_loss"] == pytest.approx(0.69314718, rel=1e-5)
    assert dpo_manifest["candidate_content_hash"] == candidate_content_hash(dpo_manifest)

    layer_module = importlib.import_module("dataevol.local_models.layer_specialist")
    monkeypatch.setattr(
        layer_module,
        "_value_and_grad",
        lambda model, loss_fn, *, objective_name: (  # noqa: ARG005
            mx.array(float("nan")),
            {"model": {"layers": [{"weight": mx.ones((1,))}]}},
        ),
    )
    failed_output = tmp_path / "non-finite"
    with pytest.raises(RuntimeError, match="optimizer update aborted"):
        _train_loaded_model(
            model=TinyLayerModel(),
            tokenizer=Tokenizer(),
            base_model="tiny-mlx-model",
            layer_index=1,
            rows=rows,
            dataset_uri=str(dataset),
            output_dir=failed_output,
            task_type="compression",
            training_mode="sft",
            max_steps=1,
            base_model_revision="a" * 40,
        )
    assert not failed_output.exists()


def test_model_fingerprint_hashes_all_indexed_weight_shards(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"tiny"}', encoding="utf-8")
    (model / "tokenizer_config.json").write_text('{"eos_token":"x"}', encoding="utf-8")
    (model / "model-1.safetensors").write_bytes(b"one")
    (model / "model-2.safetensors").write_bytes(b"two")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-1.safetensors", "b": "model-2.safetensors",
    }}), encoding="utf-8")

    first = model_fingerprint(str(model))
    assert set(first["files"]) >= {"config.json", "tokenizer_config.json", "model-1.safetensors", "model-2.safetensors"}
    (model / "model-2.safetensors").write_bytes(b"changed")
    second = model_fingerprint(str(model))
    assert first["sha256"] != second["sha256"]


def test_remote_model_fingerprint_requires_immutable_revision() -> None:
    with pytest.raises(ValueError, match="base_model_revision"):
        model_fingerprint("org/remote-model")
    fingerprint = model_fingerprint("org/remote-model", base_model_revision="b" * 40)
    assert fingerprint["resolved_revision"] == "b" * 40


def test_remote_training_passes_pinned_revision_to_mlx_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"a","completion":"b"}\n', encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = []
    mlx_lm = importlib.import_module("mlx_lm")
    layer_module = importlib.import_module("dataevol.local_models.layer_specialist")

    def fake_load(model: str, **kwargs):  # noqa: ANN003
        calls.append((model, kwargs))
        return object(), object()

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    monkeypatch.setattr(layer_module, "_train_loaded_model", lambda **kwargs: {"ok": True, "kwargs": kwargs})
    result = train_layer_specialist(
        base_model="org/remote-model",
        base_model_revision="c" * 40,
        layer_index=1,
        dataset_uri=str(dataset),
        output_dir=tmp_path / "out",
        task_type="compression",
        training_mode="sft",
    )
    assert calls == [("org/remote-model", {"revision": "c" * 40})]
    assert result["kwargs"]["base_model_revision"] == "c" * 40


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
