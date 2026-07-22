from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "dataevol.mlx_layer_specialist.v1"
FREEZE_STRATEGY = "mlx_full_layer"
DPO_SCHEMA = "dataevol.offline_dpo.v1"
PREFERENCE_DATASET_SCHEMA = "dataevol.prompt_chosen_rejected.v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonl(path: str | Path) -> str:
    return sha256_file(path)


def build_manifest(
    *,
    base_model: str,
    layer_index: int,
    task_type: str,
    training_mode: str,
    dataset_uri: str,
    output_dir: str | Path,
    tensor_files: Iterable[str],
    trainable_param_names: Iterable[str],
    trainable_param_count: int,
    frozen_param_count: int,
    param_shapes: dict[str, dict[str, Any]],
    baseline_metric: float | None,
    eval_metric: float | None,
    quantization: dict[str, Any] | None = None,
    contribution_profile_id: str | None = None,
    contribution_profile_hash: str | None = None,
    contribution: float | None = None,
    dataset_source_uri: str | None = None,
    base_model_revision: str | None = None,
    genome_id: str | None = None,
    rl_algorithm: str | None = None,
    beta: float | None = None,
    sft_coef: float | None = None,
    objective: dict[str, Any] | None = None,
    initial_policy: dict[str, Any] | None = None,
    parent_candidate_content_hash: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    files = list(tensor_files)
    name = specialist_name(base_model, task_type, layer_index)
    fingerprint = model_fingerprint(base_model, base_model_revision=base_model_revision)
    manifest = {
        "schema": SCHEMA,
        "name": name,
        "genome_id": _specialist_id(genome_id) if genome_id else None,
        "base_model_id": base_model,
        "base_model_hash": fingerprint["sha256"],
        "base_model_revision": fingerprint.get("resolved_revision"),
        "base_model_fingerprint": fingerprint,
        "layer_index": layer_index,
        "task_type": task_type,
        "training_mode": training_mode,
        "freeze_strategy": FREEZE_STRATEGY,
        "dataset_uri": dataset_uri,
        "dataset_source_uri": dataset_source_uri or dataset_uri,
        "dataset_hash": sha256_jsonl(dataset_uri),
        "contribution_profile_id": contribution_profile_id,
        "contribution_profile_hash": contribution_profile_hash,
        "contribution": contribution,
        "baseline_metric": baseline_metric,
        "eval_metric": eval_metric,
        "trainable_param_names": sorted(trainable_param_names),
        "trainable_param_count": trainable_param_count,
        "frozen_param_count": frozen_param_count,
        "param_shapes": param_shapes,
        "tensor_files": files,
        "sha256": {name: sha256_file(out / name) for name in files},
        "quantization": quantization,
        "runtime_version": runtime_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if rl_algorithm is not None:
        manifest["rl_algorithm"] = rl_algorithm
        manifest["beta"] = beta
        manifest["sft_coef"] = sft_coef
        manifest["objective"] = objective
        manifest["initial_policy"] = initial_policy
        manifest["parent_candidate_content_hash"] = parent_candidate_content_hash
    manifest["candidate_content_hash"] = candidate_content_hash(manifest)
    if manifest["genome_id"] is None:
        manifest["genome_id"] = f"layerspecialist_{manifest['candidate_content_hash'][:24]}"
    return manifest


def model_hash(base_model: str, *, base_model_revision: str | None = None) -> str:
    return str(model_fingerprint(base_model, base_model_revision=base_model_revision)["sha256"])


def model_fingerprint(base_model: str, *, base_model_revision: str | None = None) -> dict[str, Any]:
    path = Path(base_model)
    if not path.exists():
        revision = _resolved_revision(base_model_revision)
        body = {
            "schema": "dataevol.model_fingerprint.v1",
            "kind": "remote_revision",
            "model_id": base_model,
            "resolved_revision": revision,
            "files": {},
        }
        return {**body, "sha256": _canonical_hash(body)}
    if not path.is_dir():
        raise ValueError("base_model local path must be a directory")

    root = path.resolve()
    files: set[Path] = set()
    for name in (
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "tokenizer.model", "special_tokens_map.json", "added_tokens.json",
        "model.safetensors.index.json",
    ):
        candidate = root / name
        if candidate.is_file():
            files.add(candidate)
    files.update(candidate for candidate in root.glob("*token*.json") if candidate.is_file())
    index_paths = sorted(root.glob("*.safetensors.index.json"))
    for index_path in index_paths:
        files.add(index_path)
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model weight index: {index_path.name}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"model weight index has no weight_map: {index_path.name}")
        for relative in sorted({str(item) for item in weight_map.values()}):
            shard = (root / relative).resolve()
            if root not in shard.parents or not shard.is_file():
                raise ValueError(f"model weight shard is missing or unsafe: {relative}")
            files.add(shard)
    if not index_paths:
        files.update(candidate for candidate in root.glob("*.safetensors") if candidate.is_file())
    if not any(candidate.suffix == ".safetensors" for candidate in files):
        raise ValueError("local base model fingerprint requires at least one safetensors weight file")
    file_hashes = {
        candidate.relative_to(root).as_posix(): sha256_file(candidate)
        for candidate in sorted(files)
    }
    body = {
        "schema": "dataevol.model_fingerprint.v1",
        "kind": "local_files",
        "resolved_revision": _optional_revision(base_model_revision),
        "files": file_hashes,
    }
    return {**body, "sha256": _canonical_hash(body)}


def candidate_content_hash(manifest: dict[str, Any]) -> str:
    fingerprint = manifest.get("base_model_fingerprint")
    tensor_hashes = manifest.get("sha256")
    if not isinstance(fingerprint, dict) or not fingerprint.get("sha256"):
        raise ValueError("base_model_fingerprint is required for candidate identity")
    if not isinstance(tensor_hashes, dict) or not tensor_hashes:
        raise ValueError("tensor SHA-256 values are required for candidate identity")
    body = {
        "schema": "dataevol.layer_specialist_candidate.v1",
        "model_fingerprint": str(fingerprint["sha256"]),
        "base_model_revision": manifest.get("base_model_revision"),
        "layer_index": int(manifest["layer_index"]),
        "task_type": str(manifest["task_type"]),
        "training_mode": str(manifest["training_mode"]),
        "freeze_strategy": str(manifest["freeze_strategy"]),
        "tensor_sha256": {str(key): str(value).lower() for key, value in sorted(tensor_hashes.items())},
        "dataset_hash": str(manifest["dataset_hash"]).lower(),
        "contribution_profile_id": manifest.get("contribution_profile_id"),
        "contribution_profile_hash": manifest.get("contribution_profile_hash"),
    }
    if manifest.get("rl_algorithm") is not None:
        body.update(
            {
                "rl_algorithm": manifest.get("rl_algorithm"),
                "beta": manifest.get("beta"),
                "sft_coef": manifest.get("sft_coef"),
                "objective": manifest.get("objective"),
                "initial_policy": manifest.get("initial_policy"),
                "parent_candidate_content_hash": manifest.get("parent_candidate_content_hash"),
            }
        )
    return _canonical_hash(body)


def specialist_name(base_model: str, task_type: str, layer_index: int) -> str:
    short = base_model.rsplit("/", 1)[-1]
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", short).strip("._-") or "model"
    safe_task = re.sub(r"[^A-Za-z0-9._-]+", "_", task_type).strip("._-") or "task"
    return f"{safe_model}__{safe_task}__L{layer_index}"


def runtime_version() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "mlx": _package_version("mlx"),
        "mlx_lm": _package_version("mlx_lm"),
    }


def _package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", None) or "installed")
    except Exception:
        return None


def _validate_training_objective(training_mode: str, rl_algorithm: str | None, beta: float, sft_coef: float) -> None:
    if training_mode not in {"sft", "rl"}:
        raise ValueError("training_mode must be sft or rl")
    if training_mode == "rl" and rl_algorithm != "dpo":
        raise ValueError("training_mode='rl' requires rl_algorithm='dpo'")
    if training_mode == "sft" and rl_algorithm is not None:
        raise ValueError("rl_algorithm is only valid when training_mode='rl'")
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be a positive finite number")
    if not math.isfinite(float(sft_coef)) or sft_coef < 0:
        raise ValueError("sft_coef must be a non-negative finite number")


def validate_initial_specialist_manifest(
    manifest_path: str | Path,
    *,
    base_model: str,
    base_model_revision: str | None,
    layer_index: int,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("initial_specialist_manifest must be an existing manifest file")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("initial_specialist_manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("initial specialist uses an unsupported schema")
    if manifest.get("freeze_strategy") != FREEZE_STRATEGY:
        raise ValueError("initial specialist uses an unsupported freeze_strategy")
    if manifest.get("training_mode") != "sft":
        raise ValueError("initial specialist for DPO must be an SFT specialist")
    if int(manifest.get("layer_index", -1)) != layer_index:
        raise ValueError("initial specialist layer_index does not match the DPO target layer")
    fingerprint = model_fingerprint(base_model, base_model_revision=base_model_revision)
    if manifest.get("base_model_hash") != fingerprint["sha256"]:
        raise ValueError("initial specialist base model fingerprint mismatch")
    if manifest.get("base_model_revision") != fingerprint.get("resolved_revision"):
        raise ValueError("initial specialist base model revision mismatch")
    content_hash = str(manifest.get("candidate_content_hash") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        raise ValueError("initial specialist candidate_content_hash is invalid")
    if candidate_content_hash(manifest) != content_hash:
        raise ValueError("initial specialist candidate_content_hash mismatch")
    tensor_files = manifest.get("tensor_files")
    hashes = manifest.get("sha256")
    if not isinstance(tensor_files, list) or not tensor_files or not isinstance(hashes, dict):
        raise ValueError("initial specialist tensor provenance is incomplete")
    verified_files: list[dict[str, Any]] = []
    for file_name in tensor_files:
        relative = Path(str(file_name))
        expected = str(hashes.get(str(file_name)) or "").lower()
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe initial specialist tensor path: {file_name}")
        tensor_path = (path.parent / relative).resolve()
        if path.parent not in tensor_path.parents or not tensor_path.is_file():
            raise ValueError(f"initial specialist tensor is missing: {file_name}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(tensor_path) != expected:
            raise ValueError(f"initial specialist tensor hash mismatch: {file_name}")
        verified_files.append({"name": str(file_name), "path": str(tensor_path), "sha256": expected})
    return {
        "path": str(path),
        "manifest_sha256": sha256_file(path),
        "manifest": manifest,
        "tensor_files": verified_files,
    }


def _apply_initial_specialist(
    model: Any,
    manifest_path: str | Path,
    *,
    base_model: str,
    base_model_revision: str | None,
    layer_index: int,
) -> dict[str, Any]:
    import mlx.core as mx

    verified = validate_initial_specialist_manifest(
        manifest_path,
        base_model=base_model,
        base_model_revision=base_model_revision,
        layer_index=layer_index,
    )
    tensors: dict[str, Any] = {}
    for record in verified["tensor_files"]:
        for name, value in mx.load(record["path"]).items():
            if name in tensors:
                raise ValueError(f"duplicate initial specialist tensor: {name}")
            tensors[name] = value
    current = _layer_parameters(model, layer_index)
    missing = sorted(set(current) - set(tensors))
    extra = sorted(set(tensors) - set(current))
    if missing or extra:
        raise ValueError(f"initial specialist tensors incompatible missing={missing[:5]} extra={extra[:5]}")
    invalid = []
    for name, value in tensors.items():
        wanted = current[name]
        if value.shape != wanted.shape or str(value.dtype) != str(wanted.dtype):
            invalid.append(name)
            continue
        finite = mx.all(mx.isfinite(value))
        mx.eval(finite)
        if not bool(finite.item()):
            invalid.append(name)
    if invalid:
        raise ValueError(f"initial specialist tensor shape/dtype/value mismatch: {invalid[:5]}")
    model.load_weights(list(tensors.items()), strict=False)
    mx.eval(model.parameters())
    manifest = verified["manifest"]
    return {
        "schema": "dataevol.dpo_initial_policy.v1",
        "kind": "sft_layer_specialist",
        "candidate_content_hash": str(manifest["candidate_content_hash"]),
        "genome_id": manifest.get("genome_id"),
        "manifest_sha256": verified["manifest_sha256"],
        "base_model_hash": manifest["base_model_hash"],
        "base_model_revision": manifest.get("base_model_revision"),
        "layer_index": layer_index,
    }


def train_layer_specialist(
    *,
    base_model: str,
    layer_index: int,
    dataset_uri: str,
    output_dir: str | Path,
    task_type: str,
    training_mode: str,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    max_steps: int = 100,
    max_seq_len: int = 512,
    eval_split: float = 0.1,
    seed: int = 17,
    contribution_profile_id: str | None = None,
    contribution_profile_hash: str | None = None,
    contribution: float | None = None,
    dataset_source_uri: str | None = None,
    base_model_revision: str | None = None,
    genome_id: str | None = None,
    rl_algorithm: str | None = None,
    beta: float = 0.1,
    sft_coef: float = 0.0,
    initial_specialist_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Train a full replacement tensor set for one MLX decoder layer.

    The real path requires mlx/mlx_lm and a model that exposes
    `model.model.layers`. Tests can still exercise manifest/export helpers
    without loading a large model.
    """
    _validate_training_objective(training_mode, rl_algorithm, beta, sft_coef)

    try:
        import mlx.core as mx
        from mlx_lm import load
    except Exception as exc:  # pragma: no cover - depends on optional local MLX install.
        raise RuntimeError(f"MLX specialist training requires mlx and mlx_lm: {exc}") from exc

    mx.random.seed(seed)
    load_kwargs = {"revision": _resolved_revision(base_model_revision)} if not Path(base_model).exists() else {}
    model, tokenizer = load(base_model, **load_kwargs)
    rows = _load_jsonl(dataset_uri)
    return _train_loaded_model(
        model=model,
        tokenizer=tokenizer,
        base_model=base_model,
        layer_index=layer_index,
        rows=rows,
        dataset_uri=dataset_uri,
        output_dir=output_dir,
        task_type=task_type,
        training_mode=training_mode,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_steps=max_steps,
        max_seq_len=max_seq_len,
        eval_split=eval_split,
        seed=seed,
        contribution_profile_id=contribution_profile_id,
        contribution_profile_hash=contribution_profile_hash,
        contribution=contribution,
        dataset_source_uri=dataset_source_uri,
        base_model_revision=base_model_revision,
        genome_id=genome_id,
        rl_algorithm=rl_algorithm,
        beta=beta,
        sft_coef=sft_coef,
        initial_specialist_manifest=initial_specialist_manifest,
    )


def _train_loaded_model(
    *,
    model: Any,
    tokenizer: Any,
    base_model: str,
    layer_index: int,
    rows: list[dict[str, Any]],
    dataset_uri: str,
    output_dir: str | Path,
    task_type: str,
    training_mode: str,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    max_steps: int = 100,
    max_seq_len: int = 512,
    eval_split: float = 0.1,
    seed: int = 17,
    contribution_profile_id: str | None = None,
    contribution_profile_hash: str | None = None,
    contribution: float | None = None,
    dataset_source_uri: str | None = None,
    base_model_revision: str | None = None,
    genome_id: str | None = None,
    rl_algorithm: str | None = None,
    beta: float = 0.1,
    sft_coef: float = 0.0,
    initial_specialist_manifest: str | Path | None = None,
) -> dict[str, Any]:
    _validate_training_objective(training_mode, rl_algorithm, beta, sft_coef)

    import mlx.core as mx
    import mlx.optimizers as optim

    layers = _decoder_layers(model)
    if layers is None:
        raise RuntimeError("loaded model does not expose model.model.layers")
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f"layer_index {layer_index} is outside model layer range 0..{len(layers)-1}")
    if not rows:
        raise ValueError("dataset contains no examples")
    dequantized_module_count = _dequantize_quantized_linears(layers[layer_index])
    layers[layer_index].set_dtype(mx.float32)
    fingerprint = model_fingerprint(base_model, base_model_revision=base_model_revision)
    parent_policy = None
    if initial_specialist_manifest is not None:
        if training_mode != "rl":
            raise ValueError("initial_specialist_manifest is only supported for RL/DPO training")
        parent_policy = _apply_initial_specialist(
            model,
            initial_specialist_manifest,
            base_model=base_model,
            base_model_revision=base_model_revision,
            layer_index=layer_index,
        )

    warmup_steps = max(1, min(16, max_steps // 8)) if max_steps > 1 else 0
    if warmup_steps:
        learning_rate_schedule = optim.join_schedules(
            [
                optim.linear_schedule(learning_rate * 0.1, learning_rate, warmup_steps),
                optim.cosine_decay(learning_rate, max(1, max_steps - warmup_steps), end=learning_rate * 0.1),
            ],
            [warmup_steps],
        )
    else:
        learning_rate_schedule = learning_rate
    optimizer = optim.AdamW(
        learning_rate=learning_rate_schedule,
        betas=[0.9, 0.95],
        eps=1e-6,
        weight_decay=0.0,
        bias_correction=True,
    )
    objective = None
    initial_policy = None
    parent_candidate_content_hash = None
    if training_mode == "sft":
        valid_rows = [row for row in rows if _supervised_token_ids(row, tokenizer, max_seq_len) is not None]
        filtered_row_count = len(rows) - len(valid_rows)
        if not valid_rows:
            raise ValueError("dataset contains no examples with supervised completion tokens inside max_seq_len")
        shuffled_rows = list(valid_rows)
        random.Random(seed).shuffle(shuffled_rows)
        split_at = max(1, min(len(shuffled_rows) - 1, int(len(shuffled_rows) * (1.0 - eval_split)))) if len(shuffled_rows) > 1 else 1
        train_rows = shuffled_rows[:split_at]
        eval_rows = shuffled_rows[split_at:] or shuffled_rows[:1]
        baseline_metric = _eval_loss(model, tokenizer, eval_rows, max_seq_len)
        model.freeze()
        layers[layer_index].unfreeze()
        _assert_only_layer_trainable(model, layer_index)
        for step in range(1, max_steps + 1):
            batch = _batch_rows(train_rows, step, batch_size)
            loss, grads = _value_and_grad(
                model,
                lambda m: _loss(m, tokenizer, batch, max_seq_len),
                objective_name="SFT",
            )
            _assert_finite_update(loss, grads, objective_name="SFT")
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=0.5)
            mx.eval(grad_norm)
            if not math.isfinite(float(grad_norm)) or float(grad_norm) > 100:
                raise RuntimeError(f"SFT gradient norm is unsafe: {float(grad_norm):.6f}")
            optimizer.update(model, grads)
            _materialize_finite_parameters(layers[layer_index], objective_name="SFT")
            mx.eval(optimizer.state)
            print(f"step {step} loss {float(loss):.6f} grad_norm {float(grad_norm):.6f}", flush=True)
        eval_metric = _eval_loss(model, tokenizer, eval_rows, max_seq_len)
    else:
        prepared = _prepare_preference_rows(rows, tokenizer, max_seq_len)
        train_rows, eval_rows, split_provenance = _grouped_preference_split(prepared, eval_split, seed)
        model.freeze()
        _precompute_reference_logratios(model, train_rows + eval_rows)
        reference_logratios_hash = _reference_logratios_hash(train_rows + eval_rows)
        baseline_metrics = _eval_dpo(model, eval_rows, beta=beta, sft_coef=sft_coef)
        layers[layer_index].unfreeze()
        _assert_only_layer_trainable(model, layer_index)
        for step in range(1, max_steps + 1):
            batch = _batch_rows(train_rows, step, batch_size)
            loss, grads = _value_and_grad(
                model,
                lambda m: _dpo_loss(m, batch, beta=beta, sft_coef=sft_coef),
                objective_name="DPO",
            )
            _assert_finite_update(loss, grads, objective_name="DPO")
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=0.5)
            mx.eval(grad_norm)
            if not math.isfinite(float(grad_norm)) or float(grad_norm) > 100:
                raise RuntimeError(f"DPO gradient norm is unsafe: {float(grad_norm):.6f}")
            optimizer.update(model, grads)
            _materialize_finite_parameters(layers[layer_index], objective_name="DPO")
            mx.eval(optimizer.state)
            print(f"step {step} loss {float(loss):.6f} grad_norm {float(grad_norm):.6f}", flush=True)
        eval_metrics = _eval_dpo(model, eval_rows, beta=beta, sft_coef=sft_coef)
        baseline_metric = baseline_metrics["objective_loss"]
        eval_metric = eval_metrics["objective_loss"]
        parent_candidate_content_hash = parent_policy["candidate_content_hash"] if parent_policy else None
        initial_policy = parent_policy or {
            "schema": "dataevol.dpo_initial_policy.v1",
            "kind": "base_model",
            "base_model_hash": fingerprint["sha256"],
            "base_model_revision": fingerprint.get("resolved_revision"),
        }
        objective = {
            "schema": DPO_SCHEMA,
            "beta": beta,
            "sft_coef": sft_coef,
            "reference": "frozen_initial_policy",
            "log_probability_reduction": "completion_token_sum",
            "max_seq_len": max_seq_len,
            "preference_dataset_schema": PREFERENCE_DATASET_SCHEMA,
            "reference_logratios_hash": reference_logratios_hash,
            "train_pair_count": len(train_rows),
            "eval_pair_count": len(eval_rows),
            **split_provenance,
            "baseline_metrics": baseline_metrics,
            "eval_metrics": eval_metrics,
        }
    layer_params = _layer_parameters(model, layer_index)
    all_params = dict(_flatten_params(model.parameters()))
    frozen_param_count = sum(_numel(v) for name, v in all_params.items() if name not in layer_params)
    trainable_param_count = sum(_numel(v) for v in layer_params.values())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_name = specialist_name(base_model, task_type, layer_index)
    tensor_name = f"{artifact_name}.safetensors"
    tensor_path = out / tensor_name
    temporary_tensor_path = out / f".{artifact_name}.tmp.safetensors"
    mx.save_safetensors(str(temporary_tensor_path), layer_params)
    temporary_tensor_path.replace(tensor_path)
    manifest = build_manifest(
        base_model=base_model,
        layer_index=layer_index,
        task_type=task_type,
        training_mode=training_mode,
        dataset_uri=dataset_uri,
        output_dir=out,
        tensor_files=[tensor_name],
        quantization=_detect_quantization(model, base_model),
        trainable_param_names=layer_params.keys(),
        trainable_param_count=trainable_param_count,
        frozen_param_count=frozen_param_count,
        param_shapes={name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in layer_params.items()},
        baseline_metric=float(baseline_metric),
        eval_metric=float(eval_metric),
        contribution_profile_id=contribution_profile_id,
        contribution_profile_hash=contribution_profile_hash,
        contribution=contribution,
        dataset_source_uri=dataset_source_uri,
        base_model_revision=base_model_revision,
        genome_id=genome_id,
        rl_algorithm=rl_algorithm,
        beta=beta if training_mode == "rl" else None,
        sft_coef=sft_coef if training_mode == "rl" else None,
        objective=objective,
        initial_policy=initial_policy,
        parent_candidate_content_hash=parent_candidate_content_hash,
    )
    manifest["dequantized_module_count"] = dequantized_module_count
    if training_mode == "sft":
        manifest["supervised_row_count"] = len(valid_rows)
        manifest["filtered_unsupervised_row_count"] = filtered_row_count
    manifest["optimizer"] = {
        "kind": "adamw",
        "betas": [0.9, 0.95],
        "eps": 1e-6,
        "weight_decay": 0.0,
        "bias_correction": True,
        "gradient_clip_norm": 0.5,
        "target_dtype": "float32",
        "peak_learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "end_learning_rate": learning_rate * 0.1,
    }
    manifest_path = out / f"{manifest['name']}.manifest.json"
    temporary_manifest_path = out / f".{manifest['name']}.manifest.json.tmp"
    temporary_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest_path.replace(manifest_path)
    return {"ok": True, "status": "completed", "manifest_path": str(manifest_path), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one MLX full-layer specialist.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--model", required=True)
    train.add_argument("--layer-index", type=int, required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--task-type", required=True)
    train.add_argument("--training-mode", choices=("sft", "rl"), required=True)
    train.add_argument("--rl-algorithm", choices=("dpo",))
    train.add_argument("--beta", type=float, default=0.1)
    train.add_argument("--sft-coef", type=float, default=0.0)
    train.add_argument("--initial-specialist-manifest")
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=100)
    train.add_argument("--max-seq-length", type=int, default=512)
    train.add_argument("--eval-split", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--contribution-profile-id")
    train.add_argument("--contribution-profile-hash")
    train.add_argument("--contribution", type=float)
    train.add_argument("--dataset-source-uri")
    train.add_argument("--base-model-revision")
    train.add_argument("--genome-id")
    args = parser.parse_args()
    result = train_layer_specialist(
        base_model=args.model,
        layer_index=args.layer_index,
        dataset_uri=args.data,
        output_dir=args.output,
        task_type=args.task_type,
        training_mode=args.training_mode,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_seq_len=args.max_seq_length,
        eval_split=args.eval_split,
        seed=args.seed,
        contribution_profile_id=args.contribution_profile_id,
        contribution_profile_hash=args.contribution_profile_hash,
        contribution=args.contribution,
        dataset_source_uri=args.dataset_source_uri,
        base_model_revision=args.base_model_revision,
        genome_id=args.genome_id,
        rl_algorithm=args.rl_algorithm,
        beta=args.beta,
        sft_coef=args.sft_coef,
        initial_specialist_manifest=args.initial_specialist_manifest,
    )
    print(json.dumps({"status": result["status"], "manifest_path": result["manifest_path"]}, sort_keys=True), flush=True)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_completion(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or row.get("input") or row.get("instruction") or "")
    completion = str(row.get("completion") or row.get("output") or row.get("response") or "")
    return f"{prompt}\n{completion}".strip()


def _supervised_texts(row: dict[str, Any], tokenizer: Any) -> tuple[str, str]:
    messages = row.get("messages")
    if isinstance(messages, list):
        normalized = [
            {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
            for item in messages
            if isinstance(item, dict)
        ]
        assistant_index = next(
            (index for index in range(len(normalized) - 1, -1, -1) if normalized[index]["role"] == "assistant"),
            -1,
        )
        if assistant_index >= 0 and normalized[assistant_index]["content"].strip():
            prompt_messages = normalized[:assistant_index]
            supervised_messages = normalized[:assistant_index + 1]
            apply_template = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_template):
                prefix = str(apply_template(prompt_messages, tokenize=False, add_generation_prompt=True))
                full = str(apply_template(supervised_messages, tokenize=False, add_generation_prompt=False))
                return prefix, full
            prefix = "\n".join(f"{item['role']}: {item['content']}" for item in prompt_messages)
            full = "\n".join(f"{item['role']}: {item['content']}" for item in supervised_messages)
            return f"{prefix}\nassistant: ", full

    prompt = str(row.get("prompt") or row.get("input") or row.get("instruction") or "")
    completion = str(row.get("completion") or row.get("output") or row.get("response") or "")
    prefix = f"{prompt}\n" if prompt else ""
    return prefix, f"{prefix}{completion}"


def _loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    losses = []
    for row in rows:
        prepared = _supervised_token_ids(row, tokenizer, max_seq_len)
        if prepared is None:
            continue
        ids, first_supervised_target = prepared
        x = mx.array(ids[:-1])[None, :]
        y = mx.array(ids[1:])[None, :]
        logits = model(x)
        token_losses = nn.losses.cross_entropy(logits, y, reduction="none")
        losses.append(token_losses[:, first_supervised_target:].mean())
    if not losses:
        raise ValueError("batch contains no supervised completion tokens")
    return sum(losses) / len(losses)


def _supervised_token_ids(
    row: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
) -> tuple[list[int], int] | None:
    prefix, full = _supervised_texts(row, tokenizer)
    ids = list(tokenizer.encode(full))[:max_seq_len]
    prefix_ids = list(tokenizer.encode(prefix))[:max_seq_len]
    if len(ids) < 2:
        return None
    common_prefix = 0
    for prefix_token, full_token in zip(prefix_ids, ids):
        if prefix_token != full_token:
            break
        common_prefix += 1
    first_supervised_target = max(0, common_prefix - 1)
    if first_supervised_target >= len(ids) - 1:
        return None
    return ids, first_supervised_target


def _eval_loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int) -> float:
    import mlx.core as mx

    values = []
    for row in rows:
        loss = _loss(model, tokenizer, [row], max_seq_len)
        mx.eval(loss)
        value = float(loss)
        if not math.isfinite(value):
            raise RuntimeError("SFT evaluation produced a non-finite loss")
        values.append(value)
    if not values:
        raise ValueError("SFT evaluation contains no supervised examples")
    return statistics.fmean(values)


def _prepare_preference_rows(rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> list[dict[str, Any]]:
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2 for DPO")
    prepared = []
    seen_pair_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not all(isinstance(value, str) and value.strip() for value in (prompt, chosen, rejected)):
            raise ValueError(f"DPO row {index} requires non-empty string prompt, chosen, and rejected fields")
        if chosen.strip() == rejected.strip():
            raise ValueError(f"DPO row {index} chosen and rejected responses must differ")
        chosen_tokens = _preference_response_tokens(prompt, chosen, tokenizer, max_seq_len, index, "chosen")
        rejected_tokens = _preference_response_tokens(prompt, rejected, tokenizer, max_seq_len, index, "rejected")
        identity = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        pair_id = str(row.get("pair_id") or _canonical_hash(identity))
        if pair_id in seen_pair_ids:
            raise ValueError(f"duplicate DPO pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        prepared.append(
            {
                "pair_id": pair_id,
                "prompt_group": _canonical_hash({"prompt": prompt.strip()}),
                "chosen_ids": chosen_tokens["ids"],
                "chosen_first_target": chosen_tokens["first_target"],
                "chosen_target_count": chosen_tokens["target_count"],
                "rejected_ids": rejected_tokens["ids"],
                "rejected_first_target": rejected_tokens["first_target"],
                "rejected_target_count": rejected_tokens["target_count"],
            }
        )
    return prepared


def _preference_response_tokens(
    prompt: str,
    response: str,
    tokenizer: Any,
    max_seq_len: int,
    row_index: int,
    response_name: str,
) -> dict[str, Any]:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        prompt_messages = [{"role": "user", "content": prompt}]
        full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
        prefix = str(apply_template(prompt_messages, tokenize=False, add_generation_prompt=True))
        full = str(apply_template(full_messages, tokenize=False, add_generation_prompt=False))
    else:
        prefix = f"{prompt}\n"
        full = f"{prefix}{response}"
    ids = list(tokenizer.encode(full))[:max_seq_len]
    prefix_ids = list(tokenizer.encode(prefix))[:max_seq_len]
    common = _common_prefix_length(ids, prefix_ids)
    first_target = max(0, common - 1)
    target_count = max(0, len(ids) - 1 - first_target)
    if len(ids) < 2 or target_count < 1:
        raise ValueError(
            f"DPO row {row_index} {response_name} response has no completion tokens after truncation"
        )
    return {"ids": ids, "first_target": first_target, "target_count": target_count}


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def _grouped_preference_split(
    rows: list[dict[str, Any]], eval_split: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["prompt_group"]), []).append(row)
    if len(groups) < 2:
        raise ValueError("DPO requires at least two distinct prompt groups for leakage-free evaluation")
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    eval_group_count = max(1, min(len(group_ids) - 1, round(len(group_ids) * eval_split)))
    eval_groups = set(group_ids[:eval_group_count])
    train = sorted(
        (row for group_id, items in groups.items() if group_id not in eval_groups for row in items),
        key=lambda row: str(row["pair_id"]),
    )
    evaluate = sorted(
        (row for group_id, items in groups.items() if group_id in eval_groups for row in items),
        key=lambda row: str(row["pair_id"]),
    )
    split_body = {
        "seed": seed,
        "eval_split": eval_split,
        "train_prompt_group_count": len(group_ids) - eval_group_count,
        "eval_prompt_group_count": eval_group_count,
        "train_pair_ids": [str(row["pair_id"]) for row in train],
        "eval_pair_ids": [str(row["pair_id"]) for row in evaluate],
    }
    return train, evaluate, {
        "split_strategy": "deterministic_prompt_group",
        "split_seed": seed,
        "train_prompt_group_count": split_body["train_prompt_group_count"],
        "eval_prompt_group_count": split_body["eval_prompt_group_count"],
        "split_hash": _canonical_hash(split_body),
    }


def _sequence_logprob(model: Any, ids: list[int], first_target: int) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    x = mx.array(ids[:-1])[None, :]
    y = mx.array(ids[1:])[None, :]
    logits = model(x)
    token_losses = nn.losses.cross_entropy(logits, y, reduction="none").astype(mx.float32)
    return -token_losses[:, first_target:].sum()


def _precompute_reference_logratios(model: Any, rows: list[dict[str, Any]]) -> None:
    import mlx.core as mx

    for row in rows:
        chosen = _sequence_logprob(model, row["chosen_ids"], row["chosen_first_target"])
        rejected = _sequence_logprob(model, row["rejected_ids"], row["rejected_first_target"])
        mx.eval(chosen, rejected)
        chosen_value = float(chosen)
        rejected_value = float(rejected)
        if not math.isfinite(chosen_value) or not math.isfinite(rejected_value):
            raise RuntimeError("DPO reference policy produced non-finite completion log-probabilities")
        row["reference_chosen_logprob"] = chosen_value
        row["reference_rejected_logprob"] = rejected_value
        row["reference_logratio"] = chosen_value - rejected_value


def _reference_logratios_hash(rows: list[dict[str, Any]]) -> str:
    return _canonical_hash(
        {
            "schema": "dataevol.dpo_reference_logratios.v1",
            "pairs": {
                str(row["pair_id"]): float(row["reference_logratio"]).hex()
                for row in sorted(rows, key=lambda item: str(item["pair_id"]))
            },
        }
    )


def _dpo_loss_from_logratios(policy_logratio: Any, reference_logratio: Any, beta: float) -> Any:
    import mlx.core as mx

    z = (beta * (policy_logratio - reference_logratio)).astype(mx.float32)
    return mx.logaddexp(mx.array(0.0, dtype=mx.float32), -z)


def _dpo_loss(model: Any, rows: list[dict[str, Any]], *, beta: float, sft_coef: float) -> Any:
    losses = []
    for row in rows:
        chosen = _sequence_logprob(model, row["chosen_ids"], row["chosen_first_target"])
        rejected = _sequence_logprob(model, row["rejected_ids"], row["rejected_first_target"])
        pair_loss = _dpo_loss_from_logratios(chosen - rejected, row["reference_logratio"], beta)
        if sft_coef:
            pair_loss = pair_loss + sft_coef * (-chosen / row["chosen_target_count"])
        losses.append(pair_loss)
    if not losses:
        raise ValueError("DPO batch contains no preference pairs")
    return sum(losses) / len(losses)


def _eval_dpo(model: Any, rows: list[dict[str, Any]], *, beta: float, sft_coef: float) -> dict[str, float]:
    import mlx.core as mx

    objectives = []
    dpo_losses = []
    margins = []
    policy_ratios = []
    reference_ratios = []
    chosen_nlls = []
    for row in rows:
        chosen = _sequence_logprob(model, row["chosen_ids"], row["chosen_first_target"])
        rejected = _sequence_logprob(model, row["rejected_ids"], row["rejected_first_target"])
        mx.eval(chosen, rejected)
        chosen_value = float(chosen)
        rejected_value = float(rejected)
        policy_ratio = chosen_value - rejected_value
        reference_ratio = float(row["reference_logratio"])
        margin = beta * (policy_ratio - reference_ratio)
        dpo_loss = _softplus_negative(margin)
        chosen_nll = -chosen_value / int(row["chosen_target_count"])
        objectives.append(dpo_loss + sft_coef * chosen_nll)
        dpo_losses.append(dpo_loss)
        margins.append(margin)
        policy_ratios.append(policy_ratio)
        reference_ratios.append(reference_ratio)
        chosen_nlls.append(chosen_nll)
    count = len(rows)
    metrics = {
        "objective_loss": sum(objectives) / count,
        "dpo_loss": sum(dpo_losses) / count,
        "preference_accuracy": sum(1 for value in margins if value > 0) / count,
        "reward_margin": sum(margins) / count,
        "policy_logratio": sum(policy_ratios) / count,
        "reference_logratio": sum(reference_ratios) / count,
        "chosen_nll": sum(chosen_nlls) / count,
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("DPO evaluation produced non-finite metrics")
    return metrics


def _softplus_negative(value: float) -> float:
    if value >= 0:
        return math.log1p(math.exp(-value))
    return -value + math.log1p(math.exp(value))


def _value_and_grad(model: Any, loss_fn: Any, *, objective_name: str) -> tuple[Any, Any]:
    import mlx.nn as nn

    try:
        return nn.value_and_grad(model, loss_fn)(model)
    except ValueError as exc:
        if "CustomKernel" in str(exc):
            raise RuntimeError(
                "MLX cannot backpropagate through this model path because one or more custom kernels "
                "do not implement VJP. For Ornith, linear-attention layers after or inside the target "
                f"layer are not trainable with this {objective_name} path; use a full-attention layer near "
                "the end of the model or a future local-layer objective."
            ) from exc
        raise


def _assert_finite_update(loss: Any, grads: Any, *, objective_name: str) -> None:
    import mlx.core as mx

    gradient_items = _flatten_params(grads)
    if not gradient_items:
        raise RuntimeError(f"{objective_name} produced no gradients; optimizer update aborted")
    finite_loss = mx.all(mx.isfinite(loss))
    finite_gradients = [(name, mx.all(mx.isfinite(value))) for name, value in gradient_items]
    mx.eval(finite_loss, *(value for _, value in finite_gradients))
    if not bool(finite_loss.item()):
        raise RuntimeError(f"{objective_name} produced a non-finite training loss; optimizer update aborted")
    offenders = [name for name, value in finite_gradients if not bool(value.item())]
    if offenders:
        raise RuntimeError(
            f"{objective_name} produced non-finite gradients; optimizer update aborted: {offenders[:8]}"
        )


def _materialize_finite_parameters(layer: Any, *, objective_name: str) -> None:
    import mlx.core as mx

    parameter_items = _flatten_params(layer.parameters())
    finite_parameters = [(name, mx.all(mx.isfinite(value))) for name, value in parameter_items]
    mx.eval(*(value for _, value in finite_parameters))
    offenders = [name for name, value in finite_parameters if not bool(value.item())]
    if offenders:
        raise RuntimeError(
            f"{objective_name} produced non-finite parameters after optimizer update: {offenders[:8]}"
        )


def _layer_parameters(model: Any, layer_index: int) -> dict[str, Any]:
    needle = f"layers.{layer_index}."
    params = {name: value for name, value in _flatten_params(model.parameters()) if name.startswith(needle) or f".{needle}" in name}
    if not params:
        raise RuntimeError(f"no parameters matched layer {layer_index}")
    return params


def _decoder_layers(model: Any) -> Any:
    for root in (model, getattr(model, "language_model", None), getattr(model, "model", None)):
        if root is None:
            continue
        candidate = getattr(getattr(root, "model", None), "layers", None)
        if candidate is not None:
            return candidate
        candidate = getattr(root, "layers", None)
        if candidate is not None:
            return candidate
    return None


def _dequantize_quantized_linears(module: Any) -> int:
    import mlx.core as mx
    import mlx.nn as nn

    replacements: list[tuple[str, Any]] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.QuantizedLinear):
            weight = mx.dequantize(
                child.weight,
                child.scales,
                child.get("biases"),
                child.group_size,
                child.bits,
                mode=child.mode,
            )
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in child)
            linear.weight = weight
            if "bias" in child:
                linear.bias = child.bias
            replacements.append((name, linear))
    for name, replacement in replacements:
        _set_module_path(module, name, replacement)
    return len(replacements)


def _set_module_path(root: Any, dotted_path: str, value: Any) -> None:
    parent = root
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
    last = parts[-1]
    if isinstance(parent, list):
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def _assert_only_layer_trainable(model: Any, layer_index: int) -> None:
    trainable = dict(_flatten_params(model.trainable_parameters()))
    if not trainable:
        raise RuntimeError("nothing trainable after layer unfreeze")
    wanted = f"layers.{layer_index}."
    unexpected = [name for name in trainable if wanted not in name]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters outside {wanted}: {unexpected[:8]}")


def _batch_rows(rows: list[dict[str, Any]], step: int, batch_size: int) -> list[dict[str, Any]]:
    start = ((step - 1) * max(1, batch_size)) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(max(1, batch_size))]


def _flatten_params(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(tree, dict):
        out: list[tuple[str, Any]] = []
        for key, value in tree.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_flatten_params(value, next_prefix))
        return out
    if isinstance(tree, (list, tuple)):
        out: list[tuple[str, Any]] = []
        for index, value in enumerate(tree):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            out.extend(_flatten_params(value, next_prefix))
        return out
    return [(prefix, tree)]


def _numel(value: Any) -> int:
    n = 1
    for dim in getattr(value, "shape", ()):
        n *= int(dim)
    return n


def _detect_quantization(model: Any, base_model: str) -> dict[str, Any] | None:
    for source in (getattr(model, "config", None), _read_model_config(base_model)):
        if not source:
            continue
        raw = source if isinstance(source, dict) else vars(source)
        config = raw.get("quantization") or raw.get("quantization_config") or raw
        if not isinstance(config, dict):
            continue
        bits = config.get("bits") or config.get("num_bits")
        group_size = config.get("group_size") or config.get("q_group_size")
        if bits is not None or group_size is not None:
            return {
                "bits": bits,
                "group_size": group_size,
            }
    return None


def _read_model_config(base_model: str) -> dict[str, Any] | None:
    path = Path(base_model) / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _resolved_revision(value: str | None) -> str:
    revision = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,64}", revision):
        raise ValueError("base_model_revision must be an immutable 7-64 character hexadecimal commit")
    return revision


def _optional_revision(value: str | None) -> str | None:
    return _resolved_revision(value) if value else None


def _specialist_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not sanitized:
        raise ValueError("genome_id must contain at least one safe character")
    return sanitized[:200]


if __name__ == "__main__":
    main()
