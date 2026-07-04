from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "dataevol.mlx_layer_specialist.v1"
FREEZE_STRATEGY = "mlx_full_layer"


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
    contribution_profile_id: str | None = None,
    contribution_profile_hash: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    short = Path(base_model).name if "/" not in base_model else base_model.split("/")[-1]
    files = list(tensor_files)
    return {
        "schema": SCHEMA,
        "name": f"{short}__{task_type}__L{layer_index}",
        "base_model_id": base_model,
        "base_model_hash": model_hash(base_model),
        "layer_index": layer_index,
        "task_type": task_type,
        "training_mode": training_mode,
        "freeze_strategy": FREEZE_STRATEGY,
        "dataset_uri": dataset_uri,
        "dataset_hash": sha256_jsonl(dataset_uri),
        "contribution_profile_id": contribution_profile_id,
        "contribution_profile_hash": contribution_profile_hash,
        "baseline_metric": baseline_metric,
        "eval_metric": eval_metric,
        "trainable_param_names": sorted(trainable_param_names),
        "trainable_param_count": trainable_param_count,
        "frozen_param_count": frozen_param_count,
        "param_shapes": param_shapes,
        "tensor_files": files,
        "sha256": {name: sha256_file(out / name) for name in files},
        "runtime_version": runtime_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def model_hash(base_model: str) -> str | None:
    path = Path(base_model)
    if not path.exists():
        return None
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        candidate = path / name
        if candidate.exists() and candidate.is_file():
            return sha256_file(candidate)
    return None


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
) -> dict[str, Any]:
    """Train a full replacement tensor set for one MLX decoder layer.

    The real path requires mlx/mlx_lm and a model that exposes
    `model.model.layers`. Tests can still exercise manifest/export helpers
    without loading a large model.
    """
    if training_mode == "rl":
        raise NotImplementedError("RL layer-specialist training is not implemented in phase 1; use training_mode='sft'.")
    if training_mode != "sft":
        raise ValueError("training_mode must be sft or rl")

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load
        from safetensors.numpy import save_file
    except Exception as exc:  # pragma: no cover - depends on optional local MLX install.
        raise RuntimeError(f"MLX specialist training requires mlx, mlx_lm, and safetensors: {exc}") from exc

    mx.random.seed(seed)
    model, tokenizer = load(base_model)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("loaded model does not expose model.model.layers")
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f"layer_index {layer_index} is outside model layer range 0..{len(layers)-1}")

    rows = _load_jsonl(dataset_uri)
    if not rows:
        raise ValueError("dataset contains no examples")
    split_at = max(1, min(len(rows) - 1, int(len(rows) * (1.0 - eval_split)))) if len(rows) > 1 else 1
    train_rows = rows[:split_at]
    eval_rows = rows[split_at:] or rows[:1]

    baseline_metric = _eval_loss(model, tokenizer, eval_rows, max_seq_len)
    model.freeze()
    model.unfreeze(f"layers.{layer_index}")
    optimizer = optim.Adam(learning_rate=learning_rate)

    for step in range(1, max_steps + 1):
        batch = train_rows[(step - 1) % len(train_rows)]
        loss, grads = nn.value_and_grad(model, lambda m: _loss(m, tokenizer, [batch], max_seq_len))(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print(f"step {step} loss {float(loss):.6f}", flush=True)

    eval_metric = _eval_loss(model, tokenizer, eval_rows, max_seq_len)
    layer_params = _layer_parameters(model, layer_index)
    all_params = dict(_flatten_params(model.parameters()))
    frozen_param_count = sum(_numel(v) for name, v in all_params.items() if name not in layer_params)
    trainable_param_count = sum(_numel(v) for v in layer_params.values())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor_name = f"layer_{layer_index}.safetensors"
    tensors = {name: _to_numpy(value) for name, value in layer_params.items()}
    save_file(tensors, str(out / tensor_name))
    manifest = build_manifest(
        base_model=base_model,
        layer_index=layer_index,
        task_type=task_type,
        training_mode=training_mode,
        dataset_uri=dataset_uri,
        output_dir=out,
        tensor_files=[tensor_name],
        trainable_param_names=layer_params.keys(),
        trainable_param_count=trainable_param_count,
        frozen_param_count=frozen_param_count,
        param_shapes={name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in layer_params.items()},
        baseline_metric=float(baseline_metric),
        eval_metric=float(eval_metric),
        contribution_profile_id=contribution_profile_id,
        contribution_profile_hash=contribution_profile_hash,
    )
    manifest_path = out / f"{manifest['name']}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=100)
    train.add_argument("--max-seq-length", type=int, default=512)
    train.add_argument("--eval-split", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--contribution-profile-id")
    train.add_argument("--contribution-profile-hash")
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


def _loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    losses = []
    for row in rows:
        ids = tokenizer.encode(_prompt_completion(row))[:max_seq_len]
        if len(ids) < 2:
            continue
        x = mx.array(ids[:-1])[None, :]
        y = mx.array(ids[1:])[None, :]
        logits = model(x)
        losses.append(nn.losses.cross_entropy(logits, y).mean())
    if not losses:
        return mx.array(0.0)
    return sum(losses) / len(losses)


def _eval_loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int) -> float:
    import mlx.core as mx

    loss = _loss(model, tokenizer, rows, max_seq_len)
    mx.eval(loss)
    return float(loss)


def _layer_parameters(model: Any, layer_index: int) -> dict[str, Any]:
    prefix = f"layers.{layer_index}."
    params = {name: value for name, value in _flatten_params(model.parameters()) if name.startswith(prefix)}
    if not params:
        alt_prefix = f"model.layers.{layer_index}."
        params = {name: value for name, value in _flatten_params(model.parameters()) if name.startswith(alt_prefix)}
    if not params:
        raise RuntimeError(f"no parameters matched layer {layer_index}")
    return params


def _flatten_params(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(tree, dict):
        out: list[tuple[str, Any]] = []
        for key, value in tree.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_flatten_params(value, next_prefix))
        return out
    return [(prefix, tree)]


def _numel(value: Any) -> int:
    n = 1
    for dim in getattr(value, "shape", ()):
        n *= int(dim)
    return n


def _to_numpy(value: Any) -> Any:
    import numpy as np

    return np.asarray(value)


if __name__ == "__main__":
    main()
