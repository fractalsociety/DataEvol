from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_map_with_path, tree_unflatten


KL_NUMERICAL_FLOOR = 1e-3


@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float = 0.2
    beta: float = 0.02
    learning_rate: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    kl_numerical_floor: float = KL_NUMERICAL_FLOOR


@dataclass(frozen=True)
class RolloutBatch:
    tokens: mx.array
    prompt_lengths: mx.array
    completion_lengths: mx.array
    old_logprobs: mx.array
    reference_logprobs: mx.array
    reference_logits: mx.array
    advantages: mx.array


class StatefulSampler:
    """NumPy-backed categorical sampler with explicit, checkpointable state."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.counter = 0
        self.generator = np.random.default_rng(self.seed)

    def sample(self, logits: Sequence[float], temperature: float) -> int:
        if temperature <= 0:
            return int(np.argmax(np.asarray(logits)))
        values = np.asarray(logits, dtype=np.float64) / temperature
        values -= values.max()
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        self.counter += 1
        return int(self.generator.choice(len(probabilities), p=probabilities))

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "counter": self.counter,
            "bit_generator_state": self.generator.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.counter = int(state["counter"])
        self.generator = np.random.default_rng()
        self.generator.bit_generator.state = dict(state["bit_generator_state"])


class ToyCausalPolicy(nn.Module):
    """Small causal policy used to audit the same RL math as model-scale jobs."""

    def __init__(self, vocab_size: int = 16, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = mx.cumsum(self.embedding(tokens), axis=1)
        return self.output(hidden)


class ToyUniformLoRAPolicy(nn.Module):
    """Frozen toy base with a trainable low-rank output adapter."""

    def __init__(self, vocab_size: int = 16, hidden_size: int = 12, rank: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.base_output = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lora_a = nn.Linear(hidden_size, rank, bias=False)
        self.lora_b = nn.Linear(rank, vocab_size, bias=False)
        self.embedding.freeze()
        self.base_output.freeze()
        self.lora_b.weight = mx.zeros_like(self.lora_b.weight)

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = mx.cumsum(self.embedding(tokens), axis=1)
        return self.base_output(hidden) + self.lora_b(self.lora_a(hidden))


def clone_frozen(model: nn.Module, factory: type[nn.Module] | None = None) -> nn.Module:
    copy = (factory or type(model))()
    parameters = tree_map(lambda value: mx.array(value), model.parameters())
    copy.update(parameters)
    copy.freeze()
    mx.eval(copy.parameters())
    return copy


def completion_mask(
    sequence_length: int,
    prompt_lengths: mx.array,
    completion_lengths: mx.array,
) -> mx.array:
    """Mask logits that predict completion tokens in a causal sequence."""
    positions = mx.arange(sequence_length - 1)[None, :]
    starts = prompt_lengths[:, None] - 1
    stops = starts + completion_lengths[:, None]
    return (positions >= starts) & (positions < stops)


def token_logprobs(logits: mx.array, tokens: mx.array) -> mx.array:
    predictive = nn.log_softmax(logits[:, :-1, :].astype(mx.float32), axis=-1)
    return mx.take_along_axis(predictive, tokens[:, 1:, None], axis=-1).squeeze(-1)


def completion_logprobs(
    logits: mx.array,
    tokens: mx.array,
    prompt_lengths: mx.array,
    completion_lengths: mx.array,
) -> tuple[mx.array, mx.array]:
    mask = completion_mask(tokens.shape[1], prompt_lengths, completion_lengths)
    return token_logprobs(logits, tokens), mask


def masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(values.dtype)
    return mx.sum(values * weights) / mx.maximum(mx.sum(weights), mx.array(1.0))


def categorical_kl(
    current_logits: mx.array,
    reference_logits: mx.array,
    mask: mx.array,
    numerical_floor: float = KL_NUMERICAL_FLOOR,
) -> mx.array:
    current_log = nn.log_softmax(current_logits[:, :-1, :].astype(mx.float32), axis=-1)
    reference_log = nn.log_softmax(reference_logits[:, :-1, :].astype(mx.float32), axis=-1)
    probabilities = mx.exp(current_log)
    per_position = mx.sum(probabilities * (current_log - reference_log), axis=-1)
    return mx.maximum(masked_mean(per_position, mask) - numerical_floor, 0.0)


def ppo_loss(
    current_logits: mx.array,
    batch: RolloutBatch,
    config: PPOConfig,
    *,
    include_kl: bool = True,
) -> tuple[mx.array, dict[str, mx.array]]:
    current_logprobs, mask = completion_logprobs(
        current_logits, batch.tokens, batch.prompt_lengths, batch.completion_lengths
    )
    ratio = mx.exp(current_logprobs - mx.stop_gradient(batch.old_logprobs))
    advantages = batch.advantages[:, None]
    unclipped = ratio * advantages
    clipped = mx.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
    policy_loss = -masked_mean(mx.minimum(unclipped, clipped), mask)
    kl = categorical_kl(
        current_logits,
        mx.stop_gradient(batch.reference_logits),
        mask,
        numerical_floor=config.kl_numerical_floor,
    )
    total = policy_loss + (config.beta * kl if include_kl else 0.0)
    clip_fraction = masked_mean((mx.abs(ratio - 1.0) > config.clip_epsilon).astype(mx.float32), mask)
    return total, {
        "policy_loss": policy_loss,
        "kl": kl,
        "clip_fraction": clip_fraction,
        "ratio_mean": masked_mean(ratio, mask),
    }


def make_rollout_batch(
    policy: nn.Module,
    reference: nn.Module,
    tokens: mx.array,
    prompt_lengths: mx.array,
    completion_lengths: mx.array,
    rewards: mx.array,
) -> RolloutBatch:
    policy_logits = policy(tokens)
    reference_logits = reference(tokens)
    old_logprobs, _ = completion_logprobs(policy_logits, tokens, prompt_lengths, completion_lengths)
    reference_logprobs, _ = completion_logprobs(reference_logits, tokens, prompt_lengths, completion_lengths)
    # Preserve the magnitude of bounded shaping rewards. Variance normalization
    # would turn tiny shaping-only differences into unit-scale policy updates.
    advantages = rewards - mx.mean(rewards)
    mx.eval(policy_logits, reference_logits, old_logprobs, reference_logprobs, advantages)
    return RolloutBatch(
        tokens=tokens,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        old_logprobs=mx.stop_gradient(old_logprobs),
        reference_logprobs=mx.stop_gradient(reference_logprobs),
        reference_logits=mx.stop_gradient(reference_logits),
        advantages=mx.stop_gradient(advantages),
    )


def global_norm(tree: Mapping[str, Any]) -> mx.array:
    leaves = [value for _, value in tree_flatten(tree)]
    return mx.sqrt(sum(mx.sum(value * value) for value in leaves))


def clip_gradients(gradients: Mapping[str, Any], maximum: float) -> tuple[Mapping[str, Any], mx.array]:
    norm = global_norm(gradients)
    scale = mx.minimum(1.0, maximum / (norm + 1e-8))
    return tree_map(lambda value: value * scale, gradients), norm


def sample_completion(
    model: nn.Module,
    prompt: Sequence[int],
    sampler: StatefulSampler,
    *,
    max_tokens: int,
    temperature: float,
) -> list[int]:
    tokens = list(prompt)
    generated: list[int] = []
    for _ in range(max_tokens):
        logits = model(mx.array([tokens], dtype=mx.int32))[0, -1]
        mx.eval(logits)
        token = sampler.sample(np.asarray(logits), temperature)
        tokens.append(token)
        generated.append(token)
    return generated


def save_checkpoint(
    root: str | Path,
    *,
    update: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    gradient_accumulator: Mapping[str, Any],
    sampler: StatefulSampler,
    scheduler_state: Mapping[str, Any],
    training_state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"checkpoint-{update:08d}"
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
    try:
        if not optimizer._initialized:
            optimizer.init(model.trainable_parameters())
        _save_tree(temporary / "adapter.safetensors", model.trainable_parameters())
        _save_tree(temporary / "optimizer.safetensors", optimizer.state)
        _save_tree(temporary / "gradient_accumulator.safetensors", gradient_accumulator)
        mx.save_safetensors(
            str(temporary / "mlx_random_state.safetensors"),
            {"seed": mx.array(sampler.seed, dtype=mx.uint64), "counter": mx.array(sampler.counter, dtype=mx.uint64)},
        )
        _write_pickle(temporary / "python_random_state.pkl", random.getstate())
        _write_pickle(temporary / "numpy_random_state.pkl", np.random.get_state())
        _write_json(temporary / "scheduler_state.json", scheduler_state)
        _write_json(temporary / "sampler_state.json", sampler.state_dict())
        _write_json(temporary / "training_state.json", training_state)
        _write_json(temporary / "config.lock.yaml", config)
        checksums = {
            item.name: _file_hash(item)
            for item in sorted(temporary.iterdir())
            if item.is_file()
        }
        _write_json(temporary / "checksums.json", checksums)
        _validate_checkpoint_directory(temporary)
        _fsync_directory(temporary)
        (temporary / "COMPLETE").write_text("complete\n", encoding="ascii")
        _fsync_file(temporary / "COMPLETE")
        _fsync_directory(temporary)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        _atomic_latest(root, destination.name)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint(
    checkpoint: str | Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    sampler: StatefulSampler,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    _validate_checkpoint_directory(checkpoint, require_complete=True)
    model.load_weights(list(mx.load(str(checkpoint / "adapter.safetensors")).items()), strict=False)
    saved_optimizer = mx.load(str(checkpoint / "optimizer.safetensors"))
    optimizer.init(model.trainable_parameters())
    expected_paths = {name for name, _ in tree_flatten(optimizer.state)}
    if expected_paths != set(saved_optimizer):
        raise ValueError("checkpoint optimizer state does not match current trainable parameters")
    optimizer._state = tree_map_with_path(lambda path, value: saved_optimizer[path], optimizer.state)
    optimizer._initialized = True
    sampler.load_state_dict(json.loads((checkpoint / "sampler_state.json").read_text(encoding="utf-8")))
    random.setstate(pickle.loads((checkpoint / "python_random_state.pkl").read_bytes()))
    np.random.set_state(pickle.loads((checkpoint / "numpy_random_state.pkl").read_bytes()))
    mx.eval(model.parameters(), optimizer.state)
    return {
        "training_state": json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8")),
        "scheduler_state": json.loads((checkpoint / "scheduler_state.json").read_text(encoding="utf-8")),
        "gradient_accumulator": _load_tree(checkpoint / "gradient_accumulator.safetensors"),
    }


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in tree_flatten(model.trainable_parameters()):
        digest.update(name.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def tree_allclose(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float = 0.0) -> bool:
    left_flat = dict(tree_flatten(left))
    right_flat = dict(tree_flatten(right))
    if set(left_flat) != set(right_flat):
        return False
    return all(
        np.allclose(np.asarray(left_flat[key]), np.asarray(right_flat[key]), atol=tolerance, rtol=tolerance)
        for key in left_flat
    )


def _save_tree(path: Path, tree: Mapping[str, Any]) -> None:
    values = dict(tree_flatten(tree))
    if not values:
        values = {"__empty__": mx.array(0, dtype=mx.int32)}
    mx.save_safetensors(str(path), values)


def _load_tree(path: Path) -> dict[str, Any]:
    values = list(mx.load(str(path)).items())
    if len(values) == 1 and values[0][0] == "__empty__":
        return {}
    return tree_unflatten(values)


def _validate_checkpoint_directory(path: Path, require_complete: bool = False) -> None:
    required = {
        "adapter.safetensors",
        "optimizer.safetensors",
        "gradient_accumulator.safetensors",
        "mlx_random_state.safetensors",
        "python_random_state.pkl",
        "numpy_random_state.pkl",
        "scheduler_state.json",
        "sampler_state.json",
        "training_state.json",
        "config.lock.yaml",
        "checksums.json",
    }
    missing = required - {item.name for item in path.iterdir()}
    if missing:
        raise ValueError(f"incomplete checkpoint {path}: {sorted(missing)}")
    if require_complete and not (path / "COMPLETE").is_file():
        raise ValueError(f"checkpoint lacks COMPLETE sentinel: {path}")
    checksums = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        if _file_hash(path / name) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    for name in ("adapter.safetensors", "optimizer.safetensors", "gradient_accumulator.safetensors", "mlx_random_state.safetensors"):
        mx.load(str(path / name))
    pickle.loads((path / "python_random_state.pkl").read_bytes())
    pickle.loads((path / "numpy_random_state.pkl").read_bytes())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _fsync_file(path)


def _write_pickle(path: Path, value: Any) -> None:
    path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    _fsync_file(path)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_latest(root: Path, checkpoint_name: str) -> None:
    temporary = root / ".LATEST.tmp"
    temporary.write_text(checkpoint_name + "\n", encoding="ascii")
    _fsync_file(temporary)
    temporary.replace(root / "LATEST")
    _fsync_directory(root)
