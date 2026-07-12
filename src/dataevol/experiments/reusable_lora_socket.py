from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_SCHEMA = "dataevol.reusable_lora_socket_experiment.v1"
SOCKET_SCHEMA = "dataevol.reusable_lora_socket.v1"
MODEL_ID = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
MODEL_REVISION = "01a708812690b4685ab18e3b0a27848f92c50746"
BANDS = ((0, 3), (4, 7), (8, 11), (12, 14), (15, 18), (19, 21))
FAMILY_MODULES = {
    "A": ("self_attn.q_proj", "self_attn.v_proj"),
    "B": ("self_attn.o_proj",),
    "C": ("mlp.down_proj",),
}
# TinyLlama dimensions: q 2048x2048, v 256x2048, o 2048x2048,
# down 2048x5632. A LoRA matrix has r*(input+output) parameters.
FAMILY_PARAMS_PER_RANK = {"A": 6_400, "B": 4_096, "C": 7_680}
DEFAULT_RANK = 64


@dataclass(frozen=True)
class TrainConfig:
    optimizer_steps: int
    batch_size: int = 1
    grad_accumulation: int = 1
    max_seq_length: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


class _CaptureCallback:
    def __init__(self) -> None:
        self.train: list[dict[str, Any]] = []
        self.validation: list[dict[str, Any]] = []

    def on_train_loss_report(self, info: dict[str, Any]) -> None:
        self.train.append(dict(info))

    def on_val_loss_report(self, info: dict[str, Any]) -> None:
        self.validation.append(dict(info))


def prepare_datasets(output_dir: str | Path, *, seed: int = 1701) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "dataevol.reusable_socket_datasets.v1",
        "seed": seed,
        "splits": {},
        "held_out_from_socket_discovery": ["json"],
    }
    sizes = {"train": 2_000, "valid": 300, "test": 500}
    for task in ("arithmetic", "python", "json"):
        task_dir = output / task
        task_dir.mkdir(exist_ok=True)
        manifest["splits"][task] = {}
        for split_index, (split, count) in enumerate(sizes.items()):
            rows = _generate_rows(task, split, count, seed + split_index * 10_000)
            payload = _jsonl(rows)
            path = task_dir / f"{split}.jsonl"
            _write_immutable(path, payload)
            manifest["splits"][task][split] = {
                "path": str(path), "rows": count, "sha256": hashlib.sha256(payload).hexdigest()
            }
    body = dict(manifest)
    manifest["dataset_hash"] = _hash(body)
    _write_immutable(output / "manifest.json", _json(manifest))
    return manifest


def generate_socket_candidates(*, count: int = 12, seed: int = 1701, target_parameters: int = 2_400_000) -> list[dict[str, Any]]:
    if count < 3:
        raise ValueError("socket search requires at least three candidates")
    rng = random.Random(seed)
    candidates = []
    modes = (["balanced"] * math.ceil(count / 3) + ["attention-heavy"] * math.ceil(count / 3) + ["mlp-heavy"] * count)[:count]
    for index, mode in enumerate(modes):
        layers = [rng.randint(low, high) for low, high in BANDS]
        if mode == "balanced":
            families = list("AABBCC")
        elif mode == "attention-heavy":
            families = list("AAAABC")
        else:
            families = list("ABCCCC")
        rng.shuffle(families)
        per_rank = sum(FAMILY_PARAMS_PER_RANK[family] for family in families)
        rank = _closest_rank(target_parameters, per_rank)
        entries = [
            {"layer": layer, "family": family, "rank": rank}
            for layer, family in zip(layers, families)
        ]
        candidates.append(create_socket(f"candidate-{index:02d}", entries, generation=mode))
    return candidates


def create_socket(
    socket_id: str,
    entries: Iterable[Mapping[str, Any]],
    *,
    generation: str,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    normalized = []
    covered_bands = set()
    for raw in entries:
        layer = int(raw["layer"])
        family = str(raw["family"])
        rank = int(raw["rank"])
        if layer < 0 or layer > 21 or family not in FAMILY_MODULES or rank <= 0:
            raise ValueError("invalid reusable socket entry")
        normalized.append({"layer": layer, "family": family, "rank": rank})
        covered_bands.add(_band_index(layer))
    normalized.sort(key=lambda item: (item["layer"], item["family"]))
    is_single_layer = "baseline-single-layer" in generation
    is_uniform = "baseline-uniform" in generation
    if not is_single_layer and len(covered_bands) < 4:
        raise ValueError("socket must cover at least four depth bands")
    if not is_uniform and (not ({item["family"] for item in normalized} & {"A", "B"}) or not any(item["family"] == "C" for item in normalized)):
        raise ValueError("socket requires attention and MLP modules")
    if len({(item["layer"], item["family"]) for item in normalized}) != len(normalized):
        raise ValueError("socket contains duplicate layer-family entries")
    trainable = socket_parameter_count(normalized)
    if not 2_000_000 <= trainable <= 4_000_000:
        raise ValueError(f"socket trainable parameter count {trainable} is outside 2-4 million")
    body = {
        "schema": SOCKET_SCHEMA,
        "socket_id": socket_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "generation": generation,
        "entries": normalized,
        "trainable_parameters": trainable,
    }
    return {**body, "socket_hash": _hash(body)}


def socket_parameter_count(entries: Iterable[Mapping[str, Any]]) -> int:
    return sum(FAMILY_PARAMS_PER_RANK[str(item["family"])] * int(item["rank"]) for item in entries)


def parameter_matched_uniform(target: int) -> dict[str, Any]:
    layers = (0, 4, 8, 12, 16, 20)
    rank = _closest_rank(target, len(layers) * FAMILY_PARAMS_PER_RANK["A"])
    return create_socket("uniform", ({"layer": layer, "family": "A", "rank": rank} for layer in layers), generation="baseline-uniform")


def parameter_matched_single(layer: int, target: int, *, socket_id: str | None = None) -> dict[str, Any]:
    per_rank = sum(FAMILY_PARAMS_PER_RANK.values())
    rank = _closest_rank(target, per_rank)
    entries = ({"layer": layer, "family": family, "rank": rank} for family in FAMILY_MODULES)
    return create_socket(socket_id or f"single-{layer}", entries, generation="baseline-single-layer")


def parameter_matched_random(reference: Mapping[str, Any], *, count: int = 3, seed: int = 991) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    families = [(str(item["family"]), int(item["rank"])) for item in reference["entries"]]
    sockets = []
    reference_layers = [int(item["layer"]) for item in reference["entries"]]
    for index in range(count):
        while True:
            layers = [rng.randint(low, high) for low, high in BANDS]
            if layers != reference_layers:
                break
        entries = [
            {"layer": layer, "family": family, "rank": rank}
            for layer, (family, rank) in zip(layers, families)
        ]
        sockets.append(create_socket(f"random-{index}", entries, generation="baseline-random-matched"))
    return sockets


def train_socket_expert(
    *,
    model_path: str | Path,
    socket: Mapping[str, Any],
    data_dir: str | Path,
    output_dir: str | Path,
    task: str,
    config: TrainConfig,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset, CompletionsDataset
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train

    output = Path(output_dir)
    result_path = output / "training_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, tokenizer = load(str(model_path))
    _assert_tinyllama(model)
    _apply_socket(model, socket)
    trainable = sum(int(math.prod(value.shape)) for _, value in tree_flatten(model.trainable_parameters()))
    if trainable != int(socket["trainable_parameters"]):
        raise ValueError(f"runtime trainable parameter count {trainable} does not match socket manifest")
    data_path = Path(data_dir)
    train_rows = _read_jsonl(data_path / "train.jsonl")
    valid_rows = _read_jsonl(data_path / "valid.jsonl")
    train_set = CacheDataset(CompletionsDataset(train_rows, tokenizer, "prompt", "completion", True))
    valid_set = CacheDataset(CompletionsDataset(valid_rows, tokenizer, "prompt", "completion", True))
    callback = _CaptureCallback()
    total_iterations = config.optimizer_steps * config.grad_accumulation
    warmup = max(1, round(total_iterations * 0.03))
    decay = optim.cosine_decay(config.learning_rate, max(1, total_iterations - warmup), end=config.learning_rate * 0.1)
    schedule = optim.join_schedules(
        [optim.linear_schedule(config.learning_rate * 0.1, config.learning_rate, warmup), decay], [warmup]
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=config.weight_decay)
    mx.reset_peak_memory()
    started = time.perf_counter()
    args = TrainingArgs(
        batch_size=config.batch_size,
        iters=total_iterations,
        val_batches=min(30, len(valid_rows)),
        steps_per_report=max(1, total_iterations // 5),
        steps_per_eval=total_iterations,
        steps_per_save=total_iterations,
        adapter_file=output / "adapters.safetensors",
        max_seq_length=config.max_seq_length,
        grad_accumulation_steps=config.grad_accumulation,
        clear_cache_threshold=0,
    )
    train(model, optimizer, train_set, valid_set, args, training_callback=callback)
    final_loss = evaluate(model, valid_set, config.batch_size, min(50, len(valid_rows)), config.max_seq_length)
    elapsed = time.perf_counter() - started
    adapter_hash = _file_hash(output / "adapters.safetensors")
    body = {
        "schema": "dataevol.reusable_socket_training.v1",
        "task": task,
        "socket_id": socket["socket_id"],
        "socket_hash": socket["socket_hash"],
        "model_revision": socket["model_revision"],
        "seed": seed,
        "data_order_seed": seed,
        "config": config.__dict__,
        "optimizer_iterations": total_iterations,
        "optimizer_updates": config.optimizer_steps,
        "trainable_parameters": trainable,
        "validation_loss": float(final_loss),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "adapter_sha256": adapter_hash,
        "train_reports": callback.train,
        "validation_reports": callback.validation,
    }
    result = {**body, "result_hash": _hash(body)}
    _write_immutable(result_path, _json(result))
    _write_immutable(output / "socket.json", _json(dict(socket)))
    return result


def evaluate_loss(
    model_path: str | Path,
    data_dir: str | Path,
    *,
    max_seq_length: int = 256,
    batches: int = 50,
) -> float:
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset, CompletionsDataset
    from mlx_lm.tuner.trainer import evaluate

    model, tokenizer = load(str(model_path))
    rows = _read_jsonl(Path(data_dir) / "valid.jsonl")
    dataset = CacheDataset(CompletionsDataset(rows, tokenizer, "prompt", "completion", True))
    return float(evaluate(model, dataset, 1, min(batches, len(rows)), max_seq_length))


def evaluate_json_expert(
    *,
    model_path: str | Path,
    socket: Mapping[str, Any] | None,
    adapter_path: str | Path | None,
    dataset_path: str | Path,
    limit: int = 500,
    generation_batch_size: int = 32,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(model_path))
    if socket is not None:
        _apply_socket(model, socket)
        if adapter_path is None:
            raise ValueError("socket evaluation requires adapter weights")
        model.load_weights(list(mx.load(str(adapter_path)).items()), strict=False)
        mx.eval(model.parameters())
    rows = _read_jsonl(Path(dataset_path))[:limit]
    valid_json = schema_valid = exact_fields = complete = 0
    field_total = 0
    outputs = []
    sampler = make_sampler(temp=0.0)
    started = time.perf_counter()
    for offset in range(0, len(rows), generation_batch_size):
        batch_rows = rows[offset : offset + generation_batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, return_dict=False
            )
            for row in batch_rows
        ]
        response = batch_generate(model, tokenizer, prompts, max_tokens=96, sampler=sampler, verbose=False)
        for row, output in zip(batch_rows, response.texts):
            parsed = _extract_json(output)
            expected = json.loads(row["completion"])
            if parsed is not None:
                valid_json += 1
                if set(parsed) == set(expected):
                    schema_valid += 1
                matches = sum(parsed.get(key) == value for key, value in expected.items())
                exact_fields += matches
                field_total += len(expected)
                if parsed == expected:
                    complete += 1
            else:
                field_total += len(expected)
            if len(outputs) < 25:
                outputs.append({"id": row["id"], "output": output, "parsed": parsed, "expected": expected})
    count = len(rows)
    return {
        "rows": count,
        "valid_json_rate": valid_json / count if count else 0.0,
        "schema_valid_rate": schema_valid / count if count else 0.0,
        "exact_field_accuracy": exact_fields / field_total if field_total else 0.0,
        "complete_record_accuracy": complete / count if count else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "sample_outputs": outputs,
    }


def evaluate_discovery_expert(
    *,
    model_path: str | Path,
    socket: Mapping[str, Any],
    adapter_path: str | Path,
    dataset_path: str | Path,
    task: str,
    limit: int = 500,
    generation_batch_size: int = 32,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    if task not in {"arithmetic", "python"}:
        raise ValueError("discovery evaluation supports arithmetic or python")
    model, tokenizer = load(str(model_path))
    _apply_socket(model, socket)
    model.load_weights(list(mx.load(str(adapter_path)).items()), strict=False)
    mx.eval(model.parameters())
    rows = _read_jsonl(Path(dataset_path))[:limit]
    passed = 0
    samples = []
    sampler = make_sampler(temp=0.0)
    started = time.perf_counter()
    for offset in range(0, len(rows), generation_batch_size):
        batch_rows = rows[offset : offset + generation_batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, return_dict=False
            )
            for row in batch_rows
        ]
        response = batch_generate(model, tokenizer, prompts, max_tokens=96, sampler=sampler, verbose=False)
        for row, output in zip(batch_rows, response.texts):
            correct = _arithmetic_correct(output, row["answer"]) if task == "arithmetic" else _python_unit_test_pass(output, row["completion"])
            passed += int(correct)
            if len(samples) < 25:
                samples.append({"id": row["id"], "output": output, "expected": row["completion"], "passed": correct})
    return {
        "task": task,
        "metric": "exact_numerical_answer_accuracy" if task == "arithmetic" else "restricted_subprocess_unit_test_pass_rate",
        "rows": len(rows),
        "passed": passed,
        "accuracy": passed / len(rows) if rows else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "sample_outputs": samples,
    }


def run_search(
    *,
    model_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    candidates: Iterable[Mapping[str, Any]],
    tasks: tuple[str, ...],
    config: TrainConfig,
    seed: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "search_report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    base_losses = {task: evaluate_loss(model_path, Path(data_root) / task, max_seq_length=config.max_seq_length) for task in tasks}
    rows = []
    for candidate in candidates:
        task_results = {}
        improvements = []
        for task in tasks:
            result = train_socket_expert(
                model_path=model_path,
                socket=candidate,
                data_dir=Path(data_root) / task,
                output_dir=output / str(candidate["socket_id"]) / task,
                task=task,
                config=config,
                seed=seed,
            )
            task_results[task] = result
            improvements.append(max(0.0, (base_losses[task] - result["validation_loss"]) / base_losses[task]))
        score = math.prod(max(value, 1e-12) for value in improvements) ** (1 / len(improvements))
        rows.append({"socket": dict(candidate), "task_results": task_results, "normalized_improvements": improvements, "score": score})
    rows.sort(key=lambda item: (-item["score"], item["socket"]["socket_hash"]))
    body = {
        "schema": "dataevol.reusable_socket_search.v1",
        "tasks": list(tasks),
        "json_excluded": "json" not in tasks,
        "base_validation_losses": base_losses,
        "config": config.__dict__,
        "ranked_candidates": rows,
        "winner": rows[0]["socket"],
    }
    report = {**body, "search_hash": _hash(body)}
    _write_immutable(report_path, _json(report))
    return report


def confirm_discovery_finalists(
    *,
    model_path: str | Path,
    data_root: str | Path,
    proxy_report: Mapping[str, Any],
    output_dir: str | Path,
    config: TrainConfig = TrainConfig(500),
    seeds: tuple[int, ...] = (17, 29, 43),
    finalist_count: int = 3,
) -> dict[str, Any]:
    if proxy_report.get("tasks") != ["arithmetic", "python"] or not proxy_report.get("json_excluded"):
        raise ValueError("discovery confirmation requires a JSON-excluded arithmetic/Python proxy report")
    finalists = [dict(row["socket"]) for row in proxy_report["ranked_candidates"][:finalist_count]]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "confirmation_report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    base_losses = dict(proxy_report["base_validation_losses"])
    ranked = []
    for socket in finalists:
        seed_scores = []
        seed_results = []
        for seed in seeds:
            task_results = {}
            improvements = []
            for task in ("arithmetic", "python"):
                result = train_socket_expert(
                    model_path=model_path,
                    socket=socket,
                    data_dir=Path(data_root) / task,
                    output_dir=output / str(socket["socket_id"]) / f"seed-{seed}" / task,
                    task=task,
                    config=config,
                    seed=seed,
                )
                task_results[task] = result
                improvements.append(max(0.0, (float(base_losses[task]) - result["validation_loss"]) / float(base_losses[task])))
            score = math.sqrt(max(improvements[0], 1e-12) * max(improvements[1], 1e-12))
            seed_scores.append(score)
            seed_results.append({"seed": seed, "score": score, "task_results": task_results})
        ranked.append({
            "socket": socket,
            "mean_score": sum(seed_scores) / len(seed_scores),
            "score_std": _population_std(seed_scores),
            "seed_results": seed_results,
        })
    ranked.sort(key=lambda row: (-row["mean_score"], row["score_std"], row["socket"]["socket_hash"]))
    body = {
        "schema": "dataevol.reusable_socket_confirmation.v1",
        "tasks": ["arithmetic", "python"],
        "json_excluded": True,
        "proxy_search_hash": proxy_report["search_hash"],
        "config": config.__dict__,
        "seeds": list(seeds),
        "ranked_finalists": ranked,
        "winner": ranked[0]["socket"],
    }
    report = {**body, "confirmation_hash": _hash(body)}
    _write_immutable(report_path, _json(report))
    return report


def run_mvp(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    proxy_steps: int = 100,
    final_steps: int = 500,
    eval_limit: int = 500,
    seeds: tuple[int, ...] = (17, 29, 43),
) -> dict[str, Any]:
    root = Path(output_dir)
    datasets = prepare_datasets(root / "datasets")
    candidates = generate_socket_candidates()
    _write_immutable(root / "candidate_sockets.json", _json({"candidates": candidates}))
    proxy = TrainConfig(proxy_steps, max_seq_length=256)
    discovery = run_search(
        model_path=model_path, data_root=root / "datasets", output_dir=root / "discovery_search",
        candidates=candidates, tasks=("arithmetic", "python"), config=proxy, seed=17,
    )
    universal = discovery["winner"]
    target = int(universal["trainable_parameters"])
    baselines = [parameter_matched_uniform(target)]
    single_candidates = [parameter_matched_single(layer, target) for layer in (2, 6, 10, 15, 20)]
    single_search = run_search(
        model_path=model_path, data_root=root / "datasets", output_dir=root / "single_layer_search",
        candidates=single_candidates, tasks=("json",), config=proxy, seed=17,
    )
    best_single = single_search["winner"]
    json_search = run_search(
        model_path=model_path, data_root=root / "datasets", output_dir=root / "json_specific_search",
        candidates=candidates, tasks=("json",), config=proxy, seed=17,
    )
    json_specific = json_search["winner"]
    baselines.extend([best_single, *parameter_matched_random(universal), universal, json_specific])
    for baseline in baselines:
        delta = abs(int(baseline["trainable_parameters"]) - target) / target
        if delta > 0.03:
            raise ValueError(f"baseline {baseline['socket_id']} differs from universal budget by {delta:.2%}")
    named = [("uniform", baselines[0]), ("best-single", best_single)]
    named.extend((f"random-{index}", socket) for index, socket in enumerate(baselines[2:5]))
    named.extend((("universal", universal), ("json-specific", json_specific)))
    final_config = TrainConfig(final_steps, max_seq_length=256)
    results = []
    for method, socket in named:
        for seed in seeds:
            training = train_socket_expert(
                model_path=model_path, socket=socket, data_dir=root / "datasets/json",
                output_dir=root / "final" / method / f"seed-{seed}", task="json",
                config=final_config, seed=seed,
            )
            metrics_path = root / "final" / method / f"seed-{seed}" / "test_metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                metrics = evaluate_json_expert(
                    model_path=model_path, socket=socket,
                    adapter_path=root / "final" / method / f"seed-{seed}" / "adapters.safetensors",
                    dataset_path=root / "datasets/json/test.jsonl", limit=eval_limit,
                )
                _write_immutable(metrics_path, _json(metrics))
            results.append({"method": method, "seed": seed, "socket": socket, "training": training, "metrics": metrics})
    frozen_path = root / "frozen_base_json_metrics.json"
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    else:
        frozen = evaluate_json_expert(
            model_path=model_path, socket=None, adapter_path=None,
            dataset_path=root / "datasets/json/test.jsonl", limit=eval_limit,
        )
        _write_immutable(frozen_path, _json(frozen))
    summary = _summarize(results, frozen, discovery, json_search)
    body = {
        "schema": EXPERIMENT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_layers": 22,
        "dataset_hash": datasets["dataset_hash"],
        "proxy_steps": proxy_steps,
        "final_steps": final_steps,
        "eval_limit": eval_limit,
        "seeds": list(seeds),
        "universal_socket": universal,
        "json_specific_socket": json_specific,
        "results": results,
        "frozen_base": frozen,
        "summary": summary,
    }
    report = {**body, "experiment_hash": _hash(body)}
    _write_immutable(root / "experiment_report.json", _json(report))
    return report


def finalize_confirmed_experiment(
    *,
    model_path: str | Path,
    experiment_dir: str | Path,
    final_steps: int = 500,
    eval_limit: int = 500,
) -> dict[str, Any]:
    root = Path(experiment_dir)
    output_path = root / "confirmed_experiment_report.json"
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))
    original = json.loads((root / "experiment_report.json").read_text(encoding="utf-8"))
    confirmation = json.loads((root / "discovery_confirmation/confirmation_report.json").read_text(encoding="utf-8"))
    proxy = json.loads((root / "discovery_search/search_report.json").read_text(encoding="utf-8"))
    json_search = json.loads((root / "json_specific_search/search_report.json").read_text(encoding="utf-8"))
    confirmed = confirmation["winner"]
    config = TrainConfig(final_steps, max_seq_length=256)
    replacement = []
    for seed in tuple(int(value) for value in original["seeds"]):
        run_dir = root / "final_confirmed/universal" / f"seed-{seed}"
        training = train_socket_expert(
            model_path=model_path, socket=confirmed, data_dir=root / "datasets/json",
            output_dir=run_dir, task="json", config=config, seed=seed,
        )
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = evaluate_json_expert(
                model_path=model_path, socket=confirmed, adapter_path=run_dir / "adapters.safetensors",
                dataset_path=root / "datasets/json/test.jsonl", limit=eval_limit,
            )
            _write_immutable(metrics_path, _json(metrics))
        replacement.append({"method": "universal", "seed": seed, "socket": confirmed, "training": training, "metrics": metrics})
    results = [row for row in original["results"] if row["method"] != "universal"] + replacement
    summary = _summarize(results, original["frozen_base"], proxy, json_search)
    confirmation_seconds = sum(
        float(task["elapsed_seconds"])
        for row in confirmation["ranked_finalists"]
        for seed_row in row["seed_results"]
        for task in seed_row["task_results"].values()
    )
    total_discovery = summary["discovery_search_seconds"] + confirmation_seconds
    summary["discovery_confirmation_seconds"] = confirmation_seconds
    summary["total_discovery_seconds"] = total_discovery
    summary["search_break_even_experts"] = math.ceil(total_discovery / summary["json_specific_search_seconds"])
    means = {key: value["mean_complete_record_accuracy"] for key, value in summary["methods"].items()}
    comparisons = {
        "beats_uniform_mean": means["universal"] > means["uniform"],
        "beats_best_single_mean": means["universal"] > means["best-single"],
        "beats_each_sampled_random_mean": means["universal"] > max(means[f"random-{index}"] for index in range(3)),
        "beats_json_specific_mean": means["universal"] > means["json-specific"],
    }
    broad = all(comparisons.values())
    summary["broad_comparisons"] = comparisons
    summary["broad_hypothesis_supported"] = broad
    summary["verdict"] = "SUPPORTED" if broad else "PARTIALLY_SUPPORTED" if summary["hypothesis_supported"] else "NOT_SUPPORTED"
    summary["ceiling_warning"] = max(means.values()) >= 0.99
    body = {
        "schema": "dataevol.reusable_lora_socket_confirmed_experiment.v1",
        "source_experiment_hash": original["experiment_hash"],
        "confirmation_hash": confirmation["confirmation_hash"],
        "confirmed_universal_socket": confirmed,
        "json_specific_socket": original["json_specific_socket"],
        "frozen_base": original["frozen_base"],
        "results": results,
        "summary": summary,
    }
    report = {**body, "experiment_hash": _hash(body)}
    _write_immutable(output_path, _json(report))
    return report


def adjudicate_confirmed_experiment(
    *,
    experiment_dir: str | Path,
    arithmetic_floor: float = 0.50,
    python_floor: float = 0.50,
) -> dict[str, Any]:
    """Issue a binding verdict from immutable performance and behavior reports."""
    root = Path(experiment_dir)
    confirmed = json.loads((root / "confirmed_experiment_report.json").read_text(encoding="utf-8"))
    behavior = json.loads(
        (root / "discovery_confirmation/confirmed_winner_test_metrics.json").read_text(encoding="utf-8")
    )
    observed = {
        "arithmetic": float(behavior["results"]["arithmetic"]["accuracy"]),
        "python": float(behavior["results"]["python"]["accuracy"]),
    }
    floors = {"arithmetic": arithmetic_floor, "python": python_floor}
    discovery_behavior_passed = all(observed[task] >= floors[task] for task in floors)
    broad_comparison_passed = bool(confirmed["summary"]["broad_hypothesis_supported"])
    eligible = discovery_behavior_passed and broad_comparison_passed
    reasons = []
    if not discovery_behavior_passed:
        reasons.append("confirmed discovery adapters failed task-level performance floors")
    if not broad_comparison_passed:
        reasons.append("universal socket did not beat every required parameter-matched baseline")
    body = {
        "schema": "dataevol.reusable_lora_socket_adjudication.v1",
        "source_experiment_hash": confirmed["experiment_hash"],
        "discovery_test_hash": behavior["test_hash"],
        "verdict": "ELIGIBLE" if eligible else "REJECTED",
        "discovery_behavior": {
            "observed": observed,
            "required_floors": floors,
            "passed": discovery_behavior_passed,
        },
        "broad_comparison_passed": broad_comparison_passed,
        "reasons": reasons,
    }
    report = {**body, "adjudication_hash": _hash(body)}
    _write_immutable(root / "adjudication_report.json", _json(report))
    return report


def evaluate_confirmed_discovery_behavior(
    *,
    model_path: str | Path,
    experiment_dir: str | Path,
    seed: int = 17,
    eval_limit: int = 500,
) -> dict[str, Any]:
    root = Path(experiment_dir)
    output_path = root / "discovery_confirmation/confirmed_winner_test_metrics.json"
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))
    confirmation = json.loads(
        (root / "discovery_confirmation/confirmation_report.json").read_text(encoding="utf-8")
    )
    winner = confirmation["winner"]
    results = {
        task: evaluate_discovery_expert(
            model_path=model_path,
            socket=winner,
            adapter_path=root / f"discovery_confirmation/{winner['socket_id']}/seed-{seed}/{task}/adapters.safetensors",
            dataset_path=root / f"datasets/{task}/test.jsonl",
            task=task,
            limit=eval_limit,
        )
        for task in ("arithmetic", "python")
    }
    body = {
        "schema": "dataevol.reusable_socket_discovery_test.v1",
        "confirmation_hash": confirmation["confirmation_hash"],
        "socket_hash": winner["socket_hash"],
        "seed": seed,
        "results": results,
    }
    report = {**body, "test_hash": _hash(body)}
    _write_immutable(output_path, _json(report))
    return report


def run_precision_check(
    *,
    model_path: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    seed: int = 17,
    steps: int = 500,
    eval_limit: int = 500,
) -> dict[str, Any]:
    root = Path(experiment_dir)
    output = Path(output_dir)
    report_path = output / "precision_report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    confirmed = json.loads((root / "confirmed_experiment_report.json").read_text(encoding="utf-8"))
    original = json.loads((root / "experiment_report.json").read_text(encoding="utf-8"))
    source_sockets = {
        "universal": confirmed["confirmed_universal_socket"],
        "uniform": next(row["socket"] for row in original["results"] if row["method"] == "uniform"),
        "best-single": next(row["socket"] for row in original["results"] if row["method"] == "best-single"),
    }
    fingerprint = _directory_fingerprint(Path(model_path))
    config = TrainConfig(steps, max_seq_length=256)
    results = []
    for method, source in source_sockets.items():
        socket = create_socket(
            f"{source['socket_id']}-8bit",
            source["entries"],
            generation=f"precision-check:{source['generation']}",
            model_id="TinyLlama-1.1B-Chat-v1.0-MLX-8bit-local",
            model_revision=fingerprint,
        )
        run_dir = output / method / f"seed-{seed}"
        training = train_socket_expert(
            model_path=model_path, socket=socket, data_dir=root / "datasets/json",
            output_dir=run_dir, task="json", config=config, seed=seed,
        )
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = evaluate_json_expert(
                model_path=model_path, socket=socket, adapter_path=run_dir / "adapters.safetensors",
                dataset_path=root / "datasets/json/test.jsonl", limit=eval_limit,
            )
            _write_immutable(metrics_path, _json(metrics))
        results.append({"method": method, "socket": socket, "training": training, "metrics": metrics})
    body = {
        "schema": "dataevol.reusable_socket_precision_check.v1",
        "source_experiment_hash": confirmed["experiment_hash"],
        "model_directory_fingerprint": fingerprint,
        "precision": "8-bit affine, group size 64",
        "seed": seed,
        "steps": steps,
        "eval_limit": eval_limit,
        "results": results,
        "single_seed_only": True,
    }
    report = {**body, "precision_hash": _hash(body)}
    _write_immutable(report_path, _json(report))
    return report


def _summarize(results: list[dict[str, Any]], frozen: Mapping[str, Any], discovery: Mapping[str, Any], json_search: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    times: dict[str, list[float]] = {}
    for row in results:
        grouped.setdefault(row["method"], []).append(float(row["metrics"]["complete_record_accuracy"]))
        times.setdefault(row["method"], []).append(float(row["training"]["elapsed_seconds"]))
    methods = {
        method: {
            "mean_complete_record_accuracy": sum(values) / len(values),
            "std_complete_record_accuracy": _population_std(values),
            "mean_training_seconds": sum(times[method]) / len(times[method]),
        }
        for method, values in grouped.items()
    }
    base = float(frozen["complete_record_accuracy"])
    universal = methods["universal"]["mean_complete_record_accuracy"]
    specific = methods["json-specific"]["mean_complete_record_accuracy"]
    denominator = specific - base
    reuse = (universal - base) / denominator if denominator > 0 else None
    uniform = methods["uniform"]["mean_complete_record_accuracy"]
    success = {
        "performance_victory": universal - uniform >= 0.03,
        "stability_victory": universal >= uniform and methods["universal"]["std_complete_record_accuracy"] < methods["uniform"]["std_complete_record_accuracy"],
        "reusability_score": reuse,
    }
    discovery_seconds = sum(
        float(task["elapsed_seconds"])
        for row in discovery["ranked_candidates"] for task in row["task_results"].values()
    )
    specific_search_seconds = sum(
        float(task["elapsed_seconds"])
        for row in json_search["ranked_candidates"] for task in row["task_results"].values()
    )
    return {
        "methods": methods,
        "success_conditions": success,
        "discovery_search_seconds": discovery_seconds,
        "json_specific_search_seconds": specific_search_seconds,
        "search_break_even_experts": math.ceil(discovery_seconds / specific_search_seconds) if specific_search_seconds > 0 else None,
        "hypothesis_supported": bool(success["performance_victory"] or success["stability_victory"] or (reuse is not None and reuse >= 0.9)),
    }


def _apply_socket(
    model: Any,
    socket: Mapping[str, Any],
    *,
    scale: float = 20.0,
    dropout: float = 0.0,
) -> None:
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    model.freeze()
    layers = model.model.layers
    for entry in socket["entries"]:
        layer = layers[int(entry["layer"])]
        modules = dict(layer.named_modules())
        replacements = []
        for key in FAMILY_MODULES[str(entry["family"])]:
            base = modules.get(key)
            if base is None:
                raise ValueError(f"TinyLlama layer lacks socket module {key}")
            replacements.append(
                (
                    key,
                    LoRALinear.from_base(
                        base,
                        r=int(entry["rank"]),
                        scale=scale,
                        dropout=dropout,
                    ),
                )
            )
        layer.update_modules(tree_unflatten(replacements))


def _assert_tinyllama(model: Any) -> None:
    if not hasattr(model, "model") or len(model.model.layers) != 22:
        raise ValueError("experiment requires a 22-layer TinyLlama model")
    args = model.args
    if int(args.hidden_size) != 2_048 or int(args.intermediate_size) != 5_632:
        raise ValueError("TinyLlama dimensions do not match the pinned experiment")


def _generate_rows(task: str, split: str, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{task}:{split}")
    rows = []
    for index in range(count):
        if task == "arithmetic":
            row = _arithmetic_row(rng, split, index)
        elif task == "python":
            row = _python_row(rng, split, index)
        elif task == "json":
            row = _json_row(rng, split, index)
        else:
            raise ValueError(f"unsupported socket task: {task}")
        rows.append({"id": f"{task}-{split}-{index:04d}", **row})
    return rows


def _arithmetic_row(rng: random.Random, split: str, index: int) -> dict[str, str]:
    a = rng.randint(3, 80) + index % 17
    b = rng.randint(2, 30)
    c = rng.randint(1, 15)
    templates = {
        "train": [
            ("A shop had {a} pens, sold {b}, then received {c}. How many pens are there?", a - b + c),
            ("Nora collected {a} shells and {b} more, then gave away {c}. How many remain?", a + b - c),
        ],
        "valid": [("A bin starts with {a} parts. {b} are removed and {c} added. What is the final count?", a - b + c)],
        "test": [("After adding {b} tickets to {a} and using {c}, how many tickets remain?", a + b - c)],
    }
    template, answer = rng.choice(templates[split])
    prompt = "Solve the problem. Return only the integer answer.\n" + template.format(a=a, b=b, c=c)
    return {"prompt": prompt, "completion": str(answer), "answer": str(answer)}


def _python_row(rng: random.Random, split: str, index: int) -> dict[str, str]:
    variants = {
        "train": [
            ("def clamp(x, low, high):\n    return min(low, max(x, high))", "def clamp(x, low, high):\n    return max(low, min(x, high))"),
            ("def is_even(n):\n    return n % 2 == 1", "def is_even(n):\n    return n % 2 == 0"),
            ("def last_item(items):\n    return items[0]", "def last_item(items):\n    return items[-1]"),
            ("def add_tax(price, rate):\n    return price - price * rate", "def add_tax(price, rate):\n    return price + price * rate"),
        ],
        "valid": [
            ("def contains(items, value):\n    return value not in items", "def contains(items, value):\n    return value in items"),
            ("def square(n):\n    return n + n", "def square(n):\n    return n * n"),
        ],
        "test": [
            ("def first_item(items):\n    return items[-1]", "def first_item(items):\n    return items[0]"),
            ("def subtract(a, b):\n    return b - a", "def subtract(a, b):\n    return a - b"),
        ],
    }
    buggy, fixed = variants[split][index % len(variants[split])]
    salt = rng.randint(10, 99)
    prompt = f"Repair the localized bug. Return only the corrected Python function. Case {salt}.\n```python\n{buggy}\n```"
    return {"prompt": prompt, "completion": fixed}


def _json_row(rng: random.Random, split: str, index: int) -> dict[str, str]:
    first = ("Maria", "Aiden", "Priya", "Jonas", "Elena", "Omar", "Lina", "Theo")[(index + rng.randrange(8)) % 8]
    last = ("Chen", "Rivera", "Patel", "Meyer", "Silva", "Hassan", "Kim", "Brown")[(index * 3 + rng.randrange(8)) % 8]
    quantity = 1 + index % 9
    color = ("blue", "red", "green", "black", "white")[index % 5]
    item = ("cables", "folders", "lamps", "notebooks", "adapters")[index % 5]
    month = ("May", "June", "July", "August", "September")[index % 5]
    day = 1 + (index * 7) % 28
    order = f"{chr(65 + index % 26)}{chr(65 + (index // 7) % 26)}-{100 + index}"
    templates = {
        "train": [
            "{name} ordered {q} {color} {item} on {date}. The order number is {order}.",
            "Order {order}: ship {q} {color} {item} to {name}; purchase date {date}.",
            "Customer {name} bought {color} {item}, quantity {q}, dated {date}, reference {order}.",
        ],
        "valid": ["Reference {order} belongs to {name}. On {date}, they requested {q} units of {color} {item}."],
        "test": ["Purchase record for {name}\nID: {order}\nItem: {color} {item}\nCount: {q}\nDate placed: {date}"],
    }
    name = f"{first} {last}"
    values = {"name": name, "q": quantity, "color": color, "item": item, "date": f"{month} {day}", "order": order}
    text = rng.choice(templates[split]).format(**values)
    target = {"customer": name, "quantity": quantity, "item": f"{color} {item}", "date": values["date"], "order_id": order}
    prompt = "Extract the record. Return only valid JSON with keys customer, quantity, item, date, order_id.\n" + text
    return {"prompt": prompt, "completion": json.dumps(target, separators=(",", ":"), ensure_ascii=True)}


def _extract_json(output: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*?\}", output, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _arithmetic_correct(output: str, expected: str) -> bool:
    match = re.search(r"(?<![\d.])-?\d+(?![\d.])", output.strip())
    return bool(match and match.group(0) == str(expected))


def _python_unit_test_pass(output: str, expected: str, *, timeout_seconds: float = 1.0) -> bool:
    code = _extract_python(output)
    try:
        tree = ast.parse(code)
        expected_tree = ast.parse(expected)
    except SyntaxError:
        return False
    allowed = (
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Name, ast.Load,
        ast.BinOp, ast.Sub, ast.Subscript, ast.Constant, ast.UnaryOp, ast.USub,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return False
    functions = [node.name for node in expected_tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] != functions:
        return False
    function = functions[0]
    tests = {
        "first_item": "assert first_item([4, 5, 6]) == 4\nassert first_item(['x']) == 'x'",
        "subtract": "assert subtract(5, 2) == 3\nassert subtract(-1, 4) == -5",
    }.get(function)
    if tests is None:
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", f"{code}\n{tests}\n"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout_seconds, check=False, env={"PATH": "/usr/bin:/bin"},
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _extract_python(output: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", output, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else output).strip()


def _band_index(layer: int) -> int:
    return next(index for index, (low, high) in enumerate(BANDS) if low <= layer <= high)


def _closest_rank(target: int, parameters_per_rank: int) -> int:
    candidates = (max(1, target // parameters_per_rank), max(1, round(target / parameters_per_rank)), max(1, math.ceil(target / parameters_per_rank)))
    return min(candidates, key=lambda value: abs(value * parameters_per_rank - target))


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()


def _json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2) + "\n").encode()


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory contains no files: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"experiment artifact exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Mac reusable cross-layer LoRA socket experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", required=True, type=Path)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--model", required=True, type=Path)
    smoke.add_argument("--output", required=True, type=Path)
    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--model", required=True, type=Path)
    confirm.add_argument("--experiment", required=True, type=Path)
    confirm.add_argument("--steps", type=int, default=500)
    finalize = subparsers.add_parser("finalize-confirmed")
    finalize.add_argument("--model", required=True, type=Path)
    finalize.add_argument("--experiment", required=True, type=Path)
    finalize.add_argument("--steps", type=int, default=500)
    finalize.add_argument("--eval-limit", type=int, default=500)
    behavior = subparsers.add_parser("evaluate-discovery")
    behavior.add_argument("--model", required=True, type=Path)
    behavior.add_argument("--experiment", required=True, type=Path)
    behavior.add_argument("--seed", type=int, default=17)
    behavior.add_argument("--eval-limit", type=int, default=500)
    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--experiment", required=True, type=Path)
    adjudicate.add_argument("--arithmetic-floor", type=float, default=0.50)
    adjudicate.add_argument("--python-floor", type=float, default=0.50)
    precision = subparsers.add_parser("precision-check")
    precision.add_argument("--model", required=True, type=Path)
    precision.add_argument("--experiment", required=True, type=Path)
    precision.add_argument("--output", required=True, type=Path)
    precision.add_argument("--steps", type=int, default=500)
    precision.add_argument("--eval-limit", type=int, default=500)
    run = subparsers.add_parser("run-mvp")
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--proxy-steps", type=int, default=100)
    run.add_argument("--final-steps", type=int, default=500)
    run.add_argument("--eval-limit", type=int, default=500)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_datasets(args.output)
    elif args.command == "smoke":
        prepare_datasets(args.output / "datasets")
        socket = generate_socket_candidates(count=3)[0]
        result = train_socket_expert(
            model_path=args.model, socket=socket, data_dir=args.output / "datasets/arithmetic",
            output_dir=args.output / "smoke", task="arithmetic", config=TrainConfig(20), seed=17,
        )
    elif args.command == "confirm":
        proxy = json.loads((args.experiment / "discovery_search/search_report.json").read_text(encoding="utf-8"))
        result = confirm_discovery_finalists(
            model_path=args.model,
            data_root=args.experiment / "datasets",
            proxy_report=proxy,
            output_dir=args.experiment / "discovery_confirmation",
            config=TrainConfig(args.steps),
        )
    elif args.command == "finalize-confirmed":
        result = finalize_confirmed_experiment(
            model_path=args.model,
            experiment_dir=args.experiment,
            final_steps=args.steps,
            eval_limit=args.eval_limit,
        )
    elif args.command == "evaluate-discovery":
        result = evaluate_confirmed_discovery_behavior(
            model_path=args.model,
            experiment_dir=args.experiment,
            seed=args.seed,
            eval_limit=args.eval_limit,
        )
    elif args.command == "adjudicate":
        result = adjudicate_confirmed_experiment(
            experiment_dir=args.experiment,
            arithmetic_floor=args.arithmetic_floor,
            python_floor=args.python_floor,
        )
    elif args.command == "precision-check":
        result = run_precision_check(
            model_path=args.model,
            experiment_dir=args.experiment,
            output_dir=args.output,
            steps=args.steps,
            eval_limit=args.eval_limit,
        )
    else:
        result = run_mvp(
            model_path=args.model, output_dir=args.output, proxy_steps=args.proxy_steps,
            final_steps=args.final_steps, eval_limit=args.eval_limit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
