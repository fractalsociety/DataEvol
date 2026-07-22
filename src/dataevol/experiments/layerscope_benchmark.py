from __future__ import annotations

import argparse
import hashlib
import hmac
import itertools
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROMPT_TEMPLATE = "Summarize this dialogue in one concise paragraph.\n\nDialogue:\n{dialogue}\n\nSummary:"


def prepare_dialogsum(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 1701,
    selection_count: int = 512,
    train_count: int = 2048,
    preference_count: int = 384,
) -> dict[str, Any]:
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if min(selection_count, train_count, preference_count) <= 0:
        raise ValueError("all partition sample counts must be positive")
    source_paths = {
        "train": source / "dialogsum.train.jsonl",
        "dev": source / "dialogsum.dev.jsonl",
        "test": source / "dialogsum.test.jsonl",
    }
    train_source = _read_unique_jsonl(source_paths["train"], id_field="fname")
    dev_source = _read_unique_jsonl(source_paths["dev"], id_field="fname")
    test_source = _read_unique_jsonl(source_paths["test"], id_field="fname")
    train_source, dev_source, test_source, deduplication = _deduplicate_dialogsum_splits(
        train_source,
        dev_source,
        test_source,
    )
    _validate_dialogsum_splits(train_source, dev_source, test_source)
    rng = random.Random(seed)
    shuffled = list(train_source)
    rng.shuffle(shuffled)
    if selection_count > len(shuffled) or train_count + preference_count > len(shuffled):
        raise ValueError("requested sample count exceeds DialogSum training rows")

    selection = [_training_row(row) for row in shuffled[:selection_count]]
    final_train = [_training_row(row) for row in shuffled[:train_count]]
    preference_prompts = [_training_row(row) for row in shuffled[train_count:train_count + preference_count]]
    if len(preference_prompts) < preference_count:
        raise ValueError("preference rows must not overlap the final SFT sample")
    validation = [_training_row(row) for row in dev_source]
    validation_references = [_blind_reference(row) for row in dev_source]
    selection_validation = validation[:128]
    selection_validation_references = validation_references[:128]
    blind_inputs = [_blind_input(row) for row in test_source]
    blind_references = [_blind_reference(row) for row in test_source]

    paths = {
        "selection": output / "selection_train.jsonl",
        "sft_train": output / "sft_train.jsonl",
        "preference_prompts": output / "preference_prompts.jsonl",
        "validation": output / "validation.jsonl",
        "validation_references": output / "validation_references.jsonl",
        "selection_validation": output / "selection_validation.jsonl",
        "selection_validation_references": output / "selection_validation_references.jsonl",
        "blind_inputs": output / "blind_inputs.jsonl",
        "blind_references": output / "blind_references.sealed.jsonl",
    }
    values = {
        "selection": selection,
        "sft_train": final_train,
        "preference_prompts": preference_prompts,
        "validation": validation,
        "validation_references": validation_references,
        "selection_validation": selection_validation,
        "selection_validation_references": selection_validation_references,
        "blind_inputs": blind_inputs,
        "blind_references": blind_references,
    }
    for name, path in paths.items():
        _write_jsonl(path, values[name])
    os.chmod(paths["blind_references"], 0o400)
    manifest = {
        "schema": "dataevol.layerscope_benchmark.v1",
        "seed": seed,
        "prompt_template_sha256": _sha256_text(PROMPT_TEMPLATE),
        "sources": {
            path.name: {"sha256": _sha256_path(path), "rows": len(_read_unique_jsonl(path, id_field="fname"))}
            for path in source_paths.values()
        },
        "partitions": {
            name: {"path": str(path), "sha256": _sha256_path(path), "rows": len(values[name])}
            for name, path in paths.items()
        },
        "reference_labels_withheld_from_generation_inputs": True,
        "dataset_revision": "cylnlp/dialogsum@848cc8c05e0a04def79326539de827f1b28786a0",
        "deduplication": deduplication,
        "created_at_unix": time.time(),
    }
    manifest["manifest_hash"] = _hash_object(manifest)
    _write_json(output / "benchmark_manifest.json", manifest)
    return manifest


def generate_outputs(
    *,
    model_path: str,
    input_path: str | Path,
    output_path: str | Path,
    specialist_manifest: str | Path | None = None,
    max_tokens: int = 96,
    limit: int | None = None,
) -> dict[str, Any]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from dataevol.local_models.layer_specialist import _decoder_layers, model_fingerprint
    from dataevol.specialist_server.swapper import MlxLayerSwapper

    rows = _read_unique_jsonl(Path(input_path), id_field="id")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    destination = Path(output_path)
    manifest_revision: str | None = None
    if specialist_manifest:
        manifest_body = json.loads(Path(specialist_manifest).read_text(encoding="utf-8"))
        revision_value = manifest_body.get("base_model_revision")
        manifest_revision = str(revision_value) if revision_value is not None else None
    fingerprint = model_fingerprint(model_path, base_model_revision=manifest_revision)
    model, tokenizer = load(model_path)
    layers = _decoder_layers(model)
    if layers is None:
        raise RuntimeError("model does not expose decoder layers")
    variant = "base"
    candidate_hash = fingerprint["sha256"]
    if specialist_manifest:
        swapper = MlxLayerSwapper(
            model,
            base_model_id=model_path,
            base_model_hash=str(fingerprint["sha256"]),
            base_model_revision=fingerprint.get("resolved_revision"),
            num_layers=len(layers),
        )
        specialist = swapper.register(specialist_manifest)
        swapper.activate(specialist.name)
        variant = specialist.name
        candidate_hash = specialist.candidate_content_hash
    expected_rows = [
        {"id": str(row["id"]), "prompt_sha256": _sha256_text(str(row["prompt"]))}
        for row in rows
    ]
    run_config = {
        "schema": "dataevol.layerscope_generation_config.v1",
        "model_fingerprint": fingerprint["sha256"],
        "variant": variant,
        "candidate_hash": candidate_hash,
        "input_file_sha256": _sha256_path(input_path),
        "input_rows_hash": _hash_object({"rows": expected_rows}),
        "expected_ids_hash": _hash_object({"ids": [row["id"] for row in expected_rows]}),
        "expected_rows": len(expected_rows),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "prompt_format": "tokenizer_chat_template_v1",
    }
    run_config_hash = _hash_object(run_config)
    sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if destination.exists():
        if not sidecar.is_file():
            raise ValueError("existing generation output has no immutable run manifest")
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("run_config_hash") != run_config_hash:
            raise ValueError("existing generation output belongs to a different run configuration")
        for record in _read_unique_jsonl(destination, id_field="id"):
            row_id = str(record["id"])
            expected = next((item for item in expected_rows if item["id"] == row_id), None)
            if expected is None or record.get("prompt_sha256") != expected["prompt_sha256"]:
                raise ValueError(f"existing generation row does not match committed input: {row_id}")
            if record.get("run_config_hash") != run_config_hash or record.get("candidate_hash") != candidate_hash:
                raise ValueError(f"existing generation row has mixed candidate/config identity: {row_id}")
            completed[row_id] = record
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(lock_fd)
    except FileExistsError as exc:
        raise RuntimeError(f"generation run is already locked: {lock_path}") from exc
    _write_json(sidecar, {
        "schema": "dataevol.layerscope_generation_run.v1",
        "status": "running",
        "run_config": run_config,
        "run_config_hash": run_config_hash,
        "completed_rows": len(completed),
    })
    sampler = make_sampler(temp=0.0)
    mode = "a" if destination.exists() else "w"
    started = time.time()
    try:
        with destination.open(mode, encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                row_id = str(row["id"])
                if row_id in completed:
                    continue
                prompt = str(row["prompt"])
                generation_prompt = _generation_prompt(tokenizer, prompt)
                item_started = time.time()
                text = generate(
                    model,
                    tokenizer,
                    prompt=generation_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                ).strip()
                record = {
                    "id": row_id,
                    "output": text,
                    "variant": variant,
                    "candidate_hash": candidate_hash,
                    "prompt_sha256": _sha256_text(prompt),
                    "run_config_hash": run_config_hash,
                    "latency_ms": round((time.time() - item_started) * 1000, 3),
                    "output_tokens": len(tokenizer.encode(text)),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                if index == 1 or index % 25 == 0:
                    print(json.dumps({"generated": index, "total": len(rows), "variant": variant}), flush=True)
        final_rows = _read_unique_jsonl(destination, id_field="id")
        if {str(row["id"]) for row in final_rows} != {row["id"] for row in expected_rows}:
            raise RuntimeError("generation output is incomplete or contains unexpected IDs")
        result = {
            "schema": "dataevol.layerscope_generation_run.v1",
            "status": "completed",
            "run_config": run_config,
            "run_config_hash": run_config_hash,
            "output_sha256": _sha256_path(destination),
            "rows": len(final_rows),
            "session_elapsed_seconds": time.time() - started,
        }
        _write_json(sidecar, result)
        return result
    finally:
        lock_path.unlink(missing_ok=True)


def _generation_prompt(tokenizer: Any, prompt: str) -> str:
    """Match the chat boundary used by layer-specialist SFT and DPO."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return str(apply_template(messages, tokenize=False, add_generation_prompt=True))
    except TypeError:
        return str(apply_template(messages, tokenize=False))


def build_preferences(
    prompt_path: str | Path,
    base_output_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    prompt_rows = _read_unique_jsonl(prompt_path, id_field="id")
    output_rows = _read_unique_jsonl(base_output_path, id_field="id")
    prompts = {str(row["id"]): row for row in prompt_rows}
    outputs = {str(row["id"]): row for row in output_rows}
    if set(prompts) != set(outputs):
        raise ValueError("preference base outputs must exactly cover the committed prompt IDs")
    rows = []
    for row_id in sorted(prompts):
        output = outputs[row_id]
        if output.get("prompt_sha256") != _sha256_text(str(prompts[row_id]["prompt"])):
            raise ValueError(f"preference output prompt hash mismatch: {row_id}")
        if not str(output.get("output") or "").strip():
            raise ValueError(f"preference output is empty: {row_id}")
        prompt = prompts[row_id]
        chosen = str(prompt["completion"]).strip()
        rejected = str(output["output"]).strip()
        if _normalize(chosen) == _normalize(rejected):
            continue
        rows.append({"pair_id": row_id, "prompt": prompt["prompt"], "chosen": chosen, "rejected": rejected})
    _write_jsonl(Path(output_path), rows)
    result = {
        "schema": "dataevol.preference_dataset.v1",
        "rows": len(rows),
        "source_prompts_sha256": _sha256_path(prompt_path),
        "source_base_outputs_sha256": _sha256_path(base_output_path),
        "sha256": _sha256_path(output_path),
    }
    _write_json(Path(output_path).with_suffix(".manifest.json"), result)
    return result


def judge_blind_pairs(
    *,
    model_path: str,
    pair_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from dataevol.local_models.layer_specialist import model_fingerprint

    pairs = _read_unique_jsonl(pair_path, id_field="pair_id")
    if limit is not None:
        if limit <= 0:
            raise ValueError("judge limit must be positive")
        pairs = pairs[:limit]
    destination = Path(output_path)
    if destination.exists():
        raise ValueError("blind judgment output already exists; refusing to mix judge sessions")
    fingerprint = model_fingerprint(model_path)
    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.0)
    rows = []
    for index, pair in enumerate(pairs, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a blind dialogue-summary evaluator. Prefer factual accuracy, coverage of the "
                    "main outcome, concision, and no invented details. Return exactly one token: A, B, or TIE."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{pair['prompt']}\n\nSummary A:\n{pair['A']}\n\nSummary B:\n{pair['B']}\n\n"
                    "Which summary is better?"
                ),
            },
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=4, sampler=sampler, verbose=False).strip()
        verdict = _parse_exact_verdict(raw)
        rows.append({
            "pair_id": pair["pair_id"],
            "case_id": pair["case_id"],
            "verdict": verdict,
            "raw_verdict": raw,
            "raw_verdict_sha256": _sha256_text(raw),
        })
        if index == 1 or index % 25 == 0:
            print(json.dumps({"judged": index, "total": len(pairs)}), flush=True)
    _write_jsonl(destination, rows)
    result = {
        "schema": "dataevol.layerscope_blind_judge.v1",
        "judge_model_fingerprint": fingerprint["sha256"],
        "pair_sha256": _sha256_path(pair_path),
        "judgment_sha256": _sha256_path(destination),
        "rows": len(rows),
        "valid_rows": sum(row["verdict"] != "INVALID" for row in rows),
        "temperature": 0.0,
    }
    _write_json(destination.with_suffix(destination.suffix + ".manifest.json"), result)
    return result


def score_blind_judgments(
    *,
    variants: dict[str, str | Path],
    input_path: str | Path,
    judgment_path: str | Path,
    blinding_key_file: str | Path,
    seed: int = 1701,
) -> dict[str, Any]:
    input_rows = _read_unique_jsonl(input_path, id_field="id")
    judgments = {str(row["pair_id"]): row for row in _read_unique_jsonl(judgment_path, id_field="pair_id")}
    key = Path(blinding_key_file).read_bytes()
    if len(key) < 32:
        raise ValueError("blinding key must contain at least 32 bytes")
    assignments: dict[str, tuple[str, str]] = {}
    for row in input_rows:
        row_id = str(row["id"])
        for left, right in itertools.combinations(variants, 2):
            pair_id, _, names = _blind_assignment(row_id, left, right, key, seed)
            assignments[pair_id] = (names[0], names[1])
    unexpected = set(judgments) - set(assignments)
    if unexpected:
        raise ValueError(f"judgments contain unknown blinded pair IDs: {sorted(unexpected)[:5]}")
    missing = set(assignments) - set(judgments)
    if missing:
        raise ValueError(f"judgments do not exactly cover blinded pair IDs: {sorted(missing)[:5]}")
    wins = Counter()
    pairwise: dict[str, Counter[str]] = {}
    invalid = 0
    for pair_id, judgment in judgments.items():
        left, right = assignments[pair_id]
        key_name = "-vs-".join(sorted((left, right)))
        tally = pairwise.setdefault(key_name, Counter())
        verdict = judgment.get("verdict")
        if verdict == "A":
            wins[left] += 1
            tally[left] += 1
        elif verdict == "B":
            wins[right] += 1
            tally[right] += 1
        elif verdict == "TIE":
            tally["TIE"] += 1
        else:
            invalid += 1
            tally["INVALID"] += 1
    if invalid:
        raise ValueError(f"judgments contain {invalid} invalid verdicts")
    result = {
        "schema": "dataevol.layerscope_blind_judge_result.v1",
        "judgment_sha256": _sha256_path(judgment_path),
        "blinding_key_commitment": hashlib.sha256(key).hexdigest(),
        "revealed_after_judging": True,
        "wins": dict(wins),
        "pairwise": {name: dict(tally) for name, tally in pairwise.items()},
        "judged_rows": len(judgments),
        "invalid_rows": invalid,
    }
    result["result_hash"] = _hash_object(result)
    return result


def _parse_exact_verdict(raw: str) -> str:
    match = re.fullmatch(r"\s*(TIE|A|B)\s*[.!]?\s*", raw.upper())
    return match.group(1) if match else "INVALID"


def compare_outputs(
    variants: dict[str, str | Path],
    input_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    *,
    blinding_key_file: str | Path,
    seed: int = 1701,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    if len(variants) < 2:
        raise ValueError("at least two variants are required")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    input_rows = _read_unique_jsonl(input_path, id_field="id")
    benchmark_inputs = {str(row["id"]): row for row in input_rows}
    reference_rows = _read_unique_jsonl(reference_path, id_field="id")
    references = {str(row["id"]): row["references"] for row in reference_rows}
    if set(benchmark_inputs) != set(references):
        raise ValueError("benchmark inputs and references must contain exactly the same IDs")
    outputs: dict[str, dict[str, dict[str, Any]]] = {}
    run_manifests: dict[str, dict[str, Any]] = {}
    for name, path in variants.items():
        rows = _read_unique_jsonl(path, id_field="id")
        outputs[name] = {str(row["id"]): row for row in rows}
        sidecar = Path(path).with_suffix(Path(path).suffix + ".manifest.json")
        if not sidecar.is_file():
            raise ValueError(f"variant {name} has no generation manifest")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("output_sha256") != _sha256_path(path):
            raise ValueError(f"variant {name} generation manifest is incomplete or stale")
        run_manifests[name] = manifest
    benchmark_ids = set(references)
    for name, rows in outputs.items():
        if set(rows) != benchmark_ids:
            missing = sorted(benchmark_ids - set(rows))[:5]
            extra = sorted(set(rows) - benchmark_ids)[:5]
            raise ValueError(f"variant {name} does not exactly cover benchmark IDs missing={missing} extra={extra}")
    common_ids = sorted(benchmark_ids)
    input_hashes = {manifest.get("run_config", {}).get("input_rows_hash") for manifest in run_manifests.values()}
    if len(input_hashes) != 1 or None in input_hashes:
        raise ValueError("variant generation manifests do not share one committed input row set")
    candidate_hashes = {manifest.get("run_config", {}).get("candidate_hash") for manifest in run_manifests.values()}
    if len(candidate_hashes) != len(variants) or None in candidate_hashes:
        raise ValueError("variants must have distinct committed candidate identities")
    for row_id in common_ids:
        prompt_hashes = {rows[row_id].get("prompt_sha256") for rows in outputs.values()}
        expected_prompt_hash = _sha256_text(str(benchmark_inputs[row_id]["prompt"]))
        if prompt_hashes != {expected_prompt_hash}:
            raise ValueError(f"variants did not answer the same committed prompt: {row_id}")
    per_example: dict[str, dict[str, dict[str, float]]] = {}
    for row_id in common_ids:
        per_example[row_id] = {
            name: _score_summary(str(rows[row_id].get("output") or ""), references[row_id])
            for name, rows in outputs.items()
        }
    aggregate = {
        name: {
            metric: statistics.fmean(per_example[row_id][name][metric] for row_id in common_ids)
            for metric in ("rouge1", "rouge2", "rougeL", "reference_length_ratio")
        } | {
            "empty_rate": statistics.fmean(1.0 if not str(outputs[name][row_id].get("output") or "").strip() else 0.0 for row_id in common_ids),
            "median_latency_ms": statistics.median(float(outputs[name][row_id].get("latency_ms") or 0.0) for row_id in common_ids),
        }
        for name in variants
    }
    pairwise = {}
    rng = random.Random(seed)
    for left, right in itertools.combinations(variants, 2):
        deltas = [per_example[row_id][right]["rougeL"] - per_example[row_id][left]["rougeL"] for row_id in common_ids]
        boot = [statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(bootstrap_samples)]
        boot.sort()
        pairwise[f"{right}-minus-{left}"] = {
            "mean_delta_rougeL": statistics.fmean(deltas),
            "ci95": [boot[int(0.025 * len(boot))], boot[min(len(boot) - 1, int(0.975 * len(boot)))]] ,
            "wins": sum(delta > 1e-12 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
            "losses": sum(delta < -1e-12 for delta in deltas),
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blinding_key = Path(blinding_key_file).read_bytes()
    if len(blinding_key) < 32:
        raise ValueError("blinding key must contain at least 32 bytes")
    blinded = _blind_pairs(
        common_ids,
        variants,
        outputs,
        benchmark_inputs,
        blinding_key=blinding_key,
        seed=seed,
    )
    _write_jsonl(output / "blind_pairs.jsonl", blinded)
    result = {
        "schema": "dataevol.layerscope_blind_comparison.v1",
        "rows": len(common_ids),
        "reference_sha256": _sha256_path(reference_path),
        "input_sha256": _sha256_path(input_path),
        "variant_sha256": {name: _sha256_path(path) for name, path in variants.items()},
        "aggregate": aggregate,
        "pairwise": pairwise,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "scorer": "rouge_score.RougeScorer(use_stemmer=True), best-of-multiple-references",
        "blind_pairs_sha256": _sha256_path(output / "blind_pairs.jsonl"),
        "blinding_key_commitment": hashlib.sha256(blinding_key).hexdigest(),
    }
    result["result_hash"] = _hash_object(result)
    _write_json(output / "comparison.json", result)
    _write_jsonl(output / "per_example_scores.jsonl", [
        {"id": row_id, "variants": per_example[row_id]} for row_id in common_ids
    ])
    return result


def _blind_pairs(
    row_ids: list[str],
    variants: dict[str, str | Path],
    outputs: dict[str, dict[str, dict[str, Any]]],
    inputs: dict[str, dict[str, Any]],
    *,
    blinding_key: bytes,
    seed: int,
) -> list[dict[str, Any]]:
    blinded: list[dict[str, Any]] = []
    for row_id in row_ids:
        for left, right in itertools.combinations(variants, 2):
            pair_id, opaque_case_id, names = _blind_assignment(row_id, left, right, blinding_key, seed)
            blinded.append({
                "pair_id": pair_id,
                "case_id": opaque_case_id,
                "prompt": inputs[row_id]["prompt"],
                "A": outputs[names[0]][row_id]["output"],
                "B": outputs[names[1]][row_id]["output"],
            })
    random.Random(int.from_bytes(hmac.new(blinding_key, b"order", hashlib.sha256).digest()[:8], "big")).shuffle(blinded)
    return blinded


def _blind_assignment(
    row_id: str,
    left: str,
    right: str,
    blinding_key: bytes,
    seed: int,
) -> tuple[str, str, list[str]]:
    names = [left, right]
    pair_digest = hmac.new(blinding_key, f"{row_id}:{left}:{right}:{seed}".encode(), hashlib.sha256).digest()
    random.Random(int.from_bytes(pair_digest[:8], "big")).shuffle(names)
    pair_id = hmac.new(blinding_key, f"pair:{row_id}:{left}:{right}".encode(), hashlib.sha256).hexdigest()[:24]
    case_id = hmac.new(blinding_key, f"case:{row_id}".encode(), hashlib.sha256).hexdigest()[:24]
    return pair_id, case_id, names


def _training_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row["fname"]),
        "prompt": PROMPT_TEMPLATE.format(dialogue=str(row["dialogue"]).strip()),
        "completion": str(row.get("summary") or row.get("summary1") or "").strip(),
    }


def _blind_input(row: dict[str, Any]) -> dict[str, str]:
    return {"id": str(row["fname"]), "prompt": PROMPT_TEMPLATE.format(dialogue=str(row["dialogue"]).strip())}


def _blind_reference(row: dict[str, Any]) -> dict[str, Any]:
    references = [str(row[key]).strip() for key in ("summary", "summary1", "summary2", "summary3") if row.get(key)]
    return {"id": str(row["fname"]), "references": references}


def _score_summary(candidate: str, references: Iterable[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = []
    candidate_tokens = _tokens(candidate)
    reference_values = [str(reference) for reference in references]
    mean_reference_length = statistics.fmean(max(1, len(_tokens(reference))) for reference in reference_values)
    for reference in reference_values:
        reference_tokens = _tokens(reference)
        rouge = scorer.score(reference, candidate)
        scores.append({
            "rouge1": rouge["rouge1"].fmeasure,
            "rouge2": rouge["rouge2"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure,
        })
    if not scores:
        raise ValueError("at least one reference summary is required")
    result = {metric: max(score[metric] for score in scores) for metric in scores[0]}
    result["reference_length_ratio"] = len(candidate_tokens) / mean_reference_length
    return result


def _tokens(value: str) -> list[str]:
    token = ""
    tokens = []
    for character in value.lower():
        if character.isalnum() or character == "'":
            token += character
        elif token:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)
    return tokens


def _ngram_f1(candidate: list[str], reference: list[str], n: int) -> float:
    candidate_counts = Counter(tuple(candidate[index:index + n]) for index in range(max(0, len(candidate) - n + 1)))
    reference_counts = Counter(tuple(reference[index:index + n]) for index in range(max(0, len(reference) - n + 1)))
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / max(1, sum(candidate_counts.values()))
    recall = overlap / max(1, sum(reference_counts.values()))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _lcs_f1(candidate: list[str], reference: list[str]) -> float:
    previous = [0] * (len(reference) + 1)
    for candidate_token in candidate:
        current = [0]
        for index, reference_token in enumerate(reference, start=1):
            current.append(previous[index - 1] + 1 if candidate_token == reference_token else max(current[-1], previous[index]))
        previous = current
    overlap = previous[-1]
    precision = overlap / max(1, len(candidate))
    recall = overlap / max(1, len(reference))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_unique_jsonl(path: str | Path, *, id_field: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required JSONL file is missing: {source}")
    rows = _read_jsonl(source)
    if not rows:
        raise ValueError(f"required JSONL file is empty: {source}")
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number} must be an object")
        row_id = str(row.get(id_field) or "").strip()
        if not row_id:
            raise ValueError(f"{source}:{line_number} is missing {id_field}")
        if row_id in seen:
            raise ValueError(f"duplicate {id_field} in {source}: {row_id}")
        seen.add(row_id)
    return rows


def _validate_dialogsum_splits(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    split_hashes: list[set[str]] = []
    for split_name, rows in (("train", train), ("dev", dev), ("test", test)):
        hashes: set[str] = set()
        for row in rows:
            dialogue = str(row.get("dialogue") or "").strip()
            summaries = [str(row.get(key) or "").strip() for key in ("summary", "summary1", "summary2", "summary3")]
            if not dialogue or not any(summaries):
                raise ValueError(f"DialogSum {split_name} row is missing dialogue or summary: {row.get('fname')}")
            digest = _sha256_text(_normalize(dialogue))
            if digest in hashes:
                raise ValueError(f"duplicate normalized dialogue in DialogSum {split_name}: {row.get('fname')}")
            hashes.add(digest)
        split_hashes.append(hashes)
    for (left_name, left), (right_name, right) in itertools.combinations(
        zip(("train", "dev", "test"), split_hashes),
        2,
    ):
        if left & right:
            raise ValueError(f"normalized dialogue overlap between DialogSum {left_name} and {right_name}")


def _deduplicate_dialogsum_splits(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    def unique(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        seen: set[str] = set()
        kept = []
        for row in rows:
            digest = _sha256_text(_normalize(str(row.get("dialogue") or "")))
            if digest in seen:
                continue
            seen.add(digest)
            kept.append(row)
        return kept, len(rows) - len(kept)

    test_unique, test_internal = unique(test)
    test_hashes = {_sha256_text(_normalize(str(row["dialogue"]))) for row in test_unique}
    dev_unique, dev_internal = unique(dev)
    dev_without_test = [
        row for row in dev_unique
        if _sha256_text(_normalize(str(row["dialogue"]))) not in test_hashes
    ]
    dev_hashes = {_sha256_text(_normalize(str(row["dialogue"]))) for row in dev_without_test}
    train_unique, train_internal = unique(train)
    train_without_holdout = [
        row for row in train_unique
        if _sha256_text(_normalize(str(row["dialogue"]))) not in test_hashes | dev_hashes
    ]
    report = {
        "train_internal_duplicates_removed": train_internal,
        "dev_internal_duplicates_removed": dev_internal,
        "test_internal_duplicates_removed": test_internal,
        "dev_rows_overlapping_test_removed": len(dev_unique) - len(dev_without_test),
        "train_rows_overlapping_dev_or_test_removed": len(train_unique) - len(train_without_holdout),
    }
    return train_without_holdout, dev_without_test, test_unique, report


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_object(value: dict[str, Any]) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible LayerScope blind benchmark harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-dialogsum")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=1701)
    prepare.add_argument("--selection-count", type=int, default=512)
    prepare.add_argument("--train-count", type=int, default=2048)
    prepare.add_argument("--preference-count", type=int, default=384)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--input", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--specialist-manifest")
    generate_parser.add_argument("--max-tokens", type=int, default=96)
    generate_parser.add_argument("--limit", type=int)
    preferences = subparsers.add_parser("build-preferences")
    preferences.add_argument("--prompts", required=True)
    preferences.add_argument("--base-outputs", required=True)
    preferences.add_argument("--output", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--variant", action="append", required=True, help="NAME=JSONL")
    compare.add_argument("--inputs", required=True)
    compare.add_argument("--references", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--blinding-key-file", required=True)
    compare.add_argument("--seed", type=int, default=1701)
    compare.add_argument("--bootstrap-samples", type=int, default=2000)
    judge = subparsers.add_parser("judge")
    judge.add_argument("--model", required=True)
    judge.add_argument("--pairs", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--limit", type=int)
    score_judge = subparsers.add_parser("score-judgments")
    score_judge.add_argument("--variant", action="append", required=True, help="NAME=JSONL")
    score_judge.add_argument("--inputs", required=True)
    score_judge.add_argument("--judgments", required=True)
    score_judge.add_argument("--blinding-key-file", required=True)
    score_judge.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.command == "prepare-dialogsum":
        result = prepare_dialogsum(args.source, args.output, seed=args.seed, selection_count=args.selection_count, train_count=args.train_count, preference_count=args.preference_count)
    elif args.command == "generate":
        result = generate_outputs(model_path=args.model, input_path=args.input, output_path=args.output, specialist_manifest=args.specialist_manifest, max_tokens=args.max_tokens, limit=args.limit)
    elif args.command == "build-preferences":
        result = build_preferences(args.prompts, args.base_outputs, args.output)
    elif args.command == "judge":
        result = judge_blind_pairs(model_path=args.model, pair_path=args.pairs, output_path=args.output, limit=args.limit)
    elif args.command == "compare":
        variants = {}
        for item in args.variant:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                raise ValueError("--variant must be NAME=JSONL")
            variants[name] = path
        result = compare_outputs(
            variants,
            args.inputs,
            args.references,
            args.output,
            blinding_key_file=args.blinding_key_file,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        variants = {}
        for item in args.variant:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                raise ValueError("--variant must be NAME=JSONL")
            variants[name] = path
        result = score_blind_judgments(
            variants=variants,
            input_path=args.inputs,
            judgment_path=args.judgments,
            blinding_key_file=args.blinding_key_file,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
