from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


SCORE_RUN_SCHEMA = "dataevol.codex_route_ranker_run.v1"
SCORE_ROW_SCHEMA = "dataevol.codex_route_ranker_score.v1"
EVALUATION_SCHEMA = "dataevol.codex_route_ranker_evaluation.v1"


def score_ranking_candidates(
    *,
    model_path: str,
    input_path: str | Path,
    output_path: str | Path,
    specialist_manifest: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    from dataevol.local_models.layer_specialist import _decoder_layers, model_fingerprint
    from dataevol.specialist_server.swapper import MlxLayerSwapper

    rows = _read_unique_jsonl(Path(input_path), "id")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    destination = Path(output_path)
    manifest_revision = None
    if specialist_manifest:
        manifest_body = json.loads(Path(specialist_manifest).read_text(encoding="utf-8"))
        manifest_revision = manifest_body.get("base_model_revision")
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
            base_model_hash=fingerprint["sha256"],
            base_model_revision=fingerprint.get("resolved_revision"),
            num_layers=len(layers),
        )
        specialist = swapper.register(specialist_manifest)
        swapper.activate(specialist.name)
        variant = specialist.name
        candidate_hash = specialist.candidate_content_hash
    expected = [{"id": row["id"], "prompt_sha256": _sha256_text(str(row["prompt"]))} for row in rows]
    config = {
        "schema": SCORE_RUN_SCHEMA,
        "model_fingerprint": fingerprint["sha256"],
        "variant": variant,
        "candidate_hash": candidate_hash,
        "input_sha256": _sha256_path(input_path),
        "expected_rows": len(rows),
        "expected_ids_hash": _hash_object({"rows": expected}),
        "scoring": "mean_choice_label_log_probability_v2",
    }
    run_hash = _hash_object(config)
    sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or sidecar.exists():
        raise ValueError(f"ranking score output is immutable and already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_suffix(destination.suffix + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"ranking score run is already locked: {lock}") from exc
    started = time.time()
    try:
        with destination.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                prompt = str(row["prompt"])
                choices = [dict(value) for value in row.get("eligible_options") or []]
                choice_ids = [str(value) for value in row.get("eligible_choice_ids") or []]
                if not choice_ids or len(choice_ids) != len(set(choice_ids)) or {choice["choice_id"] for choice in choices} != set(choice_ids):
                    raise ValueError(f"ranking row {row['id']} requires unique eligible choices")
                option_by_choice = {str(choice["choice_id"]): str(choice["option_id"]) for choice in choices}
                scores = _score_options_with_shared_prefix(model, tokenizer, prompt, choice_ids, index)
                scores.sort(key=lambda item: (-item["mean_log_probability"], item["choice_id"]))
                probabilities = _softmax([item["mean_log_probability"] for item in scores])
                for item, probability in zip(scores, probabilities):
                    item["normalized_probability"] = probability
                margin = scores[0]["mean_log_probability"] - scores[1]["mean_log_probability"] if len(scores) > 1 else math.inf
                record = {
                    "schema": SCORE_ROW_SCHEMA,
                    "id": row["id"],
                    "task_group": row["task_group"],
                    "subtask_id": row["subtask_id"],
                    "catalog_hash": row["catalog_hash"],
                    "policy_hash": row["policy_hash"],
                    "candidate_set_hash": row["candidate_set_hash"],
                    "prompt_sha256": _sha256_text(prompt),
                    "ranked_option_ids": [option_by_choice[item["choice_id"]] for item in scores],
                    "scores": [{**item, "option_id": option_by_choice[item["choice_id"]]} for item in scores],
                    "confidence": probabilities[0],
                    "score_margin": margin,
                    "variant": variant,
                    "candidate_hash": candidate_hash,
                    "run_config_hash": run_hash,
                }
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                if index == 1 or index % 25 == 0:
                    print(json.dumps({"scored": index, "total": len(rows), "variant": variant}), flush=True)
        result = {
            "schema": SCORE_RUN_SCHEMA,
            "status": "completed",
            "run_config": config,
            "run_config_hash": run_hash,
            "output_sha256": _sha256_path(destination),
            "elapsed_seconds": time.time() - started,
        }
        sidecar.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception:
        destination.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _score_options_with_shared_prefix(
    model: Any,
    tokenizer: Any,
    prompt: str,
    option_ids: list[str],
    row_index: int,
) -> list[dict[str, Any]]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    prepared = []
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        prefix_text = str(apply_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
    else:
        prefix_text = f"{prompt}\n"
    prefix_ids = list(tokenizer.encode(prefix_text))
    for option_id in option_ids:
        full_ids = list(tokenizer.encode(f"{prefix_text}{option_id}"))
        common = 0
        for left, right in zip(prefix_ids, full_ids):
            if left != right:
                break
            common += 1
        prefix = full_ids[:common]
        suffix = full_ids[common:]
        if not prefix or not suffix:
            raise ValueError(f"ranking option {option_id} has no scoreable completion tokens")
        prepared.append((option_id, prefix, suffix))
    shared_prefix = prepared[0][1]
    if any(prefix != shared_prefix for _, prefix, _ in prepared[1:]):
        raise ValueError("ranking options do not share an identical tokenized prompt prefix")
    prompt_cache = make_prompt_cache(model)
    prefix_logits = model(mx.array(shared_prefix)[None, :], cache=prompt_cache)[:, -1:, :]
    mx.eval(prefix_logits)
    saved = [(cache.state, cache.meta_state) for cache in prompt_cache]
    scores = []
    for option_id, _, suffix in prepared:
        for cache, (state, meta_state) in zip(prompt_cache, saved):
            cache.state = state
            cache.meta_state = meta_state
        if len(suffix) > 1:
            continuation_logits = model(mx.array(suffix[:-1])[None, :], cache=prompt_cache)
            logits = mx.concatenate([prefix_logits, continuation_logits], axis=1)
        else:
            logits = prefix_logits
        targets = mx.array(suffix)[None, :, None]
        selected = mx.take_along_axis(logits, targets, axis=-1).squeeze(-1)
        log_probabilities = selected - mx.logsumexp(logits, axis=-1)
        score_value = log_probabilities.mean()
        mx.eval(score_value)
        score = float(score_value)
        if not math.isfinite(score):
            raise RuntimeError(f"non-finite candidate score for row {row_index} {option_id}")
        scores.append({"choice_id": option_id, "mean_log_probability": score})
    return scores


def evaluate_ranking_scores(
    references: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    refs = _unique_by(references, "id")
    preds = _unique_by(predictions, "id")
    if set(refs) != set(preds):
        raise ValueError("ranking predictions must exactly cover reference IDs")
    exact = model_correct = effort_correct = 0
    confidences = []
    correctness = []
    margins = []
    per_example = []
    for row_id in sorted(refs):
        reference = refs[row_id]
        prediction = preds[row_id]
        for field in ("catalog_hash", "policy_hash", "candidate_set_hash"):
            if prediction.get(field) != reference.get(field):
                raise ValueError(f"ranking prediction {field} mismatch: {row_id}")
        if prediction.get("prompt_sha256") != _sha256_text(str(reference["prompt"])):
            raise ValueError(f"ranking prediction prompt hash mismatch: {row_id}")
        ranked = prediction.get("ranked_option_ids")
        eligible = [str(value) for value in reference.get("eligible_option_ids") or []]
        if not isinstance(ranked, list) or sorted(ranked) != sorted(eligible):
            raise ValueError(f"ranking prediction must rank every eligible option exactly once: {row_id}")
        selected = str(ranked[0])
        expected = str(reference["chosen_option_id"])
        lookup = {str(item["option_id"]): item for item in reference.get("eligible_options") or []}
        is_exact = selected == expected
        exact += int(is_exact)
        model_correct += int(lookup[selected]["model_id"] == lookup[expected]["model_id"])
        effort_correct += int(lookup[selected]["reasoning_effort"] == lookup[expected]["reasoning_effort"])
        confidence = _bounded(prediction.get("confidence"), f"prediction {row_id} confidence")
        margin = float(prediction.get("score_margin", 0.0))
        confidences.append(confidence)
        correctness.append(1.0 if is_exact else 0.0)
        margins.append(margin)
        per_example.append({"id": row_id, "selected_option_id": selected, "expected_option_id": expected, "correct": is_exact, "confidence": confidence, "score_margin": margin})
    count = len(refs)
    metrics = {
        "option_accuracy": exact / count,
        "model_accuracy": model_correct / count,
        "reasoning_effort_accuracy": effort_correct / count,
        "mean_confidence": statistics.fmean(confidences),
        "mean_score_margin": statistics.fmean(margins),
        "brier_score": statistics.fmean((confidence - correct) ** 2 for confidence, correct in zip(confidences, correctness)),
        "adaptive_ece": _ece(confidences, correctness),
    }
    return {"schema": EVALUATION_SCHEMA, "rows": count, "metrics": metrics, "per_example": per_example, "evaluation_hash": _hash_object({"rows": count, "metrics": metrics, "per_example": per_example})}


def _ece(confidences: list[float], correctness: list[float], bins: int = 10) -> float:
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [position for position, confidence in enumerate(confidences) if low <= confidence < high or (index == bins - 1 and confidence == 1.0)]
        if members:
            error += len(members) / total * abs(statistics.fmean(confidences[position] for position in members) - statistics.fmean(correctness[position] for position in members))
    return error


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def _read_unique_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    return list(_unique_by((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), field).values())


def _unique_by(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in rows:
        row = dict(raw)
        key = str(row.get(field) or "")
        if not key or key in result:
            raise ValueError(f"rows require unique {field}: {key}")
        result[key] = row
    if not result:
        raise ValueError("at least one ranking row is required")
    return result


def _bounded(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be between 0 and 1") from exc
    if not math.isfinite(number) or number < 0 or number > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _hash_object(value: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score finite Codex routing options without free-form generation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--model", required=True)
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--specialist-manifest")
    score.add_argument("--limit", type=int)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--references", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "score":
        result = score_ranking_candidates(model_path=args.model, input_path=args.input, output_path=args.output, specialist_manifest=args.specialist_manifest, limit=args.limit)
    else:
        result = evaluate_ranking_scores(_read_unique_jsonl(Path(args.references), "id"), _read_unique_jsonl(Path(args.predictions), "id"))
        destination = Path(args.output)
        if destination.exists():
            raise ValueError(f"ranking evaluation is immutable and already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
