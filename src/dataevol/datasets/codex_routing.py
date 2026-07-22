from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping


CATALOG_SCHEMA = "dataevol.codex_model_catalog.v1"
SFT_SCHEMA = "dataevol.codex_routing_sft.v1"
DPO_SCHEMA = "dataevol.codex_routing_preference.v1"
FEEDBACK_SCHEMA = "dataevol.codex_routing_outcome.v1"
DATASET_SCHEMA = "dataevol.codex_routing_datasets.v1"
EVALUATION_SCHEMA = "dataevol.codex_routing_evaluation.v1"
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
REASONING_EFFORT_LEVEL = {"none": 0.0, "minimal": 0.25, "low": 1.0, "medium": 2.0, "high": 3.0, "xhigh": 4.0}


@dataclass(frozen=True)
class CodexRoutingDatasetResult:
    manifest_path: Path
    sft_train_path: Path
    sft_eval_path: Path
    dpo_train_path: Path
    dpo_eval_path: Path
    dataset_content_hash: str


def freeze_model_catalog(
    catalog: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate and freeze a caller-provided current model catalog snapshot."""
    normalized = _normalize_catalog(catalog)
    path = Path(output_path)
    if path.exists():
        existing = _load_json(path)
        if existing != normalized:
            raise ValueError(f"model catalog snapshot already exists with different content: {path}")
        return existing
    _atomic_write(path, _json_bytes(normalized, pretty=True))
    return normalized


def build_task_decomposition_rows(
    tasks: Iterable[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    catalog = _load_catalog(catalog_snapshot)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(tasks, start=1):
        task = _normalize_task(raw, index=index)
        route = _normalize_route(raw.get("route") or raw.get("routing"), task["subtasks"], catalog)
        prompt = _routing_prompt(task, catalog)
        completion = _canonical_json(route)
        identity = {
            "schema": SFT_SCHEMA,
            "catalog_hash": catalog["catalog_hash"],
            "task_group": task["task_group"],
            "prompt": prompt,
            "completion": completion,
        }
        row_id = f"codex_route_sft_{_hash_object(identity)[:24]}"
        if row_id in seen_ids:
            raise ValueError(f"duplicate task decomposition training row: {row_id}")
        seen_ids.add(row_id)
        rows.append(
            {
                "schema": SFT_SCHEMA,
                "id": row_id,
                "task_id": task["task_id"],
                "task_group": task["task_group"],
                "catalog_hash": catalog["catalog_hash"],
                "prompt": prompt,
                "completion": completion,
                "subtasks": task["subtasks"],
                "source": str(raw.get("source") or "task_decomposition"),
            }
        )
    if not rows:
        raise ValueError("at least one task decomposition row is required")
    return sorted(rows, key=lambda row: str(row["id"]))


def build_preference_rows(
    preferences: Iterable[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    catalog = _load_catalog(catalog_snapshot)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(preferences, start=1):
        if raw.get("schema") == DPO_SCHEMA:
            row = _normalize_prebuilt_preference(raw, catalog, index=index)
            pair_id = str(row["pair_id"])
            if pair_id in seen_ids:
                raise ValueError(f"duplicate routing preference pair_id: {pair_id}")
            seen_ids.add(pair_id)
            rows.append(row)
            continue
        task = _normalize_task(raw, index=index)
        chosen = _normalize_route(raw.get("chosen_route"), task["subtasks"], catalog)
        rejected = _normalize_route(raw.get("rejected_route"), task["subtasks"], catalog)
        if _route_decision(chosen) == _route_decision(rejected):
            raise ValueError(f"preference row {index} chosen and rejected routes must differ")
        prompt = _routing_prompt(task, catalog)
        identity = {
            "schema": DPO_SCHEMA,
            "catalog_hash": catalog["catalog_hash"],
            "task_group": task["task_group"],
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        pair_id = str(raw.get("pair_id") or f"codex_route_dpo_{_hash_object(identity)[:24]}")
        if pair_id in seen_ids:
            raise ValueError(f"duplicate routing preference pair_id: {pair_id}")
        seen_ids.add(pair_id)
        rows.append(
            {
                "schema": DPO_SCHEMA,
                "pair_id": pair_id,
                "task_id": task["task_id"],
                "task_group": task["task_group"],
                "catalog_hash": catalog["catalog_hash"],
                "prompt": prompt,
                "chosen": _canonical_json(chosen),
                "rejected": _canonical_json(rejected),
                "subtasks": task["subtasks"],
                "provenance": dict(raw.get("provenance") or {}),
            }
        )
    if not rows:
        raise ValueError("at least one routing preference row is required")
    return sorted(rows, key=lambda row: str(row["pair_id"]))


def build_codex_routing_datasets(
    catalog_snapshot: Mapping[str, Any] | str | Path,
    tasks: Iterable[Mapping[str, Any]],
    preferences: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 1701,
    eval_fraction: float = 0.2,
) -> CodexRoutingDatasetResult:
    """Build frozen, group-disjoint SFT and DPO files without launching training."""
    if not 0.0 < float(eval_fraction) < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    catalog = _load_catalog(catalog_snapshot)
    sft_rows = build_task_decomposition_rows(tasks, catalog)
    dpo_rows = build_preference_rows(preferences, catalog)
    groups = sorted({str(row["task_group"]) for row in [*sft_rows, *dpo_rows]})
    if len(groups) < 2:
        raise ValueError("routing datasets require at least two task groups")
    ranked_groups = sorted(groups, key=lambda value: _sha256_text(f"{seed}:{value}"))
    eval_count = max(1, min(len(groups) - 1, round(len(groups) * float(eval_fraction))))
    eval_groups = set(ranked_groups[:eval_count])

    sft_train, sft_eval = _partition(sft_rows, eval_groups)
    dpo_train, dpo_eval = _partition(dpo_rows, eval_groups)
    if not all((sft_train, sft_eval, dpo_train, dpo_eval)):
        raise ValueError(
            "each SFT/DPO split must contain rows; add preference coverage across train and eval task groups"
        )

    output = Path(output_dir)
    paths = {
        "sft_train": output / "codex_routing_sft_train.jsonl",
        "sft_eval": output / "codex_routing_sft_eval.jsonl",
        "dpo_train": output / "codex_routing_dpo_train.jsonl",
        "dpo_eval": output / "codex_routing_dpo_eval.jsonl",
    }
    values = {
        "sft_train": sft_train,
        "sft_eval": sft_eval,
        "dpo_train": dpo_train,
        "dpo_eval": dpo_eval,
    }
    payloads = {name: _jsonl_bytes(rows) for name, rows in values.items()}
    files = {
        name: {"path": str(paths[name]), "rows": len(values[name]), "sha256": _sha256_bytes(payloads[name])}
        for name in paths
    }
    identity = {
        "schema": DATASET_SCHEMA,
        "catalog_hash": catalog["catalog_hash"],
        "snapshot_hash": catalog["snapshot_hash"],
        "seed": int(seed),
        "eval_fraction": float(eval_fraction),
        "eval_task_groups": sorted(eval_groups),
        "files": {name: {"rows": item["rows"], "sha256": item["sha256"]} for name, item in files.items()},
    }
    dataset_content_hash = _hash_object(identity)
    manifest = {
        **identity,
        "dataset_content_hash": dataset_content_hash,
        "files": files,
        "split_strategy": "deterministic_task_group",
        "layer_specialist_training": {
            "task_type": "codex-task-model-routing",
            "sft": {"training_mode": "sft", "dataset": str(paths["sft_train"])},
            "dpo": {
                "training_mode": "rl",
                "dataset": str(paths["dpo_train"]),
                "requires_initial_sft_specialist_manifest": True,
            },
        },
    }
    manifest_path = output / "codex_routing_datasets.manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        if existing.get("dataset_content_hash") != dataset_content_hash:
            raise ValueError(f"routing dataset manifest already exists with different content: {manifest_path}")
        for name, item in existing.get("files", {}).items():
            if name not in paths or not paths[name].is_file() or _sha256_path(paths[name]) != item.get("sha256"):
                raise ValueError(f"existing routing dataset artifact failed integrity check: {name}")
        return CodexRoutingDatasetResult(
            manifest_path, paths["sft_train"], paths["sft_eval"], paths["dpo_train"], paths["dpo_eval"], dataset_content_hash
        )
    if any(path.exists() for path in paths.values()):
        raise ValueError("routing dataset files exist without a matching manifest")
    for name, path in paths.items():
        _atomic_write(path, payloads[name])
    _atomic_write(manifest_path, _json_bytes(manifest, pretty=True))
    return CodexRoutingDatasetResult(
        manifest_path, paths["sft_train"], paths["sft_eval"], paths["dpo_train"], paths["dpo_eval"], dataset_content_hash
    )


def normalize_outcome_feedback(
    outcomes: Iterable[Mapping[str, Any]],
    task_rows: Iterable[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any] | str | Path,
    *,
    cost_cap_usd: float = 1.0,
    latency_cap_ms: float = 60_000.0,
) -> list[dict[str, Any]]:
    """Normalize observed routing outcomes; the score is observational, not causal."""
    if cost_cap_usd <= 0 or latency_cap_ms <= 0:
        raise ValueError("feedback cost and latency caps must be positive")
    catalog = _load_catalog(catalog_snapshot)
    tasks = _task_row_index(task_rows)
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for index, raw in enumerate(outcomes, start=1):
        task = _feedback_task(raw, tasks)
        route = _normalize_route(raw.get("route") or raw.get("routing"), task["subtasks"], catalog)
        verified = _required_bool(raw.get("verified"), f"outcome {index} verified")
        quality = _bounded_number(raw.get("quality", 1.0 if verified else 0.0), f"outcome {index} quality")
        cost = _non_negative_number(raw.get("cost_usd", 0.0), f"outcome {index} cost_usd")
        latency = _non_negative_number(raw.get("latency_ms", 0.0), f"outcome {index} latency_ms")
        reasoning_tokens = _non_negative_number(raw.get("reasoning_tokens"), f"outcome {index} reasoning_tokens")
        total_tokens = _non_negative_number(raw.get("total_tokens"), f"outcome {index} total_tokens")
        if reasoning_tokens > total_tokens:
            raise ValueError(f"outcome {index} reasoning_tokens cannot exceed total_tokens")
        utility_components = {
            "verified": 1.0 if verified else 0.0,
            "quality": quality,
            "cost_penalty": min(cost / cost_cap_usd, 1.0),
            "latency_penalty": min(latency / latency_cap_ms, 1.0),
            "reasoning_token_penalty": min(reasoning_tokens / 100_000.0, 1.0),
        }
        utility = (
            0.55 * utility_components["verified"]
            + 0.35 * utility_components["quality"]
            - 0.04 * utility_components["cost_penalty"]
            - 0.03 * utility_components["latency_penalty"]
            - 0.03 * utility_components["reasoning_token_penalty"]
        )
        unsigned = {
            "schema": FEEDBACK_SCHEMA,
            "outcome_id": str(raw.get("outcome_id") or raw.get("id") or f"outcome_{index:06d}"),
            "task_id": task["task_id"],
            "task_group": task["task_group"],
            "catalog_hash": catalog["catalog_hash"],
            "route": route,
            "route_hash": _hash_object(_route_decision(route)),
            "verified": verified,
            "quality": quality,
            "cost_usd": cost,
            "latency_ms": latency,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "failure_type": str(raw.get("failure_type") or "") or None,
            "utility": round(utility, 8),
            "utility_components": utility_components,
            "utility_policy": "0.55*verified + 0.35*quality - 0.04*cost_penalty - 0.03*latency_penalty - 0.03*reasoning_token_penalty",
        }
        record_hash = _hash_object(unsigned)
        outcome_id = unsigned["outcome_id"]
        if outcome_id in seen and seen[outcome_id] != record_hash:
            raise ValueError(f"outcome_id has conflicting feedback: {outcome_id}")
        if outcome_id not in seen:
            records.append({**unsigned, "feedback_hash": record_hash})
            seen[outcome_id] = record_hash
    return sorted(records, key=lambda row: str(row["outcome_id"]))


def preferences_from_outcomes(
    task_rows: Iterable[Mapping[str, Any]],
    feedback: Iterable[Mapping[str, Any]],
    *,
    min_utility_gap: float = 0.05,
) -> list[dict[str, Any]]:
    """Create conservative DPO rows only from verified, distinct observed routes."""
    tasks = _task_row_index(task_rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in feedback:
        row = dict(item)
        if row.get("schema") != FEEDBACK_SCHEMA:
            raise ValueError("feedback row has an unsupported schema")
        groups.setdefault(str(row.get("task_group") or ""), []).append(row)
    preferences: list[dict[str, Any]] = []
    for task_group, rows in sorted(groups.items()):
        distinct = {str(row.get("route_hash")): row for row in rows}
        if len(distinct) < 2 or task_group not in tasks:
            continue
        ranked = sorted(distinct.values(), key=lambda row: (float(row.get("utility", 0.0)), str(row.get("outcome_id"))))
        rejected, chosen = ranked[0], ranked[-1]
        gap = float(chosen["utility"]) - float(rejected["utility"])
        if not chosen.get("verified") or gap < min_utility_gap:
            continue
        task = tasks[task_group]
        prompt = str(task["prompt"])
        identity = {
            "task_group": task_group,
            "chosen_feedback_hash": chosen["feedback_hash"],
            "rejected_feedback_hash": rejected["feedback_hash"],
        }
        preferences.append(
            {
                "schema": DPO_SCHEMA,
                "pair_id": f"codex_feedback_dpo_{_hash_object(identity)[:24]}",
                "task_id": task["task_id"],
                "task_group": task_group,
                "catalog_hash": task["catalog_hash"],
                "prompt": prompt,
                "chosen": _canonical_json(chosen["route"]),
                "rejected": _canonical_json(rejected["route"]),
                "subtasks": task["subtasks"],
                "provenance": {
                    "chosen_outcome_id": chosen["outcome_id"],
                    "rejected_outcome_id": rejected["outcome_id"],
                    "utility_gap": round(gap, 8),
                    "observational_not_causal": True,
                },
            }
        )
    return preferences


def evaluate_routing_predictions(
    reference_rows: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any] | str | Path,
    *,
    feedback: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Evaluate complete offline predictions against frozen decomposition routes."""
    catalog = _load_catalog(catalog_snapshot)
    references = _unique_by(reference_rows, "id", "reference")
    predicted = _unique_by(predictions, "id", "prediction")
    if set(references) != set(predicted):
        missing = sorted(set(references) - set(predicted))
        extra = sorted(set(predicted) - set(references))
        raise ValueError(f"prediction IDs must exactly cover references missing={missing[:5]} extra={extra[:5]}")
    feedback_index = {
        (str(row.get("task_group")), str(row.get("route_hash"))): dict(row)
        for row in feedback
    }
    per_example = []
    assignment_total = 0
    assignment_correct = 0
    model_correct = 0
    effort_correct = 0
    predicted_effort_levels: list[float] = []
    reference_effort_levels: list[float] = []
    valid_count = 0
    exact_count = 0
    constraint_total = 0
    constraint_satisfied = 0
    matched_feedback = []
    for row_id in sorted(references):
        reference = references[row_id]
        prediction = predicted[row_id]
        if str(reference.get("catalog_hash")) != catalog["catalog_hash"]:
            raise ValueError(f"reference catalog hash mismatch: {row_id}")
        if str(prediction.get("catalog_hash")) != catalog["catalog_hash"]:
            raise ValueError(f"prediction catalog hash mismatch: {row_id}")
        prompt = str(reference.get("prompt") or "")
        if str(prediction.get("prompt_sha256")) != _sha256_text(prompt):
            raise ValueError(f"prediction prompt hash mismatch: {row_id}")
        subtasks = list(reference.get("subtasks") or [])
        expected = _normalize_route(json.loads(str(reference["completion"])), subtasks, catalog)
        valid = True
        try:
            actual = _normalize_route(json.loads(str(prediction.get("output") or "")), subtasks, catalog)
        except (TypeError, ValueError, json.JSONDecodeError):
            valid = False
            actual = {"assignments": [], "fallback_model_id": None, "requires_verification": False}
        valid_count += int(valid)
        expected_by_subtask = {
            row["subtask_id"]: (row["model_id"], row["reasoning_effort"])
            for row in expected["assignments"]
        }
        actual_by_subtask = {
            row["subtask_id"]: (row["model_id"], row["reasoning_effort"])
            for row in actual["assignments"]
        }
        correct = sum(actual_by_subtask.get(key) == value for key, value in expected_by_subtask.items())
        correct_models = sum(actual_by_subtask.get(key, (None, None))[0] == value[0] for key, value in expected_by_subtask.items())
        correct_efforts = sum(actual_by_subtask.get(key, (None, None))[1] == value[1] for key, value in expected_by_subtask.items())
        assignment_total += len(expected_by_subtask)
        assignment_correct += correct
        model_correct += correct_models
        effort_correct += correct_efforts
        reference_effort_levels.extend(REASONING_EFFORT_LEVEL[value[1]] for value in expected_by_subtask.values())
        predicted_effort_levels.extend(REASONING_EFFORT_LEVEL[value[1]] for value in actual_by_subtask.values())
        constraints = _route_constraint_counts(actual, subtasks, catalog) if valid else (len(subtasks), 0)
        constraint_total += constraints[0]
        constraint_satisfied += constraints[1]
        exact = valid and _route_decision(actual) == _route_decision(expected)
        exact_count += int(exact)
        route_hash = _hash_object(_route_decision(actual)) if valid else None
        observed = feedback_index.get((str(reference.get("task_group")), str(route_hash)))
        if observed:
            matched_feedback.append(observed)
        per_example.append(
            {
                "id": row_id,
                "valid_route": valid,
                "exact_route": exact,
                "correct_assignments": correct,
                "correct_models": correct_models,
                "correct_reasoning_efforts": correct_efforts,
                "assignment_count": len(expected_by_subtask),
                "constraint_satisfied": constraints[1],
                "constraint_count": constraints[0],
                "matched_observational_feedback": bool(observed),
            }
        )
    count = len(references)
    observational = {
        "coverage": len(matched_feedback) / count if count else 0.0,
        "verified_rate": fmean(1.0 if row.get("verified") else 0.0 for row in matched_feedback) if matched_feedback else None,
        "quality": fmean(float(row.get("quality", 0.0)) for row in matched_feedback) if matched_feedback else None,
        "cost_usd": fmean(float(row.get("cost_usd", 0.0)) for row in matched_feedback) if matched_feedback else None,
        "latency_ms": fmean(float(row.get("latency_ms", 0.0)) for row in matched_feedback) if matched_feedback else None,
        "reasoning_tokens": fmean(float(row.get("reasoning_tokens", 0.0)) for row in matched_feedback) if matched_feedback else None,
        "total_tokens": fmean(float(row.get("total_tokens", 0.0)) for row in matched_feedback) if matched_feedback else None,
        "causal_claim_allowed": False,
    }
    result = {
        "schema": EVALUATION_SCHEMA,
        "catalog_hash": catalog["catalog_hash"],
        "rows": count,
        "metrics": {
            "valid_route_rate": valid_count / count if count else 0.0,
            "exact_route_accuracy": exact_count / count if count else 0.0,
            "subtask_assignment_accuracy": assignment_correct / assignment_total if assignment_total else 0.0,
            "subtask_model_accuracy": model_correct / assignment_total if assignment_total else 0.0,
            "subtask_reasoning_effort_accuracy": effort_correct / assignment_total if assignment_total else 0.0,
            "average_reasoning_effort_level": fmean(predicted_effort_levels) if predicted_effort_levels else None,
            "reference_average_reasoning_effort_level": fmean(reference_effort_levels) if reference_effort_levels else None,
            "reasoning_effort_level_delta": (
                fmean(predicted_effort_levels) - fmean(reference_effort_levels)
                if len(predicted_effort_levels) == len(reference_effort_levels) and predicted_effort_levels
                else None
            ),
            "constraint_satisfaction_rate": constraint_satisfied / constraint_total if constraint_total else 0.0,
        },
        "observational_outcomes": observational,
        "per_example": per_example,
    }
    result["evaluation_hash"] = _hash_object(result)
    return result


def _normalize_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _required_text(value.get("source"), "catalog source")
    source_revision = _required_text(value.get("source_revision"), "catalog source_revision")
    captured_at = _required_text(value.get("captured_at"), "catalog captured_at")
    configured_model_id = str(value.get("configured_model_id") or "") or None
    _validate_timestamp(captured_at)
    raw_models = value.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("catalog models must be a non-empty list")
    models = []
    seen: set[str] = set()
    excluded_hidden_model_count = 0
    for index, raw in enumerate(raw_models, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"catalog model {index} must be an object")
        if raw.get("hidden") is True or str(raw.get("availability") or "").lower() == "hidden":
            excluded_hidden_model_count += 1
            continue
        model_id = _required_text(raw.get("model_id") or raw.get("id"), f"catalog model {index} model_id")
        if model_id in seen:
            raise ValueError(f"duplicate catalog model_id: {model_id}")
        seen.add(model_id)
        availability = str(raw.get("availability") or "listed")
        if availability not in {"listed", "configured_verified"}:
            raise ValueError(f"{model_id} availability must be listed or configured_verified")
        capabilities = sorted({_required_text(item, f"{model_id} capability") for item in (raw.get("capabilities") or [])})
        raw_efforts = raw.get("supported_reasoning_efforts") or raw.get("supported_reasoning_levels")
        if not isinstance(raw_efforts, list) or not raw_efforts:
            raise ValueError(f"{model_id} supported_reasoning_efforts must be a non-empty list")
        supported_efforts: list[str] = []
        for raw_effort in raw_efforts:
            effort = str(raw_effort.get("effort")) if isinstance(raw_effort, Mapping) else str(raw_effort)
            if effort not in REASONING_EFFORTS:
                raise ValueError(f"{model_id} has unsupported reasoning effort: {effort}")
            supported_efforts.append(effort)
        supported_efforts = sorted(set(supported_efforts), key=lambda effort: REASONING_EFFORT_LEVEL[effort])
        default_effort = str(raw.get("default_reasoning_effort") or raw.get("default_reasoning_level") or "")
        if default_effort not in supported_efforts:
            raise ValueError(f"{model_id} default_reasoning_effort must be supported")
        context_window = int(_positive_number(raw.get("context_window", 1), f"{model_id} context_window"))
        pricing = raw.get("pricing")
        if pricing is not None and not isinstance(pricing, Mapping):
            raise ValueError(f"{model_id} pricing must be an object")
        raw_pricing = dict(pricing or {})
        legacy_api_pricing = None
        if any(key in raw for key in ("input_cost_per_million", "cached_input_cost_per_million", "output_cost_per_million")):
            legacy_api_pricing = {
                "input": _non_negative_number(raw.get("input_cost_per_million", 0.0), f"{model_id} input cost"),
                "cached_input": _non_negative_number(raw.get("cached_input_cost_per_million", 0.0), f"{model_id} cached input cost"),
                "output": _non_negative_number(raw.get("output_cost_per_million", 0.0), f"{model_id} output cost"),
            }
        models.append(
            {
                "model_id": model_id,
                "provider": _required_text(raw.get("provider"), f"{model_id} provider"),
                "revision": str(raw.get("revision") or "") or None,
                "enabled": _required_bool(raw.get("enabled", True), f"{model_id} enabled"),
                "availability": availability,
                "capabilities": capabilities,
                "supported_reasoning_efforts": supported_efforts,
                "default_reasoning_effort": default_effort,
                "context_window": context_window,
                "pricing": {
                    "codex_credit": _normalize_price_schedule(raw_pricing.get("codex_credit"), f"{model_id} codex_credit"),
                    "api_usd_equivalent": _normalize_price_schedule(
                        raw_pricing.get("api_usd_equivalent", legacy_api_pricing), f"{model_id} api_usd_equivalent"
                    ),
                },
                "latency_p50_ms": _non_negative_number(raw.get("latency_p50_ms", 0.0), f"{model_id} latency"),
                "risk_tiers": sorted({_required_text(item, f"{model_id} risk tier") for item in (raw.get("risk_tiers") or [])}),
                "metadata": dict(raw.get("metadata") or {}),
            }
        )
    models.sort(key=lambda row: row["model_id"])
    if not models:
        raise ValueError("catalog contains no non-hidden models")
    if configured_model_id is not None and configured_model_id not in {row["model_id"] for row in models}:
        raise ValueError("configured_model_id must resolve to a non-hidden catalog model")
    content = {
        "schema": CATALOG_SCHEMA,
        "source": source,
        "source_revision": source_revision,
        "configured_model_id": configured_model_id,
        "models": models,
    }
    catalog_hash = _hash_object(content)
    snapshot = {
        **content,
        "captured_at": captured_at,
        "excluded_hidden_model_count": excluded_hidden_model_count,
        "catalog_hash": catalog_hash,
    }
    return {**snapshot, "snapshot_hash": _hash_object(snapshot)}


def _load_catalog(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    raw = _load_json(Path(value)) if isinstance(value, (str, Path)) else dict(value)
    normalized = _normalize_catalog(raw)
    for field in ("catalog_hash", "snapshot_hash"):
        if raw.get(field) is not None and str(raw[field]) != normalized[field]:
            raise ValueError(f"model catalog {field} failed integrity validation")
    return normalized


def _normalize_task(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    task_id = _required_text(raw.get("task_id") or raw.get("id"), f"task {index} task_id")
    task_text = _required_text(raw.get("task") or raw.get("objective") or raw.get("prompt"), f"task {task_id}")
    task_group = str(raw.get("task_group") or _sha256_text(task_id))
    raw_subtasks = raw.get("subtasks") or raw.get("decomposition")
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ValueError(f"task {task_id} requires a non-empty subtasks/decomposition list")
    subtasks = []
    seen: set[str] = set()
    for sub_index, item in enumerate(raw_subtasks, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"task {task_id} subtask {sub_index} must be an object")
        subtask_id = _required_text(item.get("subtask_id") or item.get("id"), f"task {task_id} subtask_id")
        if subtask_id in seen:
            raise ValueError(f"task {task_id} has duplicate subtask_id: {subtask_id}")
        seen.add(subtask_id)
        dependencies = sorted({_required_text(dep, f"{subtask_id} dependency") for dep in (item.get("depends_on") or [])})
        subtasks.append(
            {
                "subtask_id": subtask_id,
                "objective": _required_text(item.get("objective") or item.get("task"), f"subtask {subtask_id} objective"),
                "depends_on": dependencies,
                "required_capabilities": sorted({_required_text(cap, f"{subtask_id} capability") for cap in (item.get("required_capabilities") or [])}),
                "risk": str(item.get("risk") or "medium"),
                "estimated_input_tokens": int(_non_negative_number(item.get("estimated_input_tokens", 0), f"{subtask_id} estimated_input_tokens")),
            }
        )
    subtask_ids = {row["subtask_id"] for row in subtasks}
    for subtask in subtasks:
        unknown = set(subtask["depends_on"]) - subtask_ids
        if unknown or subtask["subtask_id"] in subtask["depends_on"]:
            raise ValueError(f"subtask {subtask['subtask_id']} has invalid dependencies: {sorted(unknown)}")
    _assert_acyclic(subtasks)
    return {"task_id": task_id, "task_group": task_group, "task": task_text, "subtasks": sorted(subtasks, key=lambda row: row["subtask_id"])}


def _normalize_prebuilt_preference(raw: Mapping[str, Any], catalog: dict[str, Any], *, index: int) -> dict[str, Any]:
    if str(raw.get("catalog_hash")) != catalog["catalog_hash"]:
        raise ValueError(f"prebuilt preference {index} catalog hash mismatch")
    task_id = _required_text(raw.get("task_id"), f"prebuilt preference {index} task_id")
    task_group = _required_text(raw.get("task_group"), f"prebuilt preference {index} task_group")
    prompt = _required_text(raw.get("prompt"), f"prebuilt preference {index} prompt")
    subtasks = raw.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError(f"prebuilt preference {index} requires subtasks")
    try:
        chosen_raw = json.loads(_required_text(raw.get("chosen"), f"prebuilt preference {index} chosen"))
        rejected_raw = json.loads(_required_text(raw.get("rejected"), f"prebuilt preference {index} rejected"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"prebuilt preference {index} routes must be JSON objects") from exc
    chosen = _normalize_route(chosen_raw, subtasks, catalog)
    rejected = _normalize_route(rejected_raw, subtasks, catalog)
    if _route_decision(chosen) == _route_decision(rejected):
        raise ValueError(f"prebuilt preference {index} chosen and rejected routes must differ")
    identity = {
        "schema": DPO_SCHEMA,
        "catalog_hash": catalog["catalog_hash"],
        "task_group": task_group,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    return {
        "schema": DPO_SCHEMA,
        "pair_id": str(raw.get("pair_id") or f"codex_route_dpo_{_hash_object(identity)[:24]}"),
        "task_id": task_id,
        "task_group": task_group,
        "catalog_hash": catalog["catalog_hash"],
        "prompt": prompt,
        "chosen": _canonical_json(chosen),
        "rejected": _canonical_json(rejected),
        "subtasks": subtasks,
        "provenance": dict(raw.get("provenance") or {}),
    }


def _normalize_route(raw: Any, subtasks: list[dict[str, Any]], catalog: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("route must be an object")
    model_index = {row["model_id"]: row for row in catalog["models"]}
    assignments = raw.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("route assignments must be a list")
    normalized = []
    seen: set[str] = set()
    expected = {row["subtask_id"]: row for row in subtasks}
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ValueError("route assignment must be an object")
        subtask_id = _required_text(item.get("subtask_id"), "route subtask_id")
        model_id = _required_text(item.get("model_id") or item.get("model"), f"route {subtask_id} model_id")
        reasoning_effort = _required_text(item.get("reasoning_effort"), f"route {subtask_id} reasoning_effort")
        if subtask_id in seen or subtask_id not in expected:
            raise ValueError(f"route has duplicate or unknown subtask_id: {subtask_id}")
        model = model_index.get(model_id)
        if model is None or not model["enabled"]:
            raise ValueError(f"route selects unavailable model: {model_id}")
        if reasoning_effort not in model["supported_reasoning_efforts"]:
            raise ValueError(f"model {model_id} does not support reasoning_effort={reasoning_effort}")
        _assert_model_constraints(expected[subtask_id], model)
        seen.add(subtask_id)
        normalized.append(
            {
                "subtask_id": subtask_id,
                "model_id": model_id,
                "reasoning_effort": reasoning_effort,
                "reason": str(item.get("reason") or ""),
            }
        )
    if seen != set(expected):
        raise ValueError(f"route must assign every subtask; missing={sorted(set(expected) - seen)}")
    fallback = str(raw.get("fallback_model_id") or raw.get("fallback_model") or "") or None
    if fallback is not None and (fallback not in model_index or not model_index[fallback]["enabled"]):
        raise ValueError(f"route fallback model is unavailable: {fallback}")
    return {
        "assignments": sorted(normalized, key=lambda row: row["subtask_id"]),
        "fallback_model_id": fallback,
        "requires_verification": _required_bool(raw.get("requires_verification", True), "route requires_verification"),
    }


def _routing_prompt(task: dict[str, Any], catalog: dict[str, Any]) -> str:
    request = {
        "schema": "dataevol.codex_task_model_route_request.v1",
        "catalog_hash": catalog["catalog_hash"],
        "task": task["task"],
        "subtasks": task["subtasks"],
        "available_models": [
            {
                key: model[key]
                for key in ("model_id", "provider", "availability", "capabilities", "supported_reasoning_efforts", "default_reasoning_effort", "context_window", "pricing", "latency_p50_ms", "risk_tiers")
            }
            for model in catalog["models"]
            if model["enabled"]
        ],
    }
    return (
        "Assign exactly one available model and reasoning_effort to every subtask. Respect capability, context, dependency, risk, and supported-effort constraints. "
        "Minimize reasoning effort and token use when verified quality can be preserved. Return strict JSON with assignments, fallback_model_id, and requires_verification.\n\n"
        + _canonical_json(request)
    )


def _route_decision(route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "assignments": [
            {
                "subtask_id": row["subtask_id"],
                "model_id": row["model_id"],
                "reasoning_effort": row["reasoning_effort"],
            }
            for row in route.get("assignments") or []
        ],
        "fallback_model_id": route.get("fallback_model_id"),
        "requires_verification": bool(route.get("requires_verification")),
    }


def _assert_model_constraints(subtask: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    missing = set(subtask.get("required_capabilities") or []) - set(model.get("capabilities") or [])
    if missing:
        raise ValueError(f"model {model['model_id']} lacks capabilities for {subtask['subtask_id']}: {sorted(missing)}")
    if int(subtask.get("estimated_input_tokens") or 0) > int(model.get("context_window") or 0):
        raise ValueError(f"model {model['model_id']} context window is too small for {subtask['subtask_id']}")
    risk_tiers = set(model.get("risk_tiers") or [])
    if risk_tiers and str(subtask.get("risk")) not in risk_tiers:
        raise ValueError(f"model {model['model_id']} is not enabled for {subtask['risk']} risk")


def _route_constraint_counts(route: Mapping[str, Any], subtasks: list[dict[str, Any]], catalog: dict[str, Any]) -> tuple[int, int]:
    model_index = {row["model_id"]: row for row in catalog["models"]}
    subtask_index = {row["subtask_id"]: row for row in subtasks}
    satisfied = 0
    for assignment in route.get("assignments") or []:
        try:
            _assert_model_constraints(subtask_index[assignment["subtask_id"]], model_index[assignment["model_id"]])
            satisfied += 1
        except (KeyError, ValueError):
            pass
    return len(subtasks), satisfied


def _task_row_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        group = _required_text(item.get("task_group"), "task row task_group")
        if group in result and result[group] != item:
            raise ValueError(f"conflicting task rows for task_group: {group}")
        result[group] = item
    return result


def _feedback_task(raw: Mapping[str, Any], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    group = str(raw.get("task_group") or "")
    if group and group in tasks:
        return tasks[group]
    task_id = str(raw.get("task_id") or "")
    matches = [row for row in tasks.values() if str(row.get("task_id")) == task_id]
    if len(matches) != 1:
        raise ValueError(f"feedback task does not resolve uniquely: {task_id or group}")
    return matches[0]


def _partition(rows: list[dict[str, Any]], eval_groups: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if str(row["task_group"]) not in eval_groups]
    evaluate = [row for row in rows if str(row["task_group"]) in eval_groups]
    return train, evaluate


def _unique_by(rows: Iterable[Mapping[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        key = _required_text(item.get(field), f"{label} {field}")
        if key in result:
            raise ValueError(f"duplicate {label} {field}: {key}")
        result[key] = item
    if not result:
        raise ValueError(f"at least one {label} row is required")
    return result


def _assert_acyclic(subtasks: list[dict[str, Any]]) -> None:
    dependencies = {row["subtask_id"]: set(row["depends_on"]) for row in subtasks}
    ready = [key for key, value in dependencies.items() if not value]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for key, values in dependencies.items():
            if current in values:
                values.remove(current)
                if not values:
                    ready.append(key)
    if visited != len(subtasks):
        raise ValueError("task decomposition dependencies must be acyclic")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _required_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _positive_number(value: Any, field: str) -> float:
    result = _number(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _non_negative_number(value: Any, field: str) -> float:
    result = _number(value, field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _bounded_number(value: Any, field: str) -> float:
    result = _number(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def _normalize_price_schedule(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} pricing schedule must be a non-empty object or null")
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = _required_text(str(raw_key), f"{field} schedule key")
        if isinstance(raw_value, Mapping):
            result[key] = _normalize_price_schedule(raw_value, f"{field}.{key}")
        elif raw_value is None:
            result[key] = None
        else:
            result[key] = _non_negative_number(raw_value, f"{field}.{key}")
    return result


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("catalog captured_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("catalog captured_at must include a timezone")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")


def _json_bytes(value: Mapping[str, Any], *, pretty: bool) -> bytes:
    if pretty:
        return (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return (_canonical_json(value) + "\n").encode("utf-8")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash_object(value: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CATALOG_SCHEMA",
    "DATASET_SCHEMA",
    "DPO_SCHEMA",
    "EVALUATION_SCHEMA",
    "FEEDBACK_SCHEMA",
    "SFT_SCHEMA",
    "CodexRoutingDatasetResult",
    "build_codex_routing_datasets",
    "build_preference_rows",
    "build_task_decomposition_rows",
    "evaluate_routing_predictions",
    "freeze_model_catalog",
    "normalize_outcome_feedback",
    "preferences_from_outcomes",
]
