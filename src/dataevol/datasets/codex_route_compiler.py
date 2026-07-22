from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .codex_routing import (
    REASONING_EFFORT_LEVEL,
    _assert_model_constraints,
    _hash_object,
    _load_catalog,
    _normalize_task,
)


POLICY_SCHEMA = "dataevol.codex_route_policy.v2"
CANDIDATE_SET_SCHEMA = "dataevol.codex_route_candidate_set.v2"
RANKING_SCHEMA = "dataevol.codex_route_ranking.v2"
EXECUTABLE_ROUTE_SCHEMA = "dataevol.codex_executable_route.v2"
RANKING_SFT_SCHEMA = "dataevol.codex_route_ranking_sft.v7"
RANKING_DPO_SCHEMA = "dataevol.codex_route_ranking_preference.v7"
RANKING_DATASET_SCHEMA = "dataevol.codex_route_ranking_datasets.v7"

VERIFICATION_LEVEL = {"none": 0, "standard": 1, "strong": 2, "independent": 3}
BOUNDARY_STATES = frozenset({"trusted", "conditional", "exploration-only", "unmapped"})


@dataclass(frozen=True)
class CodexRankingDatasetResult:
    manifest_path: Path
    sft_train_path: Path
    sft_eval_path: Path
    dpo_train_path: Path
    dpo_eval_path: Path
    dataset_content_hash: str


def freeze_route_policy(policy: Mapping[str, Any], output_path: str | Path | None = None) -> dict[str, Any]:
    normalized = _normalize_policy(policy)
    if output_path is None:
        return normalized
    path = Path(output_path)
    payload = _json_bytes(normalized, pretty=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"route policy already exists with different content: {path}")
        return normalized
    _atomic_write(path, payload)
    return normalized


def build_route_candidate_set(
    task: Mapping[str, Any],
    catalog_snapshot: Mapping[str, Any] | str | Path,
    policy: Mapping[str, Any],
    *,
    boundary_states: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    catalog = _load_catalog(catalog_snapshot)
    route_policy = _normalize_policy(policy)
    normalized_task = _normalize_task(task, index=1)
    boundary = {str(key): str(value) for key, value in (boundary_states or {}).items()}
    unknown_states = sorted(set(boundary.values()) - BOUNDARY_STATES)
    if unknown_states:
        raise ValueError(f"unsupported capability boundary states: {unknown_states}")
    strength = {model_id: index for index, model_id in enumerate(route_policy["model_strength_order"])}
    missing_strength = sorted({model["model_id"] for model in catalog["models"] if model["enabled"]} - set(strength))
    if missing_strength:
        raise ValueError(f"route policy strength order is missing enabled models: {missing_strength}")
    subtasks = []
    for subtask in normalized_task["subtasks"]:
        candidates = []
        rejection_reasons: dict[str, str] = {}
        for model in catalog["models"]:
            if not model["enabled"]:
                continue
            model_id = model["model_id"]
            try:
                _assert_model_constraints(subtask, model)
            except ValueError as exc:
                rejection_reasons[model_id] = str(exc)
                continue
            state_key = f"{model_id}@{model.get('revision') or 'unversioned'}:{_capability_cell(subtask)}"
            state = boundary.get(state_key, "unmapped")
            if subtask["risk"] in route_policy["trusted_boundary_required_for"] and state != "trusted":
                rejection_reasons[model_id] = f"boundary state {state} is not trusted"
                continue
            for effort in model["supported_reasoning_efforts"]:
                option_id = _option_id(catalog["catalog_hash"], model_id, effort)
                candidates.append(
                    {
                        "option_id": option_id,
                        "model_id": model_id,
                        "model_revision": model.get("revision"),
                        "reasoning_effort": effort,
                        "boundary_state": state,
                        "strength_rank": strength[model_id],
                        "estimated_unit_cost": _estimated_unit_cost(model, subtask),
                    }
                )
        candidates.sort(key=lambda row: (row["estimated_unit_cost"], row["strength_rank"], REASONING_EFFORT_LEVEL[row["reasoning_effort"]], row["option_id"]))
        if not candidates:
            raise ValueError(f"no eligible route options for subtask {subtask['subtask_id']}: {rejection_reasons}")
        subtasks.append(
            {
                **subtask,
                "capability_cell": _capability_cell(subtask),
                "minimum_verification_tier": route_policy["risk_verification_tiers"][subtask["risk"]],
                "candidates": candidates,
                "rejected_models": rejection_reasons,
            }
        )
    body = {
        "schema": CANDIDATE_SET_SCHEMA,
        "task_id": normalized_task["task_id"],
        "task_group": normalized_task["task_group"],
        "task": normalized_task["task"],
        "catalog_hash": catalog["catalog_hash"],
        "policy_hash": route_policy["policy_hash"],
        "subtasks": subtasks,
    }
    return {**body, "candidate_set_hash": _hash_object(body)}


def build_ranking_rows(
    tasks: Iterable[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any] | str | Path,
    policy: Mapping[str, Any],
    *,
    boundary_states: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog = _load_catalog(catalog_snapshot)
    route_policy = _normalize_policy(policy)
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    for task_index, raw in enumerate(tasks, start=1):
        normalized_task = _normalize_task(raw, index=task_index)
        route = raw.get("route") or raw.get("routing")
        if not isinstance(route, Mapping) or not isinstance(route.get("assignments"), list):
            raise ValueError(f"task {normalized_task['task_id']} requires teacher route assignments")
        teacher = {str(row["subtask_id"]): dict(row) for row in route["assignments"] if isinstance(row, Mapping)}
        candidate_set = build_route_candidate_set(raw, catalog, route_policy, boundary_states=boundary_states)
        for subtask in candidate_set["subtasks"]:
            choices = [dict(option, choice_id=f"C{choice_index:02d}") for choice_index, option in enumerate(subtask["candidates"])]
            subtask_id = subtask["subtask_id"]
            assignment = teacher.get(subtask_id)
            if assignment is None:
                raise ValueError(f"teacher route is missing subtask {subtask_id}")
            chosen = next(
                (
                    option for option in choices
                    if option["model_id"] == assignment.get("model_id")
                    and option["reasoning_effort"] == assignment.get("reasoning_effort")
                ),
                None,
            )
            if chosen is None:
                raise ValueError(f"teacher route selects an ineligible option for {subtask_id}")
            alternatives = [option for option in choices if option["option_id"] != chosen["option_id"]]
            if not alternatives:
                raise ValueError(f"ranking task {subtask_id} requires at least two eligible options")
            rejected = max(
                alternatives,
                key=lambda option: (
                    abs(option["strength_rank"] - chosen["strength_rank"]),
                    abs(REASONING_EFFORT_LEVEL[option["reasoning_effort"]] - REASONING_EFFORT_LEVEL[chosen["reasoning_effort"]]),
                    option["option_id"],
                ),
            )
            prompt = _ranking_prompt(candidate_set, subtask)
            identity = {
                "catalog_hash": catalog["catalog_hash"],
                "policy_hash": route_policy["policy_hash"],
                "task_group": normalized_task["task_group"],
                "subtask_id": subtask_id,
                "prompt": prompt,
                "chosen": chosen["option_id"],
            }
            row_id = f"codex_rank_{_hash_object(identity)[:24]}"
            base = {
                "task_id": normalized_task["task_id"],
                "task_group": normalized_task["task_group"],
                "subtask_id": subtask_id,
                "catalog_hash": catalog["catalog_hash"],
                "policy_hash": route_policy["policy_hash"],
                "candidate_set_hash": candidate_set["candidate_set_hash"],
                "features": _classifier_features(candidate_set, subtask),
                "prompt": prompt,
                "chosen_option_id": chosen["option_id"],
                "chosen_choice_id": chosen["choice_id"],
                "eligible_option_ids": [option["option_id"] for option in choices],
                "eligible_choice_ids": [option["choice_id"] for option in choices],
                "eligible_options": [
                    {
                        "choice_id": option["choice_id"],
                        "option_id": option["option_id"],
                        "model_id": option["model_id"],
                        "model_revision": option["model_revision"],
                        "reasoning_effort": option["reasoning_effort"],
                        "boundary_state": option["boundary_state"],
                        "estimated_unit_cost": option["estimated_unit_cost"],
                    }
                    for option in choices
                ],
            }
            sft_rows.append({"schema": RANKING_SFT_SCHEMA, "id": row_id, **base, "completion": chosen["choice_id"]})
            dpo_rows.append(
                {
                    "schema": RANKING_DPO_SCHEMA,
                    "pair_id": f"{row_id}_preference",
                    **base,
                    "chosen": chosen["choice_id"],
                    "rejected": rejected["choice_id"],
                }
            )
    return sorted(sft_rows, key=lambda row: row["id"]), sorted(dpo_rows, key=lambda row: row["pair_id"])


def build_ranking_datasets(
    tasks: Iterable[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any] | str | Path,
    policy: Mapping[str, Any],
    output_dir: str | Path,
    *,
    seed: int = 1701,
    eval_fraction: float = 0.2,
    boundary_states: Mapping[str, str] | None = None,
) -> CodexRankingDatasetResult:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    catalog = _load_catalog(catalog_snapshot)
    route_policy = _normalize_policy(policy)
    sft, dpo = build_ranking_rows(tasks, catalog, route_policy, boundary_states=boundary_states)
    groups = sorted({row["task_group"] for row in sft})
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    eval_count = max(1, min(len(groups) - 1, round(len(groups) * eval_fraction)))
    eval_groups = set(ranked[:eval_count])
    values = {
        "sft_train": [row for row in sft if row["task_group"] not in eval_groups],
        "sft_eval": [row for row in sft if row["task_group"] in eval_groups],
        "dpo_train": [row for row in dpo if row["task_group"] not in eval_groups],
        "dpo_eval": [row for row in dpo if row["task_group"] in eval_groups],
    }
    if not all(values.values()):
        raise ValueError("ranking dataset requires non-empty group-disjoint train and eval splits")
    output = Path(output_dir)
    paths = {name: output / f"codex_route_ranking_{name}.jsonl" for name in values}
    payloads = {name: _jsonl_bytes(rows) for name, rows in values.items()}
    files = {name: {"path": str(paths[name]), "rows": len(values[name]), "sha256": hashlib.sha256(payload).hexdigest()} for name, payload in payloads.items()}
    identity = {
        "schema": RANKING_DATASET_SCHEMA,
        "catalog_hash": catalog["catalog_hash"],
        "policy_hash": route_policy["policy_hash"],
        "seed": seed,
        "eval_fraction": eval_fraction,
        "eval_task_groups": sorted(eval_groups),
        "files": {name: {"rows": item["rows"], "sha256": item["sha256"]} for name, item in files.items()},
    }
    content_hash = _hash_object(identity)
    manifest = {**identity, "dataset_content_hash": content_hash, "files": files, "split_strategy": "deterministic_task_group"}
    manifest_path = output / "codex_route_ranking_datasets.manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"ranking dataset manifest already exists with different content: {manifest_path}")
        for name, path in paths.items():
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != files[name]["sha256"]:
                raise ValueError(f"ranking dataset artifact failed integrity check: {name}")
    else:
        if any(path.exists() for path in paths.values()):
            raise ValueError("ranking dataset files exist without their manifest")
        for name, path in paths.items():
            _atomic_write(path, payloads[name])
        _atomic_write(manifest_path, _json_bytes(manifest, pretty=True))
    return CodexRankingDatasetResult(manifest_path, paths["sft_train"], paths["sft_eval"], paths["dpo_train"], paths["dpo_eval"], content_hash)


def compile_executable_route(
    candidate_set: Mapping[str, Any],
    rankings: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    route_policy = _normalize_policy(policy)
    candidates = _validate_candidate_set(candidate_set, route_policy)
    ranking_index: dict[str, dict[str, Any]] = {}
    for raw in rankings:
        row = dict(raw)
        if set(row) - {"schema", "subtask_id", "ranked_option_ids", "confidence", "requested_verification_tier"}:
            raise ValueError(f"ranking contains unknown fields: {sorted(set(row) - {'schema', 'subtask_id', 'ranked_option_ids', 'confidence', 'requested_verification_tier'})}")
        if row.get("schema") != RANKING_SCHEMA:
            raise ValueError("ranking schema mismatch")
        subtask_id = _required_text(row.get("subtask_id"), "ranking subtask_id")
        if subtask_id in ranking_index:
            raise ValueError(f"duplicate ranking for subtask {subtask_id}")
        option_ids = row.get("ranked_option_ids")
        if not isinstance(option_ids, list) or not option_ids or len(option_ids) != len(set(option_ids)):
            raise ValueError(f"ranking {subtask_id} requires unique ranked_option_ids")
        confidence = _bounded(row.get("confidence"), f"ranking {subtask_id} confidence")
        requested = str(row.get("requested_verification_tier") or "none")
        if requested not in VERIFICATION_LEVEL:
            raise ValueError(f"ranking {subtask_id} has invalid requested verification tier")
        ranking_index[subtask_id] = {**row, "ranked_option_ids": [str(value) for value in option_ids], "confidence": confidence, "requested_verification_tier": requested}
    expected_ids = {row["subtask_id"] for row in candidates["subtasks"]}
    if set(ranking_index) != expected_ids:
        raise ValueError(f"rankings must exactly cover subtasks missing={sorted(expected_ids - set(ranking_index))} extra={sorted(set(ranking_index) - expected_ids)}")
    assignments = []
    route_requires_verifier = False
    for subtask in candidates["subtasks"]:
        ranking = ranking_index[subtask["subtask_id"]]
        option_index = {option["option_id"]: option for option in subtask["candidates"]}
        unknown = [option_id for option_id in ranking["ranked_option_ids"] if option_id not in option_index]
        if unknown:
            raise ValueError(f"ranking selects ineligible options for {subtask['subtask_id']}: {unknown}")
        if set(ranking["ranked_option_ids"]) != set(option_index):
            raise ValueError(f"ranking must score every eligible option for {subtask['subtask_id']}")
        selected = option_index[ranking["ranked_option_ids"][0]]
        fallback = _conservative_fallback(subtask["candidates"], selected, route_policy)
        minimum = subtask["minimum_verification_tier"]
        requested = ranking["requested_verification_tier"]
        tier = max((minimum, requested), key=lambda value: VERIFICATION_LEVEL[value])
        triggers = []
        if ranking["confidence"] < route_policy["semantic_verifier_confidence_threshold"]:
            triggers.append("low-confidence")
        if selected["boundary_state"] != "trusted":
            triggers.append(f"boundary-{selected['boundary_state']}")
        if subtask["risk"] in route_policy["semantic_verifier_risk_tiers"]:
            triggers.append(f"risk-{subtask['risk']}")
        if triggers:
            route_requires_verifier = True
        assignments.append(
            {
                "subtask_id": subtask["subtask_id"],
                "option_id": selected["option_id"],
                "model_id": selected["model_id"],
                "model_revision": selected["model_revision"],
                "reasoning_effort": selected["reasoning_effort"],
                "fallback_option_id": fallback["option_id"],
                "fallback_model_id": fallback["model_id"],
                "fallback_model_revision": fallback["model_revision"],
                "fallback_reasoning_effort": fallback["reasoning_effort"],
                "verification_tier": tier,
                "confidence": ranking["confidence"],
                "verifier_triggers": sorted(set(triggers)),
                "depends_on": list(subtask["depends_on"]),
            }
        )
    body = {
        "schema": EXECUTABLE_ROUTE_SCHEMA,
        "task_id": candidates["task_id"],
        "candidate_set_hash": candidates["candidate_set_hash"],
        "catalog_hash": candidates["catalog_hash"],
        "policy_hash": candidates["policy_hash"],
        "assignments": sorted(assignments, key=lambda row: row["subtask_id"]),
        "requires_semantic_verifier": route_requires_verifier,
    }
    route = {**body, "route_hash": _hash_object(body)}
    validate_executable_route(route, candidates, route_policy)
    return route


def validate_executable_route(route: Mapping[str, Any], candidate_set: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    route_policy = _normalize_policy(policy)
    candidates = _validate_candidate_set(candidate_set, route_policy)
    required_top = {"schema", "task_id", "candidate_set_hash", "catalog_hash", "policy_hash", "assignments", "requires_semantic_verifier", "route_hash"}
    if set(route) != required_top:
        raise ValueError(f"executable route fields mismatch missing={sorted(required_top - set(route))} unknown={sorted(set(route) - required_top)}")
    if route.get("schema") != EXECUTABLE_ROUTE_SCHEMA or route.get("task_id") != candidates["task_id"]:
        raise ValueError("executable route identity mismatch")
    for field in ("candidate_set_hash", "catalog_hash", "policy_hash"):
        if route.get(field) != candidates[field]:
            raise ValueError(f"executable route {field} mismatch")
    assignments = route.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("executable route assignments must be a list")
    expected = {row["subtask_id"]: row for row in candidates["subtasks"]}
    seen = set()
    assignment_fields = {
        "subtask_id", "option_id", "model_id", "model_revision", "reasoning_effort",
        "fallback_option_id", "fallback_model_id", "fallback_model_revision", "fallback_reasoning_effort",
        "verification_tier", "confidence", "verifier_triggers", "depends_on",
    }
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or set(assignment) != assignment_fields:
            raise ValueError("executable assignment fields mismatch")
        subtask_id = _required_text(assignment.get("subtask_id"), "assignment subtask_id")
        if subtask_id in seen or subtask_id not in expected:
            raise ValueError(f"duplicate or unknown executable subtask {subtask_id}")
        seen.add(subtask_id)
        option_index = {option["option_id"]: option for option in expected[subtask_id]["candidates"]}
        selected = option_index.get(assignment.get("option_id"))
        fallback = option_index.get(assignment.get("fallback_option_id"))
        if selected is None or fallback is None or selected["option_id"] == fallback["option_id"]:
            raise ValueError(f"selected and fallback options must be distinct and eligible for {subtask_id}")
        for prefix, option in (("", selected), ("fallback_", fallback)):
            if assignment.get(f"{prefix}model_id") != option["model_id"] or assignment.get(f"{prefix}model_revision") != option["model_revision"] or assignment.get(f"{prefix}reasoning_effort") != option["reasoning_effort"]:
                raise ValueError(f"compiled option expansion mismatch for {subtask_id}")
        tier = str(assignment.get("verification_tier"))
        if tier not in VERIFICATION_LEVEL or VERIFICATION_LEVEL[tier] < VERIFICATION_LEVEL[expected[subtask_id]["minimum_verification_tier"]]:
            raise ValueError(f"verification tier is below policy minimum for {subtask_id}")
        _bounded(assignment.get("confidence"), f"assignment {subtask_id} confidence")
        if assignment.get("depends_on") != expected[subtask_id]["depends_on"]:
            raise ValueError(f"compiled dependencies changed for {subtask_id}")
    if seen != set(expected):
        raise ValueError(f"executable route must cover every subtask missing={sorted(set(expected) - seen)}")
    body = {key: route[key] for key in required_top if key != "route_hash"}
    if route.get("route_hash") != _hash_object(body):
        raise ValueError("executable route hash mismatch")


def _normalize_policy(raw: Mapping[str, Any]) -> dict[str, Any]:
    policy_id = _required_text(raw.get("policy_id"), "policy_id")
    policy_version = _required_text(raw.get("policy_version"), "policy_version")
    strength = [_required_text(value, "model_strength_order") for value in raw.get("model_strength_order") or []]
    if not strength or len(strength) != len(set(strength)):
        raise ValueError("model_strength_order must contain unique model IDs")
    tiers = dict(raw.get("risk_verification_tiers") or {})
    expected_risks = {"low", "medium", "high", "critical"}
    if set(tiers) != expected_risks or any(value not in VERIFICATION_LEVEL for value in tiers.values()):
        raise ValueError("risk_verification_tiers must define low, medium, high, and critical")
    semantic_risks = sorted({_required_text(value, "semantic_verifier_risk_tiers") for value in raw.get("semantic_verifier_risk_tiers") or []})
    if set(semantic_risks) - expected_risks:
        raise ValueError("semantic_verifier_risk_tiers contains an unsupported risk")
    trusted_required = sorted({_required_text(value, "trusted_boundary_required_for") for value in raw.get("trusted_boundary_required_for") or []})
    if set(trusted_required) - expected_risks:
        raise ValueError("trusted_boundary_required_for contains an unsupported risk")
    threshold = _bounded(raw.get("semantic_verifier_confidence_threshold", 0.65), "semantic_verifier_confidence_threshold")
    body = {
        "schema": POLICY_SCHEMA,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "model_strength_order": strength,
        "risk_verification_tiers": {risk: str(tiers[risk]) for risk in sorted(tiers)},
        "semantic_verifier_risk_tiers": semantic_risks,
        "semantic_verifier_confidence_threshold": threshold,
        "trusted_boundary_required_for": trusted_required,
    }
    return {**body, "policy_hash": _hash_object(body)}


def _validate_candidate_set(raw: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("schema") != CANDIDATE_SET_SCHEMA or raw.get("policy_hash") != policy["policy_hash"]:
        raise ValueError("candidate set schema or policy hash mismatch")
    body = {key: raw[key] for key in raw if key != "candidate_set_hash"}
    if raw.get("candidate_set_hash") != _hash_object(body):
        raise ValueError("candidate set hash mismatch")
    return dict(raw)


def _ranking_prompt(candidate_set: Mapping[str, Any], subtask: Mapping[str, Any]) -> str:
    effort_code = {"none": "N", "minimal": "I", "low": "L", "medium": "M", "high": "H", "xhigh": "X"}
    boundary_code = {"trusted": "T", "conditional": "C", "exploration-only": "E", "unmapped": "U"}
    model_aliases = {option["model_id"]: f"M{option['strength_rank']}" for option in subtask["candidates"]}
    header = [
        "ROUTE_CHOICE_V5",
        f"obj={subtask['objective']}",
        f"risk={subtask['risk'][0].upper()} tok={subtask['estimated_input_tokens']} cap={'+'.join(subtask['required_capabilities']) or 'general'} dep={'+'.join(subtask['depends_on']) or '-'}",
        " ".join(f"{alias}={model_id}" for model_id, alias in sorted(model_aliases.items(), key=lambda item: item[1])),
    ]
    options = [
        f"C{index:02d}|{model_aliases[option['model_id']]}|{effort_code[option['reasoning_effort']]}|{boundary_code[option['boundary_state']]}|{option['estimated_unit_cost']:.6g}"
        for index, option in enumerate(subtask["candidates"])
    ]
    return "\n".join([*header, *options, "Return C## only."])


def _classifier_features(candidate_set: Mapping[str, Any], subtask: Mapping[str, Any]) -> dict[str, Any]:
    searchable = f"{candidate_set['task']} {subtask['objective']}".lower()
    signals = {
        signal: signal in searchable
        for signal in ("security", "migration", "research", "architecture", "incident", "verification")
    }
    return {
        "task_text": candidate_set["task"],
        "objective": subtask["objective"],
        "risk": subtask["risk"],
        "required_capabilities": list(subtask["required_capabilities"]),
        "estimated_input_tokens": subtask["estimated_input_tokens"],
        "dependency_count": len(subtask["depends_on"]),
        "subtask_count": len(candidate_set["subtasks"]),
        "verification_floor": subtask["minimum_verification_tier"],
        "boundary_states": sorted({option["boundary_state"] for option in subtask["candidates"]}),
        "eligible_option_count": len(subtask["candidates"]),
        "signals": signals,
    }


def _conservative_fallback(candidates: list[dict[str, Any]], selected: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    alternatives = [option for option in candidates if option["option_id"] != selected["option_id"]]
    if not alternatives:
        raise ValueError(f"no distinct fallback exists for option {selected['option_id']}")
    return max(alternatives, key=lambda option: (option["strength_rank"], REASONING_EFFORT_LEVEL[option["reasoning_effort"]], -option["estimated_unit_cost"], option["option_id"]))


def _capability_cell(subtask: Mapping[str, Any]) -> str:
    capabilities = "+".join(sorted(subtask.get("required_capabilities") or [])) or "general"
    tokens = int(subtask.get("estimated_input_tokens") or 0)
    token_bucket = "small" if tokens <= 8_000 else "medium" if tokens <= 32_000 else "large"
    return f"{capabilities}:{subtask.get('risk')}:{token_bucket}"


def _estimated_unit_cost(model: Mapping[str, Any], subtask: Mapping[str, Any]) -> float:
    pricing = (model.get("pricing") or {}).get("api_usd_equivalent")
    if not isinstance(pricing, Mapping):
        return 1_000_000_000_000.0
    input_tokens = int(subtask.get("estimated_input_tokens") or 0)
    output_tokens = max(64, min(4_096, input_tokens // 8))
    return round((input_tokens * float(pricing.get("input", 0)) + output_tokens * float(pricing.get("output", 0))) / 1_000_000, 12)


def _option_id(catalog_hash: str, model_id: str, effort: str) -> str:
    return "opt_" + hashlib.sha256(f"{catalog_hash}:{model_id}:{effort}".encode()).hexdigest()[:12]


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} is required")
    return value


def _bounded(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be between 0 and 1") from exc
    if not math.isfinite(result) or result < 0 or result > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()


def _json_bytes(value: Mapping[str, Any], *, pretty: bool) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2 if pretty else None, separators=None if pretty else (",", ":")) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
