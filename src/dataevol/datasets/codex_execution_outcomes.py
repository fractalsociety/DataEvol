from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


OUTCOME_SCHEMA = "dataevol.codex_execution_outcome.v1"
CELL_REPORT_SCHEMA = "dataevol.codex_capability_cell_report.v1"
COUNTERFACTUAL_PLAN_SCHEMA = "dataevol.codex_counterfactual_plan.v1"
CHEAPEST_TARGET_SCHEMA = "dataevol.codex_cheapest_acceptable_target.v1"
LEAKAGE_AUDIT_SCHEMA = "dataevol.codex_classifier_leakage_audit.v1"
ROLLOUT_SCHEMA = "dataevol.codex_classifier_rollout.v1"

RISKS = frozenset({"low", "medium", "high", "critical"})
EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
ARMS = frozenset({"teacher", "classifier", "cheaper-candidate"})
ASSIGNMENTS = frozenset({"randomized", "deterministic-shadow", "observational"})
SERIOUS_SEVERITIES = frozenset({"serious", "high", "critical"})
TARGET_DERIVED_FEATURE_KEYS = (
    "chosen_option",
    "selected_option",
    "teacher_option",
    "classifier_option",
    "option_hash",
    "choice_id",
    "compact_label",
    "target_label",
    "fallback_model",
    "selected_model",
    "teacher_model",
)


def audit_classifier_dataset(
    train_rows: Iterable[Mapping[str, Any]],
    eval_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    train = [dict(row) for row in train_rows]
    evaluate = [dict(row) for row in eval_rows]
    train_groups = {str(row.get("task_group") or "") for row in train}
    eval_groups = {str(row.get("task_group") or "") for row in evaluate}
    prohibited = []
    embedded_targets = []
    for split, rows in (("train", train), ("eval", evaluate)):
        for row in rows:
            features = row.get("features")
            if not isinstance(features, Mapping):
                prohibited.append({"split": split, "row_id": row.get("id"), "path": "features", "reason": "missing-features"})
                continue
            for path, value in _walk(features):
                lowered = path.lower()
                if any(fragment in lowered for fragment in TARGET_DERIVED_FEATURE_KEYS):
                    prohibited.append({"split": split, "row_id": row.get("id"), "path": path, "reason": "target-derived-key"})
                if isinstance(value, str) and value in {
                    str(row.get("chosen_option_id") or ""),
                    str(row.get("chosen_choice_id") or ""),
                    str(row.get("completion") or ""),
                } and value:
                    embedded_targets.append({"split": split, "row_id": row.get("id"), "path": path})

    exact_train = {_hash(_features(row)) for row in train}
    exact_eval = {_hash(_features(row)) for row in evaluate}
    structural_train = {_hash(_structural_features(row)) for row in train}
    structural_eval = {_hash(_structural_features(row)) for row in evaluate}
    exact_overlap = exact_train & exact_eval
    structural_overlap = structural_train & structural_eval
    group_disjoint = not (train_groups & eval_groups)
    direct_leakage = bool(prohibited or embedded_targets)
    production_generalization_safe = group_disjoint and not direct_leakage and not structural_overlap
    body = {
        "schema": LEAKAGE_AUDIT_SCHEMA,
        "train_rows": len(train),
        "eval_rows": len(evaluate),
        "group_disjoint": group_disjoint,
        "overlapping_task_groups": sorted(train_groups & eval_groups),
        "prohibited_feature_paths": prohibited,
        "embedded_target_values": embedded_targets,
        "exact_feature_overlap_count": len(exact_overlap),
        "structural_feature_overlap_count": len(structural_overlap),
        "safe_for_teacher_imitation": group_disjoint and not direct_leakage,
        "safe_for_production_generalization": production_generalization_safe,
        "findings": [
            *( ["task-group split leakage"] if not group_disjoint else []),
            *( ["target-derived classifier features"] if direct_leakage else []),
            *( ["train/eval share synthetic feature archetypes"] if structural_overlap else []),
        ],
    }
    return {**body, "audit_hash": _hash(body)}


def normalize_execution_outcomes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        outcome_id = _text(_get(row, "outcome_id", "outcomeId"), "outcome_id")
        if outcome_id in seen:
            raise ValueError(f"duplicate outcome_id: {outcome_id}")
        seen.add(outcome_id)
        risk = _text(_get(row, "risk"), "risk")
        effort = _text(_get(row, "reasoning_effort", "reasoningEffort"), "reasoning_effort")
        arm = _text(_get(row, "arm"), "arm")
        assignment = _text(_get(row, "assignment_mechanism", "assignmentMechanism"), "assignment_mechanism")
        if risk not in RISKS or effort not in EFFORTS or arm not in ARMS or assignment not in ASSIGNMENTS:
            raise ValueError(f"invalid routing outcome enum in {outcome_id}")
        capabilities = sorted({_text(value, "required_capabilities") for value in (_get(row, "required_capabilities", "requiredCapabilities") or [])})
        estimated_tokens = _nonnegative_int(_get(row, "estimated_input_tokens", "estimatedInputTokens"), "estimated_input_tokens")
        model_family = _text(_get(row, "model_family", "modelFamily"), "model_family")
        cell_id = capability_cell_id(capabilities, risk, estimated_tokens, model_family, effort)
        supplied_cell = _get(row, "capability_cell_id", "capabilityCellId")
        if supplied_cell is not None and supplied_cell != cell_id:
            raise ValueError(f"capability cell mismatch in {outcome_id}")
        verifier_score = _unit(_get(row, "verifier_score", "verifierScore"), "verifier_score")
        quality_floor = _unit(_get(row, "quality_floor", "qualityFloor"), "quality_floor")
        confidence_raw = _get(row, "classifier_confidence", "classifierConfidence")
        confidence = None if confidence_raw is None else _unit(confidence_raw, "classifier_confidence")
        safety = _violations(_get(row, "safety_violations", "safetyViolations") or [], "safety_violations")
        policy = _violations(_get(row, "policy_violations", "policyViolations") or [], "policy_violations")
        body = {
            "schema": OUTCOME_SCHEMA,
            "outcome_id": outcome_id,
            "experiment_id": _text(_get(row, "experiment_id", "experimentId"), "experiment_id"),
            "counterfactual_group_id": _text(_get(row, "counterfactual_group_id", "counterfactualGroupId"), "counterfactual_group_id"),
            "arm": arm,
            "assignment_mechanism": assignment,
            "task_id": _text(_get(row, "task_id", "taskId"), "task_id"),
            "task_group": _text(_get(row, "task_group", "taskGroup"), "task_group"),
            "subtask_id": _text(_get(row, "subtask_id", "subtaskId"), "subtask_id"),
            "subtask_hash": _hash_text(_get(row, "subtask_hash", "subtaskHash"), "subtask_hash"),
            "plan_hash": _hash_text(_get(row, "plan_hash", "planHash"), "plan_hash"),
            "decision_hash": _hash_text(_get(row, "decision_hash", "decisionHash"), "decision_hash"),
            "usage_receipt_hash": _hash_text(_get(row, "usage_receipt_hash", "usageReceiptHash"), "usage_receipt_hash"),
            "catalog_hash": _hash_text(_get(row, "catalog_hash", "catalogHash"), "catalog_hash"),
            "pricing_hash": _hash_text(_get(row, "pricing_hash", "pricingHash"), "pricing_hash"),
            "policy_hash": _hash_text(_get(row, "policy_hash", "policyHash"), "policy_hash"),
            "candidate_set_hash": _hash_text(_get(row, "candidate_set_hash", "candidateSetHash"), "candidate_set_hash"),
            "source_evidence_hash": _hash_text(_get(row, "source_evidence_hash", "evidenceHash"), "source_evidence_hash"),
            "required_capabilities": capabilities,
            "risk": risk,
            "estimated_input_tokens": estimated_tokens,
            "model_family": model_family,
            "reasoning_effort": effort,
            "capability_cell_id": cell_id,
            "teacher_option_id": _text(_get(row, "teacher_option_id", "teacherOptionId"), "teacher_option_id"),
            "classifier_option_id": _optional_text(_get(row, "classifier_option_id", "classifierOptionId")),
            "classifier_confidence": confidence,
            "executed_option_id": _text(_get(row, "executed_option_id", "executedOptionId"), "executed_option_id"),
            "model_id": _text(_get(row, "model_id", "modelId"), "model_id"),
            "model_revision": _hash_text(_get(row, "model_revision", "modelRevision"), "model_revision"),
            "verified": _boolean(_get(row, "verified"), "verified"),
            "independent_verifier": _boolean(_get(row, "independent_verifier", "independentVerifier"), "independent_verifier"),
            "success": _boolean(_get(row, "success"), "success"),
            "verifier_score": verifier_score,
            "quality_floor": quality_floor,
            "cost_amount": _nonnegative_number(_get(row, "cost_amount", "costAmount"), "cost_amount"),
            "cost_unit": _text(_get(row, "cost_unit", "costUnit"), "cost_unit"),
            "latency_ms": _nonnegative_number(_get(row, "latency_ms", "latencyMs"), "latency_ms"),
            "retries": _nonnegative_int(_get(row, "retries"), "retries"),
            "tool_failures": sorted({_text(value, "tool_failures") for value in (_get(row, "tool_failures", "toolFailures") or [])}),
            "safety_violations": safety,
            "policy_violations": policy,
            "cheaper_option_tested": _boolean(_get(row, "cheaper_option_tested", "cheaperOptionTested"), "cheaper_option_tested"),
            "completed_at": _nonnegative_int(_get(row, "completed_at", "completedAt"), "completed_at"),
        }
        normalized.append({**body, "acceptable": outcome_is_acceptable(body), "record_hash": _hash(body)})
    return sorted(normalized, key=lambda item: (item["completed_at"], item["outcome_id"]))


def capability_cell_id(
    capabilities: Iterable[str], risk: str, estimated_input_tokens: int, model_family: str, effort: str
) -> str:
    capability = "+".join(sorted(set(capabilities))) or "general"
    context = "lt8k" if estimated_input_tokens < 8_000 else "8k-32k" if estimated_input_tokens <= 32_000 else "gt32k"
    descriptor = f"{capability}|{risk}|{context}|{model_family}|{effort}"
    return f"cell_{hashlib.sha256(descriptor.encode()).hexdigest()[:20]}:{descriptor}"


def outcome_is_acceptable(outcome: Mapping[str, Any]) -> bool:
    serious = any(item.get("severity") in SERIOUS_SEVERITIES for key in ("safety_violations", "policy_violations") for item in outcome.get(key, []))
    return bool(
        outcome.get("verified")
        and outcome.get("independent_verifier")
        and outcome.get("success")
        and float(outcome.get("verifier_score", 0)) >= float(outcome.get("quality_floor", 1))
        and not serious
    )


def build_capability_cell_report(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    minimum_verified: int = 100,
    minimum_precision: float = 0.97,
    minimum_wilson_lower_bound: float = 0.95,
    maximum_brier_score: float = 0.10,
    maximum_recent_regression: float = 0.02,
) -> dict[str, Any]:
    rows = [dict(row) for row in outcomes]
    for row in rows:
        if row.get("schema") != OUTCOME_SCHEMA:
            raise ValueError("capability reports require normalized execution outcomes")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["capability_cell_id"])].append(row)
    cells = []
    for cell_id, cell_rows in sorted(grouped.items()):
        independent = _independent_rows(cell_rows)
        verified = [row for row in independent if row["verified"] and row["independent_verifier"]]
        accepted = sum(bool(row["acceptable"]) for row in verified)
        precision = accepted / len(verified) if verified else 0.0
        wilson = _wilson_lower(accepted, len(verified))
        serious = sum(
            item["severity"] in SERIOUS_SEVERITIES
            for row in independent for key in ("safety_violations", "policy_violations") for item in row[key]
        )
        calibrated = [row for row in verified if row["classifier_confidence"] is not None]
        brier = sum((float(row["classifier_confidence"]) - float(row["acceptable"])) ** 2 for row in calibrated) / len(calibrated) if calibrated else None
        recent_regression = _recent_regression(verified)
        risk = str(cell_rows[0]["risk"])
        reasons = []
        if len(verified) < minimum_verified: reasons.append("insufficient-independent-verified-samples")
        if precision < minimum_precision: reasons.append("precision-below-floor")
        if wilson < minimum_wilson_lower_bound: reasons.append("wilson-lower-bound-below-floor")
        if serious: reasons.append("serious-safety-or-policy-failure")
        if not calibrated or brier is None or brier > maximum_brier_score: reasons.append("confidence-not-calibrated")
        if recent_regression > maximum_recent_regression: reasons.append("recent-regression")
        statistically_trusted = not reasons
        state = "trusted" if statistically_trusted and risk in {"low", "medium"} else "conditional" if statistically_trusted else "exploration-only"
        cells.append({
            "capability_cell_id": cell_id,
            "risk": risk,
            "state": state,
            "autonomous_authority": state == "trusted",
            "independent_verified_samples": len(verified),
            "unique_task_groups": len({row["task_group"] for row in verified}),
            "accepted_samples": accepted,
            "selective_precision": round(precision, 6),
            "wilson_lower_bound": round(wilson, 6),
            "brier_score": None if brier is None else round(brier, 6),
            "recent_regression": round(recent_regression, 6),
            "serious_failure_count": serious,
            "requires_independent_verification": risk in {"high", "critical"},
            "reasons": reasons + (["high-risk-cells-remain-conditional"] if statistically_trusted and risk in {"high", "critical"} else []),
        })
    trusted = sum(cell["state"] == "trusted" for cell in cells)
    body = {
        "schema": CELL_REPORT_SCHEMA,
        "thresholds": {
            "minimum_verified": minimum_verified,
            "minimum_precision": minimum_precision,
            "minimum_wilson_lower_bound": minimum_wilson_lower_bound,
            "maximum_brier_score": maximum_brier_score,
            "maximum_recent_regression": maximum_recent_regression,
        },
        "source_outcome_hashes": sorted(row["record_hash"] for row in rows),
        "cells": cells,
        "trusted_cell_count": trusted,
        "recommended_rollout": "canary" if trusted else "shadow",
        "maximum_classifier_coverage": 0.05 if trusted else 0.0,
    }
    return {**body, "report_hash": _hash(body)}


def build_cheapest_acceptable_rows(outcomes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        item = dict(row)
        if item.get("schema") != OUTCOME_SCHEMA:
            raise ValueError("training targets require normalized execution outcomes")
        grouped[str(item["counterfactual_group_id"])].append(item)
    targets = []
    for group_id, rows in sorted(grouped.items()):
        if len(rows) < 2 or {row["assignment_mechanism"] for row in rows} != {"randomized"}:
            continue
        if len({row["executed_option_id"] for row in rows}) < 2:
            continue
        comparison_pins = {
            (row["task_group"], row["subtask_id"], row["catalog_hash"], row["policy_hash"], row["candidate_set_hash"], row["cost_unit"])
            for row in rows
        }
        if len(comparison_pins) != 1 or not all(row["cheaper_option_tested"] for row in rows):
            continue
        acceptable = [row for row in rows if row["acceptable"]]
        if not acceptable:
            continue
        chosen = min(acceptable, key=lambda row: (row["cost_amount"], row["latency_ms"], row["executed_option_id"]))
        body = {
            "schema": CHEAPEST_TARGET_SCHEMA,
            "counterfactual_group_id": group_id,
            "task_group": chosen["task_group"],
            "subtask_id": chosen["subtask_id"],
            "capability_cell_id": chosen["capability_cell_id"],
            "chosen_option_id": chosen["executed_option_id"],
            "observed_options": sorted({row["executed_option_id"] for row in rows}),
            "acceptable_option_ids": sorted({row["executed_option_id"] for row in acceptable}),
            "source_record_hashes": sorted(row["record_hash"] for row in rows),
            "target_basis": "cheapest-independent-verified-acceptable-outcome",
            "causal": True,
        }
        targets.append({**body, "target_hash": _hash(body)})
    return targets


def plan_counterfactual_trials(
    ranking_rows: Iterable[Mapping[str, Any]],
    *,
    experiment_id: str,
    sample_rate: float = 0.05,
    seed: int = 1701,
    maximum_trials: int | None = None,
) -> list[dict[str, Any]]:
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be between 0 and 1")
    plans = []
    for row in ranking_rows:
        features = row.get("features") or {}
        if features.get("risk") != "low":
            continue
        capabilities = set(features.get("required_capabilities") or [])
        signals = features.get("signals") or {}
        if capabilities & {"security", "migration"} or signals.get("security") or signals.get("migration"):
            continue
        chosen_id = str(row.get("chosen_option_id") or "")
        options = [dict(option) for option in row.get("eligible_options") or []]
        teacher = next((option for option in options if option.get("option_id") == chosen_id), None)
        if teacher is None or "estimated_unit_cost" not in teacher:
            continue
        cheaper = [option for option in options if float(option.get("estimated_unit_cost", math.inf)) < float(teacher["estimated_unit_cost"])]
        if not cheaper:
            continue
        sample_key = f"{seed}:{row.get('task_group')}:{row.get('subtask_id')}"
        if int(hashlib.sha256(sample_key.encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF >= sample_rate:
            continue
        candidate = min(cheaper, key=lambda option: (float(option["estimated_unit_cost"]), str(option["option_id"])))
        group_id = "cf_" + hashlib.sha256(f"{experiment_id}:{sample_key}".encode()).hexdigest()[:20]
        body = {
            "schema": COUNTERFACTUAL_PLAN_SCHEMA,
            "experiment_id": _text(experiment_id, "experiment_id"),
            "counterfactual_group_id": group_id,
            "task_id": str(row.get("task_id")),
            "task_group": str(row.get("task_group")),
            "subtask_id": str(row.get("subtask_id")),
            "teacher_option_id": chosen_id,
            "candidate_option_id": str(candidate["option_id"]),
            "teacher_estimated_unit_cost": float(teacher["estimated_unit_cost"]),
            "candidate_estimated_unit_cost": float(candidate["estimated_unit_cost"]),
            "assignment_mechanism": "randomized",
            "execution_mode": "isolated-shadow",
            "production_authority": "teacher",
            "requires_independent_verifier": True,
            "catalog_hash": str(row.get("catalog_hash")),
            "policy_hash": str(row.get("policy_hash")),
            "candidate_set_hash": str(row.get("candidate_set_hash")),
        }
        plans.append({**body, "plan_hash": _hash(body)})
    plans.sort(key=lambda item: item["plan_hash"])
    return plans[:maximum_trials] if maximum_trials is not None else plans


def evaluate_rollout(cell_report: Mapping[str, Any], *, current_stage: str = "shadow", successful_evidence_epochs: int = 0) -> dict[str, Any]:
    stages = {"shadow": 0.0, "canary": 0.05, "limited": 0.25, "expanded": 0.50, "broad": 1.0}
    if current_stage not in stages or successful_evidence_epochs < 0:
        raise ValueError("invalid rollout state")
    trusted = int(cell_report.get("trusted_cell_count") or 0)
    target = "shadow"
    if trusted and successful_evidence_epochs >= 1: target = "canary"
    if trusted and successful_evidence_epochs >= 3: target = "limited"
    if trusted and successful_evidence_epochs >= 6: target = "expanded"
    if trusted and successful_evidence_epochs >= 12: target = "broad"
    body = {
        "schema": ROLLOUT_SCHEMA,
        "current_stage": current_stage,
        "authorized_stage": target,
        "maximum_classifier_coverage": stages[target],
        "trusted_cell_count": trusted,
        "successful_evidence_epochs": successful_evidence_epochs,
        "dataevol_verdict_required": True,
        "fractalwork_canary_required": target != "shadow",
        "authority": "teacher" if target == "shadow" else "fractalwork-canary",
    }
    return {**body, "rollout_hash": _hash(body)}


def _independent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["completed_at"], item["outcome_id"]), reverse=True):
        by_group.setdefault(str(row["task_group"]), row)
    return sorted(by_group.values(), key=lambda item: (item["completed_at"], item["outcome_id"]))


def _recent_regression(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 40:
        return 0.0
    ordered = sorted(rows, key=lambda item: (item["completed_at"], item["outcome_id"]))
    midpoint = len(ordered) // 2
    prior = sum(row["acceptable"] for row in ordered[:midpoint]) / midpoint
    recent = sum(row["acceptable"] for row in ordered[midpoint:]) / (len(ordered) - midpoint)
    return max(0.0, prior - recent)


def _wilson_lower(successes: int, count: int, z: float = 1.96) -> float:
    if count == 0:
        return 0.0
    p = successes / count
    denominator = 1 + z * z / count
    centre = p + z * z / (2 * count)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * count)) / count)
    return (centre - margin) / denominator


def _features(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("features")
    return value if isinstance(value, Mapping) else {}


def _structural_features(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in _features(row).items() if key not in {"task_text", "objective"}}


def _walk(value: Mapping[str, Any], prefix: str = "features"):
    for key, item in value.items():
        path = f"{prefix}.{key}"
        if isinstance(item, Mapping):
            yield from _walk(item, path)
        else:
            yield path, item


def _get(row: Mapping[str, Any], *names: str) -> Any:
    return next((row[name] for name in names if name in row), None)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} is required")
    return value


def _optional_text(value: Any) -> str | None:
    return None if value is None else _text(value, "optional text")


def _hash_text(value: Any, field: str) -> str:
    result = _text(value, field)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _unit(value: Any, field: str) -> float:
    result = _nonnegative_number(value, field)
    if result > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _violations(values: Iterable[Mapping[str, Any]], field: str) -> list[dict[str, str]]:
    result = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} entries must be objects")
        severity = _text(value.get("severity"), f"{field}.severity").lower()
        result.append({"code": _text(value.get("code"), f"{field}.code"), "severity": severity})
    return sorted(result, key=lambda item: (item["severity"], item["code"]))


def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()
