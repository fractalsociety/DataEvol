from __future__ import annotations

from dataevol.datasets.codex_execution_outcomes import (
    OUTCOME_SCHEMA,
    audit_classifier_dataset,
    build_capability_cell_report,
    build_cheapest_acceptable_rows,
    evaluate_rollout,
    normalize_execution_outcomes,
    plan_counterfactual_trials,
)
from dataevol.experiments.codex_outcome_pipeline import run_outcome_pipeline


HASH = "a" * 64


def _raw_outcome(index: int, *, risk: str = "low", arm: str = "classifier", cost: float = 1.0) -> dict:
    return {
        "outcome_id": f"outcome-{index}",
        "experiment_id": "exp-1",
        "counterfactual_group_id": f"pair-{index // 2}",
        "arm": arm,
        "assignment_mechanism": "randomized",
        "task_id": f"task-{index}",
        "task_group": f"group-{index}",
        "subtask_id": "test",
        "subtask_hash": HASH,
        "plan_hash": HASH,
        "decision_hash": HASH,
        "usage_receipt_hash": HASH,
        "catalog_hash": HASH,
        "pricing_hash": HASH,
        "policy_hash": HASH,
        "candidate_set_hash": HASH,
        "source_evidence_hash": HASH,
        "required_capabilities": ["tests"],
        "risk": risk,
        "estimated_input_tokens": 2_000,
        "model_family": "mini",
        "reasoning_effort": "medium",
        "teacher_option_id": "teacher",
        "classifier_option_id": "cheap",
        "classifier_confidence": 0.99,
        "executed_option_id": "cheap" if arm != "teacher" else "teacher",
        "model_id": "gpt-mini" if arm != "teacher" else "gpt-standard",
        "model_revision": HASH,
        "verified": True,
        "independent_verifier": True,
        "success": True,
        "verifier_score": 0.99,
        "quality_floor": 0.9,
        "cost_amount": cost,
        "cost_unit": "usd-micros",
        "latency_ms": 25,
        "retries": 0,
        "tool_failures": [],
        "safety_violations": [],
        "policy_violations": [],
        "cheaper_option_tested": True,
        "completed_at": 1_000 + index,
    }


def test_normalizes_runtime_evidence_and_rejects_cell_mismatch() -> None:
    outcome = normalize_execution_outcomes([_raw_outcome(1)])[0]

    assert outcome["schema"] == OUTCOME_SCHEMA
    assert outcome["acceptable"] is True
    assert outcome["capability_cell_id"].endswith(":tests|low|lt8k|mini|medium")
    assert len(outcome["record_hash"]) == 64

    malformed = _raw_outcome(2)
    malformed["capability_cell_id"] = "wrong"
    try:
        normalize_execution_outcomes([malformed])
    except ValueError as exc:
        assert "capability cell mismatch" in str(exc)
    else:
        raise AssertionError("mismatched capability cell was accepted")


def test_normalizer_accepts_fractalwork_camel_case_wire_contract() -> None:
    raw = _raw_outcome(3)
    camel_names = {
        "outcome_id": "outcomeId", "experiment_id": "experimentId", "counterfactual_group_id": "counterfactualGroupId",
        "assignment_mechanism": "assignmentMechanism", "task_id": "taskId", "task_group": "taskGroup",
        "subtask_id": "subtaskId", "subtask_hash": "subtaskHash", "plan_hash": "planHash",
        "decision_hash": "decisionHash", "usage_receipt_hash": "usageReceiptHash", "catalog_hash": "catalogHash",
        "pricing_hash": "pricingHash", "policy_hash": "policyHash", "candidate_set_hash": "candidateSetHash",
        "source_evidence_hash": "evidenceHash", "required_capabilities": "requiredCapabilities",
        "estimated_input_tokens": "estimatedInputTokens", "model_family": "modelFamily", "reasoning_effort": "reasoningEffort",
        "teacher_option_id": "teacherOptionId", "classifier_option_id": "classifierOptionId",
        "classifier_confidence": "classifierConfidence", "executed_option_id": "executedOptionId", "model_id": "modelId",
        "model_revision": "modelRevision", "independent_verifier": "independentVerifier", "verifier_score": "verifierScore",
        "quality_floor": "qualityFloor", "cost_amount": "costAmount", "cost_unit": "costUnit", "latency_ms": "latencyMs",
        "tool_failures": "toolFailures", "safety_violations": "safetyViolations", "policy_violations": "policyViolations",
        "cheaper_option_tested": "cheaperOptionTested", "completed_at": "completedAt",
    }
    wire = {camel_names.get(key, key): value for key, value in raw.items()}
    wire["schema"] = "codex.execution_evidence.v1"

    normalized = normalize_execution_outcomes([wire])[0]
    assert normalized["outcome_id"] == raw["outcome_id"]
    assert normalized["source_evidence_hash"] == raw["source_evidence_hash"]
    assert normalized["acceptable"] is True


def test_capability_cells_require_independent_samples_calibration_and_no_serious_failures() -> None:
    outcomes = normalize_execution_outcomes([_raw_outcome(index) for index in range(100)])
    report = build_capability_cell_report(outcomes)

    assert report["trusted_cell_count"] == 1
    assert report["recommended_rollout"] == "canary"
    assert report["cells"][0]["state"] == "trusted"
    assert report["cells"][0]["independent_verified_samples"] == 100
    assert report["cells"][0]["selective_precision"] == 1.0

    failed = _raw_outcome(99)
    failed["outcome_id"] = "replacement-with-serious-failure"
    failed["safety_violations"] = [{"code": "credential-leak", "severity": "critical"}]
    blocked = build_capability_cell_report(normalize_execution_outcomes([*[_raw_outcome(index) for index in range(99)], failed]))
    assert blocked["trusted_cell_count"] == 0
    assert "serious-safety-or-policy-failure" in blocked["cells"][0]["reasons"]


def test_high_risk_cells_never_gain_autonomous_authority_and_repeats_are_not_independent() -> None:
    high = normalize_execution_outcomes([_raw_outcome(index, risk="high") for index in range(100)])
    high_report = build_capability_cell_report(high)
    assert high_report["cells"][0]["state"] == "conditional"
    assert high_report["cells"][0]["autonomous_authority"] is False

    repeats = [_raw_outcome(index) for index in range(100)]
    for row in repeats:
        row["task_group"] = "same-generated-archetype"
    repeat_report = build_capability_cell_report(normalize_execution_outcomes(repeats))
    assert repeat_report["cells"][0]["independent_verified_samples"] == 1
    assert repeat_report["cells"][0]["state"] == "exploration-only"


def test_cheapest_targets_only_use_randomized_paired_acceptable_outcomes() -> None:
    teacher = _raw_outcome(0, arm="teacher", cost=10)
    cheap = _raw_outcome(1, arm="cheaper-candidate", cost=2)
    cheap["counterfactual_group_id"] = teacher["counterfactual_group_id"]
    cheap["task_group"] = teacher["task_group"]
    targets = build_cheapest_acceptable_rows(normalize_execution_outcomes([teacher, cheap]))

    assert len(targets) == 1
    assert targets[0]["chosen_option_id"] == "cheap"
    assert targets[0]["causal"] is True

    cheap["assignment_mechanism"] = "observational"
    assert build_cheapest_acceptable_rows(normalize_execution_outcomes([teacher, cheap])) == []


def test_counterfactual_plans_are_low_risk_isolated_and_keep_teacher_authority() -> None:
    row = {
        "task_id": "task-1",
        "task_group": "group-1",
        "subtask_id": "inspect",
        "catalog_hash": HASH,
        "policy_hash": HASH,
        "candidate_set_hash": HASH,
        "chosen_option_id": "teacher",
        "features": {"risk": "low", "required_capabilities": ["tests"], "signals": {}},
        "eligible_options": [
            {"option_id": "cheap", "estimated_unit_cost": 1},
            {"option_id": "teacher", "estimated_unit_cost": 5},
        ],
    }
    plans = plan_counterfactual_trials([row], experiment_id="exp-1", sample_rate=1)

    assert len(plans) == 1
    assert plans[0]["candidate_option_id"] == "cheap"
    assert plans[0]["execution_mode"] == "isolated-shadow"
    assert plans[0]["production_authority"] == "teacher"
    assert plans[0]["requires_independent_verifier"] is True

    row["features"]["risk"] = "high"
    assert plan_counterfactual_trials([row], experiment_id="exp-1", sample_rate=1) == []


def test_leakage_audit_distinguishes_teacher_imitation_from_production_generalization() -> None:
    base = {
        "chosen_option_id": "opt-teacher",
        "chosen_choice_id": "C01",
        "features": {
            "task_text": "different wording",
            "objective": "different objective",
            "risk": "low",
            "required_capabilities": ["tests"],
            "verification_floor": "standard",
        },
    }
    train = [{**base, "id": "train", "task_group": "train-group"}]
    evaluate = [{**base, "id": "eval", "task_group": "eval-group", "features": {**base["features"], "task_text": "paraphrase"}}]
    audit = audit_classifier_dataset(train, evaluate)

    assert audit["group_disjoint"] is True
    assert audit["safe_for_teacher_imitation"] is True
    assert audit["safe_for_production_generalization"] is False
    assert "train/eval share synthetic feature archetypes" in audit["findings"]

    train[0]["features"]["teacher_option_id"] = "opt-teacher"
    assert audit_classifier_dataset(train, evaluate)["safe_for_teacher_imitation"] is False


def test_rollout_starts_at_zero_and_caps_first_authority_at_five_percent() -> None:
    empty = build_capability_cell_report([])
    assert evaluate_rollout(empty, successful_evidence_epochs=10)["maximum_classifier_coverage"] == 0

    trusted = build_capability_cell_report(normalize_execution_outcomes([_raw_outcome(index) for index in range(100)]))
    canary = evaluate_rollout(trusted, successful_evidence_epochs=1)
    limited = evaluate_rollout(trusted, current_stage="canary", successful_evidence_epochs=3)
    assert canary["authorized_stage"] == "canary"
    assert canary["maximum_classifier_coverage"] == 0.05
    assert limited["maximum_classifier_coverage"] == 0.25


def test_outcome_pipeline_writes_reproducible_shadow_artifacts(tmp_path) -> None:
    manifest = run_outcome_pipeline([_raw_outcome(1)], tmp_path / "outcomes")

    assert manifest["outcome_count"] == 1
    assert manifest["causal_target_count"] == 0
    assert manifest["trusted_cell_count"] == 0
    assert manifest["authorized_stage"] == "shadow"
    assert manifest["maximum_classifier_coverage"] == 0
    assert run_outcome_pipeline([_raw_outcome(1)], tmp_path / "outcomes") == manifest
