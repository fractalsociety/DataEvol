from __future__ import annotations

from dataevol.experiments.codex_route_classifiers import run_classifier_benchmark, selective_tabular_decision


def test_classifier_benchmark_reports_exact_and_selective_metrics() -> None:
    rows = [_row(index) for index in range(90)]
    report, fitted = run_classifier_benchmark(rows[:72], rows[72:], seed=1701)

    assert report["models"]["deterministic_rule"]["metrics"]["option_accuracy"] == 1.0
    assert report["models"]["tabular_logistic"]["metrics"]["valid_combination_rate"] == 1.0
    accepted = report["models"]["deterministic_rule"]["selective_low_medium"]["0.97"]
    assert accepted["coverage"] == 1.0
    assert accepted["precision"] == 1.0
    assert "tabular_hierarchical" in fitted
    assert report["report_hash"]

    shadow = selective_tabular_decision(rows[72], fitted, confidence_threshold=0.97)
    assert shadow["accepted"] is False
    assert shadow["authority"] == "teacher"
    assert "capability-boundary-not-trusted" in shadow["reasons"]
    accepted = selective_tabular_decision(rows[72], fitted, confidence_threshold=0.97, require_trusted_boundary=False)
    assert accepted["accepted"] is True
    assert accepted["authority"] == "classifier"


def _row(index: int) -> dict:
    kind = index % 3
    if kind == 0:
        risk, capability, model, effort = "low", "code", "gpt-5.4-mini", "medium"
    elif kind == 1:
        risk, capability, model, effort = "medium", "integration", "gpt-5.4", "medium"
    else:
        risk, capability, model, effort = "high", "security", "gpt-5.6-sol", "high"
    options = [
        {"choice_id": "C00", "option_id": "opt-mini-low", "model_id": "gpt-5.4-mini", "reasoning_effort": "low"},
        {"choice_id": "C01", "option_id": "opt-mini-medium", "model_id": "gpt-5.4-mini", "reasoning_effort": "medium"},
        {"choice_id": "C02", "option_id": "opt-standard-medium", "model_id": "gpt-5.4", "reasoning_effort": "medium"},
        {"choice_id": "C03", "option_id": "opt-frontier-high", "model_id": "gpt-5.6-sol", "reasoning_effort": "high"},
    ]
    chosen = next(option for option in options if option["model_id"] == model and option["reasoning_effort"] == effort)
    return {
        "id": f"row-{index:03d}",
        "task_group": f"task-{index // 3:03d}",
        "subtask_id": f"subtask-{index:03d}",
        "catalog_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "candidate_set_hash": "c" * 64,
        "chosen_option_id": chosen["option_id"],
        "eligible_options": options,
        "features": {
            "task_text": f"Task {index}",
            "objective": f"Handle {capability}",
            "risk": risk,
            "required_capabilities": [capability],
            "estimated_input_tokens": 1_000 * (kind + 1),
            "dependency_count": kind,
            "subtask_count": 3,
            "verification_floor": "strong" if risk == "high" else "standard" if risk == "medium" else "none",
            "boundary_states": ["unmapped"],
            "eligible_option_count": len(options),
            "signals": {"security": capability == "security", "migration": False, "research": False, "architecture": False, "incident": False, "verification": False},
        },
    }
