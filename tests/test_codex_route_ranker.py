from __future__ import annotations

import hashlib
import json

import pytest

from dataevol.experiments.codex_route_ranker import evaluate_ranking_scores


def test_ranking_evaluation_scores_decisions_and_calibration_separately() -> None:
    prompt = "choose"
    reference = {
        "id": "row-1",
        "catalog_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "candidate_set_hash": "c" * 64,
        "prompt": prompt,
        "chosen_option_id": "opt_a",
        "eligible_option_ids": ["opt_a", "opt_b"],
        "eligible_options": [
            {"choice_id": "C00", "option_id": "opt_a", "model_id": "mini", "reasoning_effort": "low"},
            {"choice_id": "C01", "option_id": "opt_b", "model_id": "mini", "reasoning_effort": "medium"},
        ],
    }
    prediction = {
        "id": "row-1",
        "catalog_hash": reference["catalog_hash"],
        "policy_hash": reference["policy_hash"],
        "candidate_set_hash": reference["candidate_set_hash"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "ranked_option_ids": ["opt_b", "opt_a"],
        "confidence": 0.6,
        "score_margin": 0.2,
    }

    result = evaluate_ranking_scores([reference], [prediction])

    assert result["metrics"]["option_accuracy"] == 0.0
    assert result["metrics"]["model_accuracy"] == 1.0
    assert result["metrics"]["reasoning_effort_accuracy"] == 0.0
    assert result["metrics"]["brier_score"] == pytest.approx(0.36)


def test_ranking_evaluation_fails_closed_on_selective_options() -> None:
    reference = {
        "id": "row-1", "catalog_hash": "a", "policy_hash": "b", "candidate_set_hash": "c",
        "prompt": "choose", "chosen_option_id": "a", "eligible_option_ids": ["a", "b"],
        "eligible_options": [
            {"choice_id": "C00", "option_id": "a", "model_id": "m1", "reasoning_effort": "low"},
            {"choice_id": "C01", "option_id": "b", "model_id": "m2", "reasoning_effort": "high"},
        ],
    }
    prediction = {
        "id": "row-1", "catalog_hash": "a", "policy_hash": "b", "candidate_set_hash": "c",
        "prompt_sha256": hashlib.sha256(b"choose").hexdigest(), "ranked_option_ids": ["a"],
        "confidence": 1.0, "score_margin": 1.0,
    }
    with pytest.raises(ValueError, match="every eligible option"):
        evaluate_ranking_scores([reference], [prediction])
