from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dataevol.datasets.codex_routing import (
    DPO_SCHEMA,
    build_codex_routing_datasets,
    build_preference_rows,
    build_task_decomposition_rows,
    evaluate_routing_predictions,
    freeze_model_catalog,
    normalize_outcome_feedback,
    preferences_from_outcomes,
)
from dataevol.datasets.codex_route_compiler import (
    EXECUTABLE_ROUTE_SCHEMA,
    RANKING_SCHEMA,
    build_ranking_datasets,
    build_route_candidate_set,
    compile_executable_route,
    freeze_route_policy,
    validate_executable_route,
)


def test_catalog_snapshot_is_order_stable_immutable_and_tamper_checked(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.json"
    frozen = freeze_model_catalog(catalog, path)
    reordered = {**catalog, "models": list(reversed(catalog["models"]))}

    assert freeze_model_catalog(reordered, path) == frozen
    assert len(frozen["catalog_hash"]) == 64
    assert len(frozen["snapshot_hash"]) == 64

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["models"][0]["context_window"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        freeze_model_catalog(catalog, path)


def test_task_rows_enforce_decomposition_and_catalog_constraints(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_catalog(), tmp_path / "catalog.json")
    rows = build_task_decomposition_rows([_task(1)], catalog)

    assert rows[0]["schema"].endswith("routing_sft.v1")
    assert rows[0]["catalog_hash"] == catalog["catalog_hash"]
    assert json.loads(rows[0]["completion"])["assignments"][0]["subtask_id"] == "implement"

    invalid = _task(2)
    invalid["route"]["assignments"][0]["model_id"] = "codex-fast"
    invalid["route"]["assignments"][0]["reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="lacks capabilities"):
        build_task_decomposition_rows([invalid], catalog)

    missing_effort = _task(4)
    missing_effort["route"]["assignments"][0].pop("reasoning_effort")
    with pytest.raises(ValueError, match="reasoning_effort is required"):
        build_task_decomposition_rows([missing_effort], catalog)

    unsupported_effort = _task(5)
    unsupported_effort["route"]["assignments"][1]["reasoning_effort"] = "xhigh"
    with pytest.raises(ValueError, match="does not support reasoning_effort"):
        build_task_decomposition_rows([unsupported_effort], catalog)

    effort_preference = _task(3)
    rejected = json.loads(json.dumps(effort_preference["route"]))
    next(item for item in rejected["assignments"] if item["subtask_id"] == "implement")["reasoning_effort"] = "medium"
    dpo = build_preference_rows(
        [{**effort_preference, "chosen_route": effort_preference["route"], "rejected_route": rejected}], catalog
    )
    assert json.loads(dpo[0]["chosen"])["assignments"] != json.loads(dpo[0]["rejected"])["assignments"]


def test_catalog_distinguishes_listed_configured_and_hidden_models(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_current_codex_catalog(), tmp_path / "current_catalog.json")
    models = {row["model_id"]: row for row in catalog["models"]}

    assert catalog["configured_model_id"] == "gpt-5.6-sol"
    assert catalog["excluded_hidden_model_count"] == 1
    assert "auto-review" not in models
    assert models["gpt-5.6-sol"]["availability"] == "configured_verified"
    assert models["gpt-5.5"]["availability"] == "listed"
    assert models["gpt-5.5"]["pricing"]["api_usd_equivalent"] == {
        "cached_input": 0.5,
        "input": 5.0,
        "output": 30.0,
    }
    assert models["gpt-5.4-mini"]["pricing"]["codex_credit"] == {
        "cached_input": 1.875,
        "input": 18.75,
        "output": 113.0,
    }
    assert models["gpt-5.3-codex-spark"]["pricing"]["api_usd_equivalent"] is None


def test_builds_group_disjoint_layerscope_compatible_sft_and_dpo(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_catalog(), tmp_path / "catalog.json")
    tasks = [_task(index) for index in range(4)]
    preferences = [_preference(index) for index in range(4)]

    result = build_codex_routing_datasets(catalog, tasks, preferences, tmp_path / "datasets", eval_fraction=0.25)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    sft_train = _rows(result.sft_train_path)
    sft_eval = _rows(result.sft_eval_path)
    dpo_train = _rows(result.dpo_train_path)
    dpo_eval = _rows(result.dpo_eval_path)

    assert {row["task_group"] for row in sft_train}.isdisjoint(row["task_group"] for row in sft_eval)
    assert {row["task_group"] for row in dpo_train}.isdisjoint(row["task_group"] for row in dpo_eval)
    assert {row["task_group"] for row in sft_train}.isdisjoint(row["task_group"] for row in dpo_eval)
    assert all(set(("prompt", "completion")) <= set(row) for row in [*sft_train, *sft_eval])
    assert all(set(("prompt", "chosen", "rejected")) <= set(row) for row in [*dpo_train, *dpo_eval])
    assert manifest["layer_specialist_training"]["sft"]["training_mode"] == "sft"
    assert manifest["layer_specialist_training"]["dpo"]["training_mode"] == "rl"
    assert manifest["layer_specialist_training"]["dpo"]["requires_initial_sft_specialist_manifest"] is True
    assert build_codex_routing_datasets(catalog, tasks, preferences, tmp_path / "datasets", eval_fraction=0.25) == result


def test_feedback_can_create_dpo_preferences_and_offline_metrics(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_catalog(), tmp_path / "catalog.json")
    task_rows = build_task_decomposition_rows([_task(1), _task(2)], catalog)
    selected = task_rows[0]
    chosen_route = json.loads(selected["completion"])
    rejected_route = json.loads(selected["completion"])
    next(item for item in rejected_route["assignments"] if item["subtask_id"] == "implement")["model_id"] = "codex-reason"
    outcomes = normalize_outcome_feedback(
        [
            {
                "outcome_id": "good",
                "task_group": selected["task_group"],
                "route": chosen_route,
                "verified": True,
                "quality": 0.95,
                "cost_usd": 0.01,
                "latency_ms": 800,
                "reasoning_tokens": 1_000,
                "total_tokens": 5_000,
            },
            {
                "outcome_id": "bad",
                "task_group": selected["task_group"],
                "route": rejected_route,
                "verified": False,
                "quality": 0.2,
                "cost_usd": 0.1,
                "latency_ms": 2_000,
                "reasoning_tokens": 8_000,
                "total_tokens": 12_000,
                "failure_type": "failed_verification",
            },
        ],
        task_rows,
        catalog,
    )
    preferences = preferences_from_outcomes(task_rows, outcomes)

    assert len(preferences) == 1
    assert preferences[0]["schema"] == DPO_SCHEMA
    assert preferences[0]["provenance"]["observational_not_causal"] is True
    assert build_preference_rows(preferences, catalog)[0]["pair_id"] == preferences[0]["pair_id"]

    prediction = {
        "id": selected["id"],
        "catalog_hash": catalog["catalog_hash"],
        "prompt_sha256": hashlib.sha256(selected["prompt"].encode("utf-8")).hexdigest(),
        "output": selected["completion"],
    }
    evaluation = evaluate_routing_predictions([selected], [prediction], catalog, feedback=outcomes)
    assert evaluation["metrics"]["valid_route_rate"] == 1.0
    assert evaluation["metrics"]["exact_route_accuracy"] == 1.0
    assert evaluation["metrics"]["subtask_assignment_accuracy"] == 1.0
    assert evaluation["metrics"]["subtask_model_accuracy"] == 1.0
    assert evaluation["metrics"]["subtask_reasoning_effort_accuracy"] == 1.0
    assert evaluation["metrics"]["constraint_satisfaction_rate"] == 1.0
    assert evaluation["observational_outcomes"]["coverage"] == 1.0
    assert evaluation["observational_outcomes"]["reasoning_tokens"] == 1_000
    assert evaluation["observational_outcomes"]["causal_claim_allowed"] is False

    bad_prediction = {**prediction, "output": prediction["output"].replace("codex-fast", "missing-model")}
    invalid = evaluate_routing_predictions([selected], [bad_prediction], catalog)
    assert invalid["metrics"]["valid_route_rate"] == 0.0
    assert invalid["metrics"]["exact_route_accuracy"] == 0.0

    effort_only = json.loads(prediction["output"])
    next(item for item in effort_only["assignments"] if item["subtask_id"] == "implement")["reasoning_effort"] = "medium"
    effort_eval = evaluate_routing_predictions([selected], [{**prediction, "output": json.dumps(effort_only)}], catalog)
    assert effort_eval["metrics"]["subtask_model_accuracy"] == 1.0
    assert effort_eval["metrics"]["subtask_assignment_accuracy"] < 1.0
    assert effort_eval["metrics"]["subtask_reasoning_effort_accuracy"] < 1.0


def test_offline_evaluation_rejects_selective_or_mismatched_predictions(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_catalog(), tmp_path / "catalog.json")
    rows = build_task_decomposition_rows([_task(1), _task(2)], catalog)
    prediction = {
        "id": rows[0]["id"],
        "catalog_hash": catalog["catalog_hash"],
        "prompt_sha256": "0" * 64,
        "output": rows[0]["completion"],
    }
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_routing_predictions(rows, [prediction], catalog)
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        evaluate_routing_predictions([rows[0]], [prediction], catalog)


def test_candidate_compiler_masks_ineligible_models_and_builds_strict_route(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_catalog(), tmp_path / "catalog.json")
    policy = freeze_route_policy(_policy(), tmp_path / "policy.json")
    candidates = build_route_candidate_set(_task(1), catalog, policy)
    by_subtask = {row["subtask_id"]: row for row in candidates["subtasks"]}
    assert {row["model_id"] for row in by_subtask["implement"]["candidates"]} == {"codex-fast", "codex-reason"}
    assert {row["model_id"] for row in by_subtask["verify"]["candidates"]} == {"codex-review"}

    rankings = []
    for subtask in candidates["subtasks"]:
        option_ids = [row["option_id"] for row in subtask["candidates"]]
        rankings.append({
            "schema": RANKING_SCHEMA,
            "subtask_id": subtask["subtask_id"],
            "ranked_option_ids": option_ids,
            "confidence": 0.9,
            "requested_verification_tier": "none",
        })
    route = compile_executable_route(candidates, rankings, policy)

    assert route["schema"] == EXECUTABLE_ROUTE_SCHEMA
    assert len(route["assignments"]) == 3
    verify = next(row for row in route["assignments"] if row["subtask_id"] == "verify")
    assert verify["verification_tier"] == "strong"
    assert verify["depends_on"] == ["implement"]
    assert verify["option_id"] != verify["fallback_option_id"]
    validate_executable_route(route, candidates, policy)

    with pytest.raises(ValueError, match="fields mismatch"):
        validate_executable_route({**route, "unexpected": True}, candidates, policy)


def test_candidate_compiler_escalates_low_confidence_and_preserves_policy_floor(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_catalog(), tmp_path / "catalog.json")
    policy = freeze_route_policy(_policy(), tmp_path / "policy.json")
    candidates = build_route_candidate_set(_task(1), catalog, policy)
    rankings = [
        {
            "schema": RANKING_SCHEMA,
            "subtask_id": subtask["subtask_id"],
            "ranked_option_ids": [row["option_id"] for row in subtask["candidates"]],
            "confidence": 0.4,
            "requested_verification_tier": "standard",
        }
        for subtask in candidates["subtasks"]
    ]
    route = compile_executable_route(candidates, rankings, policy)
    assert route["requires_semantic_verifier"] is True
    assert all("low-confidence" in row["verifier_triggers"] for row in route["assignments"])
    assert next(row for row in route["assignments"] if row["subtask_id"] == "plan")["verification_tier"] == "strong"


def test_ranking_datasets_are_compact_group_disjoint_and_repeatable(tmp_path: Path) -> None:
    catalog = freeze_model_catalog(_catalog(), tmp_path / "catalog.json")
    policy = freeze_route_policy(_policy(), tmp_path / "policy.json")
    tasks = [_task(index) for index in range(1, 7)]
    result = build_ranking_datasets(tasks, catalog, policy, tmp_path / "ranking", eval_fraction=0.34)
    repeated = build_ranking_datasets(tasks, catalog, policy, tmp_path / "ranking", eval_fraction=0.34)
    assert repeated.dataset_content_hash == result.dataset_content_hash
    train = _rows(result.sft_train_path)
    evaluate = _rows(result.sft_eval_path)
    assert {row["task_group"] for row in train}.isdisjoint({row["task_group"] for row in evaluate})
    assert all(row["completion"].startswith("C") for row in train + evaluate)
    assert max(len(row["completion"]) for row in train + evaluate) == 3


def _catalog() -> dict:
    return {
        "source": "codex-model-catalog",
        "source_revision": "catalog-2026-07-11",
        "captured_at": "2026-07-11T12:00:00+00:00",
        "models": [
            {
                "model_id": "codex-fast",
                "provider": "openai",
                "revision": "fast-1",
                "capabilities": ["code"],
                "supported_reasoning_efforts": ["low", "medium"],
                "default_reasoning_effort": "low",
                "context_window": 32_000,
                "input_cost_per_million": 0.2,
                "output_cost_per_million": 0.8,
                "latency_p50_ms": 300,
                "risk_tiers": ["low"],
            },
            {
                "model_id": "codex-reason",
                "provider": "openai",
                "revision": "reason-1",
                "capabilities": ["code", "reasoning"],
                "supported_reasoning_efforts": ["low", "medium", "high", "xhigh"],
                "default_reasoning_effort": "medium",
                "context_window": 128_000,
                "input_cost_per_million": 1.0,
                "output_cost_per_million": 4.0,
                "latency_p50_ms": 1_200,
                "risk_tiers": ["low", "high"],
            },
            {
                "model_id": "codex-review",
                "provider": "openai",
                "revision": "review-1",
                "capabilities": ["verification"],
                "supported_reasoning_efforts": ["low", "medium", "high"],
                "default_reasoning_effort": "medium",
                "context_window": 64_000,
                "input_cost_per_million": 0.5,
                "output_cost_per_million": 2.0,
                "latency_p50_ms": 600,
                "risk_tiers": ["low", "high"],
            },
        ],
    }


def _policy() -> dict:
    return {
        "policy_id": "codex-routing",
        "policy_version": "test-v1",
        "model_strength_order": ["codex-fast", "codex-review", "codex-reason"],
        "risk_verification_tiers": {
            "low": "none",
            "medium": "standard",
            "high": "strong",
            "critical": "independent",
        },
        "semantic_verifier_risk_tiers": ["high", "critical"],
        "semantic_verifier_confidence_threshold": 0.65,
        "trusted_boundary_required_for": [],
    }


def _current_codex_catalog() -> dict:
    def priced(
        model_id: str,
        availability: str,
        api: tuple[float, float, float],
        credits: tuple[float, float, float],
    ) -> dict:
        return {
            "model_id": model_id,
            "provider": "openai",
            "availability": availability,
            "capabilities": ["code", "reasoning"],
            "supported_reasoning_efforts": ["low", "medium", "high", "xhigh"],
            "default_reasoning_effort": "medium",
            "context_window": 128_000,
            "risk_tiers": ["low", "high"],
            "pricing": {
                "codex_credit": {"input": credits[0], "cached_input": credits[1], "output": credits[2]},
                "api_usd_equivalent": {"input": api[0], "cached_input": api[1], "output": api[2]},
            },
        }

    return {
        "source": "~/.codex/models_cache.json + ~/.codex/config.toml + official OpenAI pricing",
        "source_revision": "2026-07-11",
        "captured_at": "2026-07-11T12:00:00+00:00",
        "configured_model_id": "gpt-5.6-sol",
        "models": [
            priced("gpt-5.6-sol", "configured_verified", (5.0, 0.5, 30.0), (125.0, 12.5, 750.0)),
            priced("gpt-5.5", "listed", (5.0, 0.5, 30.0), (125.0, 12.5, 750.0)),
            priced("gpt-5.4", "listed", (2.5, 0.25, 15.0), (62.5, 6.25, 375.0)),
            priced("gpt-5.4-mini", "listed", (0.75, 0.075, 4.5), (18.75, 1.875, 113.0)),
            {
                "model_id": "gpt-5.3-codex-spark",
                "provider": "openai",
                "availability": "listed",
                "capabilities": ["code"],
                "supported_reasoning_efforts": ["low", "medium", "high", "xhigh"],
                "default_reasoning_effort": "high",
                "context_window": 32_000,
                "risk_tiers": ["low"],
                "pricing": {"codex_credit": None, "api_usd_equivalent": None},
                "metadata": {"release_channel": "research_preview", "unpriced": True},
            },
            {
                "model_id": "auto-review",
                "provider": "openai",
                "availability": "hidden",
                "hidden": True,
            },
        ],
    }


def _task(index: int) -> dict:
    return {
        "task_id": f"task-{index}",
        "task": f"Implement and verify feature {index}",
        "subtasks": [
            {
                "subtask_id": "plan",
                "objective": "Analyze dependencies",
                "required_capabilities": ["reasoning"],
                "risk": "high",
                "estimated_input_tokens": 4_000,
            },
            {
                "subtask_id": "implement",
                "objective": "Implement the change",
                "depends_on": ["plan"],
                "required_capabilities": ["code"],
                "risk": "low",
                "estimated_input_tokens": 8_000,
            },
            {
                "subtask_id": "verify",
                "objective": "Review tests and risks",
                "depends_on": ["implement"],
                "required_capabilities": ["verification"],
                "risk": "high",
                "estimated_input_tokens": 6_000,
            },
        ],
        "route": {
            "assignments": [
                {"subtask_id": "plan", "model_id": "codex-reason", "reasoning_effort": "high", "reason": "reasoning"},
                {"subtask_id": "implement", "model_id": "codex-fast", "reasoning_effort": "low", "reason": "cost"},
                {"subtask_id": "verify", "model_id": "codex-review", "reasoning_effort": "medium", "reason": "independent verification"},
            ],
            "fallback_model_id": "codex-reason",
            "requires_verification": True,
        },
    }


def _preference(index: int) -> dict:
    task = _task(index)
    rejected = json.loads(json.dumps(task["route"]))
    next(item for item in rejected["assignments"] if item["subtask_id"] == "implement")["model_id"] = "codex-reason"
    return {
        **{key: task[key] for key in ("task_id", "task", "subtasks")},
        "chosen_route": task["route"],
        "rejected_route": rejected,
        "provenance": {"source": "measured_outcome_pair"},
    }


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
