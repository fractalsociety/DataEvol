from __future__ import annotations

import json

from dataevol.experiments.reusable_lora_socket import generate_socket_candidates
from dataevol.experiments.rl_socket_pipeline import (
    FAMILIES,
    _arithmetic_behavioral_reward,
    _contiguous_socket,
    _health_abort_reason,
    _mechanistic_socket,
    _python_public_test_fraction,
    _training_reward,
    _validate_config,
    build_candidate_manifest,
    prepare_joint_sft_curriculum,
)


def test_equal_search_arms_and_half_percent_parameter_contract(tmp_path) -> None:
    source = tmp_path / "source"
    report = {
        "ranked_candidates": [
            {"socket": socket}
            for socket in generate_socket_candidates(count=8, seed=71)
        ]
    }
    (source / "discovery_search").mkdir(parents=True)
    (source / "discovery_search/search_report.json").write_text(json.dumps(report))
    config = {
        "sft_source_experiment": str(source),
        "search": {"candidates_per_arm": 8, "random_seed": 72},
        "adapter": {"target_parameters": 2_400_000},
    }

    manifest = build_candidate_manifest(config, tmp_path / "run")

    assert manifest["guided_count"] == manifest["random_count"] == 8
    assert len({row["socket_hash"] for row in manifest["candidates"]}) == 16
    for candidate in manifest["candidates"]:
        assert abs(candidate["trainable_parameters"] - 2_400_000) / 2_400_000 <= 0.005
        assert candidate["topology_families"]
    observed_families = {family for candidate in manifest["candidates"] for family in candidate["topology_families"]}
    assert set(FAMILIES) <= observed_families


def test_mechanistic_sockets_are_exactly_parameter_matched() -> None:
    sockets = [
        _mechanistic_socket("mlp-only", "C", 2_400_000),
        _mechanistic_socket("attention-only", "A", 2_400_000),
        _contiguous_socket(2_400_000),
    ]

    for socket in sockets:
        assert abs(socket["trainable_parameters"] - 2_400_000) / 2_400_000 <= 0.005


def test_config_rejects_small_groups_and_loose_parameter_matching() -> None:
    config = {
        "schema": "dataevol.rl_socket_v2.config.v1",
        "search": {"candidates_per_arm": 8},
        "rollout": {"group_size": 8},
        "adapter": {"maximum_parameter_mismatch": 0.005},
        "selection": {
            "arithmetic_weight": 0.45,
            "python_weight": 0.35,
            "behavioral_improvement_efficiency_weight": 0.10,
            "retention_weight": 0.05,
            "seed_stability_weight": 0.05,
            "efficiency_scale": 0.02,
        },
    }
    _validate_config(config)
    config["rollout"]["group_size"] = 7
    try:
        _validate_config(config)
    except ValueError as error:
        assert "at least eight" in str(error)
    else:
        raise AssertionError("undersized rollout group was accepted")


def test_training_rewards_are_behavioral_and_shaping_is_gated() -> None:
    assert _arithmetic_behavioral_reward("42", "42") == 1.0
    assert _arithmetic_behavioral_reward("40", "42") > _arithmetic_behavioral_reward("4", "42") > 0.0
    assert _training_reward("arithmetic", "no number", {"answer": "42"}, 2, 48) == 0.0

    expected = "def subtract(a, b):\n    return a - b"
    correct = "def subtract(a, b):\n    return a - b"
    partial = "def subtract(a, b):\n    return 3"
    assert _python_public_test_fraction(correct, expected) == 1.0
    assert _python_public_test_fraction(partial, expected) == 0.5
    assert _training_reward("python", "not python", {"completion": expected}, 2, 48) == 0.0


def test_every_python_training_function_has_public_behavior_tests() -> None:
    functions = [
        "def clamp(x, low, high):\n    return max(low, min(x, high))",
        "def is_even(n):\n    return n % 2 == 0",
        "def last_item(items):\n    return items[-1]",
        "def add_tax(price, rate):\n    return price + price * rate",
        "def contains(items, value):\n    return value in items",
        "def square(n):\n    return n * n",
    ]
    assert all(_python_public_test_fraction(code, code) == 1.0 for code in functions)


def test_health_abort_requires_sustained_failure() -> None:
    health = {
        "kl_threshold": 0.2,
        "kl_abort_consecutive": 3,
        "zero_variance_window": 20,
        "zero_variance_abort_threshold": 0.9,
        "diversity_window": 10,
        "diversity_abort_threshold": 0.2,
    }
    history = [
        {"kl_divergence": value, "zero_variance_group": 0.0, "completion_diversity": 1.0}
        for value in (0.1, 0.3, 0.4)
    ]
    assert _health_abort_reason(history, health) is None
    history.append({"kl_divergence": 0.5, "zero_variance_group": 0.0, "completion_diversity": 1.0})
    assert _health_abort_reason(history, health).startswith("KL exceeded")


def test_joint_sft_curriculum_covers_python_families_in_every_split(tmp_path) -> None:
    manifest = prepare_joint_sft_curriculum(tmp_path)
    assert prepare_joint_sft_curriculum(tmp_path) == manifest

    assert manifest["python_function_families"] == 8
    assert manifest["splits"]["train"]["rows"] == 4_000
    assert manifest["splits"]["valid"]["rows"] == 600
    assert manifest["splits"]["test"]["rows"] == 1_000
    for split in ("train", "valid", "test"):
        text = (tmp_path / f"datasets/python/{split}.jsonl").read_text()
        for function in ("clamp", "is_even", "last_item", "add_tax", "contains", "square", "first_item", "subtract"):
            assert f"def {function}" in text
