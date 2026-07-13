from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from dataevol.experiments.reusable_lora_socket import generate_socket_candidates
from dataevol.experiments.rl_socket_pipeline import (
    FAMILIES,
    _arithmetic_behavioral_reward,
    _arithmetic_paired_interval,
    _contiguous_socket,
    _health_abort_reason,
    _mechanistic_socket,
    _python_public_test_fraction,
    _training_reward,
    _validate_config,
    build_candidate_manifest,
    entry_key,
    gradient_norms_by_entry,
    mask_frozen_entry_gradients,
    prepare_joint_sft_curriculum,
    prepare_targeted_arithmetic_curriculum,
    prepare_uniform_kl_sweep_curriculum,
    select_python_protected_entries,
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


def test_targeted_confirmation_budget_is_pinned_to_sixty_updates() -> None:
    config = json.loads(Path("configs/rl_socket_targeted_arithmetic.yaml").read_text())
    assert config["targeted_arithmetic"]["updates"] == 60
    assert config["uniform_kl_sweep"]["updates"] == 60
    assert [row["learning_rate"] for row in config["uniform_kl_sweep"]["schedules"]] == [1e-5, 5e-6, 2e-6]


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

    health["kl_abort_window"] = 4
    spiky = [
        {"kl_divergence": value, "zero_variance_group": 0.0, "completion_diversity": 1.0}
        for value in (0.05, 0.5, 0.05, 0.05)
    ]
    assert _health_abort_reason(spiky, health) is None
    spiky[-1]["kl_divergence"] = 0.4
    assert _health_abort_reason(spiky, health).startswith("mean KL exceeded")


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


def test_targeted_arithmetic_curriculum_is_disjoint_and_in_learnable_range(tmp_path) -> None:
    source = tmp_path / "source"
    prepare_joint_sft_curriculum(source)

    manifest = prepare_targeted_arithmetic_curriculum(tmp_path / "run", source)

    assert manifest["operand_range"] == [0, 4]
    train = (tmp_path / "run/datasets/arithmetic/train.jsonl").read_text()
    valid = (tmp_path / "run/datasets/arithmetic/valid.jsonl").read_text()
    test = (tmp_path / "run/datasets/arithmetic/test.jsonl").read_text()
    assert "Calculate the sum" in train
    assert "What is" in valid
    assert "Compute" in test
    assert manifest["splits"]["train"]["sha256"] != manifest["splits"]["valid"]["sha256"]


def test_uniform_sweep_uses_new_locked_test_panel(tmp_path) -> None:
    source = tmp_path / "source"
    prepare_joint_sft_curriculum(source)
    prepare_targeted_arithmetic_curriculum(source, source)

    manifest = prepare_uniform_kl_sweep_curriculum(tmp_path / "sweep", source)

    test = (tmp_path / "sweep/datasets/arithmetic/test.jsonl").read_text()
    assert "Find the total" in test
    assert manifest["test_rows"] == 500
    assert manifest["test_sha256"] != json.loads(
        (source / "datasets/targeted_manifest.json").read_text()
    )["splits"]["test"]["sha256"]


def test_arithmetic_interval_is_paired_against_same_prompts() -> None:
    baseline = {"outcomes": [False, False, True, False]}
    rows = [
        {"behavior": {"arithmetic": {"outcomes": [True, True, True, False]}}},
        {"behavior": {"arithmetic": {"outcomes": [True, False, True, True]}}},
    ]
    interval = _arithmetic_paired_interval(rows, baseline, draws=2_000)
    assert interval["mean_difference"] == 0.5
    assert interval["lower"] > 0


def test_entry_gradient_logging_and_masking_use_socket_entries() -> None:
    candidate = {
        "entries": [
            {"layer": 3, "family": "C", "rank": 1},
            {"layer": 5, "family": "B", "rank": 1},
        ]
    }
    gradients = tree_unflatten([
        ("model.layers.3.mlp.down_proj.lora_a", mx.array([3.0, 4.0])),
        ("model.layers.5.self_attn.o_proj.lora_a", mx.array([12.0])),
    ])

    norms = gradient_norms_by_entry(gradients, candidate)
    masked = dict(tree_flatten(mask_frozen_entry_gradients(
        gradients, candidate, {entry_key(candidate["entries"][0])}
    )))

    assert norms == {"layer-3:family-C": 5.0, "layer-5:family-B": 12.0}
    assert bool(mx.all(masked["model.layers.3.mlp.down_proj.lora_a"] == 0))
    assert bool(mx.all(masked["model.layers.5.self_attn.o_proj.lora_a"] == 12))


def test_python_protection_uses_task_specific_gradient_ratio() -> None:
    profile = {
        "gradient_norms": {
            "arithmetic": {"python-specific": 1.0, "shared": 8.0},
            "python": {"python-specific": 9.0, "shared": 8.0},
        }
    }
    assert select_python_protected_entries(profile, count=1) == ["python-specific"]
