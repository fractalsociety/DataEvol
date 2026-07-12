from __future__ import annotations

import json

from dataevol.experiments.reusable_lora_socket import generate_socket_candidates
from dataevol.experiments.rl_socket_pipeline import (
    FAMILIES,
    _contiguous_socket,
    _mechanistic_socket,
    _validate_config,
    build_candidate_manifest,
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
