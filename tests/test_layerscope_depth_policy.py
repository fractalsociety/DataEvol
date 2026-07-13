from __future__ import annotations

import pytest

from dataevol.experiments.layerscope_depth_policy import (
    classify_depth_profile,
    cost_adjusted_entry_scores,
    estimated_step_cost,
    select_cost_aware_entries,
)


def test_depth_classifier_routes_only_concentrated_profiles_aggressively() -> None:
    early = classify_depth_profile(
        {"layer-2:family-A": 8.0, "layer-10:family-B": 1.0, "layer-20:family-C": 1.0},
        num_layers=22,
    )
    late = classify_depth_profile(
        {"layer-2:family-A": 1.0, "layer-10:family-B": 1.0, "layer-20:family-C": 8.0},
        num_layers=22,
    )
    ambiguous = classify_depth_profile(
        {"layer-2:family-A": 4.0, "layer-10:family-B": 2.0, "layer-20:family-C": 4.0},
        num_layers=22,
    )

    assert early["profile"] == "early_concentrated"
    assert early["route"] == "direct_layer"
    assert late["profile"] == "late_concentrated"
    assert late["route"] == "aggressive_socket"
    assert ambiguous["profile"] == "bimodal_or_distributed"
    assert ambiguous["route"] == "scheduled_socket_with_reserves"
    assert ambiguous["reserve_policy"] == "depth_bands"
    assert not ambiguous["profile_confident"]


def test_depth_classifier_does_not_turn_a_middle_centroid_into_false_confidence() -> None:
    profile = classify_depth_profile(
        {
            "layer-0:family-A": 1.0,
            "layer-4:family-A": 1.0,
            "layer-8:family-A": 4.0,
            "layer-12:family-A": 4.0,
            "layer-16:family-A": 1.0,
            "layer-20:family-A": 1.0,
        },
        num_layers=22,
    )

    assert 0.4 < profile["depth_centroid"] < 0.6
    assert profile["profile"] == "bimodal_or_distributed"
    assert profile["route"] == "scheduled_socket_with_reserves"


def test_cost_adjustment_can_prefer_a_nearly_as_salient_late_socket() -> None:
    norms = {"layer-2:family-B": 1.0, "layer-20:family-B": 0.9}
    counts = {key: 100_000 for key in norms}

    scores = cost_adjusted_entry_scores(norms, counts, num_layers=22)
    selected, _ = select_cost_aware_entries(
        norms,
        counts,
        current_entries=set(),
        required_entries=set(),
        parameter_budget=100_000,
        hysteresis_ratio=0.85,
        num_layers=22,
    )

    assert scores["layer-20:family-B"]["saliency_density"] < scores["layer-2:family-B"]["saliency_density"]
    assert scores["layer-20:family-B"]["cost_adjusted_score"] > scores["layer-2:family-B"]["cost_adjusted_score"]
    assert selected == {"layer-20:family-B"}


def test_estimated_cost_keeps_forward_work_and_prices_backward_depth() -> None:
    early = estimated_step_cost(0, num_layers=22, fixed_forward_fraction=0.5)
    late = estimated_step_cost(21, num_layers=22, fixed_forward_fraction=0.5)

    assert early == 1.0
    assert late == pytest.approx(0.5 + 0.5 / 22)
    assert late > 0.5
