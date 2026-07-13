from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA = "dataevol.layerscope_depth_policy.v1"
ROUTES = {
    "early_concentrated": "direct_layer",
    "late_concentrated": "aggressive_socket",
    "bimodal_or_distributed": "scheduled_socket_with_reserves",
}


def layer_from_entry_key(key: str) -> int:
    prefix = "layer-"
    marker = ":family-"
    if not key.startswith(prefix) or marker not in key:
        raise ValueError(f"invalid LayerScope entry key: {key}")
    return int(key[len(prefix):key.index(marker)])


def normalized_depth(layer: int, *, num_layers: int) -> float:
    if num_layers < 2:
        raise ValueError("num_layers must be at least two")
    if not 0 <= layer < num_layers:
        raise ValueError(f"layer {layer} is outside a {num_layers}-layer model")
    return layer / (num_layers - 1)


def estimated_step_cost(
    layer: int,
    *,
    num_layers: int,
    fixed_forward_fraction: float = 0.5,
) -> float:
    """Estimate step cost when backward traversal stops below ``layer``.

    Forward work is unchanged. The remaining fraction models backward work from
    the output through the selected layer. This is a ranking proxy, not a claim
    of measured speedup.
    """
    if not 0.0 <= fixed_forward_fraction <= 1.0:
        raise ValueError("fixed_forward_fraction must be between zero and one")
    backward_fraction = (num_layers - layer) / num_layers
    return fixed_forward_fraction + (1.0 - fixed_forward_fraction) * backward_fraction


def cost_adjusted_entry_scores(
    gradient_norms: Mapping[str, float],
    parameter_counts: Mapping[str, int],
    *,
    num_layers: int,
    fixed_forward_fraction: float = 0.5,
    depth_cost_weight: float = 1.0,
) -> dict[str, dict[str, float | int]]:
    if depth_cost_weight < 0:
        raise ValueError("depth_cost_weight must be nonnegative")
    if set(parameter_counts) - set(gradient_norms):
        missing = sorted(set(parameter_counts) - set(gradient_norms))
        raise ValueError(f"gradient norms missing LayerScope entries: {missing}")
    scores: dict[str, dict[str, float | int]] = {}
    for key, count in parameter_counts.items():
        if count <= 0:
            raise ValueError(f"entry {key} has a nonpositive parameter count")
        layer = layer_from_entry_key(key)
        saliency_density = float(gradient_norms[key]) / math.sqrt(count)
        cost = estimated_step_cost(
            layer,
            num_layers=num_layers,
            fixed_forward_fraction=fixed_forward_fraction,
        )
        scores[key] = {
            "layer": layer,
            "normalized_depth": normalized_depth(layer, num_layers=num_layers),
            "saliency_density": saliency_density,
            "estimated_step_cost": cost,
            "cost_adjusted_score": saliency_density / (cost**depth_cost_weight),
        }
    return scores


def classify_depth_profile(
    gradient_norms: Mapping[str, float],
    *,
    num_layers: int,
    concentration_threshold: float = 0.60,
    bimodal_lobe_threshold: float = 0.25,
) -> dict[str, Any]:
    """Classify saliency mass without treating the centroid as sufficient proof."""
    if not 0.5 <= concentration_threshold <= 1.0:
        raise ValueError("concentration_threshold must be between 0.5 and one")
    if not 0.0 <= bimodal_lobe_threshold < 0.5:
        raise ValueError("bimodal_lobe_threshold must be between zero and 0.5")
    if not gradient_norms:
        raise ValueError("cannot classify an empty saliency profile")

    layer_mass: dict[int, float] = {}
    for key, raw_mass in gradient_norms.items():
        mass = max(0.0, float(raw_mass))
        layer = layer_from_entry_key(key)
        normalized_depth(layer, num_layers=num_layers)
        layer_mass[layer] = layer_mass.get(layer, 0.0) + mass
    total = sum(layer_mass.values())
    if total <= 0:
        raise ValueError("saliency profile has no positive mass")

    normalized_mass = {layer: mass / total for layer, mass in layer_mass.items()}
    bands = {
        "early": sum(mass for layer, mass in normalized_mass.items() if normalized_depth(layer, num_layers=num_layers) < 1 / 3),
        "middle": sum(mass for layer, mass in normalized_mass.items() if 1 / 3 <= normalized_depth(layer, num_layers=num_layers) < 2 / 3),
        "late": sum(mass for layer, mass in normalized_mass.items() if normalized_depth(layer, num_layers=num_layers) >= 2 / 3),
    }
    centroid = sum(
        normalized_depth(layer, num_layers=num_layers) * mass
        for layer, mass in normalized_mass.items()
    )
    dominant_layer = max(normalized_mass, key=lambda layer: (normalized_mass[layer], -layer))
    bimodal = bands["early"] >= bimodal_lobe_threshold and bands["late"] >= bimodal_lobe_threshold
    if bimodal:
        profile = "bimodal_or_distributed"
    elif bands["early"] >= concentration_threshold:
        profile = "early_concentrated"
    elif bands["late"] >= concentration_threshold:
        profile = "late_concentrated"
    else:
        profile = "bimodal_or_distributed"

    return {
        "schema": SCHEMA,
        "profile": profile,
        "route": ROUTES[profile],
        "depth_centroid": centroid,
        "saliency_mass_by_band": bands,
        "dominant_layer": dominant_layer,
        "dominant_layer_mass": normalized_mass[dominant_layer],
        "concentration_threshold": concentration_threshold,
        "bimodal_lobe_threshold": bimodal_lobe_threshold,
        "profile_confident": profile != "bimodal_or_distributed",
        "reserve_policy": "depth_bands" if profile == "bimodal_or_distributed" else "none",
    }


def select_cost_aware_entries(
    gradient_norms: Mapping[str, float],
    parameter_counts: Mapping[str, int],
    *,
    current_entries: set[str],
    required_entries: set[str],
    parameter_budget: int,
    hysteresis_ratio: float,
    num_layers: int,
    fixed_forward_fraction: float = 0.5,
    depth_cost_weight: float = 1.0,
) -> tuple[set[str], dict[str, dict[str, float | int]]]:
    if not 0.0 < hysteresis_ratio <= 1.0:
        raise ValueError("hysteresis_ratio must be between zero and one")
    missing = required_entries - set(parameter_counts)
    if missing:
        raise ValueError(f"required entries are absent from candidate: {sorted(missing)}")
    selected = set(required_entries)
    used = sum(parameter_counts[key] for key in selected)
    if used > parameter_budget:
        raise ValueError("required entries exceed the active parameter budget")

    scores = cost_adjusted_entry_scores(
        gradient_norms,
        parameter_counts,
        num_layers=num_layers,
        fixed_forward_fraction=fixed_forward_fraction,
        depth_cost_weight=depth_cost_weight,
    )
    ranked = []
    for key, row in scores.items():
        if key in required_entries:
            continue
        score = float(row["cost_adjusted_score"])
        if key in current_entries:
            score /= hysteresis_ratio
        ranked.append((score, key))
    for _, key in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if used + parameter_counts[key] <= parameter_budget:
            selected.add(key)
            used += parameter_counts[key]
    return selected, scores
