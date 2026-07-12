from __future__ import annotations

import json

from dataevol.experiments.reusable_lora_socket import (
    BANDS,
    _python_unit_test_pass,
    adjudicate_confirmed_experiment,
    generate_socket_candidates,
    parameter_matched_random,
    parameter_matched_single,
    parameter_matched_uniform,
    prepare_datasets,
)


def test_candidates_cover_bands_modules_and_budget() -> None:
    candidates = generate_socket_candidates(count=12, seed=1701)

    assert len(candidates) == 12
    assert len({item["socket_hash"] for item in candidates}) == 12
    target = candidates[0]["trainable_parameters"]
    for candidate in candidates:
        assert 2_000_000 <= candidate["trainable_parameters"] <= 4_000_000
        assert abs(candidate["trainable_parameters"] - target) / target <= 0.03
        covered = {
            index
            for entry in candidate["entries"]
            for index, (low, high) in enumerate(BANDS)
            if low <= entry["layer"] <= high
        }
        assert len(covered) >= 4
        assert {entry["family"] for entry in candidate["entries"]} & {"A", "B"}
        assert "C" in {entry["family"] for entry in candidate["entries"]}


def test_baselines_are_parameter_matched_within_three_percent() -> None:
    reference = generate_socket_candidates(count=3)[0]
    target = reference["trainable_parameters"]
    baselines = [
        parameter_matched_uniform(target),
        *(parameter_matched_single(layer, target) for layer in (2, 6, 10, 15, 20)),
        *parameter_matched_random(reference),
    ]

    for baseline in baselines:
        assert abs(baseline["trainable_parameters"] - target) / target <= 0.03


def test_datasets_have_pinned_sizes_and_disjoint_heldout_templates(tmp_path) -> None:
    manifest = prepare_datasets(tmp_path / "datasets")

    assert manifest["held_out_from_socket_discovery"] == ["json"]
    for task in ("arithmetic", "python", "json"):
        assert manifest["splits"][task]["train"]["rows"] == 2_000
        assert manifest["splits"][task]["valid"]["rows"] == 300
        assert manifest["splits"][task]["test"]["rows"] == 500
    json_train = [json.loads(line)["prompt"] for line in (tmp_path / "datasets/json/train.jsonl").read_text().splitlines()]
    json_test = [json.loads(line)["prompt"] for line in (tmp_path / "datasets/json/test.jsonl").read_text().splitlines()]
    assert all("Purchase record for" not in prompt for prompt in json_train)
    assert all("Purchase record for" in prompt for prompt in json_test)
    assert prepare_datasets(tmp_path / "datasets") == manifest


def test_python_metric_runs_only_ast_restricted_code_in_isolated_subprocess() -> None:
    expected = "def subtract(a, b):\n    return a - b"
    assert _python_unit_test_pass("```python\ndef subtract(a, b):\n    return a - b\n```", expected)
    assert not _python_unit_test_pass("import os\ndef subtract(a, b):\n    return a - b", expected)
    assert not _python_unit_test_pass("def subtract(a, b):\n    return b - a", expected)


def test_adjudication_rejects_behavioral_failure(tmp_path) -> None:
    root = tmp_path / "experiment"
    discovery = root / "discovery_confirmation"
    discovery.mkdir(parents=True)
    (root / "confirmed_experiment_report.json").write_text(
        json.dumps({"experiment_hash": "experiment", "summary": {"broad_hypothesis_supported": True}})
    )
    (discovery / "confirmed_winner_test_metrics.json").write_text(
        json.dumps(
            {
                "test_hash": "behavior",
                "results": {"arithmetic": {"accuracy": 0.49}, "python": {"accuracy": 0.99}},
            }
        )
    )

    report = adjudicate_confirmed_experiment(experiment_dir=root)

    assert report["verdict"] == "REJECTED"
    assert not report["discovery_behavior"]["passed"]
