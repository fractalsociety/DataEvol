from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataevol.harness.promotion import HarnessPromotionGate, HarnessPromotionThresholds
from dataevol.promotion.gate import PromotionRejected


def _good_report(rollback: Path) -> dict:
    return {
        "genome_id": "chal",
        "incumbent_genome_id": "inc",
        "task_type": "permit_set_review",
        "median_quality_improved": 0.05,
        "quality_delta": 0.05,
        "bootstrap": (0.05, 0.02, 0.08),  # mean_delta, ci_low>0, ci_high
        "critical_benchmark_regressions": [],
        "cost_delta": 0.03,
        "failure_rate_delta": -0.1,
        "reproducible_runs": 3,
        "rollback_snapshot": str(rollback),
        "comparison": {"quality": {"control": 0.5, "variant": 0.6}},
    }


def _write_rollback(path: Path, genome_id: str = "inc") -> None:
    path.write_text(json.dumps({"state": {"genome_id": genome_id}}), encoding="utf-8")


def test_clean_win_promotes_and_writes_file(tmp_path: Path):
    rollback = tmp_path / "rollback.json"
    _write_rollback(rollback)
    gate = HarnessPromotionGate()
    decision = gate.evaluate(_good_report(rollback))
    assert decision.promoted
    promoted = gate.promote(_good_report(rollback), tmp_path / "promotions")
    assert promoted.promotion_path is not None
    assert promoted.promotion_path.exists()


def test_rejects_when_quality_below_threshold(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback)
    report = _good_report(rollback)
    report["median_quality_improved"] = 0.005
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted
    assert any("median quality" in r for r in decision.reasons)


def test_rejects_when_ci_low_not_positive(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback)
    report = _good_report(rollback)
    report["bootstrap"] = (0.01, -0.02, 0.03)
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted
    assert any("bootstrap" in r for r in decision.reasons)


def test_rejects_on_critical_benchmark_regression(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback)
    report = _good_report(rollback)
    report["critical_benchmark_regressions"] = ["long_context"]
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted
    assert any("long_context" in r for r in decision.reasons)


def test_rejects_when_cost_too_high_without_quality_tradeoff(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback)
    report = _good_report(rollback)
    report["cost_delta"] = 0.20
    report["quality_delta"] = 0.03  # not enough to justify cost
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted
    assert any("cost" in r for r in decision.reasons)


def test_allows_high_cost_when_quality_tradeoff_met(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback)
    report = _good_report(rollback)
    report["cost_delta"] = 0.20
    report["quality_delta"] = 0.07  # > tradeoff threshold
    decision = HarnessPromotionGate().evaluate(report)
    assert decision.promoted


def test_rejects_when_failure_rate_increases(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback)
    report = _good_report(rollback)
    report["failure_rate_delta"] = 0.05
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted


def test_rejects_when_judge_is_not_independent(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback)
    report = _good_report(rollback)
    report["judge_independent"] = False
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted
    assert any("judge" in r for r in decision.reasons)


def test_rejects_when_rollback_missing(tmp_path: Path):
    report = _good_report(tmp_path / "does_not_exist.json")
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted
    assert any("rollback" in r for r in decision.reasons)


def test_accepts_durable_rollback_artifact_without_local_file(tmp_path: Path):
    report = _good_report(tmp_path / "not-local.json")
    report["rollback_artifact_hash"] = "a" * 64
    decision = HarnessPromotionGate().evaluate(report)
    assert decision.promoted


def test_rejects_malformed_rollback_artifact_hash(tmp_path: Path):
    report = _good_report(tmp_path / "not-local.json")
    report["rollback_artifact_hash"] = "not-a-sha256"
    decision = HarnessPromotionGate().evaluate(report)
    assert not decision.promoted
    assert any("artifact hash" in reason for reason in decision.reasons)


def test_rejects_rollback_for_a_different_incumbent(tmp_path: Path):
    rollback = tmp_path / "r.json"
    _write_rollback(rollback, genome_id="someone-else")
    decision = HarnessPromotionGate().evaluate(_good_report(rollback))
    assert not decision.promoted
    assert any("does not match" in reason for reason in decision.reasons)


def test_promote_raises_on_rejection(tmp_path: Path):
    report = _good_report(tmp_path / "missing.json")  # no rollback file
    with pytest.raises(PromotionRejected):
        HarnessPromotionGate().promote(report, tmp_path / "promotions")
