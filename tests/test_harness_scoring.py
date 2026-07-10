from __future__ import annotations

from dataevol.harness.scoring import (
    SCORE_LOWER_IS_BETTER,
    ScoreWeights,
    bootstrap_ci,
    composite_score,
    median,
    normalize_cost,
)


def _metrics(**kw):
    base = dict(quality=0.7, robustness=0.6, verifier_agreement=0.6, cost=0.2, latency=800.0, failure_rate=0.1)
    base.update(kw)
    return base


def test_composite_is_monotonic_in_quality_and_cost():
    w = ScoreWeights()
    base = composite_score(_metrics(), w)
    higher_q = composite_score(_metrics(quality=0.9), w)
    higher_c = composite_score(_metrics(cost=0.8), w)
    assert higher_q > base
    assert higher_c < base, "raising cost must lower the composite score"


def test_score_lower_is_better_set():
    assert SCORE_LOWER_IS_BETTER == {"cost", "latency", "failure_rate"}


def test_default_weights_sum_to_one():
    s = sum(vars(ScoreWeights()).values())
    assert abs(s - 1.0) < 1e-9
    assert ScoreWeights.default_for_task("medical safety").quality > ScoreWeights().quality
    assert ScoreWeights.default_for_task("fast chat support").latency > ScoreWeights().latency


def test_normalize_cost_latency_bounded():
    assert normalize_cost(0) == 0.0
    assert normalize_cost(2.0) == 1.0
    assert 0.0 < normalize_cost(0.5) < 1.0


def test_bootstrap_ci_clearly_better_excludes_zero():
    control = [0.50, 0.51, 0.49, 0.50, 0.52]
    candidate = [0.72, 0.66, 0.78, 0.69, 0.75]
    mean_delta, lo, hi = bootstrap_ci(control, candidate, samples=2000, confidence=0.95, seed=1)
    assert mean_delta > 0
    assert lo > 0, "clearly-better candidate should have ci_low > 0"
    assert hi > lo


def test_bootstrap_ci_tied_straddles_zero():
    control = [0.5, 0.5, 0.5, 0.5]
    candidate = [0.5, 0.5, 0.5, 0.5]
    _, lo, hi = bootstrap_ci(control, candidate, samples=500, confidence=0.95, seed=2)
    assert lo <= 0.0 <= hi


def test_median_helper():
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert median([]) == 0.0
