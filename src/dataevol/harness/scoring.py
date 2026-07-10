"""Multi-objective scoring, bootstrap statistics, and the evaluation record.

Score:   S = w_q*Q + w_r*R + w_v*V - w_c*norm(C) - w_l*norm(L) - w_f*F
where Q/R/V/F are in [0,1] (higher better) and C/L are min-max normalized
against documented caps so they are commensurate with the [0,1] metrics.
Weights are task-tunable via ScoreWeights.default_for_task(task_type).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SCORE_LOWER_IS_BETTER = {"cost", "latency", "failure_rate"}

# Caps used to normalize cost (USD-equivalent) and latency (ms-equivalent) into
# [0,1]. Exposed (not magic) so behavior is tunable.
COST_CAP = 1.0
LATENCY_CAP = 4000.0


@dataclass(frozen=True)
class ScoreWeights:
    quality: float = 0.34
    robustness: float = 0.20
    verifier_agreement: float = 0.16
    cost: float = 0.10
    latency: float = 0.06
    failure_rate: float = 0.14

    @classmethod
    def default_for_task(cls, task_type: str) -> "ScoreWeights":
        """Task-tuned default weights. Falls back to the balanced default."""
        t = (task_type or "").lower()
        if any(k in t for k in ("medical", "safety", "clinical")):
            # Correctness and abstention dominate.
            return cls(quality=0.42, robustness=0.22, verifier_agreement=0.20, cost=0.05, latency=0.03, failure_rate=0.08)
        if any(k in t for k in ("code", "coding", "software")):
            # Tests passed + low regression matter most.
            return cls(quality=0.40, robustness=0.24, verifier_agreement=0.14, cost=0.06, latency=0.04, failure_rate=0.12)
        if any(k in t for k in ("support", "chat", "fast", "realtime")):
            # Latency-weighted.
            return cls(quality=0.30, robustness=0.16, verifier_agreement=0.10, cost=0.12, latency=0.22, failure_rate=0.10)
        return cls()


def _metric(values: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return default


def composite_score(values: Mapping[str, Any], weights: ScoreWeights) -> float:
    """S = w_q Q + w_r R + w_v V - w_c norm(C) - w_l norm(L) - w_f F."""
    q = _clamp01(_metric(values, "quality"))
    r = _clamp01(_metric(values, "robustness"))
    v = _clamp01(_metric(values, "verifier_agreement"))
    c = max(0.0, _metric(values, "cost"))
    latency = max(0.0, _metric(values, "latency"))
    f = _clamp01(_metric(values, "failure_rate"))
    norm_c = min(c / COST_CAP, 1.0) if COST_CAP > 0 else 0.0
    norm_l = min(latency / LATENCY_CAP, 1.0) if LATENCY_CAP > 0 else 0.0
    return (
        weights.quality * q
        + weights.robustness * r
        + weights.verifier_agreement * v
        - weights.cost * norm_c
        - weights.latency * norm_l
        - weights.failure_rate * f
    )


def normalize_cost(value: float) -> float:
    return min(max(0.0, float(value)) / COST_CAP, 1.0) if COST_CAP > 0 else 0.0


def normalize_latency(value: float) -> float:
    return min(max(0.0, float(value)) / LATENCY_CAP, 1.0) if LATENCY_CAP > 0 else 0.0


def bootstrap_ci(
    control_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> tuple[float, float, float]:
    """Paired bootstrap of (candidate - control) deltas.

    Returns (mean_delta, ci_low, ci_high). A candidate is "statistically better"
    when ci_low > 0 at the chosen confidence level.
    """
    deltas = [float(b) - float(a) for a, b in zip(control_scores, candidate_scores)]
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(max(1, int(samples))):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, min(len(boot_means) - 1, int(alpha * len(boot_means))))
    hi_idx = max(0, min(len(boot_means) - 1, int((1.0 - alpha) * len(boot_means))))
    mean_delta = sum(deltas) / n
    return mean_delta, boot_means[lo_idx], boot_means[hi_idx]


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class HarnessEvaluation:
    genome_id: str
    quality: float
    robustness: float
    verifier_agreement: float
    cost: float
    latency: float
    failure_rate: float
    score: float
    per_category: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    failure_categories: tuple[str, ...] = ()
    run_count: int = 1
    per_run_scores: tuple[float, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "robustness": self.robustness,
            "verifier_agreement": self.verifier_agreement,
            "cost": self.cost,
            "latency": self.latency,
            "failure_rate": self.failure_rate,
            "score": self.score,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "quality": self.quality,
            "robustness": self.robustness,
            "verifier_agreement": self.verifier_agreement,
            "cost": self.cost,
            "latency": self.latency,
            "failure_rate": self.failure_rate,
            "score": self.score,
            "per_category": {k: dict(v) for k, v in self.per_category.items()},
            "failure_categories": list(self.failure_categories),
            "run_count": self.run_count,
            "per_run_scores": list(self.per_run_scores),
            "raw": dict(self.raw),
        }
