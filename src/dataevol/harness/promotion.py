"""Harness promotion gate (stricter, multi-objective).

Same shape as promotion/gate.py (PromotionDecision/PromotionRejected) but with
the spec's full promotion rules: median quality improvement, bootstrap CI,
no critical benchmark regression, cost/latency limits, failure-rate limit,
reproducibility, and rollback snapshot. The statistical decision is computed
deterministically from the report dict; the judge's qualitative review is
recorded as ``decision_reason`` only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dataevol.promotion.gate import PromotionRejected


@dataclass(frozen=True)
class HarnessPromotionThresholds:
    min_quality_improvement: float = 0.02
    bootstrap_confidence: float = 0.95
    max_critical_benchmark_drop: float = 0.01
    max_cost_increase: float = 0.10
    quality_cost_tradeoff: float = 0.05  # allow >max_cost_increase if quality > +this
    reproducibility_requirement: int = 2


@dataclass(frozen=True)
class HarnessPromotionDecision:
    promoted: bool
    reasons: list[str]
    promotion_path: Path | None = None


class HarnessPromotionGate:
    def __init__(self, thresholds: HarnessPromotionThresholds | None = None) -> None:
        self.thresholds = thresholds or HarnessPromotionThresholds()

    def evaluate(self, report: Mapping[str, Any]) -> HarnessPromotionDecision:
        t = self.thresholds
        reasons: list[str] = []

        median_quality_improved = float(report.get("median_quality_improved", 0.0) or 0.0)
        quality_delta = float(report.get("quality_delta", median_quality_improved) or 0.0)
        if median_quality_improved < t.min_quality_improvement:
            reasons.append(
                f"median quality improved only {median_quality_improved:.1%} (< {t.min_quality_improvement:.0%})"
            )

        bootstrap = report.get("bootstrap") or (0.0, 0.0, 0.0)
        try:
            ci_low = float(bootstrap[1])
            mean_delta = float(bootstrap[0])
        except (TypeError, IndexError, ValueError):
            ci_low, mean_delta = 0.0, 0.0
        bootstrap_confidence = float(report.get("bootstrap_confidence", t.bootstrap_confidence) or 0.0)
        if bootstrap_confidence < t.bootstrap_confidence:
            reasons.append(
                f"bootstrap confidence {bootstrap_confidence:.0%} below required {t.bootstrap_confidence:.0%}"
            )
        if not (mean_delta > 0 and ci_low > 0):
            reasons.append(f"bootstrap CI does not confirm improvement (mean_delta={mean_delta:.4f}, ci_low={ci_low:.4f})")

        if report.get("judge_independent") is False:
            reasons.append("judge is not independent from mutator")

        critical = list(report.get("critical_benchmark_regressions") or [])
        if critical:
            reasons.append("critical benchmark regressed: " + ", ".join(critical))

        cost_delta = float(report.get("cost_delta", 0.0) or 0.0)
        if cost_delta > t.max_cost_increase and quality_delta <= t.quality_cost_tradeoff:
            reasons.append(
                f"cost increased {cost_delta:.1%} (> {t.max_cost_increase:.0%} without >{t.quality_cost_tradeoff:.0%} quality gain)"
            )

        failure_rate_delta = float(report.get("failure_rate_delta", 0.0) or 0.0)
        if failure_rate_delta > 0:
            reasons.append(f"failure rate increased by {failure_rate_delta:.1%}")

        reproducible_runs = int(report.get("reproducible_runs", 0) or 0)
        if reproducible_runs < t.reproducibility_requirement:
            reasons.append(f"reproducibility requirement not met ({reproducible_runs} < {t.reproducibility_requirement})")

        rollback_artifact_hash = report.get("rollback_artifact_hash")
        rollback_snapshot = report.get("rollback_snapshot")
        if rollback_artifact_hash is not None:
            if not isinstance(rollback_artifact_hash, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", rollback_artifact_hash
            ):
                reasons.append("rollback artifact hash must be a SHA-256")
        elif not rollback_snapshot or not Path(str(rollback_snapshot)).is_file():
            reasons.append("rollback snapshot or durable artifact hash missing")
        else:
            try:
                snapshot = json.loads(Path(str(rollback_snapshot)).read_text(encoding="utf-8"))
                state = snapshot.get("state") if isinstance(snapshot, Mapping) else None
                if not isinstance(state, Mapping):
                    reasons.append("rollback snapshot does not contain restorable state")
                elif report.get("incumbent_genome_id") and state.get("genome_id") != report.get("incumbent_genome_id"):
                    reasons.append("rollback snapshot does not match incumbent genome")
            except (OSError, json.JSONDecodeError):
                reasons.append("rollback snapshot is unreadable")

        return HarnessPromotionDecision(promoted=not reasons, reasons=reasons)

    def promote(self, report: Mapping[str, Any], output_dir: str | Path) -> HarnessPromotionDecision:
        decision = self.evaluate(report)
        if not decision.promoted:
            raise PromotionRejected("; ".join(decision.reasons))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        genome_id = str(report.get("genome_id") or report.get("challenger_genome_id") or "candidate")
        path = out / f"promotion_{genome_id}.json"
        payload = {
            "genome_id": genome_id,
            "parent_genome_id": report.get("incumbent_genome_id"),
            "task_type": report.get("task_type"),
            "old_version": report.get("incumbent_version"),
            "new_version": report.get("challenger_version"),
            "rollback_path": str(report.get("rollback_snapshot")),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "bootstrap": list(report.get("bootstrap") or (0.0, 0.0, 0.0)),
            "metrics": report.get("comparison") or {},
            "decision_reason": report.get("decision_reason", ""),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return HarnessPromotionDecision(True, [], path)
