from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class PromotionRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: list[str]
    promotion_path: Path | None = None


class PromotionGate:
    def evaluate(self, report: Mapping[str, Any]) -> PromotionDecision:
        reasons: list[str] = []
        if not report.get("primary_metric_improved"):
            reasons.append("primary metric did not improve")
        if report.get("regressions"):
            reasons.append("non-regression metrics worsened: " + ", ".join(report["regressions"]))
        if not report.get("safety_passed"):
            reasons.append("safety checks failed")
        if not report.get("verification_passed"):
            reasons.append("verification pass rate declined")
        if int(report.get("reproducible_runs", 0) or 0) < 2:
            reasons.append("reproducibility requirement not met")
        rollback_snapshot = report.get("rollback_snapshot")
        if not rollback_snapshot or not Path(str(rollback_snapshot)).exists():
            reasons.append("rollback snapshot missing")
        return PromotionDecision(promoted=not reasons, reasons=reasons)

    def promote(self, report: Mapping[str, Any], output_dir: str | Path) -> PromotionDecision:
        decision = self.evaluate(report)
        if not decision.promoted:
            raise PromotionRejected("; ".join(decision.reasons))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"promotion_{report.get('experiment_id', 'experiment')}.json"
        payload = {
            "experiment_id": report.get("experiment_id"),
            "promoted_component": report.get("component", "router"),
            "old_version": report.get("control_version"),
            "new_version": report.get("variant_version"),
            "rollback_path": report.get("rollback_snapshot"),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "metrics": report.get("comparison", {}),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return PromotionDecision(True, [], path)
