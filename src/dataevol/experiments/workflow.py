from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dataevol.promotion import PromotionGate, PromotionRejected


def freeze_benchmark_for_experiment(benchmark_path: str | Path, experiment_dir: str | Path) -> Path:
    src = Path(benchmark_path)
    out = Path(experiment_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "frozen_benchmark.jsonl"
    dest.write_text(src.read_text(encoding="utf-8") if src.exists() else "", encoding="utf-8")
    return dest


def compare_experiment(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    comparison = {
        "experiment_id": report.get("experiment_id"),
        "primary_metric": report.get("primary_metric"),
        "primary_metric_improved": bool(report.get("primary_metric_improved")),
        "regressions": list(report.get("regressions") or []),
        "safety_passed": bool(report.get("safety_passed")),
        "verification_passed": bool(report.get("verification_passed")),
        "verdict": "promotable" if _promotable(report) else "reject",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{comparison['experiment_id'] or 'experiment'}.compare.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison["path"] = str(path)
    return comparison


def create_rollback_snapshot(
    component: str,
    version: str,
    output_dir: str | Path,
    *,
    state: Mapping[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{component}_{version}_rollback.json"
    payload = {
        "component": component,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if state is not None:
        payload["state"] = dict(state)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def promote_experiment(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    decision = PromotionGate().promote(report, output_dir)
    return {"promoted": True, "promotion_path": str(decision.promotion_path)}


def reject_experiment(report: Mapping[str, Any], output_dir: str | Path, *, reason: str | None = None) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rejected_{report.get('experiment_id', 'experiment')}.json"
    payload = {
        "experiment_id": report.get("experiment_id"),
        "reason": reason or "promotion gate failed",
        "negative_evidence": report,
        "rejected_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rejected": True, "path": str(path)}


def _promotable(report: Mapping[str, Any]) -> bool:
    try:
        return PromotionGate().evaluate(report).promoted
    except (TypeError, ValueError, PromotionRejected):
        return False
