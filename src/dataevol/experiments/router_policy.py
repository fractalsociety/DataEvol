from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from dataevol.storage import connect, init_db


LOWER_IS_BETTER = {"cost_per_verified_task", "latency_ms", "duplicate_rate", "hallucination_rate"}


@dataclass(frozen=True)
class RouterPolicyExperimentSpec:
    experiment_id: str
    control_version: str
    variant_version: str
    primary_metric: str = "cost_per_verified_task"
    non_regression_metrics: tuple[str, ...] = ("correctness", "verification_pass_rate")
    safety_metric: str = "safety_score"
    reproducibility_requirement: int = 2
    min_primary_relative_improvement: float = 0.0


class RouterPolicyExperimentRunner:
    def run(
        self,
        fixture_metrics: Mapping[str, list[Mapping[str, float]]],
        output_dir: str | Path,
        *,
        spec: RouterPolicyExperimentSpec | None = None,
        rollback_snapshot: str | None = None,
    ) -> dict[str, Any]:
        spec = spec or RouterPolicyExperimentSpec("exp_router_policy_mvp", "control", "variant")
        control = fixture_metrics["control"]
        variant = fixture_metrics["variant"]
        comparison = _compare_metrics(control, variant)
        primary = comparison[spec.primary_metric]
        reproducible_runs = _count_reproducible_primary_runs(control, variant, spec.primary_metric)
        regressions = [
            metric
            for metric in spec.non_regression_metrics
            if metric in comparison and _worse(comparison[metric]["control"], comparison[metric]["variant"], metric)
        ]
        safety_passed = not _worse(comparison[spec.safety_metric]["control"], comparison[spec.safety_metric]["variant"], spec.safety_metric)
        verification_passed = True
        if "verification_pass_rate" in comparison:
            verification_passed = not _worse(comparison["verification_pass_rate"]["control"], comparison["verification_pass_rate"]["variant"], "verification_pass_rate")
        report = {
            "experiment_id": spec.experiment_id,
            "component": "router",
            "control_version": spec.control_version,
            "variant_version": spec.variant_version,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "comparison": comparison,
            "primary_metric": spec.primary_metric,
            "primary_metric_improved": _improved(primary["control"], primary["variant"], spec.primary_metric, spec.min_primary_relative_improvement),
            "regressions": regressions,
            "safety_passed": safety_passed,
            "verification_passed": verification_passed,
            "reproducible_runs": reproducible_runs,
            "reproducibility_requirement": spec.reproducibility_requirement,
            "rollback_snapshot": rollback_snapshot,
            "status": "ready_for_gate",
        }
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{spec.experiment_id}.report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def run_measured_router_policy_experiment(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    run_id: int | None = None,
    rollback_snapshot: str | None = None,
    spec: RouterPolicyExperimentSpec | None = None,
    variant_provider: str = "openrouter",
    min_primary_relative_improvement: float = 0.05,
) -> dict[str, Any]:
    """Run a router experiment from measured traces instead of synthetic fixtures.

    Control metrics are computed from observed trace provider/model choices.
    Variant metrics are computed by applying a conservative routing policy to
    eligible low-risk traces and replacing those traces with measured historical
    performance for the variant provider. If the run has no measured evidence
    for the variant provider, the experiment is rejected as inconclusive.
    """
    spec = spec or RouterPolicyExperimentSpec(
        "exp_router_policy_measured",
        "observed_policy",
        f"{variant_provider}_low_risk_first",
        min_primary_relative_improvement=min_primary_relative_improvement,
    )
    rows = _load_measured_trace_rows(db_path, run_id=run_id)
    if not rows:
        return _write_report(
            {
                "experiment_id": spec.experiment_id,
                "component": "router",
                "control_version": spec.control_version,
                "variant_version": spec.variant_version,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "comparison": {},
                "primary_metric": spec.primary_metric,
                "primary_metric_improved": False,
                "regressions": ["no_measured_trace_data"],
                "safety_passed": False,
                "verification_passed": False,
                "reproducible_runs": 0,
                "reproducibility_requirement": spec.reproducibility_requirement,
                "rollback_snapshot": rollback_snapshot,
                "status": "rejected_no_measured_data",
                "measurement_source": "sqlite",
            },
            output_dir,
        )

    profile = _provider_profile(rows, variant_provider)
    if profile is None:
        return _write_report(
            {
                "experiment_id": spec.experiment_id,
                "component": "router",
                "control_version": spec.control_version,
                "variant_version": spec.variant_version,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "comparison": {},
                "primary_metric": spec.primary_metric,
                "primary_metric_improved": False,
                "regressions": [f"no_measured_{variant_provider}_evidence"],
                "safety_passed": False,
                "verification_passed": False,
                "reproducible_runs": 0,
                "reproducibility_requirement": spec.reproducibility_requirement,
                "rollback_snapshot": rollback_snapshot,
                "status": "rejected_no_variant_evidence",
                "measurement_source": "sqlite",
            },
            output_dir,
        )

    chunks = _chunks(rows, max(2, spec.reproducibility_requirement))
    control_metrics = [_aggregate_metrics(chunk) for chunk in chunks]
    variant_metrics = [_aggregate_metrics(_apply_variant_policy(chunk, profile, variant_provider)) for chunk in chunks]
    comparison = _compare_metrics(control_metrics, variant_metrics)
    primary = comparison[spec.primary_metric]
    regressions = [
        metric
        for metric in spec.non_regression_metrics
        if metric in comparison and _worse(comparison[metric]["control"], comparison[metric]["variant"], metric)
    ]
    safety_passed = spec.safety_metric in comparison and not _worse(
        comparison[spec.safety_metric]["control"],
        comparison[spec.safety_metric]["variant"],
        spec.safety_metric,
    )
    verification_passed = "verification_pass_rate" in comparison and not _worse(
        comparison["verification_pass_rate"]["control"],
        comparison["verification_pass_rate"]["variant"],
        "verification_pass_rate",
    )
    report = {
        "experiment_id": spec.experiment_id,
        "component": "router",
        "control_version": spec.control_version,
        "variant_version": spec.variant_version,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "comparison": comparison,
        "primary_metric": spec.primary_metric,
        "primary_metric_improved": _improved(primary["control"], primary["variant"], spec.primary_metric, spec.min_primary_relative_improvement),
        "regressions": regressions,
        "safety_passed": safety_passed,
        "verification_passed": verification_passed,
        "reproducible_runs": _count_reproducible_primary_runs(control_metrics, variant_metrics, spec.primary_metric),
        "reproducibility_requirement": spec.reproducibility_requirement,
        "rollback_snapshot": rollback_snapshot,
        "status": "ready_for_gate",
        "measurement_source": "sqlite",
        "variant_provider_profile": profile,
        "evaluated_trace_count": len(rows),
    }
    return _write_report(report, output_dir)


def run_router_policy_experiment(
    fixture_metrics: Mapping[str, list[Mapping[str, float]]],
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return RouterPolicyExperimentRunner().run(fixture_metrics, output_dir, **kwargs)


def _write_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report.get('experiment_id', 'experiment')}.report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_measured_trace_rows(db_path: str | Path, *, run_id: int | None) -> list[dict[str, Any]]:
    init_db(db_path)
    where = "WHERE t.run_id = ?" if run_id is not None else ""
    args = (run_id,) if run_id is not None else ()
    query = f"""
        SELECT
          t.id, t.run_id, t.trace_type, t.task_id, t.provider, t.model,
          t.prompt, t.response, t.tests_run, t.raw_path, t.privacy_status,
          l.label,
          s.correctness_score, s.quality_score, s.latency_score, s.cost_score,
          s.safety_score, s.training_value_score
        FROM traces t
        LEFT JOIN labels l ON l.trace_id = t.id
        LEFT JOIN scores s ON s.trace_id = t.id
        {where}
        ORDER BY t.id
    """
    with connect(db_path) as conn:
        rows = [dict(row) for row in conn.execute(query, args).fetchall()]
    return [_with_raw_metrics(row) for row in rows]


def _with_raw_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    raw_path = row.get("raw_path")
    if raw_path:
        try:
            metrics = json.loads(Path(raw_path).read_text(encoding="utf-8")).get("metrics", {})
        except (OSError, json.JSONDecodeError):
            metrics = {}
    label = row.get("label")
    correctness = _optional_float(row.get("correctness_score"))
    if correctness is None:
        correctness = 1.0 if label in {"accepted", "good_training_candidate"} else 0.0
    safety = _float(row.get("safety_score"), 1.0)
    row["measured_cost_usd"] = _float(metrics.get("cost_usd"), 0.0)
    row["measured_latency_ms"] = _float(metrics.get("latency_ms"), 0.0)
    row["measured_correctness"] = correctness
    row["measured_safety"] = safety
    row["measured_verified"] = 1.0 if correctness >= 1.0 and label not in {"hallucinated", "unsafe_or_policy_blocked"} else 0.0
    row["measured_hallucinated"] = 1.0 if label == "hallucinated" else 0.0
    return row


def _provider_profile(rows: list[dict[str, Any]], provider: str) -> dict[str, float] | None:
    candidates = [row for row in rows if str(row.get("provider") or "").lower() == provider.lower()]
    if not candidates:
        return None
    aggregate = _aggregate_metrics(candidates)
    return {
        "provider": provider,
        "sample_count": float(len(candidates)),
        "cost_usd": aggregate["avg_cost_usd"],
        "latency_ms": aggregate["latency_ms"],
        "correctness": aggregate["correctness"],
        "verification_pass_rate": aggregate["verification_pass_rate"],
        "safety_score": aggregate["safety_score"],
        "hallucination_rate": aggregate["hallucination_rate"],
    }


def _apply_variant_policy(rows: list[dict[str, Any]], profile: dict[str, float], provider: str) -> list[dict[str, Any]]:
    variant_rows: list[dict[str, Any]] = []
    for row in rows:
        changed = dict(row)
        if _eligible_for_low_risk_variant(row, provider):
            changed["measured_cost_usd"] = profile["cost_usd"]
            changed["measured_latency_ms"] = profile["latency_ms"]
            changed["measured_correctness"] = profile["correctness"]
            changed["measured_verified"] = profile["verification_pass_rate"]
            changed["measured_safety"] = profile["safety_score"]
            changed["measured_hallucinated"] = profile["hallucination_rate"]
            changed["variant_provider"] = provider
        variant_rows.append(changed)
    return variant_rows


def _eligible_for_low_risk_variant(row: Mapping[str, Any], provider: str) -> bool:
    if str(row.get("provider") or "").lower() == provider.lower():
        return False
    label = row.get("label")
    if label in {"failed_tests", "failed_verification", "hallucinated", "unsafe_or_policy_blocked", "rescued_by_stronger_model"}:
        return False
    text = " ".join(str(row.get(key) or "").lower() for key in ("trace_type", "task_id", "prompt", "response"))
    high_risk_terms = {"unsafe", "clinical", "diagnosis", "legal", "payment", "secret"}
    if any(term in text for term in high_risk_terms):
        return False
    return row.get("trace_type") in {"router_trace", "coding_trace", "planner_trace", "verification_trace", "worker_trace"}


def _aggregate_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "cost_per_verified_task": float("inf"),
            "correctness": 0.0,
            "verification_pass_rate": 0.0,
            "safety_score": 0.0,
            "latency_ms": float("inf"),
            "hallucination_rate": 1.0,
            "avg_cost_usd": float("inf"),
        }
    total_cost = sum(_float(row.get("measured_cost_usd"), 0.0) for row in rows)
    verified = sum(_float(row.get("measured_verified"), 0.0) for row in rows)
    return {
        "cost_per_verified_task": total_cost / verified if verified > 0 else float("inf"),
        "correctness": mean(_float(row.get("measured_correctness"), 0.0) for row in rows),
        "verification_pass_rate": verified / len(rows),
        "safety_score": mean(_float(row.get("measured_safety"), 1.0) for row in rows),
        "latency_ms": mean(_float(row.get("measured_latency_ms"), 0.0) for row in rows),
        "hallucination_rate": mean(_float(row.get("measured_hallucinated"), 0.0) for row in rows),
        "avg_cost_usd": total_cost / len(rows),
    }


def _chunks(rows: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for index, row in enumerate(rows):
        buckets[index % count].append(row)
    return [bucket for bucket in buckets if bucket]


def _float(value: Any, default: float | None) -> float:
    if value is None:
        if default is None:
            raise TypeError("missing float")
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            raise
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare_metrics(control: list[Mapping[str, float]], variant: list[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    metrics = sorted(set().union(*(row.keys() for row in control), *(row.keys() for row in variant)))
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        c = mean(float(row[metric]) for row in control if metric in row)
        v = mean(float(row[metric]) for row in variant if metric in row)
        result[metric] = {"control": c, "variant": v, "delta": v - c}
    return result


def _count_reproducible_primary_runs(control: list[Mapping[str, float]], variant: list[Mapping[str, float]], metric: str) -> int:
    return sum(1 for c, v in zip(control, variant) if metric in c and metric in v and _improved(float(c[metric]), float(v[metric]), metric, 0.0))


def _improved(control: float, variant: float, metric: str, min_relative: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return variant <= control * (1.0 - min_relative)
    return variant >= control * (1.0 + min_relative)


def _worse(control: float, variant: float, metric: str) -> bool:
    if metric in LOWER_IS_BETTER:
        return variant > control
    return variant < control
