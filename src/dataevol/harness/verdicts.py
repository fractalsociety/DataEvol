"""Binding evaluation verdicts issued by the Harness Evolver.

The verdict is deliberately narrower than deployment state. DataEvol may
authorize a canary, reject a candidate, or decline to decide; it never marks a
harness canary or active.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .execution_contract import is_real_executor_kind
from .promotion import HarnessPromotionGate


VERDICT_SCHEMA = "dataevol.harness_verdict.v1"
VERDICTS = frozenset({"ELIGIBLE", "REJECTED", "INCONCLUSIVE"})
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_STATISTICAL_FIELDS = (
    "median_quality_improved",
    "quality_delta",
    "bootstrap",
    "bootstrap_confidence",
    "critical_benchmark_regressions",
    "cost_delta",
    "failure_rate_delta",
    "reproducible_runs",
    "judge_independent",
)


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize a verdict payload with one stable, cross-service encoding."""
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _verdict_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _content_hash(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if not _HASH_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return text.lower()


def _created_at(value: Any = None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    text = _required_text(value, "created_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return text


def is_simulation_executor(executor_kind: str) -> bool:
    """Fail closed for the built-in reference evaluator and simulations."""
    normalized = re.sub(r"[^a-z0-9]", "", executor_kind.lower())
    return normalized.startswith("reference") or "simulat" in normalized


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def incomplete_statistical_evidence(report: Mapping[str, Any]) -> list[str]:
    """Return malformed or absent inputs that make the promotion gate unsafe."""
    issues: list[str] = []
    for field in _STATISTICAL_FIELDS:
        if field not in report:
            issues.append(f"missing {field}")

    for field in ("median_quality_improved", "quality_delta", "cost_delta", "failure_rate_delta"):
        if field in report and not _finite_number(report[field]):
            issues.append(f"{field} must be a finite number")

    bootstrap = report.get("bootstrap")
    if "bootstrap" in report:
        if not isinstance(bootstrap, (list, tuple)) or len(bootstrap) != 3 or not all(_finite_number(v) for v in bootstrap):
            issues.append("bootstrap must contain three finite numbers")

    confidence = report.get("bootstrap_confidence")
    if "bootstrap_confidence" in report and (
        not _finite_number(confidence) or not 0.0 < float(confidence) <= 1.0
    ):
        issues.append("bootstrap_confidence must be in (0, 1]")

    regressions = report.get("critical_benchmark_regressions")
    if "critical_benchmark_regressions" in report and (
        not isinstance(regressions, (list, tuple)) or not all(isinstance(item, str) for item in regressions)
    ):
        issues.append("critical_benchmark_regressions must be a list of strings")

    runs = report.get("reproducible_runs")
    if "reproducible_runs" in report and (
        isinstance(runs, bool) or not isinstance(runs, int) or runs < 0
    ):
        issues.append("reproducible_runs must be a non-negative integer")

    independent = report.get("judge_independent")
    if "judge_independent" in report and not isinstance(independent, bool):
        issues.append("judge_independent must be a boolean")

    return issues


@dataclass(frozen=True)
class HarnessVerdict:
    schema: str
    verdict_id: str
    verdict: str
    task_type: str
    incumbent_genome_id: str
    candidate_genome_id: str
    candidate_content_hash: str
    benchmark_hash: str
    evidence_hash: str
    executor_kind: str
    reasons: tuple[str, ...]
    created_at: str
    verdict_hash: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict_id": self.verdict_id,
            "verdict": self.verdict,
            "task_type": self.task_type,
            "incumbent_genome_id": self.incumbent_genome_id,
            "candidate_genome_id": self.candidate_genome_id,
            "candidate_content_hash": self.candidate_content_hash,
            "benchmark_hash": self.benchmark_hash,
            "evidence_hash": self.evidence_hash,
            "executor_kind": self.executor_kind,
            "reasons": list(self.reasons),
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "verdict_hash": self.verdict_hash}

    def verify_hash(self) -> bool:
        return self.verdict_hash == _verdict_hash(self.unsigned_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, verify_hash: bool = True) -> "HarnessVerdict":
        verdict = _required_text(value.get("verdict"), "verdict").upper()
        if verdict not in VERDICTS:
            raise ValueError("verdict must be ELIGIBLE, REJECTED, or INCONCLUSIVE")
        reasons = value.get("reasons")
        if not isinstance(reasons, (list, tuple)) or not all(isinstance(reason, str) for reason in reasons):
            raise ValueError("reasons must be a list of strings")
        result = cls(
            schema=_required_text(value.get("schema"), "schema"),
            verdict_id=_required_text(value.get("verdict_id"), "verdict_id"),
            verdict=verdict,
            task_type=_required_text(value.get("task_type"), "task_type"),
            incumbent_genome_id=_required_text(value.get("incumbent_genome_id"), "incumbent_genome_id"),
            candidate_genome_id=_required_text(value.get("candidate_genome_id"), "candidate_genome_id"),
            candidate_content_hash=_content_hash(value.get("candidate_content_hash"), "candidate_content_hash"),
            benchmark_hash=_content_hash(value.get("benchmark_hash"), "benchmark_hash"),
            evidence_hash=_content_hash(value.get("evidence_hash"), "evidence_hash"),
            executor_kind=_required_text(value.get("executor_kind"), "executor_kind"),
            reasons=tuple(reasons),
            created_at=_created_at(_required_text(value.get("created_at"), "created_at")),
            verdict_hash=_content_hash(value.get("verdict_hash"), "verdict_hash"),
        )
        if result.schema != VERDICT_SCHEMA:
            raise ValueError(f"schema must be {VERDICT_SCHEMA}")
        if verify_hash and not result.verify_hash():
            raise ValueError("verdict_hash does not match the canonical verdict payload")
        return result


def issue_harness_verdict(
    payload: Mapping[str, Any],
    *,
    gate: HarnessPromotionGate | None = None,
) -> HarnessVerdict:
    """Issue a binding verdict from executor provenance and a gate report."""
    executor_kind = _required_text(payload.get("executor_kind"), "executor_kind")
    raw_report = payload.get("report")
    report = dict(raw_report) if isinstance(raw_report, Mapping) else {}

    reasons: list[str]
    if is_simulation_executor(executor_kind):
        verdict = "INCONCLUSIVE"
        reasons = ["simulation and ReferenceExecutor evidence cannot authorize a production canary"]
    elif not is_real_executor_kind(executor_kind):
        # Closed allowlist: unknown executor kinds fail closed as non-real.
        verdict = "INCONCLUSIVE"
        reasons = [f"executor kind '{executor_kind}' is not on the real-executor allowlist"]
    elif not isinstance(raw_report, Mapping):
        verdict = "INCONCLUSIVE"
        reasons = ["incomplete statistical evidence: report must be an object"]
    else:
        evidence_issues = incomplete_statistical_evidence(report)
        if evidence_issues:
            verdict = "INCONCLUSIVE"
            reasons = [f"incomplete statistical evidence: {issue}" for issue in evidence_issues]
        else:
            report_executor = report.get("executor_kind")
            if report_executor is not None and str(report_executor).strip() != executor_kind:
                verdict = "INCONCLUSIVE"
                reasons = ["executor_kind conflicts with the statistical report provenance"]
            else:
                decision = (gate or HarnessPromotionGate()).evaluate(report)
                verdict = "ELIGIBLE" if decision.promoted else "REJECTED"
                reasons = list(decision.reasons)

    unsigned = {
        "schema": VERDICT_SCHEMA,
        "verdict_id": _required_text(payload.get("verdict_id") or f"hv_{uuid.uuid4().hex}", "verdict_id"),
        "verdict": verdict,
        "task_type": _required_text(payload.get("task_type"), "task_type"),
        "incumbent_genome_id": _required_text(payload.get("incumbent_genome_id"), "incumbent_genome_id"),
        "candidate_genome_id": _required_text(payload.get("candidate_genome_id"), "candidate_genome_id"),
        "candidate_content_hash": _content_hash(payload.get("candidate_content_hash"), "candidate_content_hash"),
        "benchmark_hash": _content_hash(payload.get("benchmark_hash"), "benchmark_hash"),
        "evidence_hash": _content_hash(payload.get("evidence_hash"), "evidence_hash"),
        "executor_kind": executor_kind,
        "reasons": reasons,
        "created_at": _created_at(payload.get("created_at")),
    }
    return HarnessVerdict.from_dict({**unsigned, "verdict_hash": _verdict_hash(unsigned)})


__all__ = [
    "HarnessVerdict",
    "VERDICT_SCHEMA",
    "VERDICTS",
    "canonical_json",
    "incomplete_statistical_evidence",
    "is_simulation_executor",
    "issue_harness_verdict",
]
