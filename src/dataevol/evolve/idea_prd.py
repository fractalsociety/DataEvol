from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


REQUIRED_SECTIONS = [
    "Observation",
    "Hypothesis",
    "Affected Component",
    "Baseline",
    "Variant",
    "Benchmark",
    "Primary Metric",
    "Non-Regression Metrics",
    "Safety Checks",
    "Reproducibility Requirement",
    "Rollback Plan",
    "Promotion Rule",
    "Rejection Rule",
]

COMPONENT_DEFAULTS = {
    "router": ("Current router policy.", "Frozen router policy benchmark.", "cost_per_verified_task"),
    "prompt": ("Current prompt pack.", "Frozen prompt benchmark.", "task_success_rate"),
    "verifier": ("Current verifier policy.", "Frozen verifier benchmark.", "verification_pass_rate"),
    "local_model": ("Current local dataset/export.", "Frozen local model evaluation benchmark.", "quality_score"),
    "benchmark": ("Current benchmark suite.", "Benchmark coverage audit.", "regression_rate"),
}


def generate_idea_prd(
    opportunity: Mapping[str, Any],
    *,
    title: str | None = None,
    affected_component: str = "router",
    baseline: str = "Current router policy.",
    variant: str | None = None,
    benchmark: str = "Frozen router policy benchmark.",
    primary_metric: str | None = None,
    non_regression_metrics: list[str] | None = None,
    reproducibility_requirement: int = 2,
) -> str:
    metric = primary_metric or str(opportunity.get("expected_metric") or "cost_per_verified_task")
    variant_text = variant or str(opportunity.get("proposed_change") or "Proposed change from opportunity detector.")
    non_regression = ", ".join(non_regression_metrics or ["correctness", "verification_pass_rate", "safety_score"])
    heading = title or str(opportunity.get("id") or "Router Policy Improvement")
    return "\n".join(
        [
            f"# Idea PRD: {heading}",
            "",
            "## Observation",
            str(opportunity.get("observation") or "No observation supplied."),
            "",
            "## Hypothesis",
            str(opportunity.get("hypothesis") or "No hypothesis supplied."),
            "",
            "## Affected Component",
            affected_component,
            "",
            "## Baseline",
            baseline,
            "",
            "## Variant",
            variant_text,
            "",
            "## Benchmark",
            benchmark,
            "",
            "## Primary Metric",
            metric,
            "",
            "## Non-Regression Metrics",
            non_regression,
            "",
            "## Safety Checks",
            "Safety score must not decline and safety regressions must equal 0.",
            "",
            "## Reproducibility Requirement",
            str(reproducibility_requirement),
            "",
            "## Rollback Plan",
            "Restore the previous router policy snapshot recorded before promotion.",
            "",
            "## Promotion Rule",
            "Promote only if the primary metric improves, non-regression metrics do not worsen, safety passes, reproducibility is >= 2, and rollback snapshot exists.",
            "",
            "## Rejection Rule",
            "Reject if any promotion condition fails.",
            "",
        ]
    )


def generate_component_idea_prd(opportunity: Mapping[str, Any], component: str) -> str:
    baseline, benchmark, metric = COMPONENT_DEFAULTS[component]
    return generate_idea_prd(
        opportunity,
        affected_component=component,
        baseline=baseline,
        benchmark=benchmark,
        primary_metric=str(opportunity.get("expected_metric") or metric),
    )


def save_idea_prd(prd: str, output_dir: str | Path, *, slug: str = "idea_prd") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{slug}.md"
    path.write_text(prd, encoding="utf-8")
    return path


def validate_idea_prd(prd: str | Path) -> tuple[bool, list[str]]:
    text = Path(prd).read_text(encoding="utf-8") if isinstance(prd, Path) else prd
    missing: list[str] = []
    if not text.lstrip().startswith("# Idea PRD:"):
        missing.append("Title")
    for section in REQUIRED_SECTIONS:
        marker = f"## {section}"
        if marker not in text:
            missing.append(section)
            continue
        body = _section_body(text, marker)
        if not body.strip():
            missing.append(f"{section} content")
    return not missing, missing


def _section_body(text: str, marker: str) -> str:
    start = text.index(marker) + len(marker)
    rest = text[start:]
    next_section = rest.find("\n## ")
    return rest if next_section == -1 else rest[:next_section]
