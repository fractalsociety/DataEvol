"""Lineage + training-record dataclasses for the harness evolver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LineageNode:
    genome_id: str
    parent_id: str | None
    generation: int
    mutation: Mapping[str, Any] = field(default_factory=dict)
    hypothesis: str | None = None
    benchmark_delta: Mapping[str, float] = field(default_factory=dict)
    cost_delta: float = 0.0
    failed_categories_improved: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()
    promoted: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "parent_genome_id": self.parent_id,
            "generation": self.generation,
            "mutation": dict(self.mutation),
            "hypothesis": self.hypothesis,
            "benchmark_delta": dict(self.benchmark_delta),
            "cost_delta": self.cost_delta,
            "failed_categories_improved": list(self.failed_categories_improved),
            "regressions": list(self.regressions),
            "promoted": self.promoted,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ExperimentRecord:
    """Training record emitted by every experiment (for later fine-tuning)."""

    genome_id: str
    task_features: Mapping[str, Any]
    parent_harness: Mapping[str, Any]
    failure_analysis: Mapping[str, Any]
    proposed_mutation: Mapping[str, Any]
    mutation_hypothesis: str
    candidate_harness: Mapping[str, Any]
    benchmark_results: Mapping[str, Any]
    cost_results: Mapping[str, Any]
    promotion_decision: str  # "promoted" | "rejected"
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "task_features": dict(self.task_features),
            "parent_harness": dict(self.parent_harness),
            "failure_analysis": dict(self.failure_analysis),
            "proposed_mutation": dict(self.proposed_mutation),
            "mutation_hypothesis": self.mutation_hypothesis,
            "candidate_harness": dict(self.candidate_harness),
            "benchmark_results": dict(self.benchmark_results),
            "cost_results": dict(self.cost_results),
            "promotion_decision": self.promotion_decision,
            "decision_reason": self.decision_reason,
        }
