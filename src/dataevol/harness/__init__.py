"""Fractal Harness Evolver.

Generates, tests, diagnoses, mutates, and selects executable AI harnesses.
Every proposed change states a hypothesis, runs against a reproducible frozen
benchmark, compares paired against the incumbent, and is promoted only when it
produces a reliable improvement without unacceptable regressions.

Interacts with — but stays separate from — the DataEvol data/weights pipeline.
"""
from __future__ import annotations

from .genome import (
    MUTATION_MODES,
    AgentSpec,
    HarnessGenome,
    MemorySpec,
    MutationRecord,
    OutputSchemaSpec,
    RecoverySpec,
    RouterSpec,
    WorkflowStep,
    new_genome_id,
)
from .scoring import (
    COST_CAP,
    LATENCY_CAP,
    HarnessEvaluation,
    ScoreWeights,
    bootstrap_ci,
    composite_score,
    median,
)
from .executor import HarnessExecutor, ReferenceExecutor, parallel_evaluate
from . import storage

__all__ = [
    "MUTATION_MODES",
    "AgentSpec",
    "HarnessExecutor",
    "HarnessEvaluation",
    "HarnessGenome",
    "MemorySpec",
    "MutationRecord",
    "OutputSchemaSpec",
    "ReferenceExecutor",
    "RecoverySpec",
    "RouterSpec",
    "ScoreWeights",
    "WorkflowStep",
    "bootstrap_ci",
    "composite_score",
    "new_genome_id",
    "parallel_evaluate",
    "storage",
]
