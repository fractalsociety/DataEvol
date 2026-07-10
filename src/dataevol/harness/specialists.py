"""The five harness-builder specialists.

Each specialist wraps a ModelClient: it builds a strict-JSON prompt, calls the
model, parses+validates the response, and raises SpecialistError on malformed
output (the loop then skips that candidate). Promotion stays deterministic —
the judge supplies a qualitative review only; the statistical decision is made
by scoring.py + the gate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .genome import MUTATION_MODES, HarnessGenome, new_genome_id
from .model_client import ModelClient
from .scoring import HarnessEvaluation


# Failure taxonomy used by the FailureAnalyst (earliest-causal classification).
FAILURE_TAXONOMY = (
    "TASK_UNDERSTANDING",
    "MISSING_CONTEXT",
    "BAD_ROUTING",
    "TOOL_SELECTION",
    "TOOL_ARGUMENT_ERROR",
    "WORKFLOW_ORDER",
    "REASONING_FAILURE",
    "VERIFICATION_FAILURE",
    "OUTPUT_FORMAT",
    "EXCESSIVE_COST",
    "EXCESSIVE_LATENCY",
    "NONDETERMINISM",
)

BENCHMARK_CATEGORIES = (
    "normal", "edge", "adversarial", "tool_failure", "ambiguous",
    "long_context", "regression", "hidden_holdout",
)


class SpecialistError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_task_spec(task_spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(task_spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _parse_json(content: str, *, label: str) -> Any:
    text = (content or "").strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecialistError(f"{label} returned malformed JSON: {exc}; head={text[:160]!r}") from exc


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecialistError(f"{label} expected a JSON object, got {type(value).__name__}")
    return dict(value)


# --- records produced by specialists ----------------------------------------

@dataclass(frozen=True)
class FailureClassification:
    category: str
    earliest_cause: str
    evidence: str = ""


@dataclass(frozen=True)
class FailureAnalysis:
    failures: tuple[FailureClassification, ...]
    summary: str = ""

    def categories(self) -> tuple[str, ...]:
        return tuple(f.category for f in self.failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failures": [f.__dict__ for f in self.failures],
            "summary": self.summary,
        }


@dataclass(frozen=True)
class MutationProposal:
    hypothesis: str
    mode: str
    target: str | None
    description: str
    patch: Mapping[str, Any] = field(default_factory=dict)
    expected_effect: Mapping[str, float] = field(default_factory=dict)
    affected_tests: tuple[str, ...] = ()

    def mutation_record(self, *, parent_genome_id: str | None = None) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target": self.target,
            "description": self.description,
            "parent_genome_id": parent_genome_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "mode": self.mode,
            "target": self.target,
            "description": self.description,
            "patch": dict(self.patch),
            "expected_effect": dict(self.expected_effect),
            "affected_tests": list(self.affected_tests),
        }


@dataclass(frozen=True)
class JudgeReview:
    verdict: str  # "promotable" | "reject" | "inconclusive"
    reason: str
    confidence: float = 0.0
    independent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "reason": self.reason, "confidence": self.confidence, "independent": self.independent}


# --- specialists -------------------------------------------------------------

GENOME_SCHEMA_BRIEF = """{
  "task_type": str,
  "router": {"model": str, "confidence_threshold": float},
  "agents": [{"role": str, "model": str, "prompt_ref": str, "tools": [str], "cannot_view": [str]}],
  "workflow": [{"step_id": str, "agent_role": str, "depends_on": [str]}],
  "memory": {"type": "none|scratchpad|summary_buffer|structured_state|vector_store"},
  "recovery": {"max_retries": int, "retry_on": [str], "backoff": "fixed|exponential|linear"},
  "output": {"schema": {}, "validation": "strict|lenient"}
}"""


class HarnessArchitect:
    """Converts a task description into an initial harness design."""

    def __init__(self, client: ModelClient, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def design(self, task_spec: Mapping[str, Any]) -> HarnessGenome:
        task_type = str(task_spec.get("task_type") or "general")
        system = (
            "You are the Harness Architect. Convert the task specification into a structured "
            "AI harness genome. Respond with ONLY a JSON object matching this schema:\n"
            + GENOME_SCHEMA_BRIEF
        )
        user = f"task_type={task_type}\ntask_spec={json.dumps(dict(task_spec), sort_keys=True)}"
        result = self.client.complete(system=system, user=user, response_format="json", model=self.model, temperature=0.2)
        data = _as_mapping(_parse_json(result["content"], label="architect"), label="architect")
        data.setdefault("task_type", task_type)
        data["version"] = 1
        data["parent_id"] = None
        data["task_spec_hash"] = hash_task_spec(task_spec)
        data["created_at"] = now_iso()
        return HarnessGenome.from_dict(data)


class BenchmarkBuilder:
    """Creates the evaluation suite (normal/edge/adversarial/.../hidden_holdout).

    Partially isolated from the architect: the hidden_holdout category is
    withheld from the architect and mutator and only seen by the judge/gate.
    """

    def __init__(self, client: ModelClient, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def build(self, task_spec: Mapping[str, Any], *, n_per_category: int = 2) -> list[dict[str, Any]]:
        task_type = str(task_spec.get("task_type") or "general")
        system = (
            "You are the Benchmark Builder. Produce an evaluation suite for the task. "
            'Respond with ONLY a JSON object {"cases": [...]} where each case is '
            '{"id": str, "category": <one of ' + ", ".join(BENCHMARK_CATEGORIES) + ">, \"task\": str, \"expected\": str}. "
            "Include at least one case per category; do NOT omit hidden_holdout."
        )
        user = f"task_type={task_type}\ntask_spec={json.dumps(dict(task_spec), sort_keys=True)}\nn_per_category={n_per_category}"
        result = self.client.complete(system=system, user=user, response_format="json", model=self.model, temperature=0.3)
        parsed = _parse_json(result["content"], label="benchmark")
        if isinstance(parsed, Mapping):
            parsed = parsed.get("cases") or parsed.get("items") or []
        if not isinstance(parsed, list):
            raise SpecialistError("benchmark expected a JSON array of cases")
        cases: list[dict[str, Any]] = []
        for i, raw in enumerate(parsed):
            if not isinstance(raw, Mapping):
                continue
            category = str(raw.get("category") or "normal").lower()
            cases.append({
                "id": str(raw.get("id") or f"{category}_{i}"),
                "category": category,
                "task": str(raw.get("task") or ""),
                "expected": str(raw.get("expected") or ""),
            })
        if not cases:
            raise SpecialistError("benchmark builder returned no cases")
        return cases

    @staticmethod
    def partition_holdout(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train = [c for c in cases if c.get("category") != "hidden_holdout"]
        holdout = [c for c in cases if c.get("category") == "hidden_holdout"]
        return train, holdout


class FailureAnalyst:
    """Examines traces/evaluations and classifies earliest-causal failures."""

    def __init__(self, client: ModelClient, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def analyze(self, genome: HarnessGenome, evaluation: HarnessEvaluation) -> FailureAnalysis:
        system = (
            "You are the Failure Analyst. Classify the EARLIEST causal failure, not merely "
            "where the final answer became wrong. Categories (choose from): "
            + ", ".join(FAILURE_TAXONOMY)
            + ". Respond with ONLY "
            '{"failures": [{"category": str, "earliest_cause": str, "evidence": str}], "summary": str}.'
        )
        user = (
            f"genome={json.dumps(genome.semantic_dict(), sort_keys=True)}\n"
            f"failure_categories={list(evaluation.failure_categories)}\n"
            f"per_category={ {k: dict(v) for k, v in evaluation.per_category.items()} }"
        )
        result = self.client.complete(system=system, user=user, response_format="json", model=self.model, temperature=0.1)
        data = _as_mapping(_parse_json(result["content"], label="failure_analyst"), label="failure_analyst")
        raw_failures = data.get("failures") or []
        if not isinstance(raw_failures, list):
            raw_failures = [raw_failures]
        failures = []
        for item in raw_failures:
            if not isinstance(item, Mapping):
                continue
            failures.append(FailureClassification(
                category=str(item.get("category") or "REASONING_FAILURE"),
                earliest_cause=str(item.get("earliest_cause") or item.get("cause") or ""),
                evidence=str(item.get("evidence") or ""),
            ))
        return FailureAnalysis(failures=tuple(failures), summary=str(data.get("summary", "")))


class HarnessMutator:
    """Proposes targeted, hypothesis-backed mutations."""

    def __init__(self, client: ModelClient, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def propose(
        self,
        incumbent: HarnessGenome,
        failures: FailureAnalysis,
        *,
        number_of_candidates: int = 8,
        strategy: str = "balanced",
        second_parent: HarnessGenome | None = None,
    ) -> list[MutationProposal]:
        system = (
            "You are the Harness Mutator. Propose targeted mutations, each tied to an explicit "
            "hypothesis. Modes (choose from): " + ", ".join(MUTATION_MODES) + ". Respond with ONLY a JSON array of "
            '{"hypothesis": str, "mutation": {"mode": str, "target": str, "description": str}, '
            '"patch": {partial genome JSON to merge}, "expected_effect": {"quality": float, "cost": float}, '
            '"affected_tests": [str]}. patch keys may include router/agents/workflow/memory/recovery/output.'
        )
        user = (
            f"strategy={strategy}\n"
            f"incumbent={json.dumps(incumbent.semantic_dict(), sort_keys=True)}\n"
            f"failures={failures.to_dict()}\n"
            f"number_of_candidates={number_of_candidates}\n"
            f"second_parent={json.dumps(second_parent.semantic_dict(), sort_keys=True) if second_parent else 'null'}"
        )
        result = self.client.complete(system=system, user=user, response_format="json", model=self.model, temperature=0.4)
        parsed = _parse_json(result["content"], label="mutator")
        if isinstance(parsed, Mapping):
            # tolerate a single proposal or a wrapped list
            parsed = parsed.get("proposals") or ([parsed] if "mutation" in parsed or "hypothesis" in parsed else [])
        if not isinstance(parsed, list):
            raise SpecialistError("mutator expected a JSON array of proposals")
        proposals: list[MutationProposal] = []
        for item in parsed:
            if not isinstance(item, Mapping):
                continue
            mut = item.get("mutation") if isinstance(item.get("mutation"), Mapping) else {}
            mode = str(mut.get("mode") or "local")
            proposals.append(MutationProposal(
                hypothesis=str(item.get("hypothesis") or ""),
                mode=mode if mode in MUTATION_MODES else "local",
                target=mut.get("target"),
                description=str(mut.get("description") or ""),
                patch=_as_mapping(item.get("patch") or {}, label="mutator.patch"),
                expected_effect=_as_mapping(item.get("expected_effect") or {}, label="mutator.expected_effect"),
                affected_tests=tuple(item.get("affected_tests") or []),
            ))
        if not proposals:
            raise SpecialistError("mutator returned no proposals")
        return proposals


def apply_mutation(incumbent: HarnessGenome, proposal: MutationProposal, *, created_at: str | None = None) -> HarnessGenome:
    """Merge a proposal's patch into the incumbent to produce a candidate genome."""
    base = incumbent.to_dict()
    merged = _deep_merge(base, proposal.patch)
    merged["genome_id"] = new_genome_id()
    merged["version"] = incumbent.version + 1
    merged["parent_id"] = incumbent.genome_id
    merged["task_spec_hash"] = incumbent.task_spec_hash
    merged["mutation"] = proposal.mutation_record(parent_genome_id=incumbent.genome_id)
    merged["hypothesis"] = proposal.hypothesis
    merged["created_at"] = created_at or now_iso()
    try:
        return HarnessGenome.from_dict(merged)
    except (TypeError, ValueError) as exc:
        raise SpecialistError(f"mutator produced an invalid genome patch: {exc}") from exc


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


class ExperimentJudge:
    """Runs the qualitative comparison. NOT the model that proposed the change."""

    def __init__(self, client: ModelClient, *, judge_model: str | None = None) -> None:
        self.client = client
        self.judge_model = judge_model

    def compare(
        self,
        *,
        incumbent: HarnessEvaluation,
        challenger: HarnessEvaluation,
        bootstrap: tuple[float, float, float],
        mutator_model: str | None = None,
    ) -> JudgeReview:
        independent = bool(self.judge_model) and (mutator_model is None or self.judge_model != mutator_model)
        system = (
            "You are the Experiment Judge. Compare candidate vs incumbent on paired benchmark runs. "
            "Respond with ONLY " + '{"verdict": "promotable|reject|inconclusive", "reason": str, "confidence": float}. '
            "Your review is advisory; promotion also requires statistical confidence."
        )
        user = (
            f"incumbent={json.dumps(incumbent.to_dict(), sort_keys=True)}\n"
            f"challenger={json.dumps(challenger.to_dict(), sort_keys=True)}\n"
            f"bootstrap=(mean_delta={bootstrap[0]:.4f}, ci_low={bootstrap[1]:.4f}, ci_high={bootstrap[2]:.4f})"
        )
        result = self.client.complete(system=system, user=user, response_format="json", model=self.judge_model, temperature=0.1)
        data = _as_mapping(_parse_json(result["content"], label="judge"), label="judge")
        verdict = str(data.get("verdict") or "inconclusive").lower()
        if verdict not in {"promotable", "reject", "inconclusive"}:
            verdict = "inconclusive"
        try:
            confidence = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return JudgeReview(verdict=verdict, reason=str(data.get("reason") or ""), confidence=confidence, independent=independent)
