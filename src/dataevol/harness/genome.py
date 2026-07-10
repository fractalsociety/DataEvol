"""Structured, machine-editable harness genome.

A genome is the full specification of an AI harness: routing, agents, workflow
graph, memory, recovery policy, and output schema. Each component is
independently mutable, so experiments change one axis at a time rather than
rewriting the whole harness. The genome's ``content_hash`` ignores ancestry
fields (genome_id/parent/version/mutation/hypothesis/created_at) so two genomes
that differ only by lineage dedup equal.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Mapping


# Mutation operators available to the HarnessMutator. ``target`` (a dotted
# component path such as "router.confidence_threshold") records which axis a
# local/component mutation touched; structural/crossover/restart use target=None.
MUTATION_MODES = (
    "local",          # revise one prompt/threshold
    "component",      # replace router/verifier/memory
    "structural",     # change the agent/workflow graph
    "model",          # swap model or quantization
    "crossover",      # combine strong components from two harnesses
    "restart",        # fresh architect design
    "simplification", # remove a component to test if it was necessary
)


def new_genome_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class RouterSpec:
    model: str
    confidence_threshold: float
    fallback_model: str | None = None
    policy: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "confidence_threshold": self.confidence_threshold,
            "fallback_model": self.fallback_model,
            "policy": dict(self.policy),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouterSpec":
        return cls(
            model=str(data.get("model", "")),
            confidence_threshold=float(data.get("confidence_threshold", 0.5)),
            fallback_model=data.get("fallback_model"),
            policy=dict(data.get("policy") or {}),
        )


@dataclass(frozen=True)
class AgentSpec:
    role: str
    model: str
    prompt_ref: str
    tools: tuple[str, ...] = ()
    cannot_view: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "prompt_ref": self.prompt_ref,
            "tools": list(self.tools),
            "cannot_view": list(self.cannot_view),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentSpec":
        return cls(
            role=str(data.get("role", "")),
            model=str(data.get("model", "")),
            prompt_ref=str(data.get("prompt_ref", "")),
            tools=tuple(data.get("tools") or []),
            cannot_view=tuple(data.get("cannot_view") or []),
            params=dict(data.get("params") or {}),
        )


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    agent_role: str
    depends_on: tuple[str, ...] = ()
    description: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_role": self.agent_role,
            "depends_on": list(self.depends_on),
            "description": self.description,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowStep":
        return cls(
            step_id=str(data.get("step_id", "")),
            agent_role=str(data.get("agent_role", "")),
            depends_on=tuple(data.get("depends_on") or []),
            description=str(data.get("description", "")),
            config=dict(data.get("config") or {}),
        )


@dataclass(frozen=True)
class MemorySpec:
    type: str = "none"  # none | scratchpad | summary_buffer | structured_state | vector_store
    schema: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "schema": dict(self.schema)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemorySpec":
        return cls(type=str(data.get("type", "none")), schema=dict(data.get("schema") or {}))


@dataclass(frozen=True)
class RecoverySpec:
    max_retries: int = 0
    retry_on: tuple[str, ...] = ()  # failure-taxonomy category names
    backoff: str = "fixed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "retry_on": list(self.retry_on),
            "backoff": self.backoff,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecoverySpec":
        return cls(
            max_retries=int(data.get("max_retries", 0)),
            retry_on=tuple(data.get("retry_on") or []),
            backoff=str(data.get("backoff", "fixed")),
        )


@dataclass(frozen=True)
class OutputSchemaSpec:
    schema: Mapping[str, Any] = field(default_factory=dict)
    validation: str = "strict"  # strict | lenient

    def to_dict(self) -> dict[str, Any]:
        return {"schema": dict(self.schema), "validation": self.validation}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputSchemaSpec":
        return cls(
            schema=dict(data.get("schema") or {}),
            validation=str(data.get("validation", "strict")),
        )


@dataclass(frozen=True)
class MutationRecord:
    mode: str
    target: str | None
    description: str
    parent_genome_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target": self.target,
            "description": self.description,
            "parent_genome_id": self.parent_genome_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutationRecord":
        return cls(
            mode=str(data.get("mode", "local")),
            target=data.get("target"),
            description=str(data.get("description", "")),
            parent_genome_id=data.get("parent_genome_id"),
        )


# Fields excluded from the content hash so ancestry/metadata do not affect dedup.
_LINEAGE_FIELDS = frozenset({
    "genome_id", "parent_id", "version", "mutation", "hypothesis", "created_at"
})


@dataclass(frozen=True)
class HarnessGenome:
    genome_id: str
    version: int
    parent_id: str | None
    task_type: str
    task_spec_hash: str
    router: RouterSpec
    agents: tuple[AgentSpec, ...]
    workflow: tuple[WorkflowStep, ...]
    memory: MemorySpec
    recovery: RecoverySpec
    output: OutputSchemaSpec
    mutation: MutationRecord | None = None
    hypothesis: str | None = None
    created_at: str = ""

    def semantic_dict(self) -> dict[str, Any]:
        """Dict of content-bearing fields only (excludes ancestry/metadata)."""
        return {
            "task_type": self.task_type,
            "task_spec_hash": self.task_spec_hash,
            "router": self.router.to_dict(),
            "agents": [a.to_dict() for a in self.agents],
            "workflow": [s.to_dict() for s in self.workflow],
            "memory": self.memory.to_dict(),
            "recovery": self.recovery.to_dict(),
            "output": self.output.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "version": self.version,
            "parent_id": self.parent_id,
            "task_type": self.task_type,
            "task_spec_hash": self.task_spec_hash,
            "router": self.router.to_dict(),
            "agents": [a.to_dict() for a in self.agents],
            "workflow": [s.to_dict() for s in self.workflow],
            "memory": self.memory.to_dict(),
            "recovery": self.recovery.to_dict(),
            "output": self.output.to_dict(),
            "mutation": self.mutation.to_dict() if self.mutation else None,
            "hypothesis": self.hypothesis,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HarnessGenome":
        mutation_data = data.get("mutation")
        return cls(
            genome_id=str(data.get("genome_id") or new_genome_id()),
            version=int(data.get("version", 1)),
            parent_id=data.get("parent_id"),
            task_type=str(data.get("task_type", "")),
            task_spec_hash=str(data.get("task_spec_hash", "")),
            router=RouterSpec.from_dict(data.get("router") or {}),
            agents=tuple(AgentSpec.from_dict(a) for a in (data.get("agents") or [])),
            workflow=tuple(WorkflowStep.from_dict(s) for s in (data.get("workflow") or [])),
            memory=MemorySpec.from_dict(data.get("memory") or {}),
            recovery=RecoverySpec.from_dict(data.get("recovery") or {}),
            output=OutputSchemaSpec.from_dict(data.get("output") or {}),
            mutation=MutationRecord.from_dict(mutation_data) if mutation_data else None,
            hypothesis=data.get("hypothesis"),
            created_at=str(data.get("created_at", "")),
        )

    def with_component(self, **overrides: Any) -> "HarnessGenome":
        """Return a copy with one or more component fields replaced (used by mutator)."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(overrides)
        return HarnessGenome(**current)
