"""Typed harness compilation and deterministic activation routing.

The external teacher is allowed to author rich instructions, but the 1B worker
only receives a compact, validated state machine. Hard invariants remain data
for the deterministic controller rather than prose the worker may ignore.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .model_client import ModelClient


COMPILED_HARNESS_SCHEMA = "dataevol.compiled_harness.v1"
HARNESS_STATUSES = frozenset({"active", "archived", "quarantined"})
ACTION_TYPES = frozenset({"tool", "check", "complete", "escalate"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _identifier(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"{field} must contain only letters, numbers, dot, underscore, or hyphen")
    return text


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class HarnessTriggers:
    task_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    error_classes: tuple[str, ...] = ()
    repository_paths: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    risk_levels: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessTriggers":
        return cls(**{
            field: _string_tuple(value.get(field), f"triggers.{field}")
            for field in cls.__dataclass_fields__
        })

    def to_dict(self) -> dict[str, Any]:
        return {field: list(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class CompiledStep:
    step_id: str
    action_type: str
    tool: str | None = None
    arguments: Mapping[str, Any] | None = None
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    max_attempts: int = 1
    on_success: str | None = None
    on_failure: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], index: int) -> "CompiledStep":
        action_type = _required_text(value.get("action_type"), f"steps[{index}].action_type")
        if action_type not in ACTION_TYPES:
            raise ValueError(f"steps[{index}].action_type must be one of {sorted(ACTION_TYPES)}")
        tool = value.get("tool")
        if action_type == "tool":
            tool = _identifier(tool, f"steps[{index}].tool")
        elif tool is not None:
            raise ValueError(f"steps[{index}].tool is only valid for tool actions")
        arguments = value.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            raise ValueError(f"steps[{index}].arguments must be an object")
        max_attempts = int(value.get("max_attempts", 1))
        if max_attempts < 1 or max_attempts > 20:
            raise ValueError(f"steps[{index}].max_attempts must be in [1, 20]")
        return cls(
            step_id=_identifier(value.get("step_id"), f"steps[{index}].step_id"),
            action_type=action_type,
            tool=tool,
            arguments=dict(arguments),
            requires=_string_tuple(value.get("requires"), f"steps[{index}].requires"),
            produces=_string_tuple(value.get("produces"), f"steps[{index}].produces"),
            max_attempts=max_attempts,
            on_success=str(value["on_success"]).strip() if value.get("on_success") else None,
            on_failure=str(value["on_failure"]).strip() if value.get("on_failure") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "tool": self.tool,
            "arguments": dict(self.arguments or {}),
            "requires": list(self.requires),
            "produces": list(self.produces),
            "max_attempts": self.max_attempts,
            "on_success": self.on_success,
            "on_failure": self.on_failure,
        }


@dataclass(frozen=True)
class HarnessInvariants:
    allowed_tools: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...] = ()
    tool_requirements: Mapping[str, tuple[str, ...]] | None = None
    required_evidence: tuple[str, ...] = ()
    max_total_actions: int = 30
    max_failures_before_escalation: int = 2
    high_risk_requires_teacher: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessInvariants":
        raw_requirements = value.get("tool_requirements") or {}
        if not isinstance(raw_requirements, Mapping):
            raise ValueError("invariants.tool_requirements must be an object")
        requirements = {
            _identifier(str(tool), "invariants.tool_requirements tool"): _string_tuple(flags, f"tool_requirements.{tool}")
            for tool, flags in raw_requirements.items()
        }
        max_actions = int(value.get("max_total_actions", 30))
        max_failures = int(value.get("max_failures_before_escalation", 2))
        if max_actions < 1 or max_actions > 1000:
            raise ValueError("invariants.max_total_actions must be in [1, 1000]")
        if max_failures < 1 or max_failures > 100:
            raise ValueError("invariants.max_failures_before_escalation must be in [1, 100]")
        return cls(
            allowed_tools=_string_tuple(value.get("allowed_tools"), "invariants.allowed_tools"),
            allowed_path_prefixes=_string_tuple(value.get("allowed_path_prefixes"), "invariants.allowed_path_prefixes"),
            tool_requirements=requirements,
            required_evidence=_string_tuple(value.get("required_evidence"), "invariants.required_evidence"),
            max_total_actions=max_actions,
            max_failures_before_escalation=max_failures,
            high_risk_requires_teacher=bool(value.get("high_risk_requires_teacher", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "allowed_path_prefixes": list(self.allowed_path_prefixes),
            "tool_requirements": {key: list(value) for key, value in (self.tool_requirements or {}).items()},
            "required_evidence": list(self.required_evidence),
            "max_total_actions": self.max_total_actions,
            "max_failures_before_escalation": self.max_failures_before_escalation,
            "high_risk_requires_teacher": self.high_risk_requires_teacher,
        }


@dataclass(frozen=True)
class CompiledHarness:
    harness_id: str
    version: int
    category: str
    goal: str
    triggers: HarnessTriggers
    steps: tuple[CompiledStep, ...]
    invariants: HarnessInvariants
    status: str = "active"
    parent_id: str | None = None
    source_genome_id: str | None = None
    provenance: Mapping[str, Any] | None = None
    created_at: str = ""
    content_hash: str = ""
    schema: str = COMPILED_HARNESS_SCHEMA

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, verify_hash: bool = True) -> "CompiledHarness":
        schema = str(value.get("schema") or COMPILED_HARNESS_SCHEMA)
        if schema != COMPILED_HARNESS_SCHEMA:
            raise ValueError(f"schema must be {COMPILED_HARNESS_SCHEMA}")
        version = int(value.get("version", 1))
        if version < 1:
            raise ValueError("version must be positive")
        status = str(value.get("status") or "active").lower()
        if status not in HARNESS_STATUSES:
            raise ValueError(f"status must be one of {sorted(HARNESS_STATUSES)}")
        raw_steps = value.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("steps must be a non-empty list")
        steps = tuple(CompiledStep.from_dict(item, index) for index, item in enumerate(raw_steps) if isinstance(item, Mapping))
        if len(steps) != len(raw_steps):
            raise ValueError("each step must be an object")
        ids = [step.step_id for step in steps]
        if len(set(ids)) != len(ids):
            raise ValueError("step_id values must be unique")
        for step in steps:
            for target in (step.on_success, step.on_failure):
                if target is not None and target not in ids:
                    raise ValueError(f"step {step.step_id} references unknown target {target}")
        if steps[-1].action_type not in {"complete", "escalate"}:
            raise ValueError("the final compiled step must complete or escalate")
        provenance = value.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be an object")
        invariants = HarnessInvariants.from_dict(value.get("invariants") or {})
        undeclared_tools = sorted({step.tool for step in steps if step.tool} - set(invariants.allowed_tools))
        if undeclared_tools:
            raise ValueError("tool steps are not declared in invariants.allowed_tools: " + ", ".join(undeclared_tools))
        result = cls(
            harness_id=_identifier(value.get("harness_id"), "harness_id"),
            version=version,
            category=_identifier(value.get("category"), "category"),
            goal=_required_text(value.get("goal"), "goal"),
            triggers=HarnessTriggers.from_dict(value.get("triggers") or {}),
            steps=steps,
            invariants=invariants,
            status=status,
            parent_id=str(value["parent_id"]).strip() if value.get("parent_id") else None,
            source_genome_id=str(value["source_genome_id"]).strip() if value.get("source_genome_id") else None,
            provenance=dict(provenance),
            created_at=str(value.get("created_at") or _now_iso()),
            content_hash=str(value.get("content_hash") or ""),
            schema=schema,
        )
        calculated = result.calculate_hash()
        if verify_hash and result.content_hash and result.content_hash != calculated:
            raise ValueError("content_hash does not match compiled harness content")
        return cls(**{**result.__dict__, "content_hash": calculated})

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "harness_id": self.harness_id,
            "version": self.version,
            "category": self.category,
            "goal": self.goal,
            "triggers": self.triggers.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "invariants": self.invariants.to_dict(),
            "status": self.status,
            "parent_id": self.parent_id,
            "source_genome_id": self.source_genome_id,
            "provenance": dict(self.provenance or {}),
        }

    def calculate_hash(self) -> str:
        return hashlib.sha256(_canonical(self.semantic_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "created_at": self.created_at, "content_hash": self.content_hash or self.calculate_hash()}


class ExternalHarnessCompiler:
    """Ask a capable teacher for typed steps, then validate deterministically."""

    def __init__(self, client: ModelClient, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def compile(
        self,
        *,
        task: Mapping[str, Any],
        rich_harness: Mapping[str, Any] | str,
        harness_id: str,
        version: int = 1,
        parent_id: str | None = None,
        source_genome_id: str | None = None,
    ) -> CompiledHarness:
        response = self.client.complete(
            system=(
                "You are the DataEvol Harness Compiler. Convert a rich harness into a short typed state machine for a 1B model. "
                "Return JSON only with category, goal, triggers, steps, and invariants. Each step needs step_id, action_type, "
                "requires, produces, max_attempts, and optional tool/arguments/on_success/on_failure. Put safety rules in invariants, "
                "including allowed_tools, tool_requirements, required_evidence, and retry/action limits. Do not expose hidden tests."
            ),
            user=_canonical({"task": dict(task), "rich_harness": rich_harness}),
            response_format="json",
            model=self.model,
            temperature=0.1,
            max_tokens=2400,
        )
        try:
            generated = json.loads(str(response.get("content") or ""))
        except json.JSONDecodeError as exc:
            raise ValueError("harness compiler returned invalid JSON") from exc
        if not isinstance(generated, Mapping):
            raise ValueError("harness compiler must return an object")
        payload = {
            **dict(generated),
            "schema": COMPILED_HARNESS_SCHEMA,
            "harness_id": harness_id,
            "version": version,
            "parent_id": parent_id,
            "source_genome_id": source_genome_id,
            "status": "active",
            "provenance": {
                "kind": "external_teacher_compilation",
                "model": str(response.get("model") or self.model or "unknown"),
                "usage": dict(response.get("usage") or {}),
            },
        }
        return CompiledHarness.from_dict(payload, verify_hash=False)


@dataclass(frozen=True)
class HarnessRouteCandidate:
    harness_id: str
    version: int
    content_hash: str
    score: float
    matched: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "matched": list(self.matched)}


@dataclass(frozen=True)
class HarnessRouteDecision:
    candidates: tuple[HarnessRouteCandidate, ...]
    confidence: float
    teacher_required: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "confidence": self.confidence,
            "teacher_required": self.teacher_required,
            "reasons": list(self.reasons),
        }


class DeterministicHarnessRouter:
    """Narrow a registry to at most three candidates using explicit features."""

    _WEIGHTS = {
        "task_types": 6.0,
        "repository_paths": 4.0,
        "languages": 3.0,
        "error_classes": 3.0,
        "required_tools": 2.0,
        "risk_levels": 2.0,
        "keywords": 1.0,
    }

    def route(
        self,
        features: Mapping[str, Any],
        harnesses: Sequence[CompiledHarness],
        *,
        top_k: int = 3,
    ) -> HarnessRouteDecision:
        if top_k < 1 or top_k > 3:
            raise ValueError("top_k must be in [1, 3]")
        newest: dict[str, CompiledHarness] = {}
        for harness in harnesses:
            current = newest.get(harness.harness_id)
            if current is None or harness.version > current.version:
                newest[harness.harness_id] = harness
        scored: list[HarnessRouteCandidate] = []
        for harness in newest.values():
            if harness.status != "active":
                continue
            score, matched = self._score(features, harness)
            if score > 0:
                scored.append(HarnessRouteCandidate(
                    harness_id=harness.harness_id,
                    version=harness.version,
                    content_hash=harness.content_hash,
                    score=score,
                    matched=tuple(matched),
                ))
        scored.sort(key=lambda item: (-item.score, item.harness_id, -item.version))
        candidates = tuple(scored[:top_k])
        total = sum(candidate.score for candidate in candidates)
        confidence = candidates[0].score / total if candidates and total > 0 else 0.0
        risk = str(features.get("risk_level") or "low").lower()
        ambiguous = len(candidates) > 1 and candidates[0].score - candidates[1].score < 1.0
        teacher_required = risk in {"high", "critical"} or not candidates or confidence < 0.55 or ambiguous
        reasons: list[str] = []
        if not candidates:
            reasons.append("no compiled harness matched the task features")
        if risk in {"high", "critical"}:
            reasons.append(f"{risk}-risk task requires teacher selection")
        if ambiguous:
            reasons.append("top harness candidates are near a capability boundary")
        if candidates and confidence < 0.55:
            reasons.append("route confidence is below 0.55")
        return HarnessRouteDecision(candidates, confidence, teacher_required, tuple(reasons))

    def _score(self, features: Mapping[str, Any], harness: CompiledHarness) -> tuple[float, list[str]]:
        score = 0.0
        matched: list[str] = []
        text = str(features.get("text") or "").lower()
        feature_map = {
            "task_types": {str(features.get("task_type") or "").lower()},
            "languages": {str(features.get("language") or "").lower()},
            "error_classes": {str(features.get("error_class") or "").lower()},
            "required_tools": {str(item).lower() for item in (features.get("required_tools") or [])},
            "risk_levels": {str(features.get("risk_level") or "").lower()},
        }
        for name, values in feature_map.items():
            triggers = {value.lower() for value in getattr(harness.triggers, name)}
            if triggers and values & triggers:
                score += self._WEIGHTS[name]
                matched.append(name)
        paths = [str(path) for path in (features.get("repository_paths") or [])]
        if harness.triggers.repository_paths and any(
            path.startswith(prefix) for path in paths for prefix in harness.triggers.repository_paths
        ):
            score += self._WEIGHTS["repository_paths"]
            matched.append("repository_paths")
        if harness.triggers.keywords and any(keyword.lower() in text for keyword in harness.triggers.keywords):
            score += self._WEIGHTS["keywords"]
            matched.append("keywords")
        return score, matched


__all__ = [
    "ACTION_TYPES",
    "COMPILED_HARNESS_SCHEMA",
    "CompiledHarness",
    "CompiledStep",
    "DeterministicHarnessRouter",
    "ExternalHarnessCompiler",
    "HarnessInvariants",
    "HarnessRouteCandidate",
    "HarnessRouteDecision",
    "HarnessTriggers",
]
