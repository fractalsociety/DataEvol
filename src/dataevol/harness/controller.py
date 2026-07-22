"""Deterministic, persistent-friendly execution controller for compiled harnesses."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .compiled import CompiledHarness, CompiledStep


EXECUTION_SCHEMA = "dataevol.harness_execution.v1"
TERMINAL_STATUSES = frozenset({"COMPLETED", "ESCALATED", "FAILED"})
_PATH_KEYS = frozenset({"path", "file", "target", "directory", "cwd"})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class HarnessExecutionState:
    session_id: str
    task: Mapping[str, Any]
    harness_id: str
    harness_version: int
    harness_content_hash: str
    status: str
    current_step_id: str | None
    pending_step_id: str | None = None
    flags: tuple[str, ...] = ()
    evidence: Mapping[str, Any] | None = None
    attempts: Mapping[str, int] | None = None
    completed_steps: tuple[str, ...] = ()
    action_count: int = 0
    failure_count: int = 0
    violation_count: int = 0
    teacher_correction_count: int = 0
    schema: str = EXECUTION_SCHEMA

    @classmethod
    def start(cls, task: Mapping[str, Any], harness: CompiledHarness, *, session_id: str | None = None) -> "HarnessExecutionState":
        return cls(
            session_id=session_id or f"hexec_{uuid.uuid4().hex}",
            task=dict(task),
            harness_id=harness.harness_id,
            harness_version=harness.version,
            harness_content_hash=harness.content_hash,
            status="READY",
            current_step_id=harness.steps[0].step_id,
            evidence={},
            attempts={},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessExecutionState":
        if value.get("schema") != EXECUTION_SCHEMA:
            raise ValueError(f"execution schema must be {EXECUTION_SCHEMA}")
        return cls(
            session_id=str(value["session_id"]),
            task=dict(value.get("task") or {}),
            harness_id=str(value["harness_id"]),
            harness_version=int(value["harness_version"]),
            harness_content_hash=str(value["harness_content_hash"]),
            status=str(value["status"]),
            current_step_id=value.get("current_step_id"),
            pending_step_id=value.get("pending_step_id"),
            flags=tuple(str(item) for item in (value.get("flags") or [])),
            evidence=dict(value.get("evidence") or {}),
            attempts={str(key): int(count) for key, count in (value.get("attempts") or {}).items()},
            completed_steps=tuple(str(item) for item in (value.get("completed_steps") or [])),
            action_count=int(value.get("action_count", 0)),
            failure_count=int(value.get("failure_count", 0)),
            violation_count=int(value.get("violation_count", 0)),
            teacher_correction_count=int(value.get("teacher_correction_count", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "session_id": self.session_id,
            "task": dict(self.task),
            "harness_id": self.harness_id,
            "harness_version": self.harness_version,
            "harness_content_hash": self.harness_content_hash,
            "status": self.status,
            "current_step_id": self.current_step_id,
            "pending_step_id": self.pending_step_id,
            "flags": list(self.flags),
            "evidence": dict(self.evidence or {}),
            "attempts": dict(self.attempts or {}),
            "completed_steps": list(self.completed_steps),
            "action_count": self.action_count,
            "failure_count": self.failure_count,
            "violation_count": self.violation_count,
            "teacher_correction_count": self.teacher_correction_count,
        }


@dataclass(frozen=True)
class ActionDecision:
    accepted: bool
    violations: tuple[str, ...]
    expected_action: Mapping[str, Any]
    proposal: Mapping[str, Any]
    state: HarnessExecutionState

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "violations": list(self.violations),
            "expected_action": dict(self.expected_action),
            "proposal": dict(self.proposal),
            "state": self.state.to_dict(),
        }


@dataclass(frozen=True)
class ObservationDecision:
    state: HarnessExecutionState
    transition: str
    expected_action: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "transition": self.transition,
            "expected_action": dict(self.expected_action) if self.expected_action else None,
        }


class HarnessExecutionController:
    """Hold long-horizon state while the worker predicts one local action."""

    def expected_action(self, state: HarnessExecutionState, harness: CompiledHarness) -> dict[str, Any]:
        self._validate_identity(state, harness)
        if state.status in TERMINAL_STATUSES or state.current_step_id is None:
            return {}
        step = self._step(harness, state.current_step_id)
        return {
            "step_id": step.step_id,
            "action_type": step.action_type,
            "tool": step.tool,
            "arguments": dict(step.arguments or {}),
        }

    def propose(
        self,
        state: HarnessExecutionState,
        harness: CompiledHarness,
        proposal: Mapping[str, Any],
    ) -> ActionDecision:
        self._validate_identity(state, harness)
        expected = self.expected_action(state, harness)
        violations: list[str] = []
        if state.status != "READY":
            violations.append("execution_not_ready")
        if state.action_count >= harness.invariants.max_total_actions:
            violations.append("action_budget_exhausted")
        if not expected:
            violations.append("no_action_expected")
        if str(proposal.get("step_id") or "") != str(expected.get("step_id") or ""):
            violations.append("step_order_violation")
        if str(proposal.get("action_type") or "") != str(expected.get("action_type") or ""):
            violations.append("action_type_violation")
        step = self._step(harness, state.current_step_id) if state.current_step_id else None
        if step is not None:
            violations.extend(self._step_violations(state, harness, step, proposal))
        if violations:
            rejected = replace(state, violation_count=state.violation_count + 1)
            return ActionDecision(False, tuple(dict.fromkeys(violations)), expected, dict(proposal), rejected)

        if step is None:
            raise ValueError("current execution step is missing")
        attempts = dict(state.attempts or {})
        attempts[step.step_id] = attempts.get(step.step_id, 0) + 1
        if step.action_type == "complete":
            accepted = replace(
                state,
                status="COMPLETED",
                current_step_id=None,
                completed_steps=tuple(dict.fromkeys((*state.completed_steps, step.step_id))),
                attempts=attempts,
                action_count=state.action_count + 1,
            )
        elif step.action_type == "escalate":
            accepted = replace(
                state,
                status="ESCALATED",
                current_step_id=None,
                completed_steps=tuple(dict.fromkeys((*state.completed_steps, step.step_id))),
                attempts=attempts,
                action_count=state.action_count + 1,
            )
        else:
            accepted = replace(
                state,
                status="AWAITING_OBSERVATION",
                pending_step_id=step.step_id,
                attempts=attempts,
                action_count=state.action_count + 1,
            )
        return ActionDecision(True, (), expected, dict(proposal), accepted)

    def observe(
        self,
        state: HarnessExecutionState,
        harness: CompiledHarness,
        *,
        success: bool,
        produced_flags: Sequence[str] = (),
        evidence: Mapping[str, Any] | None = None,
    ) -> ObservationDecision:
        self._validate_identity(state, harness)
        if state.status != "AWAITING_OBSERVATION" or not state.pending_step_id:
            raise ValueError("execution is not awaiting an observation")
        step = self._step(harness, state.pending_step_id)
        flags = tuple(dict.fromkeys((*state.flags, *(str(item) for item in produced_flags))))
        evidence_state = {**dict(state.evidence or {}), **dict(evidence or {})}
        completed = state.completed_steps
        failure_count = state.failure_count

        if success:
            flags = tuple(dict.fromkeys((*flags, *step.produces)))
            completed = tuple(dict.fromkeys((*completed, step.step_id)))
            target = step.on_success or self._sequential_next(harness, step.step_id)
            transition = "success"
        else:
            failure_count += 1
            attempts = int((state.attempts or {}).get(step.step_id, 0))
            if failure_count >= harness.invariants.max_failures_before_escalation:
                target = None
                transition = "failure_escalation"
            elif attempts >= step.max_attempts and (not step.on_failure or step.on_failure == step.step_id):
                target = None
                transition = "attempts_exhausted"
            elif step.on_failure:
                target = step.on_failure
                transition = "failure_branch"
            elif attempts < step.max_attempts:
                target = step.step_id
                transition = "retry"
            else:  # pragma: no cover - exhaustive guard for future step variants.
                target = None
                transition = "attempts_exhausted"

        status = "READY"
        if target is None:
            status = "COMPLETED" if success else "ESCALATED"
        updated = replace(
            state,
            status=status,
            current_step_id=target,
            pending_step_id=None,
            flags=flags,
            evidence=evidence_state,
            completed_steps=completed,
            failure_count=failure_count,
        )
        expected = self.expected_action(updated, harness) if status == "READY" else None
        return ObservationDecision(updated, transition, expected)

    def apply_teacher_correction(
        self,
        state: HarnessExecutionState,
        *,
        flags: Sequence[str] = (),
        evidence: Mapping[str, Any] | None = None,
    ) -> HarnessExecutionState:
        return replace(
            state,
            flags=tuple(dict.fromkeys((*state.flags, *(str(item) for item in flags)))),
            evidence={**dict(state.evidence or {}), **dict(evidence or {})},
            teacher_correction_count=state.teacher_correction_count + 1,
        )

    def _step_violations(
        self,
        state: HarnessExecutionState,
        harness: CompiledHarness,
        step: CompiledStep,
        proposal: Mapping[str, Any],
    ) -> list[str]:
        violations: list[str] = []
        missing = [flag for flag in step.requires if flag not in state.flags]
        if missing:
            violations.append("missing_step_requirements:" + ",".join(missing))
        if step.action_type == "tool":
            tool = str(proposal.get("tool") or "")
            if tool != step.tool:
                violations.append("tool_selection_violation")
            if tool not in harness.invariants.allowed_tools:
                violations.append("tool_not_allowed")
            tool_missing = [
                flag for flag in (harness.invariants.tool_requirements or {}).get(tool, ())
                if flag not in state.flags
            ]
            if tool_missing:
                violations.append("tool_guard_violation:" + ",".join(tool_missing))
            arguments = proposal.get("arguments") or {}
            if not isinstance(arguments, Mapping):
                violations.append("arguments_not_object")
            else:
                violations.extend(self._argument_violations(step, harness, arguments))
        if step.action_type == "complete":
            missing_evidence = [key for key in harness.invariants.required_evidence if key not in (state.evidence or {})]
            if missing_evidence:
                violations.append("required_evidence_missing:" + ",".join(missing_evidence))
        return violations

    @staticmethod
    def _argument_violations(step: CompiledStep, harness: CompiledHarness, arguments: Mapping[str, Any]) -> list[str]:
        violations: list[str] = []
        for key, expected in (step.arguments or {}).items():
            if key not in arguments:
                violations.append(f"argument_missing:{key}")
            elif expected != "$dynamic" and arguments[key] != expected:
                violations.append(f"argument_constraint_violation:{key}")
        prefixes = tuple(prefix.rstrip("/") for prefix in harness.invariants.allowed_path_prefixes)
        if prefixes:
            for key, value in arguments.items():
                if key not in _PATH_KEYS or not isinstance(value, str):
                    continue
                path = PurePosixPath(value)
                normalized = str(path)
                if path.is_absolute() or ".." in path.parts or not any(
                    normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes
                ):
                    violations.append(f"path_not_allowed:{key}")
        return violations

    @staticmethod
    def _step(harness: CompiledHarness, step_id: str | None) -> CompiledStep:
        for step in harness.steps:
            if step.step_id == step_id:
                return step
        raise ValueError(f"compiled harness does not contain step {step_id}")

    @staticmethod
    def _sequential_next(harness: CompiledHarness, step_id: str) -> str | None:
        for index, step in enumerate(harness.steps):
            if step.step_id == step_id:
                return harness.steps[index + 1].step_id if index + 1 < len(harness.steps) else None
        raise ValueError(f"compiled harness does not contain step {step_id}")

    @staticmethod
    def _validate_identity(state: HarnessExecutionState, harness: CompiledHarness) -> None:
        if (state.harness_id, state.harness_version, state.harness_content_hash) != (
            harness.harness_id,
            harness.version,
            harness.content_hash,
        ):
            raise ValueError("execution state does not match the pinned compiled harness")


def distillation_example(
    *,
    state_before: Mapping[str, Any],
    expected_action: Mapping[str, Any],
    proposal: Mapping[str, Any],
    accepted: bool,
    violations: Sequence[str],
    state_after: Mapping[str, Any],
    teacher_correction: Mapping[str, Any] | None = None,
    verifier: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one short-horizon supervised decision with negative evidence."""
    target = dict(teacher_correction or expected_action)
    payload = {
        "schema": "dataevol.next_action_example.v1",
        "task": dict(state_before.get("task") or {}),
        "harness": {
            "harness_id": state_before.get("harness_id"),
            "version": state_before.get("harness_version"),
            "content_hash": state_before.get("harness_content_hash"),
        },
        "state": {
            "status": state_before.get("status"),
            "current_step_id": state_before.get("current_step_id"),
            "flags": list(state_before.get("flags") or []),
            "evidence_keys": sorted((state_before.get("evidence") or {}).keys()),
            "failure_count": state_before.get("failure_count", 0),
        },
        "proposed_action": dict(proposal),
        "next_action": target,
        "accepted": bool(accepted),
        "violation_labels": list(violations),
        "teacher_corrected": teacher_correction is not None,
        "verifier": dict(verifier or {}),
        "next_state_status": state_after.get("status"),
    }
    payload["example_hash"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def execution_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposals = [event for event in events if event.get("kind") == "action"]
    observations = [event for event in events if event.get("kind") == "observation"]
    accepted = sum(1 for event in proposals if event.get("accepted"))
    order_violations = sum(
        1 for event in proposals if any("step_order_violation" in str(item) for item in (event.get("violations") or []))
    )
    invariant_violations = sum(1 for event in proposals if event.get("violations"))
    sessions = {str(event.get("session_id")) for event in events if event.get("session_id")}
    completed = {
        str(event.get("session_id")) for event in events
        if (event.get("state_after") or {}).get("status") == "COMPLETED"
    }
    escalated = {
        str(event.get("session_id")) for event in events
        if (event.get("state_after") or {}).get("status") == "ESCALATED"
    }
    teacher_corrections = sum(1 for event in events if event.get("teacher_correction"))
    return {
        "schema": "dataevol.harness_execution_metrics.v1",
        "session_count": len(sessions),
        "action_proposal_count": len(proposals),
        "observation_count": len(observations),
        "step_adherence": accepted / len(proposals) if proposals else 0.0,
        "order_adherence": 1.0 - order_violations / len(proposals) if proposals else 0.0,
        "invariant_violation_rate": invariant_violations / len(proposals) if proposals else 0.0,
        "long_horizon_completion_rate": len(completed) / len(sessions) if sessions else 0.0,
        "teacher_escalation_rate": len(escalated) / len(sessions) if sessions else 0.0,
        "teacher_correction_count": teacher_corrections,
    }


__all__ = [
    "ActionDecision",
    "EXECUTION_SCHEMA",
    "HarnessExecutionController",
    "HarnessExecutionState",
    "ObservationDecision",
    "distillation_example",
    "execution_metrics",
]
