"""S1 real-execution bridge contract.

Defines the pinned execution request (PRD S1.2), the versioned execution
event schema (S1.6/S1.7), the closed executor-provenance allowlist (S1.9),
and the replay hash (S1.8). Executors, the smoke suite, and verdict issuance
all depend on this module; it depends on nothing but the stdlib.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

PINNED_REQUEST_SCHEMA = "dataevol.pinned_execution_request.v1"
EXECUTION_EVENT_SCHEMA = "dataevol.harness_execution_event.v1"

# Closed provenance allowlist: only enumerated real kinds may authorize a
# production canary. Unknown kinds fail closed as non-real.
REAL_EXECUTOR_KINDS = frozenset({"subprocess.v1", "fractalwork-runtime-v1"})
NON_REAL_EXECUTOR_KINDS = frozenset({"reference", "fixture", "mock", "simulation"})

EVENT_TYPES = frozenset({
    "ROUTE",
    "STATE",
    "PROPOSAL",
    "ACCEPTED",
    "REJECTED",
    "VIOLATION",
    "TOOL_CALL",
    "OBSERVATION",
    "CORRECTION",
    "VERIFIER",
    "REWARD",
    "SUBPROCESS_EXIT",
})

# Nondeterministic per-run telemetry, excluded from the replay hash so that
# a replay from the same seed and pinned artifacts reproduces the hash.
REPLAY_EXCLUDED_FIELDS = frozenset({"latency_ms", "cost_usd", "created_at", "peak_memory_mb"})

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def is_real_executor_kind(executor_kind: str) -> bool:
    return executor_kind in REAL_EXECUTOR_KINDS


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _required_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _content_hash_text(value: Any, field: str) -> str:
    text = _required_text(value, field).lower()
    if not _HASH_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return text


def _created_at(value: Any = None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    text = _required_text(value, "created_at")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return text


@dataclass(frozen=True)
class PinnedExecutionRequest:
    """Everything a real execution is pinned to (PRD S1.2)."""

    schema: str
    request_id: str
    executor_kind: str
    model_revision: str
    tokenizer_revision: str
    adapter_revision: str  # "" when no adapter is applied
    harness_id: str
    harness_version: int
    harness_content_hash: str
    gym_version: str
    verifier_version: str
    seed: int
    max_wall_seconds: int
    max_memory_mb: int
    max_actions: int
    content_hash: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "executor_kind": self.executor_kind,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "adapter_revision": self.adapter_revision,
            "harness_id": self.harness_id,
            "harness_version": self.harness_version,
            "harness_content_hash": self.harness_content_hash,
            "gym_version": self.gym_version,
            "verifier_version": self.verifier_version,
            "seed": self.seed,
            "max_wall_seconds": self.max_wall_seconds,
            "max_memory_mb": self.max_memory_mb,
            "max_actions": self.max_actions,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "content_hash": self.content_hash}

    def verify_hash(self) -> bool:
        return self.content_hash == _sha256(canonical_json(self.unsigned_dict()))

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "PinnedExecutionRequest":
        unsigned = {
            "schema": PINNED_REQUEST_SCHEMA,
            "request_id": _required_text(value.get("request_id") or f"pexec_{uuid.uuid4().hex}", "request_id"),
            "executor_kind": _required_text(value.get("executor_kind"), "executor_kind"),
            "model_revision": _required_text(value.get("model_revision"), "model_revision"),
            "tokenizer_revision": _required_text(value.get("tokenizer_revision"), "tokenizer_revision"),
            "adapter_revision": str(value.get("adapter_revision") or ""),
            "harness_id": _required_text(value.get("harness_id"), "harness_id"),
            "harness_version": _required_int(value.get("harness_version"), "harness_version", minimum=1),
            "harness_content_hash": _content_hash_text(value.get("harness_content_hash"), "harness_content_hash"),
            "gym_version": _required_text(value.get("gym_version"), "gym_version"),
            "verifier_version": _required_text(value.get("verifier_version"), "verifier_version"),
            "seed": _required_int(value.get("seed"), "seed"),
            "max_wall_seconds": _required_int(value.get("max_wall_seconds"), "max_wall_seconds", minimum=1),
            "max_memory_mb": _required_int(value.get("max_memory_mb"), "max_memory_mb", minimum=1),
            "max_actions": _required_int(value.get("max_actions"), "max_actions", minimum=1),
        }
        return cls(**unsigned, content_hash=_sha256(canonical_json(unsigned)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, verify_hash: bool = True) -> "PinnedExecutionRequest":
        if value.get("schema") != PINNED_REQUEST_SCHEMA:
            raise ValueError(f"schema must be {PINNED_REQUEST_SCHEMA}")
        result = cls.create({k: v for k, v in value.items() if k not in ("schema", "content_hash")})
        stored = value.get("content_hash")
        if verify_hash and stored is not None and stored != result.content_hash:
            raise ValueError("content_hash does not match the canonical request payload")
        return result


@dataclass(frozen=True)
class ExecutionEvent:
    """One versioned execution event (PRD S1.6/S1.7).

    ``payload`` carries event-type-specific detail (tool observation,
    violation labels, verifier result, stdout/stderr ranges, exit status,
    failure classification, checkpoint identity, peak_memory_mb, ...).
    """

    schema: str
    session_id: str
    request_hash: str
    sequence: int
    event_type: str
    payload: Mapping[str, Any]
    model_identity: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "session_id": self.session_id,
            "request_hash": self.request_hash,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "model_identity": self.model_identity,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at,
        }

    def replay_dict(self) -> dict[str, Any]:
        base = self.to_dict()
        payload = {k: v for k, v in base.pop("payload").items() if k not in REPLAY_EXCLUDED_FIELDS}
        return {**{k: v for k, v in base.items() if k not in REPLAY_EXCLUDED_FIELDS}, "payload": payload}

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "ExecutionEvent":
        event_type = _required_text(value.get("event_type"), "event_type").upper()
        if event_type not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(EVENT_TYPES)}")
        payload = value.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be an object")
        canonical_json(dict(payload))  # reject unserializable payloads at creation
        latency = value.get("latency_ms", 0.0)
        cost = value.get("cost_usd", 0.0)
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
            raise ValueError("latency_ms must be a non-negative number")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            raise ValueError("cost_usd must be a non-negative number")
        return cls(
            schema=EXECUTION_EVENT_SCHEMA,
            session_id=_required_text(value.get("session_id"), "session_id"),
            request_hash=_content_hash_text(value.get("request_hash"), "request_hash"),
            sequence=_required_int(value.get("sequence"), "sequence"),
            event_type=event_type,
            payload=dict(payload),
            model_identity=_required_text(value.get("model_identity"), "model_identity"),
            tokens_in=_required_int(value.get("tokens_in", 0), "tokens_in"),
            tokens_out=_required_int(value.get("tokens_out", 0), "tokens_out"),
            latency_ms=float(latency),
            cost_usd=float(cost),
            created_at=_created_at(value.get("created_at")),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionEvent":
        if value.get("schema") != EXECUTION_EVENT_SCHEMA:
            raise ValueError(f"schema must be {EXECUTION_EVENT_SCHEMA}")
        return cls.create({k: v for k, v in value.items() if k != "schema"})


def replay_hash(events: Iterable[ExecutionEvent]) -> str:
    """Deterministic hash of an event stream (PRD S1.8).

    Excludes wall-clock, cost, and host-telemetry fields so a replay from the
    same seed and pinned artifacts reproduces the hash. Requires a single
    session and a contiguous zero-based sequence.
    """
    ordered = sorted(events, key=lambda e: e.sequence)
    if not ordered:
        raise ValueError("replay_hash requires at least one event")
    sessions = {e.session_id for e in ordered}
    if len(sessions) != 1:
        raise ValueError("replay_hash requires events from exactly one session")
    if [e.sequence for e in ordered] != list(range(len(ordered))):
        raise ValueError("event sequence must be contiguous and zero-based")
    return _sha256(canonical_json([e.replay_dict() for e in ordered]))


__all__ = [
    "EXECUTION_EVENT_SCHEMA",
    "EVENT_TYPES",
    "ExecutionEvent",
    "NON_REAL_EXECUTOR_KINDS",
    "PINNED_REQUEST_SCHEMA",
    "PinnedExecutionRequest",
    "REAL_EXECUTOR_KINDS",
    "REPLAY_EXCLUDED_FIELDS",
    "canonical_json",
    "is_real_executor_kind",
    "replay_hash",
]
