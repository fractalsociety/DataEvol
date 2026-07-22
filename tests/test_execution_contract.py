"""Tests for the S1 real-execution bridge contract."""
from __future__ import annotations

import hashlib

import pytest

from dataevol.harness.execution_contract import (
    EXECUTION_EVENT_SCHEMA,
    ExecutionEvent,
    PINNED_REQUEST_SCHEMA,
    PinnedExecutionRequest,
    is_real_executor_kind,
    replay_hash,
)
from dataevol.harness.verdicts import issue_harness_verdict

_HASH = hashlib.sha256(b"pinned").hexdigest()


def _request_payload(**overrides) -> dict:
    payload = {
        "executor_kind": "subprocess.v1",
        "model_revision": "qwen2.5-0.5b@abc123",
        "tokenizer_revision": "qwen2.5-0.5b-tok@abc123",
        "adapter_revision": "",
        "harness_id": "h_test",
        "harness_version": 1,
        "harness_content_hash": _HASH,
        "gym_version": "gym.v1",
        "verifier_version": "verifier.v1",
        "seed": 17,
        "max_wall_seconds": 60,
        "max_memory_mb": 2048,
        "max_actions": 20,
    }
    payload.update(overrides)
    return payload


def _event(sequence: int = 0, **overrides) -> ExecutionEvent:
    value = {
        "session_id": "hexec_1",
        "request_hash": _HASH,
        "sequence": sequence,
        "event_type": "PROPOSAL",
        "payload": {"action": "run_tests"},
        "model_identity": "qwen2.5-0.5b@abc123",
        "tokens_in": 100,
        "tokens_out": 20,
        "latency_ms": 12.5,
        "cost_usd": 0.0001,
    }
    value.update(overrides)
    return ExecutionEvent.create(value)


def test_pinned_request_hash_roundtrip():
    request = PinnedExecutionRequest.create(_request_payload())
    assert request.schema == PINNED_REQUEST_SCHEMA
    assert request.verify_hash()
    restored = PinnedExecutionRequest.from_dict(request.to_dict())
    assert restored == request


def test_pinned_request_rejects_missing_fields():
    for field in ("model_revision", "tokenizer_revision", "gym_version", "verifier_version"):
        with pytest.raises(ValueError):
            PinnedExecutionRequest.create(_request_payload(**{field: ""}))
    with pytest.raises(ValueError):
        PinnedExecutionRequest.create(_request_payload(harness_content_hash="not-a-hash"))
    with pytest.raises(ValueError):
        PinnedExecutionRequest.create(_request_payload(max_wall_seconds=0))


def test_pinned_request_tampered_hash_rejected():
    request = PinnedExecutionRequest.create(_request_payload())
    tampered = {**request.to_dict(), "seed": 18}
    with pytest.raises(ValueError):
        PinnedExecutionRequest.from_dict(tampered)


def test_event_roundtrip_and_validation():
    event = _event()
    assert event.schema == EXECUTION_EVENT_SCHEMA
    assert ExecutionEvent.from_dict(event.to_dict()) == event
    with pytest.raises(ValueError):
        _event(event_type="NOT_A_TYPE")
    with pytest.raises(ValueError):
        _event(latency_ms=-1.0)


def test_replay_hash_ignores_wall_clock_and_cost():
    first = [_event(0), _event(1, event_type="OBSERVATION", payload={"stdout_range": [0, 42]})]
    second = [
        _event(0, latency_ms=999.0, cost_usd=1.0, created_at="2026-07-13T01:02:03+00:00"),
        _event(
            1,
            event_type="OBSERVATION",
            payload={"stdout_range": [0, 42], "peak_memory_mb": 512},
            latency_ms=1.0,
        ),
    ]
    assert replay_hash(first) == replay_hash(second)


def test_replay_hash_detects_divergence_and_gaps():
    baseline = [_event(0), _event(1)]
    diverged = [_event(0), _event(1, payload={"action": "edit_file"})]
    assert replay_hash(baseline) != replay_hash(diverged)
    with pytest.raises(ValueError):
        replay_hash([_event(0), _event(2)])
    with pytest.raises(ValueError):
        replay_hash([])
    with pytest.raises(ValueError):
        replay_hash([_event(0), _event(1, session_id="hexec_other")])


def test_executor_allowlist_fails_closed():
    assert is_real_executor_kind("subprocess.v1")
    assert is_real_executor_kind("fractalwork-runtime-v1")
    assert not is_real_executor_kind("fixture")
    assert not is_real_executor_kind("docker.v9")  # unknown -> non-real

    verdict = issue_harness_verdict({
        "task_type": "python_repair",
        "incumbent_genome_id": "g_inc",
        "candidate_genome_id": "g_cand",
        "candidate_content_hash": _HASH,
        "benchmark_hash": _HASH,
        "evidence_hash": _HASH,
        "executor_kind": "docker.v9",
        "report": {},
    })
    assert verdict.verdict == "INCONCLUSIVE"
    assert any("allowlist" in reason for reason in verdict.reasons)
