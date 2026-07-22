from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from dataevol.harness.execution_contract import ExecutionEvent, PinnedExecutionRequest
from dataevol.harness.genome import (
    AgentSpec,
    HarnessGenome,
    MemorySpec,
    OutputSchemaSpec,
    RecoverySpec,
    RouterSpec,
    WorkflowStep,
)
from dataevol.harness.subprocess_executor import SubprocessHarnessExecutor


BENCHMARK = [
    {"id": "one", "prompt": "one", "expected": "1", "category": "normal"},
    {"id": "two", "prompt": "two", "expected": "2", "category": "normal"},
    {"id": "three", "prompt": "three", "expected": "3", "category": "edge"},
    {"id": "four", "prompt": "four", "expected": "4", "category": "edge"},
]


def _request(*, executor_kind: str = "fixture", max_wall_seconds: int = 5) -> PinnedExecutionRequest:
    digest = hashlib.sha256(b"harness").hexdigest()
    return PinnedExecutionRequest.create({
        "executor_kind": executor_kind,
        "model_revision": "scripted@v1",
        "tokenizer_revision": "whitespace@v1",
        "adapter_revision": "",
        "harness_id": "h_subprocess",
        "harness_version": 1,
        "harness_content_hash": digest,
        "gym_version": "inline.v1",
        "verifier_version": "exact.v1",
        "seed": 17,
        "max_wall_seconds": max_wall_seconds,
        "max_memory_mb": 512,
        "max_actions": 3,
    })


def _genome() -> HarnessGenome:
    return HarnessGenome(
        genome_id="g_subprocess",
        version=1,
        parent_id=None,
        task_type="exact_match",
        task_spec_hash="inline",
        router=RouterSpec(model="scripted", confidence_threshold=0.5),
        agents=(AgentSpec(role="worker", model="scripted", prompt_ref="inline"),),
        workflow=(WorkflowStep(step_id="answer", agent_role="worker"),),
        memory=MemorySpec(),
        recovery=RecoverySpec(),
        output=OutputSchemaSpec(),
    )


def _worker(path) -> list[str]:
    return [sys.executable, "-m", "dataevol.harness.worker_main", "--scripted", str(path)]


def _write_script(path, values) -> None:
    path.write_text(json.dumps(values), encoding="utf-8")


def _verifier(case, output: str) -> bool:
    return output == case["expected"]


def test_end_to_end_measures_outcomes_and_valid_events(tmp_path):
    script = tmp_path / "responses.json"
    _write_script(script, {"one": "1", "two": "wrong", "three": "3", "four": "4"})
    executor = SubprocessHarnessExecutor(_request(), _worker(script), _verifier)

    evaluation = executor.evaluate(_genome(), BENCHMARK)

    assert evaluation.quality == pytest.approx(0.75)
    assert evaluation.failure_rate == pytest.approx(0.25)
    assert evaluation.verifier_agreement == 1.0
    assert evaluation.robustness == evaluation.quality
    assert len(executor.last_event_streams) == 4
    for events in executor.last_event_streams.values():
        assert [event.event_type for event in events] == [
            "ROUTE", "PROPOSAL", "OBSERVATION", "VERIFIER", "REWARD", "SUBPROCESS_EXIT",
        ]
        assert tuple(ExecutionEvent.from_dict(event.to_dict()) for event in events) == events


def test_timeout_kills_worker_and_classifies_case(tmp_path):
    script = tmp_path / "responses.json"
    _write_script(script, {"slow": {"output": "ok", "sleep_s": 2.0}})
    executor = SubprocessHarnessExecutor(_request(max_wall_seconds=1), _worker(script), lambda case, output: True)

    evaluation = executor.evaluate(_genome(), [{"id": "slow", "prompt": "wait"}])

    assert evaluation.failure_rate == 1.0
    assert evaluation.failure_categories == ("TIMEOUT",)
    exit_event = executor.last_event_streams["slow"][-1]
    assert exit_event.payload["failure_classification"] == "TIMEOUT"
    assert exit_event.payload["exit_status"] is None or exit_event.payload["exit_status"] != 0


def test_cancellation_mid_run_stops_active_worker(tmp_path):
    script = tmp_path / "responses.json"
    _write_script(script, {"slow": {"output": "ok", "sleep_s": 5.0}})
    executor = SubprocessHarnessExecutor(_request(max_wall_seconds=10), _worker(script), lambda case, output: True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.evaluate, _genome(), [{"id": "slow", "prompt": "wait"}])
        time.sleep(0.25)
        executor.cancel()
        evaluation = future.result(timeout=3.0)

    assert evaluation.failure_rate == 1.0
    assert evaluation.failure_categories == ("WORKER_CRASH",)
    assert executor.last_event_streams["slow"][-1].payload["exit_status"] != 0


def test_replay_hashes_are_deterministic(tmp_path):
    script = tmp_path / "responses.json"
    _write_script(script, {case["id"]: case["expected"] for case in BENCHMARK})
    executor = SubprocessHarnessExecutor(_request(), _worker(script), _verifier)

    first = executor.evaluate(_genome(), BENCHMARK)
    second = executor.evaluate(_genome(), BENCHMARK)

    assert first.raw["replay_hashes"] == second.raw["replay_hashes"]


def test_scripted_provenance_and_real_kind_fail_closed(tmp_path):
    script = tmp_path / "responses.json"
    _write_script(script, {"one": "1"})
    worker = _worker(script)
    evaluation = SubprocessHarnessExecutor(_request(), worker, _verifier).evaluate(_genome(), [BENCHMARK[0]])
    assert evaluation.raw["executor_kind"] == "fixture"

    with pytest.raises(ValueError, match="scripted workers require"):
        SubprocessHarnessExecutor(_request(executor_kind="subprocess.v1"), worker, _verifier)
