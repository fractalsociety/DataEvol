from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataevol.api.app import create_app
from dataevol.compat import call_core
from dataevol.config import DataEvolConfig
from dataevol.harness import storage
from dataevol.harness.compiled import CompiledHarness, DeterministicHarnessRouter
from dataevol.harness.controller import HarnessExecutionController, HarnessExecutionState
from dataevol.harness.model_client import FakeModelClient
from dataevol.harness.compiled import ExternalHarnessCompiler


def _manifest(**overrides):
    value = {
        "schema": "dataevol.compiled_harness.v1",
        "harness_id": "test-first-python",
        "version": 1,
        "category": "python_repair",
        "goal": "repair a failing Python test with evidence",
        "triggers": {
            "task_types": ["code_repair"],
            "languages": ["python"],
            "error_classes": ["test_failure"],
            "repository_paths": ["src/", "tests/"],
            "required_tools": ["test", "read", "edit"],
            "risk_levels": ["low", "medium"],
            "keywords": ["failing test"],
        },
        "steps": [
            {
                "step_id": "reproduce",
                "action_type": "tool",
                "tool": "test",
                "arguments": {"target": "$dynamic"},
                "produces": ["failure_reproduced"],
                "max_attempts": 2,
            },
            {
                "step_id": "inspect",
                "action_type": "tool",
                "tool": "read",
                "arguments": {"path": "$dynamic"},
                "requires": ["failure_reproduced"],
                "produces": ["traceback_read"],
            },
            {
                "step_id": "edit",
                "action_type": "tool",
                "tool": "edit",
                "arguments": {"path": "$dynamic"},
                "requires": ["traceback_read"],
                "produces": ["patch_written"],
            },
            {
                "step_id": "verify",
                "action_type": "tool",
                "tool": "test",
                "arguments": {"target": "$dynamic"},
                "requires": ["patch_written"],
                "produces": ["related_tests_pass"],
            },
            {
                "step_id": "complete",
                "action_type": "complete",
                "requires": ["related_tests_pass"],
            },
        ],
        "invariants": {
            "allowed_tools": ["test", "read", "edit"],
            "allowed_path_prefixes": ["src", "tests"],
            "tool_requirements": {"edit": ["failure_reproduced", "traceback_read"]},
            "required_evidence": ["test_report"],
            "max_total_actions": 8,
            "max_failures_before_escalation": 2,
            "high_risk_requires_teacher": True,
        },
        "provenance": {"kind": "human_seed"},
    }
    value.update(overrides)
    return value


def _cfg(tmp_path: Path) -> DataEvolConfig:
    return DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / "dataevol.sqlite3",
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="secret",
    )


def _proposal(step_id: str, action_type: str, tool: str | None = None, **arguments):
    return {"step_id": step_id, "action_type": action_type, "tool": tool, "arguments": arguments}


def test_compiled_manifest_is_hashed_and_rejects_undeclared_tools():
    harness = CompiledHarness.from_dict(_manifest())
    assert len(harness.content_hash) == 64
    assert CompiledHarness.from_dict(harness.to_dict()).content_hash == harness.content_hash

    invalid = _manifest()
    invalid["invariants"] = {**invalid["invariants"], "allowed_tools": ["test", "read"]}
    with pytest.raises(ValueError, match="not declared"):
        CompiledHarness.from_dict(invalid)


def test_router_activates_relevant_harness_and_requires_teacher_for_high_risk():
    harness = CompiledHarness.from_dict(_manifest())
    router = DeterministicHarnessRouter()
    low = router.route({
        "task_type": "code_repair",
        "language": "python",
        "error_class": "test_failure",
        "repository_paths": ["src/dataevol/x.py"],
        "risk_level": "low",
        "text": "fix failing test",
    }, [harness])
    assert low.candidates[0].harness_id == harness.harness_id
    assert low.teacher_required is False

    high = router.route({"task_type": "code_repair", "language": "python", "risk_level": "critical"}, [harness])
    assert high.teacher_required is True
    assert any("critical-risk" in reason for reason in high.reasons)

    newer = CompiledHarness.from_dict(_manifest(version=2, parent_id=harness.harness_id))
    latest = router.route({"task_type": "code_repair", "language": "python", "risk_level": "low"}, [harness, newer])
    assert len(latest.candidates) == 1
    assert latest.candidates[0].version == 2


def test_external_teacher_compiler_is_schema_validated():
    generated = _manifest()
    for key in ("schema", "harness_id", "version", "parent_id", "source_genome_id", "status", "provenance"):
        generated.pop(key, None)
    client = FakeModelClient(responder=lambda **_: generated, default_model="teacher-model")
    harness = ExternalHarnessCompiler(client, model="teacher-model").compile(
        task={"task_type": "code_repair"},
        rich_harness="Test first, then edit and verify.",
        harness_id="compiled-by-teacher",
    )
    assert harness.provenance["kind"] == "external_teacher_compilation"
    assert harness.provenance["model"] == "teacher-model"
    assert harness.steps[-1].action_type == "complete"


def test_controller_blocks_skipped_steps_forbidden_paths_and_premature_completion():
    harness = CompiledHarness.from_dict(_manifest())
    controller = HarnessExecutionController()
    state = HarnessExecutionState.start({"id": "task-1"}, harness, session_id="session-1")

    skipped = controller.propose(state, harness, _proposal("edit", "tool", "edit", path="src/x.py"))
    assert skipped.accepted is False
    assert "step_order_violation" in skipped.violations

    accepted = controller.propose(skipped.state, harness, _proposal("reproduce", "tool", "test", target="tests/test_x.py"))
    assert accepted.accepted is True
    state = controller.observe(accepted.state, harness, success=True).state
    state = controller.propose(state, harness, _proposal("inspect", "tool", "read", path="src/x.py")).state
    state = controller.observe(state, harness, success=True).state

    bad_path = controller.propose(state, harness, _proposal("edit", "tool", "edit", path="../secret.txt"))
    assert bad_path.accepted is False
    assert "path_not_allowed:path" in bad_path.violations

    state = controller.propose(bad_path.state, harness, _proposal("edit", "tool", "edit", path="src/x.py")).state
    state = controller.observe(state, harness, success=True).state
    state = controller.propose(state, harness, _proposal("verify", "tool", "test", target="tests/test_x.py")).state
    state = controller.observe(state, harness, success=True).state
    premature = controller.propose(state, harness, _proposal("complete", "complete"))
    assert premature.accepted is False
    assert "required_evidence_missing:test_report" in premature.violations


def test_registry_is_immutable_and_execution_survives_requests(tmp_path: Path):
    cfg = _cfg(tmp_path)
    harness = CompiledHarness.from_dict(_manifest())
    storage.register_compiled_harness(cfg.db_path, harness)
    assert storage.load_compiled_harness(cfg.db_path, harness.harness_id)["content_hash"] == harness.content_hash

    changed = CompiledHarness.from_dict(_manifest(goal="different behavior"))
    with pytest.raises(ValueError, match="already exists"):
        storage.register_compiled_harness(cfg.db_path, changed)

    features = {
        "task_type": "code_repair", "language": "python", "error_class": "test_failure",
        "repository_paths": ["src/x.py"], "risk_level": "low", "text": "failing test",
    }
    started = call_core("harness", "start_execution", {"task": {"id": "task-1"}, "features": features}, config=cfg)
    session_id = started["execution"]["session_id"]
    wrong = _proposal("edit", "tool", "edit", path="src/x.py")
    correction = _proposal("reproduce", "tool", "test", target="tests/test_x.py")
    acted = call_core("harness", "execution_action", {
        "session_id": session_id, "proposal": wrong, "teacher_correction": correction,
    }, config=cfg)
    assert acted["ok"] is True
    assert acted["decision"]["accepted"] is False
    assert acted["teacher_decision"]["accepted"] is True

    observed = call_core("harness", "execution_observation", {
        "session_id": session_id,
        "success": True,
        "verifier": {"passed": True, "score": 1.0},
    }, config=cfg)
    assert observed["state"]["status"] == "READY"

    read_back = call_core("harness", "get_execution", {"session_id": session_id}, config=cfg)
    assert read_back["execution"]["state"]["status"] == "READY"
    assert len(read_back["events"]) == 3

    output = tmp_path / "next-actions.jsonl"
    exported = call_core("harness", "export_next_actions", {
        "session_id": session_id, "output": str(output),
    }, config=cfg)
    assert exported["dataset"]["example_count"] == 2
    assert exported["dataset"]["includes_negative_examples"] is True
    assert exported["dataset"]["includes_teacher_corrections"] is True
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["violation_labels"]
    assert rows[1]["teacher_corrected"] is True
    assert rows[1]["verifier"]["passed"] is True


def test_compiled_harness_api_requires_auth_and_exposes_registry(tmp_path: Path):
    cfg = _cfg(tmp_path)
    client = TestClient(create_app(cfg))
    response = client.post("/harness/register_compiled", json={"payload": {"harness": _manifest()}})
    assert response.status_code == 401
    response = client.post(
        "/harness/register_compiled",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {"harness": _manifest()}},
    )
    assert response.status_code == 200
    registry = client.get("/harness/compiled", headers={"Authorization": "Bearer secret"})
    assert registry.status_code == 200
    assert registry.json()["count"] == 1
