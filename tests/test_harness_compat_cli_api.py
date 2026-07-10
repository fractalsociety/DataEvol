from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from dataevol.api.app import create_app
from dataevol.cli.main import app as cli_app
from dataevol.config import DataEvolConfig
from dataevol.harness.model_client import FakeModelClient
from tests.test_harness_reference_executor import _genome

BENCHMARK_CASES = [
    {"id": "n1", "category": "normal"},
    {"id": "a1", "category": "adversarial"},
    {"id": "t1", "category": "tool_failure"},
    {"id": "l1", "category": "long_context"},
    {"id": "e1", "category": "edge"},
    {"id": "h1", "category": "hidden_holdout"},
]
IMPROVING_PATCH = {
    "agents": [
        {"role": "worker", "model": "local-7b", "prompt_ref": "prompts/worker.md", "tools": []},
        {"role": "verifier", "model": "local-7b", "prompt_ref": "prompts/verify.md", "cannot_view": ["previous_agent_confidence"]},
    ],
    "workflow": [{"step_id": "do", "agent_role": "worker"}, {"step_id": "verify", "agent_role": "verifier", "depends_on": ["do"]}],
    "memory": {"type": "summary_buffer"},
    "recovery": {"max_retries": 2, "retry_on": ["TOOL_ARGUMENT_ERROR", "VERIFICATION_FAILURE"], "backoff": "exponential"},
}

runner = CliRunner()


def _cfg(tmp_path: Path, *, with_model: bool = False) -> DataEvolConfig:
    kwargs = dict(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / "dataevol.sqlite3",
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="t",
    )
    if with_model:
        kwargs.update(model_name="frontier-architect", judge_model_name="frontier-judge",
                      model_endpoint="https://example.invalid/chat/completions", model_api_key="k")
    return DataEvolConfig(**kwargs)


def test_compat_evolve_without_model_returns_model_not_configured(tmp_path: Path):
    from dataevol.compat import call_core

    cfg = _cfg(tmp_path)
    result = call_core("harness", "evolve", {"task": '{"task_type": "permit_set_review"}', "max_generations": 1}, config=cfg)
    assert result["status"] == "model_not_configured"
    assert result["ok"] is False


def test_compat_evolve_with_monkeypatched_client_runs(tmp_path: Path, monkeypatch):
    from dataevol.compat import call_core
    from dataevol.harness import dispatch

    cfg = _cfg(tmp_path, with_model=True)
    fake = FakeModelClient(scripts=[
        ("Benchmark Builder", BENCHMARK_CASES),
        ("Mutator", [{"hypothesis": "h", "mutation": {"mode": "component", "target": "agents", "description": "d"}, "patch": IMPROVING_PATCH}]),
    ])
    monkeypatch.setattr(dispatch, "resolve_model_client", lambda config: fake)
    result = call_core("harness", "evolve", {"task": '{"task_type": "permit_set_review"}', "max_generations": 2, "number_of_candidates": 1}, config=cfg)
    assert result["status"] == "completed"
    assert result["promotions"] >= 1


def test_compat_evaluate_accepts_inline_genome_json_and_registers_it(tmp_path: Path):
    from dataevol.compat import call_core

    cfg = _cfg(tmp_path)
    genome = _genome()
    result = call_core(
        "harness",
        "evaluate",
        {
            "genome": json.dumps(genome.to_dict()),
            "benchmark": json.dumps([{"id": "n1", "category": "normal"}]),
            "repeated_runs": 2,
        },
        config=cfg,
    )
    assert result["status"] == "completed"
    assert result["evaluation_id"] >= 1
    assert result["evaluation"]["genome_id"] == genome.genome_id


def test_compat_failures_accepts_evaluation_path(tmp_path: Path, monkeypatch):
    from dataevol.compat import call_core
    from dataevol.harness import dispatch

    cfg = _cfg(tmp_path, with_model=True)
    genome = _genome()
    genome_path = tmp_path / "genome.json"
    eval_path = tmp_path / "eval.json"
    genome_path.write_text(json.dumps(genome.to_dict()), encoding="utf-8")
    eval_path.write_text(json.dumps({
        "genome_id": genome.genome_id,
        "quality": 0.5,
        "robustness": 0.4,
        "verifier_agreement": 0.3,
        "cost": 0.1,
        "latency": 500,
        "failure_rate": 0.2,
        "score": 0.42,
        "failure_categories": ["VERIFICATION_FAILURE"],
    }), encoding="utf-8")
    fake = FakeModelClient(scripts=[("Failure Analyst", {
        "failures": [{"category": "VERIFICATION_FAILURE", "earliest_cause": "missing verifier"}],
        "summary": "needs verifier",
    })])
    monkeypatch.setattr(dispatch, "resolve_model_client", lambda config: fake)
    result = call_core("harness", "failures", {"genome": str(genome_path), "evaluation": str(eval_path)}, config=cfg)
    assert result["status"] == "completed"
    assert result["analysis"]["failures"][0]["earliest_cause"] == "missing verifier"


def test_compat_judge_accepts_evaluation_json_strings(tmp_path: Path, monkeypatch):
    from dataevol.compat import call_core
    from dataevol.harness import dispatch

    cfg = _cfg(tmp_path, with_model=True)
    fake = FakeModelClient(scripts=[("Judge", {"verdict": "promotable", "reason": "better", "confidence": 0.9})])
    monkeypatch.setattr(dispatch, "resolve_model_client", lambda config: fake)
    incumbent = {"genome_id": "inc", "quality": 0.4, "robustness": 0.4, "verifier_agreement": 0.4, "cost": 0.1, "latency": 500, "failure_rate": 0.2, "score": 0.3}
    challenger = {"genome_id": "chal", "quality": 0.6, "robustness": 0.5, "verifier_agreement": 0.5, "cost": 0.1, "latency": 500, "failure_rate": 0.1, "score": 0.5}
    result = call_core(
        "harness",
        "judge",
        {"incumbent_eval": json.dumps(incumbent), "challenger_eval": json.dumps(challenger)},
        config=cfg,
    )
    assert result["status"] == "completed"
    assert result["review"]["verdict"] == "promotable"
    assert result["review"]["independent"] is True


def test_cli_harness_help_lists_commands():
    result = runner.invoke(cli_app, ["harness", "--help"])
    assert result.exit_code == 0
    assert "evolve" in result.output
    assert "lineage" in result.output


def test_cli_evolve_reports_model_not_configured(tmp_path: Path, monkeypatch):
    # Point DATAEVOL-free config at a tmp file with no [model] section.
    for key in (
        "DATAEVOL_MODEL_PROVIDER",
        "DATAEVOL_MODEL_ENDPOINT",
        "DATAEVOL_MODEL_API_KEY",
        "DATAEVOL_MODEL_NAME",
        "DATAEVOL_JUDGE_MODEL_NAME",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    config_path = tmp_path / "dataevol.toml"
    config_path.write_text(
        '[paths]\ndb = "%s"\nraw = "%s"\nartifacts = "%s"\n[api]\ntoken = "t"\n[privacy]\nmode = "private-local-only"\n'
        % (tmp_path / "db", tmp_path / "raw", tmp_path / "artifacts"),
        encoding="utf-8",
    )
    result = runner.invoke(cli_app, ["harness", "evolve", "--task", '{"task_type":"x"}', "--max-generations", "1", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "model_not_configured" in result.output


def test_api_harness_evolve_model_not_configured(tmp_path: Path):
    cfg = _cfg(tmp_path)
    client = TestClient(create_app(cfg))
    resp = client.post("/harness/evolve", headers={"Authorization": "Bearer t"}, json={"payload": {"task": '{"task_type":"x"}', "max_generations": 1}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "model_not_configured"


def test_api_harness_endpoints_require_token(tmp_path: Path):
    cfg = _cfg(tmp_path)
    client = TestClient(create_app(cfg))
    resp = client.post("/harness/evolve", json={"payload": {}})
    assert resp.status_code == 401


def test_api_harness_lineage_returns_empty_list_initially(tmp_path: Path):
    cfg = _cfg(tmp_path)
    client = TestClient(create_app(cfg))
    resp = client.get("/harness/lineage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["lineage"] == []


def test_compat_lineage_and_incumbent_read_paths(tmp_path: Path):
    from dataevol.compat import call_core

    cfg = _cfg(tmp_path)
    assert call_core("harness", "lineage", {}, config=cfg)["lineage"] == []
    assert call_core("harness", "incumbent", {}, config=cfg)["incumbent"] is None
    records = call_core("harness", "training_records", {}, config=cfg)
    assert records["status"] == "completed"
