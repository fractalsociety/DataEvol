from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dataevol.config import DataEvolConfig
from dataevol.harness import storage as harness_storage
from dataevol.harness.loop import EvolutionConfig, run_harness_evolution
from dataevol.harness.model_client import FakeModelClient
from dataevol.harness.promotion import HarnessPromotionGate

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
    "workflow": [
        {"step_id": "do", "agent_role": "worker"},
        {"step_id": "verify", "agent_role": "verifier", "depends_on": ["do"]},
    ],
    "memory": {"type": "summary_buffer"},
    "recovery": {"max_retries": 2, "retry_on": ["TOOL_ARGUMENT_ERROR", "VERIFICATION_FAILURE"], "backoff": "exponential"},
}

NEUTRAL_PATCH = {"router": {"confidence_threshold": 0.5}}  # identical to architect default -> no improvement


def _cfg(tmp_path: Path) -> DataEvolConfig:
    return DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / "dataevol.sqlite3",
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="t",
        model_name="frontier-architect",
        judge_model_name="frontier-judge",
    )


def _client(patch: dict) -> FakeModelClient:
    return FakeModelClient(scripts=[
        ("Benchmark Builder", BENCHMARK_CASES),
        ("Mutator", [{
            "hypothesis": "add verifier + retries + memory",
            "mutation": {"mode": "component", "target": "agents", "description": "add independent verifier"},
            "patch": patch,
            "expected_effect": {"quality": 0.2},
            "affected_tests": ["adversarial", "edge"],
        }]),
    ])


def test_loop_promotes_improving_candidate_and_records_everything(tmp_path: Path):
    cfg = _cfg(tmp_path)
    result = run_harness_evolution(
        {"task_type": "permit_set_review"},
        config=cfg,
        model_client=_client(IMPROVING_PATCH),
        evolution=EvolutionConfig(max_generations=2, number_of_candidates=1, repeated_runs=3, plateau_window=5),
    )
    assert result.promotions >= 1
    assert any(node.promoted for node in result.lineage)
    # incumbent advanced to a genome with a verifier
    assert any(a.role == "verifier" for a in result.incumbent.agents)

    # checkpoint written for the promoted incumbent
    checkpoints = list((tmp_path / "artifacts" / "harness" / "checkpoints").glob("incumbent_*.json"))
    assert checkpoints

    # DB populated: training records, lineage, experiments
    records = harness_storage.load_training_records(cfg.db_path)
    assert len(records) >= result.generations  # one per generation
    lineage = harness_storage.load_lineage(cfg.db_path)
    assert any(row["promoted"] for row in lineage)
    incumbent = harness_storage.load_incumbent(cfg.db_path, task_id=result.task_id)
    assert incumbent is not None and any(a["role"] == "verifier" for a in incumbent["agents"])
    rollback = json.loads(Path(result.incumbent_rollback_snapshot).read_text(encoding="utf-8"))
    assert rollback["state"] == result.incumbent.to_dict()

    with sqlite3.connect(cfg.db_path) as conn:
        n_exp = conn.execute("SELECT COUNT(*) FROM harness_experiments").fetchone()[0]
    assert n_exp >= 1


def test_loop_rejects_neutral_candidate_but_still_emits_training_record(tmp_path: Path):
    cfg = _cfg(tmp_path)
    result = run_harness_evolution(
        {"task_type": "permit_set_review"},
        config=cfg,
        model_client=_client(NEUTRAL_PATCH),
        evolution=EvolutionConfig(max_generations=2, number_of_candidates=1, repeated_runs=3, plateau_window=5),
    )
    assert result.promotions == 0
    assert all(not node.promoted for node in result.lineage)
    records = harness_storage.load_training_records(cfg.db_path)
    assert len(records) >= 1
    assert all(r["promotion_decision"] == "rejected" for r in records)


def test_loop_survives_specialist_error_and_continues(tmp_path: Path):
    cfg = _cfg(tmp_path)
    # Mutator returns malformed JSON -> SpecialistError each generation; loop must not crash.
    client = FakeModelClient(scripts=[
        ("Benchmark Builder", BENCHMARK_CASES),
        ("Mutator", "not json {{{"),
    ])
    result = run_harness_evolution(
        {"task_type": "permit_set_review"},
        config=cfg,
        model_client=client,
        evolution=EvolutionConfig(max_generations=2, number_of_candidates=1, repeated_runs=2, plateau_window=5),
    )
    assert result.generations == 2
    assert result.promotions == 0


def test_rerun_resumes_stored_incumbent_instead_of_replacing_it(tmp_path: Path):
    cfg = _cfg(tmp_path)
    first = run_harness_evolution(
        {"task_type": "permit_set_review"},
        config=cfg,
        model_client=_client(IMPROVING_PATCH),
        evolution=EvolutionConfig(max_generations=1, number_of_candidates=1, repeated_runs=3),
    )
    second = run_harness_evolution(
        {"task_type": "permit_set_review"},
        config=cfg,
        model_client=_client(NEUTRAL_PATCH),
        evolution=EvolutionConfig(max_generations=0),
        benchmark=BENCHMARK_CASES,
    )
    assert second.incumbent.genome_id == first.incumbent.genome_id


def test_promotion_artifact_failure_does_not_advance_incumbent(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(HarnessPromotionGate, "promote", fail_write)
    result = run_harness_evolution(
        {"task_type": "permit_set_review"},
        config=cfg,
        model_client=_client(IMPROVING_PATCH),
        evolution=EvolutionConfig(max_generations=1, number_of_candidates=1, repeated_runs=3),
    )
    assert result.promotions == 0
    assert not any(agent.role == "verifier" for agent in result.incumbent.agents)
    assert harness_storage.load_lineage(cfg.db_path)[-1]["promoted"] is False
