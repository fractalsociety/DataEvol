from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataevol.api.app import create_app
from dataevol.compat import call_core
from dataevol.config import DataEvolConfig
from dataevol.harness import storage as harness_storage
from dataevol.harness.verdicts import VERDICT_SCHEMA, canonical_json, issue_harness_verdict


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _cfg(tmp_path: Path) -> DataEvolConfig:
    return DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / "dataevol.sqlite3",
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="secret",
    )


def _rollback(path: Path) -> str:
    path.write_text(json.dumps({"state": {"genome_id": "incumbent-1"}}), encoding="utf-8")
    return str(path)


def _good_report(tmp_path: Path) -> dict:
    return {
        "genome_id": "candidate-1",
        "incumbent_genome_id": "incumbent-1",
        "task_type": "permit_review",
        "median_quality_improved": 0.05,
        "quality_delta": 0.05,
        "bootstrap": [0.05, 0.02, 0.08],
        "bootstrap_confidence": 0.95,
        "judge_independent": True,
        "critical_benchmark_regressions": [],
        "cost_delta": 0.02,
        "failure_rate_delta": -0.01,
        "reproducible_runs": 3,
        "rollback_snapshot": _rollback(tmp_path / "rollback.json"),
    }


def _payload(tmp_path: Path, *, executor_kind: str = "fractalwork-runtime-v1") -> dict:
    return {
        "verdict_id": "verdict-1",
        "task_type": "permit_review",
        "incumbent_genome_id": "incumbent-1",
        "candidate_genome_id": "candidate-1",
        "candidate_content_hash": HASH_A.upper(),
        "benchmark_hash": HASH_B,
        "evidence_hash": HASH_C,
        "executor_kind": executor_kind,
        "created_at": "2026-07-10T12:00:00+00:00",
        "report": _good_report(tmp_path),
    }


def test_complete_measured_report_issues_eligible_canonical_verdict(tmp_path: Path):
    verdict = issue_harness_verdict(_payload(tmp_path))
    data = verdict.to_dict()

    assert set(data) == {
        "schema", "verdict_id", "verdict", "task_type", "incumbent_genome_id",
        "candidate_genome_id", "candidate_content_hash", "benchmark_hash",
        "evidence_hash", "executor_kind", "reasons", "created_at", "verdict_hash",
    }
    assert data["schema"] == VERDICT_SCHEMA
    assert data["verdict"] == "ELIGIBLE"
    assert data["candidate_content_hash"] == HASH_A
    unsigned = {key: value for key, value in data.items() if key != "verdict_hash"}
    assert data["verdict_hash"] == hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    assert verdict.verify_hash()


def test_complete_failed_gate_report_issues_rejected(tmp_path: Path):
    payload = _payload(tmp_path)
    payload["report"]["median_quality_improved"] = 0.001
    verdict = issue_harness_verdict(payload)

    assert verdict.verdict == "REJECTED"
    assert any("median quality" in reason for reason in verdict.reasons)


@pytest.mark.parametrize(
    "executor_kind",
    ["ReferenceExecutor", "reference_executor", "reference", "simulation", "simulated-runtime", "simulator"],
)
def test_simulation_executor_is_always_inconclusive(tmp_path: Path, executor_kind: str):
    payload = _payload(tmp_path, executor_kind=executor_kind)
    payload["report"]["median_quality_improved"] = 1.0
    verdict = issue_harness_verdict(payload)

    assert verdict.verdict == "INCONCLUSIVE"
    assert any("cannot authorize" in reason for reason in verdict.reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap", None),
        ("bootstrap", [0.1, 0.2]),
        ("bootstrap_confidence", float("nan")),
        ("reproducible_runs", "3"),
        ("judge_independent", None),
    ],
)
def test_incomplete_or_malformed_statistical_evidence_is_inconclusive(
    tmp_path: Path, field: str, value: object
):
    payload = _payload(tmp_path)
    if value is None:
        payload["report"].pop(field)
    else:
        payload["report"][field] = value

    verdict = issue_harness_verdict(payload)
    assert verdict.verdict == "INCONCLUSIVE"
    assert any(field in reason for reason in verdict.reasons)


def test_contract_hashes_must_be_sha256_hex(tmp_path: Path):
    payload = _payload(tmp_path)
    payload["evidence_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="evidence_hash"):
        issue_harness_verdict(payload)


def test_storage_round_trip_and_immutable_idempotency(tmp_path: Path):
    cfg = _cfg(tmp_path)
    verdict = issue_harness_verdict(_payload(tmp_path))

    assert harness_storage.register_verdict(cfg.db_path, verdict) == "verdict-1"
    assert harness_storage.register_verdict(cfg.db_path, verdict) == "verdict-1"
    assert harness_storage.load_verdict(cfg.db_path, "verdict-1") == verdict.to_dict()

    different = _payload(tmp_path)
    different["evidence_hash"] = "d" * 64
    with pytest.raises(ValueError, match="different content"):
        harness_storage.register_verdict(cfg.db_path, issue_harness_verdict(different))


def test_storage_rejects_a_tampered_persisted_verdict(tmp_path: Path):
    cfg = _cfg(tmp_path)
    harness_storage.register_verdict(cfg.db_path, issue_harness_verdict(_payload(tmp_path)))
    with sqlite3.connect(cfg.db_path) as conn:
        conn.execute(
            "UPDATE harness_verdicts SET reasons = ? WHERE verdict_id = ?",
            ('["tampered"]', "verdict-1"),
        )

    with pytest.raises(ValueError, match="verdict_hash"):
        harness_storage.load_verdict(cfg.db_path, "verdict-1")


def test_dispatch_creates_and_loads_verdict(tmp_path: Path):
    cfg = _cfg(tmp_path)
    created = call_core("harness", "verdict", _payload(tmp_path), config=cfg)
    loaded = call_core("harness", "get_verdict", {"verdict_id": "verdict-1"}, config=cfg)

    assert created["status"] == "completed"
    assert created["verdict"]["verdict"] == "ELIGIBLE"
    assert loaded["verdict"] == created["verdict"]
    assert call_core("harness", "get_verdict", {"verdict_id": "missing"}, config=cfg)["status"] == "not_found"


def test_api_post_and_get_verdict_are_authenticated(tmp_path: Path):
    cfg = _cfg(tmp_path)
    client = TestClient(create_app(cfg))

    assert client.post("/harness/verdict", json=_payload(tmp_path)).status_code == 401
    created = client.post(
        "/harness/verdict",
        headers={"Authorization": "Bearer secret"},
        json={"payload": _payload(tmp_path)},
    )
    assert created.status_code == 200
    assert created.json()["verdict"]["verdict"] == "ELIGIBLE"

    loaded = client.get(
        "/harness/verdicts/verdict-1", headers={"Authorization": "Bearer secret"}
    )
    assert loaded.status_code == 200
    assert loaded.json()["verdict"] == created.json()["verdict"]
    assert client.get("/harness/verdicts/verdict-1").status_code == 401
    missing = client.get(
        "/harness/verdicts/missing", headers={"Authorization": "Bearer secret"}
    )
    assert missing.status_code == 404
