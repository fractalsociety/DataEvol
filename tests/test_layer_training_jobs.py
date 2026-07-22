from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataevol.config import DataEvolConfig


def test_interrupted_layer_job_is_durable_recoverable_and_retry_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = importlib.import_module("dataevol.api.app")
    cfg = _cfg(tmp_path)
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n", encoding="utf-8")
    started: list[str] = []
    monkeypatch.setattr(api, "_start_layer_training_job", lambda job: started.append(str(job["job_id"])))
    client = TestClient(api.create_app(cfg))
    response = client.post(
        "/local_model/layerscope/train_layer_specialist",
        headers=_auth(),
        json={"payload": {
            "base_model": "org/remote-model",
            "base_model_revision": "a" * 40,
            "output": str(tmp_path / "output"),
            "task_type": "compression",
            "training_mode": "sft",
            "layer_index": 1,
            "dataset_uri": str(dataset),
            "execute": True,
            "timeout_seconds": 30,
        }},
    )
    assert response.status_code == 200
    original_id = response.json()["job_id"]
    api._update_training_job(original_id, status="running", started_at=1.0)

    restarted = TestClient(api.create_app(cfg))
    recovered = restarted.post(
        "/local_model/training/status", headers=_auth(), json={"payload": {"job_id": original_id}}
    ).json()
    assert recovered["status"] == "failed"
    assert recovered["recoverable"] is True
    assert "service restart" in recovered["error"]
    assert recovered["normalized_payload"]["dataset_sha256"]

    retry = restarted.post(
        "/local_model/training/retry", headers=_auth(), json={"payload": {"job_id": original_id}}
    )
    assert retry.status_code == 200
    retry_body = retry.json()
    assert retry_body["status"] == "queued"
    assert retry_body["retry_of"] == original_id
    again = restarted.post(
        "/local_model/training/retry", headers=_auth(), json={"payload": {"job_id": original_id}}
    ).json()
    assert again["job_id"] == retry_body["job_id"]
    assert started == [original_id, retry_body["job_id"]]

    with sqlite3.connect(cfg.db_path) as conn:
        rows = conn.execute("SELECT job_id, status, normalized_payload FROM local_model_training_jobs ORDER BY created_at").fetchall()
    assert [row[0] for row in rows] == [original_id, retry_body["job_id"]]
    assert json.loads(rows[1][2])["base_model_revision"] == "a" * 40


def test_interrupted_dashboard_job_uses_the_same_durable_retry_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("dataevol.api.app")
    cfg = _cfg(tmp_path)
    started: list[str] = []
    monkeypatch.setattr(api, "_start_dashboard_training_job", lambda job: started.append(str(job["job_id"])))
    client = TestClient(api.create_app(cfg))

    original = client.post(
        "/local_model/training/start",
        headers=_auth(),
        json={"payload": {"output": str(tmp_path / "dashboard-output"), "experts": ["verifier"]}},
    ).json()
    api._update_training_job(original["job_id"], status="running", started_at=1.0)

    restarted = TestClient(api.create_app(cfg))
    recovered = restarted.post(
        "/local_model/training/status",
        headers=_auth(),
        json={"payload": {"job_id": original["job_id"]}},
    ).json()
    assert recovered["status"] == "failed"
    assert recovered["recoverable"] is True

    retry = restarted.post(
        "/local_model/training/retry",
        headers=_auth(),
        json={"payload": {"job_id": original["job_id"]}},
    ).json()
    again = restarted.post(
        "/local_model/training/retry",
        headers=_auth(),
        json={"payload": {"job_id": original["job_id"]}},
    ).json()
    assert retry["job_type"] == "local_model_training"
    assert retry["retry_of"] == original["job_id"]
    assert again["job_id"] == retry["job_id"]
    assert started == [original["job_id"], retry["job_id"]]


def test_layer_subprocess_timeout_terminates_process(monkeypatch: pytest.MonkeyPatch) -> None:
    api = importlib.import_module("dataevol.api.app")
    job_id = "timeout-test"
    api.TRAINING_CANCEL_EVENTS.pop(job_id, None)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import os,time; os.close(1); os.close(2); time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with pytest.raises(TimeoutError, match="exceeded timeout"):
        api._stream_training_process(job_id, proc, timeout_seconds=0.1, on_line=lambda line: None)
    assert proc.poll() is not None


def test_layer_subprocess_cancel_terminates_process() -> None:
    api = importlib.import_module("dataevol.api.app")
    job_id = "cancel-test"
    event = threading.Event()
    event.set()
    api.TRAINING_CANCEL_EVENTS[job_id] = event
    proc = subprocess.Popen(
        [sys.executable, "-c", "import os,time; os.close(1); os.close(2); time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with pytest.raises(api._TrainingCancelled, match="cancelled"):
            api._stream_training_process(job_id, proc, timeout_seconds=30, on_line=lambda line: None)
        assert proc.poll() is not None
    finally:
        api.TRAINING_CANCEL_EVENTS.pop(job_id, None)


def _cfg(tmp_path: Path) -> DataEvolConfig:
    return DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / ".dataevol/dataevol.sqlite3",
        raw_path=tmp_path / ".dataevol/raw",
        artifacts_path=tmp_path / ".dataevol/artifacts",
        api_token="secret",
    )


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer secret"}
