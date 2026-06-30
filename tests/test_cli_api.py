from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from dataevol.api.app import create_app
from dataevol.cli.main import app
from dataevol.config import DataEvolConfig

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DataEvol trace evolution CLI" in result.output
    assert "ingest" in result.output


def test_init_writes_config_and_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "dataevol.toml"
    result = runner.invoke(app, ["init", "--config", str(config_path)])
    assert result.exit_code == 0
    assert config_path.exists()
    assert (tmp_path / ".dataevol").exists()


def test_health_unauthenticated() -> None:
    cfg = DataEvolConfig(
        path=Path("dataevol.toml"),
        db_path=Path(".dataevol/dataevol.sqlite3"),
        raw_path=Path(".dataevol/raw"),
        artifacts_path=Path(".dataevol/artifacts"),
        api_token="secret",
    )
    client = TestClient(create_app(cfg))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_ingest_trace_requires_token_and_ingests(tmp_path: Path) -> None:
    cfg = DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / ".dataevol/dataevol.sqlite3",
        raw_path=tmp_path / ".dataevol/raw",
        artifacts_path=tmp_path / ".dataevol/artifacts",
        api_token="secret",
    )
    client = TestClient(create_app(cfg))
    unauthorized = client.post("/ingest_trace", json={"trace": {"type": "router_trace"}})
    assert unauthorized.status_code == 401

    response = client.post(
        "/ingest_trace",
        headers={"Authorization": "Bearer secret"},
        json={"trace": {"trace_type": "router_trace", "objective": "demo route"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "completed"
    assert body["operation"] == "ingest_trace"
    assert body["accepted"] == 1
    assert body["trace_ids"]
