from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from dataevol.api.app import create_app
from dataevol.cli.main import app
from dataevol.compat import call_core
from dataevol.config import DataEvolConfig
from dataevol.local_models import (
    BASE_MODEL,
    EXPERTS,
    build_adapter_jobs,
    expert_examples,
    prepare_local_adapter_training,
    write_expert_datasets,
)


runner = CliRunner()


def test_expert_adapter_dataset_and_jobs(tmp_path):
    data_root = tmp_path / "data"
    adapter_root = tmp_path / "adapters"
    paths = write_expert_datasets(data_root, count=8)

    assert set(paths) == set(EXPERTS)
    assert all((path / "train.jsonl").exists() for path in paths.values())
    assert expert_examples("router", count=2)[0]["prompt"]

    jobs = build_adapter_jobs("python", data_root, adapter_root, iters=2)
    assert len(jobs) == len(EXPERTS)
    assert jobs[0].command[:5] == ["python", "-m", "mlx_lm", "lora", "--model"]
    assert BASE_MODEL in jobs[0].command
    assert "--adapter-path" in jobs[0].command


def test_local_adapter_training_plan_writes_real_driver(tmp_path: Path) -> None:
    plan = prepare_local_adapter_training(tmp_path / "local", python_bin="python", experts=("router", "critic"), count=6, iters=1)
    manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
    script = plan.script_path.read_text(encoding="utf-8")

    assert manifest["experts"] == ["router", "critic"]
    assert manifest["jobs"][0]["command"][:4] == ["python", "-m", "mlx_lm", "lora"]
    assert "run_local_adapter_training_from_manifest" in script
    assert "placeholder" not in script.lower()


def test_local_models_compat_cli_and_api_are_wired(tmp_path: Path) -> None:
    cfg = DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / ".dataevol/dataevol.sqlite3",
        raw_path=tmp_path / ".dataevol/raw",
        artifacts_path=tmp_path / ".dataevol/artifacts",
        api_token="secret",
    )
    output = tmp_path / "local"

    prepared = call_core(
        "local_models",
        "prepare",
        {"output": str(output), "experts": ["router"], "count": 4, "iters": 1},
        config=cfg,
    )
    assert prepared["status"] == "completed"
    assert Path(prepared["manifest_path"]).exists()

    planned = call_core(
        "local_models",
        "train",
        {"output": str(output), "experts": ["router"], "count": 4, "iters": 1, "execute": False},
        config=cfg,
    )
    assert planned["status"] == "planned"
    assert planned["executed"] is False
    assert planned["jobs"][0]["command"][:4] == ["python", "-m", "mlx_lm", "lora"]

    cli_result = runner.invoke(
        app,
        [
            "local-model",
            "prepare",
            "--output",
            str(tmp_path / "cli-local"),
            "--expert",
            "router",
            "--count",
            "4",
            "--iters",
            "1",
        ],
    )
    assert cli_result.exit_code == 0
    assert "adapter_training_manifest.json" in cli_result.output

    client = TestClient(create_app(cfg))
    unauthorized = client.post("/local_model/prepare", json={"payload": {"output": str(tmp_path / "api-local")}})
    assert unauthorized.status_code == 401
    response = client.post(
        "/local_model/prepare",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {"output": str(tmp_path / "api-local"), "experts": ["router"], "count": 4, "iters": 1}},
    )
    assert response.status_code == 200
    assert response.json()["jobs"][0]["expert"] == "router"
