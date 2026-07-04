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


DATAEVOL_SPECIALIST_SWARM = {
    "ingestor",
    "deduper",
    "cleaner",
    "classifier",
    "difficulty_scorer",
    "quality_scorer",
    "trace_compressor",
    "early_failure_detector",
    "verifier",
    "critic",
    "error_taxonomist",
    "synthetic_generator",
    "mutation_evolver",
    "recipe_generator",
    "recipe_verifier",
    "curriculum_builder",
    "dataset_mixer",
    "benchmark_generator",
    "regression_tester",
    "promotion_gatekeeper",
}

ECOSYSTEM_SPECIALISTS = {
    "router",
    "planner",
    "worker",
    "manager",
    "coordinator",
    "inspector",
    "compressor",
    "duplicate_detector",
    "failure_classifier",
    "prompt_improver",
    "prompt_pack_generator",
    "local_model_trainer",
    "local_evaluator",
    "model_mix_optimizer",
    "scientific_method",
    "coding_agent",
    "tool_trace_analyzer",
    "correction_linker",
    "benchmark_task",
    "privacy_redactor",
    "report_builder",
    "integration_bridge",
}


def test_expert_adapter_dataset_and_jobs(tmp_path):
    data_root = tmp_path / "data"
    adapter_root = tmp_path / "adapters"
    paths = write_expert_datasets(data_root, count=8)

    assert set(paths) == set(EXPERTS)
    assert set(EXPERTS) == DATAEVOL_SPECIALIST_SWARM | ECOSYSTEM_SPECIALISTS
    assert all((path / "train.jsonl").exists() for path in paths.values())
    assert expert_examples("ingestor", count=2)[0]["prompt"]
    assert "promotion_gatekeeper" in expert_examples("promotion_gatekeeper", count=1)[0]["prompt"]
    assert "benchmark_case" in expert_examples("benchmark_generator", count=1)[0]["completion"]
    assert "model_mix" in expert_examples("model_mix_optimizer", count=1)[0]["completion"]

    jobs = build_adapter_jobs("python", data_root, adapter_root, iters=2)
    assert len(jobs) == len(EXPERTS)
    assert jobs[0].command[:5] == ["python", "-m", "mlx_lm", "lora", "--model"]
    assert BASE_MODEL in jobs[0].command
    assert "--adapter-path" in jobs[0].command


def test_local_adapter_training_plan_writes_real_driver(tmp_path: Path) -> None:
    plan = prepare_local_adapter_training(tmp_path / "local", python_bin="python", experts=("ingestor", "critic"), count=6, iters=1)
    manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
    script = plan.script_path.read_text(encoding="utf-8")

    assert manifest["experts"] == ["ingestor", "critic"]
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
        {"output": str(output), "experts": ["ingestor"], "count": 4, "iters": 1},
        config=cfg,
    )
    assert prepared["status"] == "completed"
    assert Path(prepared["manifest_path"]).exists()

    planned = call_core(
        "local_models",
        "train",
        {"output": str(output), "experts": ["ingestor"], "count": 4, "iters": 1, "execute": False},
        config=cfg,
    )
    assert planned["status"] == "planned"
    assert planned["executed"] is False
    assert planned["jobs"][0]["command"][0].endswith("python")
    assert planned["jobs"][0]["command"][1:4] == ["-m", "mlx_lm", "lora"]

    cli_result = runner.invoke(
        app,
        [
            "local-model",
            "prepare",
            "--output",
            str(tmp_path / "cli-local"),
            "--expert",
            "ingestor",
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
        json={"payload": {"output": str(tmp_path / "api-local"), "experts": ["ingestor"], "count": 4, "iters": 1}},
    )
    assert response.status_code == 200
    assert response.json()["jobs"][0]["expert"] == "ingestor"

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "DataEvol Local Expert Training" in dashboard.text
    assert "Ornith 1.0 9B MLX 8-bit" in dashboard.text
    assert "ingestor" in dashboard.text
    assert "model_mix_optimizer" in dashboard.text
    assert "promotion_gatekeeper" in dashboard.text
    assert "Training Progress" in dashboard.text
    assert "/local_model/training/start" in dashboard.text
    assert "progressFill" in dashboard.text

    status = client.post(
        "/local_model/status",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {"output": str(output), "model": ".dataevol/models/Ornith-1.0-9B-8bit"}},
    )
    assert status.status_code == 200
    assert status.json()["experts"][0] == "ingestor"
    assert "ingestor" in status.json()["datasets"]
    assert "router" in status.json()["datasets"]
    assert "promotion_gatekeeper" in status.json()["datasets"]

    adapter_dir = output / "adapters" / "ingestor"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapters.safetensors"
    adapter_file.write_bytes(b"fake-adapter")
    export = client.post(
        "/local_model/artifacts/export",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {"output": str(output), "experts": ["ingestor"]}},
    )
    assert export.status_code == 200
    exported = export.json()
    assert exported["schema"] == "dataevol.local_model_artifact_export.v1"
    assert exported["adapters"]["ingestor"]["exists"] is True
    assert exported["adapters"]["ingestor"]["files"][0]["relative_path"] == "adapters/ingestor/adapters.safetensors"
    assert exported["adapters"]["ingestor"]["files"][0]["content_base64"]

    missing_job = client.post(
        "/local_model/training/status",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {"job_id": "missing"}},
    )
    assert missing_job.status_code == 404

    latest_job = client.post(
        "/local_model/training/latest",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {}},
    )
    assert latest_job.status_code == 200
    assert latest_job.json()["status"] == "idle"
