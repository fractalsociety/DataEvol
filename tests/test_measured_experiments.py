from __future__ import annotations

import json
from pathlib import Path

from dataevol.compat import call_core
from dataevol.config import DataEvolConfig
from dataevol.experiments import run_measured_router_policy_experiment
from dataevol.ingest import ingest_jsonl
from dataevol.label import label_run
from dataevol.score import score_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _prepare_run(tmp_path: Path, traces: list[dict]) -> tuple[Path, int]:
    db_path = tmp_path / "dataevol.sqlite"
    jsonl = tmp_path / "traces.jsonl"
    _write_jsonl(jsonl, traces)
    report = ingest_jsonl(jsonl, db_path, source_system="measured-test", raw_root=tmp_path / "raw")
    label_run(db_path, report.run_id)
    score_run(db_path, report.run_id)
    return db_path, report.run_id


def test_measured_experiment_can_win_from_observed_provider_performance(tmp_path: Path) -> None:
    traces = [
        {
            "trace_type": "router_trace",
            "task_id": "openrouter-good",
            "provider": "openrouter",
            "model": "cheap",
            "prompt": "Route low risk documentation",
            "response": "Accepted and verified.",
            "tests_run": [{"name": "verify", "status": "passed"}],
            "metrics": {"cost_usd": 0.01, "latency_ms": 500},
        },
        *[
            {
                "trace_type": "router_trace",
                "task_id": f"frontier-{index}",
                "provider": "frontier",
                "model": "expensive",
                "prompt": f"Route low risk documentation {index}",
                "response": "Accepted and verified.",
                "tests_run": [{"name": "verify", "status": "passed"}],
                "metrics": {"cost_usd": 0.10, "latency_ms": 1200},
            }
            for index in range(5)
        ],
    ]
    db_path, run_id = _prepare_run(tmp_path, traces)

    report = run_measured_router_policy_experiment(db_path, tmp_path / "experiments", run_id=run_id)

    assert report["measurement_source"] == "sqlite"
    assert report["evaluated_trace_count"] == 6
    assert report["primary_metric_improved"] is True
    assert report["regressions"] == []
    assert report["reproducible_runs"] >= 2


def test_measured_experiment_rejects_regressing_variant(tmp_path: Path) -> None:
    traces = [
        {
            "trace_type": "router_trace",
            "task_id": "openrouter-bad",
            "provider": "openrouter",
            "model": "cheap",
            "prompt": "Route low risk documentation",
            "response": "Verification failed; pytest failed.",
            "tests_run": [{"name": "verify", "status": "failed"}],
            "metrics": {"cost_usd": 0.20, "latency_ms": 2500},
        },
        *[
            {
                "trace_type": "router_trace",
                "task_id": f"frontier-good-{index}",
                "provider": "frontier",
                "model": "expensive",
                "prompt": f"Route low risk documentation {index}",
                "response": "Accepted and verified.",
                "tests_run": [{"name": "verify", "status": "passed"}],
                "metrics": {"cost_usd": 0.03, "latency_ms": 700},
            }
            for index in range(5)
        ],
    ]
    db_path, run_id = _prepare_run(tmp_path, traces)

    report = run_measured_router_policy_experiment(db_path, tmp_path / "experiments", run_id=run_id)

    assert report["primary_metric_improved"] is False
    assert "correctness" in report["regressions"]
    assert report["verification_passed"] is False
    assert report["reproducible_runs"] == 0


def test_compat_experiment_no_longer_hardcodes_winning_fixture(tmp_path: Path) -> None:
    cfg = DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / "empty.sqlite",
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="secret",
    )

    result = call_core("evolve", "experiment", {}, config=cfg)

    assert result["ok"] is True
    assert result["operation"] == "experiment"
    assert result["primary_metric_improved"] is False
    assert result["status"] == "rejected_no_measured_data"
    assert "no_measured_trace_data" in result["regressions"]


def test_compat_promote_rejects_saved_measured_regression(tmp_path: Path) -> None:
    traces = [
        {
            "trace_type": "router_trace",
            "task_id": "openrouter-bad",
            "provider": "openrouter",
            "model": "cheap",
            "prompt": "Route low risk documentation",
            "response": "Verification failed; pytest failed.",
            "tests_run": [{"name": "verify", "status": "failed"}],
            "metrics": {"cost_usd": 0.20, "latency_ms": 2500},
        },
        {
            "trace_type": "router_trace",
            "task_id": "frontier-good",
            "provider": "frontier",
            "model": "expensive",
            "prompt": "Route low risk documentation safely",
            "response": "Accepted and verified.",
            "tests_run": [{"name": "verify", "status": "passed"}],
            "metrics": {"cost_usd": 0.03, "latency_ms": 700},
        },
    ]
    db_path, run_id = _prepare_run(tmp_path, traces)
    cfg = DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=db_path,
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="secret",
    )

    experiment = call_core("evolve", "experiment", {"run_id": run_id}, config=cfg)
    assert experiment["verification_passed"] is False
    assert experiment["regressions"]

    comparison = call_core("evolve", "compare", {"experiment": experiment["experiment_id"]}, config=cfg)
    assert comparison["verdict"] == "reject"

    promotion = call_core("evolve", "promote", {"experiment": experiment["experiment_id"]}, config=cfg)
    assert promotion["ok"] is False
    assert promotion["status"] == "rejected"
    assert "verification pass rate declined" in promotion["detail"]
