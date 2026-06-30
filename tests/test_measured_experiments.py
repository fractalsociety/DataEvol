from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from dataevol.api.app import create_app
from dataevol.compat import call_core
from dataevol.config import DataEvolConfig
from dataevol.experiments import run_measured_router_policy_experiment, run_router_policy_benchmark
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
    assert Path(report["benchmark_path"]).exists()
    assert Path(report["benchmark_manifest_path"]).exists()
    assert report["benchmark_execution"]["case_count"] == 6
    assert report["benchmark_execution"]["eligible_variant_cases"] == 5
    assert report["primary_metric_improved"] is True
    assert report["regressions"] == []
    assert report["reproducible_runs"] >= 2

    replay = run_router_policy_benchmark(
        report["benchmark_path"],
        report["variant_provider_profile"],
        "openrouter",
        reproducibility_requirement=2,
    )
    assert replay["case_count"] == 6
    assert replay["variant_metrics"][0]["cost_per_verified_task"] < replay["control_metrics"][0]["cost_per_verified_task"]


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

    assert Path(report["benchmark_path"]).exists()
    assert report["benchmark_execution"]["case_count"] == 6
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
    assert Path(experiment["benchmark_path"]).exists()
    assert experiment["benchmark_execution"]["case_count"] == 2
    assert experiment["verification_passed"] is False
    assert experiment["regressions"]

    comparison = call_core("evolve", "compare", {"experiment": experiment["experiment_id"]}, config=cfg)
    assert comparison["verdict"] == "reject"

    promotion = call_core("evolve", "promote", {"experiment": experiment["experiment_id"]}, config=cfg)
    assert promotion["ok"] is False
    assert promotion["status"] == "rejected"
    assert "verification pass rate declined" in promotion["detail"]


def test_builders_experiment_and_promotion_register_in_db_reports(tmp_path: Path) -> None:
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
                "task_id": f"frontier-good-{index}",
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
    cfg = DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=db_path,
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="secret",
    )

    dataset = call_core("datasets", "build_dataset", {"type": "router", "run_id": run_id}, config=cfg)
    benchmark = call_core("benchmarks", "build_benchmark", {"type": "router", "from_runs": "last_100"}, config=cfg)
    reflection = call_core("evolve", "reflect", {"run_id": run_id}, config=cfg)
    idea = call_core("evolve", "idea_prd", {"opportunity_id": reflection["opportunity_ids"][0]}, config=cfg)
    experiment = call_core("evolve", "experiment", {"run_id": run_id, "idea": idea["path"]}, config=cfg)
    comparison = call_core("evolve", "compare", {"experiment": experiment["experiment_id"]}, config=cfg)
    promotion = call_core("evolve", "promote", {"experiment": experiment["experiment_id"]}, config=cfg)

    assert dataset["db_dataset_id"]
    assert dataset["source_trace_count"] == 6
    assert benchmark["db_benchmark_id"]
    assert idea["db_idea_prd_id"]
    assert experiment["db_experiment_id"]
    assert comparison["verdict"] == "promotable"
    assert promotion["db_promotion_id"]

    assert call_core("reports", "datasets", {}, config=cfg)["datasets"]
    assert call_core("reports", "benchmarks", {}, config=cfg)["benchmarks"]
    assert call_core("reports", "opportunities", {}, config=cfg)["opportunities"]
    assert call_core("reports", "idea_prds", {}, config=cfg)["idea_prds"]
    assert call_core("reports", "experiments", {}, config=cfg)["experiments"]
    assert call_core("reports", "promotions", {}, config=cfg)["promotions"]


def test_reached_privacy_router_prompt_and_integration_helpers(tmp_path: Path) -> None:
    traces = [
        {
            "trace_type": "router_trace",
            "task_id": "public-route",
            "provider": "openrouter",
            "model": "cheap",
            "prompt": "Route public benchmark task",
            "response": "Accepted and verified.",
            "privacy_mode": "public-benchmark-contribution",
            "tests_run": [{"name": "verify", "status": "passed"}],
            "metrics": {"cost_usd": 0.01, "latency_ms": 500},
        }
    ]
    db_path = tmp_path / "dataevol.sqlite"
    jsonl = tmp_path / "public.jsonl"
    _write_jsonl(jsonl, traces)
    report = ingest_jsonl(
        jsonl,
        db_path,
        source_system="public-test",
        privacy_mode="public-benchmark-contribution",
        raw_root=tmp_path / "raw",
    )
    label_run(db_path, report.run_id)
    score_run(db_path, report.run_id)
    cfg = DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=db_path,
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="secret",
        privacy_mode="public-benchmark-contribution",
    )

    exported = call_core("privacy", "export_training_candidates", {"run_id": report.run_id, "public": True}, config=cfg)
    performance = call_core("datasets", "router_performance", {"run_id": report.run_id}, config=cfg)
    policy = call_core("datasets", "candidate_router_policy", {"run_id": report.run_id}, config=cfg)
    prompt = call_core("prompts", "variants", {"pack": {"manager": "plan"}}, config=cfg)
    test_result = call_core(
        "prompts",
        "ab_test",
        {"control_metrics": {"success_rate": 0.7}, "variant_metrics": {"success_rate": 0.8, "hallucination_rate": 0.0}},
        config=cfg,
    )
    promoted_prompt = call_core("prompts", "promote", {"test_result": test_result}, config=cfg)
    manifest = tmp_path / "router_dataset.manifest.json"
    manifest.write_text(json.dumps({"name": "router_dataset", "path": "router.jsonl"}), encoding="utf-8")
    pulled = call_core("integrations", "router_dataset_pull", {"manifest": str(manifest)}, config=cfg)

    assert exported["candidate_count"] == 1
    assert Path(exported["path"]).exists()
    assert performance["rows"][0]["provider"] == "openrouter"
    assert policy["policy"]["preferred_providers"] == ["openrouter"]
    assert "verifiable" in prompt["pack"]["manager"]
    assert promoted_prompt["promoted"] is True
    assert pulled["dataset"]["name"] == "router_dataset"

    client = TestClient(create_app(cfg))
    api_response = client.post(
        "/prompts/variants",
        headers={"Authorization": "Bearer secret"},
        json={"payload": {"pack": {"manager": "plan"}}},
    )
    assert api_response.status_code == 200
    assert "verifiable" in api_response.json()["pack"]["manager"]


def test_integration_router_dataset_pull_uses_http(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            body = json.dumps({"name": "remote_router_dataset", "path": "remote.jsonl"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib callback signature
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = DataEvolConfig(
            path=tmp_path / "dataevol.toml",
            db_path=tmp_path / "dataevol.sqlite",
            raw_path=tmp_path / "raw",
            artifacts_path=tmp_path / "artifacts",
            api_token="secret",
        )
        result = call_core(
            "integrations",
            "router_dataset_pull",
            {"manifest": "unused", "endpoint": f"http://127.0.0.1:{server.server_port}"},
            config=cfg,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result["dataset"]["name"] == "remote_router_dataset"
