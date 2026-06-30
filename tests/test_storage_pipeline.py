from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataevol.compress import compress_run
from dataevol.ingest import ingest_jsonl
from dataevol.label import label_run
from dataevol.privacy.export import assert_public_export_allowed, build_training_candidate
from dataevol.score import score_run
from dataevol.storage import connect


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_ingest_label_score_compress_and_redact(tmp_path: Path) -> None:
    trace = {
        "trace_type": "coding_trace",
        "task_id": "task-1",
        "agent_id": "worker-a",
        "provider": "OpenRouter",
        "model": "free-model",
        "prompt": "Please fix redaction-fixture@example.invalid using api_key=redaction_fixture_api_key_000000",
        "response": "Tests passed. Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
        "tests_run": [{"name": "pytest", "status": "passed"}],
        "metrics": {"cost_usd": 0.0, "latency_ms": 1000},
    }
    failing = {
        "trace_type": "coding_trace",
        "task_id": "task-2",
        "prompt": "Fix the router timeout",
        "response": "Verification failed; pytest failed.",
        "tests_run": [{"name": "pytest", "status": "failed"}],
        "metrics": {"cost_usd": 0.01, "latency_ms": 2000},
    }
    jsonl_path = tmp_path / "traces.jsonl"
    _write_jsonl(jsonl_path, [trace, trace, failing, {"trace_type": "unknown"}])

    db_path = tmp_path / "dataevol.sqlite"
    report = ingest_jsonl(
        jsonl_path,
        db_path,
        source_system="coordinate",
        raw_root=tmp_path / ".dataevol" / "raw",
    )

    assert report.accepted == 2
    assert report.duplicates == 1
    assert report.rejected == 1

    labels = label_run(db_path, report.run_id)
    scores = score_run(db_path, report.run_id)
    compressed = compress_run(db_path, report.run_id)

    assert [item["label"] for item in labels] == ["accepted", "failed_tests"]
    assert scores[0]["correctness_score"] == 1.0
    assert scores[1]["correctness_score"] == 0.0
    assert compressed[1]["failure_type"] == "failed_code_tests"

    raw_files = list((tmp_path / ".dataevol" / "raw").glob("run_*/*.json"))
    assert len(raw_files) == 3
    raw_text = "\n".join(path.read_text(encoding="utf-8") for path in raw_files)
    assert "redaction-fixture@example.invalid" not in raw_text
    assert "redaction_fixture_api_key_000000" not in raw_text
    assert "[REDACTED_EMAIL]" in raw_text
    assert "[REDACTED_SECRET]" in raw_text

    with connect(db_path) as conn:
        trace_row = conn.execute("SELECT * FROM traces ORDER BY id LIMIT 1").fetchone()
        label_row = conn.execute("SELECT label FROM labels WHERE trace_id = ?", (trace_row["id"],)).fetchone()
        score_row = conn.execute("SELECT training_value_score FROM scores WHERE trace_id = ?", (trace_row["id"],)).fetchone()
        candidate = build_training_candidate(dict(trace_row), label_row["label"], score_row["training_value_score"])

    assert candidate["privacy_status"] == "local_only"
    try:
        assert_public_export_allowed(candidate)
    except PermissionError:
        pass
    else:
        raise AssertionError("private-local-only traces must not be public-exportable")


def test_public_privacy_mode_can_export_candidate(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "public.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "trace_type": "router_trace",
                "task_id": "public-task",
                "prompt": "Route public benchmark task",
                "response": "Assigned to local model and verified.",
                "privacy_mode": "public-benchmark-contribution",
            }
        ],
    )
    db_path = tmp_path / "dataevol.sqlite"
    report = ingest_jsonl(
        jsonl_path,
        db_path,
        source_system="fractal-router",
        privacy_mode="public-benchmark-contribution",
        raw_root=tmp_path / ".dataevol" / "raw",
    )
    label_run(db_path, report.run_id)
    score_run(db_path, report.run_id)

    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT t.*, l.label, s.training_value_score
            FROM traces t
            JOIN labels l ON l.trace_id = t.id
            JOIN scores s ON s.trace_id = t.id
            """
        ).fetchone()
        candidate = build_training_candidate(dict(row), row["label"], row["training_value_score"])

    assert candidate["privacy_status"] == "public_benchmark"
    assert_public_export_allowed(candidate)

