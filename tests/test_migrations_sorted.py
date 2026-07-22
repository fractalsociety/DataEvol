from __future__ import annotations

import sqlite3
from pathlib import Path

from dataevol.storage import init_db
from dataevol.storage.sqlite import migration_paths


LEGACY = {"runs", "traces", "labels", "scores", "compressed_traces", "datasets", "benchmarks"}
HARNESS = {"harness_tasks", "harness_genomes", "harness_benchmarks", "harness_evaluations",
           "harness_lineage", "harness_training_records", "harness_experiments", "harness_verdicts",
           "compiled_harnesses", "harness_execution_sessions", "harness_execution_events"}


def _tables(db: Path) -> set[str]:
    with sqlite3.connect(db) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


def test_migrations_apply_in_sorted_order():
    names = [p.name for p in migration_paths()]
    assert names == sorted(names)
    assert "001_init.sql" in names
    assert "002_harness.sql" in names
    assert "003_harness_verdicts.sql" in names
    assert "005_compiled_harness_execution.sql" in names


def test_init_db_creates_legacy_and_harness_tables(tmp_path: Path):
    db = tmp_path / "h.db"
    init_db(db)
    tables = _tables(db)
    assert LEGACY.issubset(tables)
    assert HARNESS.issubset(tables)


def test_init_db_is_idempotent(tmp_path: Path):
    db = tmp_path / "h.db"
    init_db(db)
    init_db(db)
    assert LEGACY.issubset(_tables(db))
    assert HARNESS.issubset(_tables(db))
