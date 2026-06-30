from __future__ import annotations

from pathlib import Path
from typing import Any

from dataevol.storage import connect


def load_evolution_context(db_path: str | Path, run_id: int | None = None) -> dict[str, list[dict[str, Any]]]:
    where = "WHERE t.run_id = ?" if run_id is not None else ""
    args = (run_id,) if run_id is not None else ()
    with connect(db_path) as conn:
        traces = [dict(row) for row in conn.execute(f"SELECT * FROM traces t {where}", args).fetchall()]
        labels = [dict(row) for row in conn.execute("SELECT * FROM labels").fetchall()]
        scores = [dict(row) for row in conn.execute("SELECT * FROM scores").fetchall()]
        benchmarks = [dict(row) for row in conn.execute("SELECT * FROM benchmarks").fetchall()]
        experiments = [dict(row) for row in conn.execute("SELECT * FROM experiment_results").fetchall()]
    return {
        "traces": traces,
        "labels": labels,
        "scores": scores,
        "benchmarks": benchmarks,
        "verifier_reports": experiments,
    }
