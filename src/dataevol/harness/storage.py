"""DB persistence for the Harness Evolver (mirrors storage/registry.py style).

Every mutator writes a row here so the harness_* tables stay populated and the
report/read endpoints work — unlike the pre-existing DataEvol surface where the
register_* helpers exist but compat never calls them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dataevol.storage import connect, init_db


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def register_task(db_path: str | Path, task_type: str, task_spec: Mapping[str, Any], task_spec_hash: str) -> int:
    init_db(db_path)
    spec_text = _json(dict(task_spec))
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO harness_tasks (task_type, task_spec, task_spec_hash, created_at) VALUES (?, ?, ?, ?)",
            (task_type, spec_text, task_spec_hash, now_iso()),
        )
        row = conn.execute("SELECT id FROM harness_tasks WHERE task_spec_hash = ?", (task_spec_hash,)).fetchone()
        return int(row["id"])


def register_genome(db_path: str | Path, genome: Mapping[str, Any], task_id: int) -> int:
    init_db(db_path)
    genome_id = str(genome.get("genome_id"))
    content_hash = str(genome.get("content_hash") or "")
    mutation = genome.get("mutation") or {}
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO harness_genomes (
              genome_id, task_id, version, parent_genome_id, task_type, content_hash,
              genome_json, mutation_mode, mutation_target, hypothesis, is_incumbent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                genome_id,
                task_id,
                int(genome.get("version", 1)),
                genome.get("parent_id"),
                str(genome.get("task_type", "")),
                content_hash,
                _json(dict(genome)),
                str(mutation.get("mode", "")) if mutation else "",
                mutation.get("target") if mutation else None,
                genome.get("hypothesis"),
                str(genome.get("created_at", "")),
            ),
        )
        row = conn.execute("SELECT id FROM harness_genomes WHERE genome_id = ?", (genome_id,)).fetchone()
        return int(row["id"])


def set_incumbent(db_path: str | Path, genome_id: str, task_id: int) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("UPDATE harness_genomes SET is_incumbent = 0 WHERE task_id = ?", (task_id,))
        conn.execute("UPDATE harness_genomes SET is_incumbent = 1 WHERE genome_id = ?", (genome_id,))


def register_benchmark(
    db_path: str | Path,
    *,
    task_id: int,
    name: str,
    version: str,
    category: str,
    path: str,
    manifest_path: str | None = None,
    sha256: str | None = None,
    item_count: int | None = None,
    frozen: bool = True,
) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO harness_benchmarks (
              task_id, name, version, category, path, manifest_path, frozen, sha256, item_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, name, version, category, path, manifest_path, 1 if frozen else 0, sha256, item_count, now_iso()),
        )
        return int(cursor.lastrowid)


def register_evaluation(db_path: str | Path, evaluation: Mapping[str, Any], *, benchmark_id: int | None = None) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO harness_evaluations (
              genome_id, benchmark_id, quality, robustness, verifier_agreement,
              cost, latency, failure_rate, score, run_count, per_run_scores,
              per_category, failure_categories, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(evaluation.get("genome_id")),
                benchmark_id,
                evaluation.get("quality"),
                evaluation.get("robustness"),
                evaluation.get("verifier_agreement"),
                evaluation.get("cost"),
                evaluation.get("latency"),
                evaluation.get("failure_rate"),
                evaluation.get("score"),
                int(evaluation.get("run_count", 1)),
                _json(list(evaluation.get("per_run_scores") or [])),
                _json(dict(evaluation.get("per_category") or {})),
                _json(list(evaluation.get("failure_categories") or [])),
                now_iso(),
            ),
        )
        return int(cursor.lastrowid)


def register_lineage(db_path: str | Path, node: Mapping[str, Any]) -> int:
    init_db(db_path)
    mutation = node.get("mutation") or {}
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO harness_lineage (
              genome_id, parent_genome_id, generation, mutation_mode, mutation_target,
              hypothesis, benchmark_delta, cost_delta, failed_categories_improved,
              regressions, promoted, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(node.get("genome_id")),
                node.get("parent_genome_id"),
                int(node.get("generation", 0)),
                str(mutation.get("mode", "")) if mutation else "",
                mutation.get("target") if mutation else None,
                node.get("hypothesis"),
                _json(dict(node.get("benchmark_delta") or {})),
                node.get("cost_delta"),
                _json(list(node.get("failed_categories_improved") or [])),
                _json(list(node.get("regressions") or [])),
                1 if node.get("promoted") else 0,
                str(node.get("created_at", now_iso())),
            ),
        )
        return int(cursor.lastrowid)


def register_training_record(db_path: str | Path, record: Mapping[str, Any]) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO harness_training_records (
              genome_id, task_features, parent_harness, failure_analysis,
              proposed_mutation, mutation_hypothesis, candidate_harness,
              benchmark_results, cost_results, promotion_decision, decision_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.get("genome_id")),
                _json(record.get("task_features") or {}),
                _json(record.get("parent_harness") or {}),
                _json(record.get("failure_analysis") or {}),
                _json(record.get("proposed_mutation") or {}),
                str(record.get("mutation_hypothesis") or ""),
                _json(record.get("candidate_harness") or {}),
                _json(record.get("benchmark_results") or {}),
                _json(record.get("cost_results") or {}),
                str(record.get("promotion_decision") or ""),
                str(record.get("decision_reason") or ""),
                now_iso(),
            ),
        )
        return int(cursor.lastrowid)


def register_experiment(db_path: str | Path, experiment: Mapping[str, Any]) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO harness_experiments (
              incumbent_genome_id, challenger_genome_id, task_id, benchmark_id,
              generation, paired, matched_seeds, incumbent_score, challenger_score,
              bootstrap_ci_low, bootstrap_ci_high, promoted, decision_reason, status,
              started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(experiment.get("incumbent_genome_id")),
                str(experiment.get("challenger_genome_id")),
                int(experiment.get("task_id")),
                int(experiment.get("benchmark_id")),
                int(experiment.get("generation", 0)),
                1 if experiment.get("paired", True) else 0,
                _json(list(experiment.get("matched_seeds") or [])),
                experiment.get("incumbent_score"),
                experiment.get("challenger_score"),
                experiment.get("bootstrap_ci_low"),
                experiment.get("bootstrap_ci_high"),
                1 if experiment.get("promoted") else 0,
                str(experiment.get("decision_reason") or ""),
                str(experiment.get("status") or "completed"),
                experiment.get("started_at"),
                experiment.get("completed_at"),
            ),
        )
        return int(cursor.lastrowid)


# --- readers -----------------------------------------------------------------

def load_genome(db_path: str | Path, genome_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT genome_json FROM harness_genomes WHERE genome_id = ?", (genome_id,)).fetchone()
    return _loads(row["genome_json"], None) if row else None


def load_incumbent(db_path: str | Path, task_id: int | None = None) -> dict[str, Any] | None:
    init_db(db_path)
    query = "SELECT genome_json FROM harness_genomes WHERE is_incumbent = 1"
    args: tuple[Any, ...] = ()
    if task_id is not None:
        query += " AND task_id = ?"
        args = (task_id,)
    query += " ORDER BY id DESC LIMIT 1"
    with connect(db_path) as conn:
        row = conn.execute(query, args).fetchone()
    return _loads(row["genome_json"], None) if row else None


def load_lineage(db_path: str | Path, task_id: int | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    query = "SELECT * FROM harness_lineage"
    args: tuple[Any, ...] = ()
    if task_id is not None:
        query += " WHERE genome_id IN (SELECT genome_id FROM harness_genomes WHERE task_id = ?)"
        args = (task_id,)
    query += " ORDER BY id ASC"
    with connect(db_path) as conn:
        rows = [dict(r) for r in conn.execute(query, args).fetchall()]
    for row in rows:
        row["benchmark_delta"] = _loads(row.get("benchmark_delta"), {})
        row["failed_categories_improved"] = _loads(row.get("failed_categories_improved"), [])
        row["regressions"] = _loads(row.get("regressions"), [])
        row["promoted"] = bool(row.get("promoted"))
    return rows


def load_training_records(db_path: str | Path, genome_id: str | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    query = "SELECT * FROM harness_training_records"
    args: tuple[Any, ...] = ()
    if genome_id is not None:
        query += " WHERE genome_id = ?"
        args = (genome_id,)
    query += " ORDER BY id ASC"
    with connect(db_path) as conn:
        rows = [dict(r) for r in conn.execute(query, args).fetchall()]
    json_keys = ("task_features", "parent_harness", "failure_analysis", "proposed_mutation",
                 "candidate_harness", "benchmark_results", "cost_results")
    for row in rows:
        for key in json_keys:
            row[key] = _loads(row.get(key), {})
    return rows
