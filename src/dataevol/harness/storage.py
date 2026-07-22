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

from .verdicts import HarnessVerdict
from .compiled import CompiledHarness
from .controller import HarnessExecutionState


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


def register_verdict(db_path: str | Path, verdict: HarnessVerdict | Mapping[str, Any]) -> str:
    """Persist an immutable verdict, allowing only byte-equivalent retries."""
    init_db(db_path)
    record = verdict if isinstance(verdict, HarnessVerdict) else HarnessVerdict.from_dict(verdict)
    if not record.verify_hash():
        raise ValueError("verdict_hash does not match the canonical verdict payload")
    data = record.to_dict()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO harness_verdicts (
              verdict_id, schema, verdict, task_type, incumbent_genome_id,
              candidate_genome_id, candidate_content_hash, benchmark_hash,
              evidence_hash, executor_kind, reasons, created_at, verdict_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.verdict_id,
                record.schema,
                record.verdict,
                record.task_type,
                record.incumbent_genome_id,
                record.candidate_genome_id,
                record.candidate_content_hash,
                record.benchmark_hash,
                record.evidence_hash,
                record.executor_kind,
                _json(data["reasons"]),
                record.created_at,
                record.verdict_hash,
            ),
        )
        existing = conn.execute(
            "SELECT verdict_hash FROM harness_verdicts WHERE verdict_id = ?", (record.verdict_id,)
        ).fetchone()
        if existing is None or existing["verdict_hash"] != record.verdict_hash:
            raise ValueError(f"verdict_id {record.verdict_id} already exists with different content")
    return record.verdict_id


def register_compiled_harness(db_path: str | Path, harness: CompiledHarness | Mapping[str, Any]) -> str:
    """Store an immutable compiled harness version and reject conflicting retries."""
    init_db(db_path)
    record = harness if isinstance(harness, CompiledHarness) else CompiledHarness.from_dict(harness)
    payload = record.to_dict()
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT content_hash FROM compiled_harnesses WHERE harness_id = ? AND version = ?",
            (record.harness_id, record.version),
        ).fetchone()
        if existing is not None and existing["content_hash"] != record.content_hash:
            raise ValueError(f"compiled harness {record.harness_id} v{record.version} already exists with different content")
        conn.execute(
            """
            INSERT OR IGNORE INTO compiled_harnesses (
              harness_id, version, category, status, content_hash, parent_id,
              source_genome_id, manifest_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.harness_id,
                record.version,
                record.category,
                record.status,
                record.content_hash,
                record.parent_id,
                record.source_genome_id,
                _json(payload),
                record.created_at,
            ),
        )
    return record.content_hash


def create_execution_session(
    db_path: str | Path,
    state: HarnessExecutionState,
    *,
    task_features: Mapping[str, Any],
    route_decision: Mapping[str, Any],
) -> str:
    init_db(db_path)
    now = now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO harness_execution_sessions (
              session_id, harness_id, harness_version, harness_content_hash,
              status, task_features, route_decision, state_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.session_id,
                state.harness_id,
                state.harness_version,
                state.harness_content_hash,
                state.status,
                _json(dict(task_features)),
                _json(dict(route_decision)),
                _json(state.to_dict()),
                now,
                now,
            ),
        )
    return state.session_id


def update_execution_session(db_path: str | Path, state: HarnessExecutionState) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE harness_execution_sessions
            SET status = ?, state_json = ?, updated_at = ?
            WHERE session_id = ? AND harness_content_hash = ?
            """,
            (state.status, _json(state.to_dict()), now_iso(), state.session_id, state.harness_content_hash),
        )
        if cursor.rowcount != 1:
            raise ValueError("execution session is missing or its pinned harness identity changed")


def register_execution_event(db_path: str | Path, event: Mapping[str, Any]) -> int:
    init_db(db_path)
    session_id = str(event.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("execution event session_id is required")
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(event_index), -1) + 1 AS next_index FROM harness_execution_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        event_index = int(row["next_index"])
        cursor = conn.execute(
            """
            INSERT INTO harness_execution_events (
              session_id, event_index, kind, state_before, proposal, accepted,
              violations, expected_action, observation, state_after,
              teacher_correction, verifier, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event_index,
                str(event.get("kind") or "action"),
                _json(dict(event.get("state_before") or {})),
                _json(dict(event.get("proposal") or {})),
                None if event.get("accepted") is None else (1 if event.get("accepted") else 0),
                _json(list(event.get("violations") or [])),
                _json(dict(event.get("expected_action") or {})),
                _json(dict(event.get("observation") or {})),
                _json(dict(event.get("state_after") or {})),
                _json(dict(event.get("teacher_correction") or {})),
                _json(dict(event.get("verifier") or {})),
                now_iso(),
            ),
        )
        return int(cursor.lastrowid)


# --- readers -----------------------------------------------------------------

def load_genome(db_path: str | Path, genome_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT genome_json FROM harness_genomes WHERE genome_id = ?", (genome_id,)).fetchone()
    return _loads(row["genome_json"], None) if row else None


def load_compiled_harness(
    db_path: str | Path,
    harness_id: str,
    version: int | None = None,
) -> dict[str, Any] | None:
    init_db(db_path)
    query = "SELECT manifest_json FROM compiled_harnesses WHERE harness_id = ?"
    args: tuple[Any, ...] = (harness_id,)
    if version is not None:
        query += " AND version = ?"
        args = (harness_id, version)
    query += " ORDER BY version DESC LIMIT 1"
    with connect(db_path) as conn:
        row = conn.execute(query, args).fetchone()
    return _loads(row["manifest_json"], None) if row else None


def list_compiled_harnesses(
    db_path: str | Path,
    *,
    status: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    init_db(db_path)
    clauses: list[str] = []
    args: list[Any] = []
    if status is not None:
        clauses.append("status = ?")
        args.append(status)
    if category is not None:
        clauses.append("category = ?")
        args.append(category)
    query = "SELECT manifest_json FROM compiled_harnesses"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY harness_id ASC, version DESC"
    with connect(db_path) as conn:
        rows = conn.execute(query, tuple(args)).fetchall()
    return [_loads(row["manifest_json"], {}) for row in rows]


def load_execution_session(db_path: str | Path, session_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM harness_execution_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["task_features"] = _loads(data.get("task_features"), {})
    data["route_decision"] = _loads(data.get("route_decision"), {})
    data["state"] = _loads(data.pop("state_json", None), {})
    return data


def load_execution_events(db_path: str | Path, session_id: str | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    query = "SELECT * FROM harness_execution_events"
    args: tuple[Any, ...] = ()
    if session_id is not None:
        query += " WHERE session_id = ?"
        args = (session_id,)
    query += " ORDER BY session_id ASC, event_index ASC"
    with connect(db_path) as conn:
        rows = [dict(row) for row in conn.execute(query, args).fetchall()]
    for row in rows:
        for key, default in (
            ("state_before", {}), ("proposal", {}), ("violations", []),
            ("expected_action", {}), ("observation", {}), ("state_after", {}),
            ("teacher_correction", {}), ("verifier", {}),
        ):
            row[key] = _loads(row.get(key), default)
        row["accepted"] = None if row.get("accepted") is None else bool(row["accepted"])
    return rows


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


def load_verdict(db_path: str | Path, verdict_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM harness_verdicts WHERE verdict_id = ?", (verdict_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["reasons"] = _loads(data.get("reasons"), [])
    return HarnessVerdict.from_dict(data).to_dict()
