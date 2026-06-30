from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .sqlite import connect, init_db


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_trace_rows(db_path: str | Path, *, run_id: int | None = None, from_runs: str | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    where = ""
    args: tuple[Any, ...] = ()
    limit = _selector_limit(from_runs)
    if run_id is not None:
        where = "WHERE t.run_id = ?"
        args = (run_id,)
    query = f"""
        SELECT
          t.*,
          l.label,
          s.quality_score,
          s.correctness_score,
          s.cost_score,
          s.latency_score,
          s.safety_score,
          s.training_value_score
        FROM traces t
        LEFT JOIN labels l ON l.trace_id = t.id
        LEFT JOIN scores s ON s.trace_id = t.id
        {where}
        ORDER BY t.id DESC
        {f"LIMIT {limit}" if limit else ""}
    """
    with connect(db_path) as conn:
        rows = [dict(row) for row in conn.execute(query, args).fetchall()]
    return [_hydrate_trace_row(row) for row in reversed(rows)]


def register_dataset(
    db_path: str | Path,
    result: Any,
    *,
    traces: list[Mapping[str, Any]] | None = None,
) -> int:
    init_db(db_path)
    manifest = _read_json(getattr(result, "manifest_path", None)) or {}
    name = str(manifest.get("name") or Path(str(result.dataset_path)).stem)
    dataset_type = str(manifest.get("dataset_type") or getattr(result, "dataset_type", "router"))
    version = str(manifest.get("version") or "v0")
    path = str(getattr(result, "dataset_path"))
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO datasets (name, dataset_type, version, path, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, dataset_type, version, path, now_iso()),
        )
        dataset_id = int(cursor.lastrowid)
        trace_ids = {str(row.get("id")) for row in traces or [] if row.get("id") is not None}
        for item in _read_jsonl(path):
            source_trace_id = item.get("source_trace_id")
            trace_id = int(source_trace_id) if str(source_trace_id) in trace_ids else None
            conn.execute(
                """
                INSERT INTO dataset_items (dataset_id, trace_id, item_type, accepted, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, trace_id, dataset_type, 1, now_iso()),
            )
    return dataset_id


def register_benchmark(db_path: str | Path, result: Any) -> int:
    init_db(db_path)
    manifest = _read_json(getattr(result, "manifest_path", None)) or {}
    name = str(manifest.get("name") or Path(str(result.benchmark_path)).stem)
    benchmark_type = str(manifest.get("benchmark_type") or "router_policy")
    version = str(manifest.get("version") or "v0")
    frozen = 1 if manifest.get("frozen", True) else 0
    path = str(getattr(result, "benchmark_path"))
    return _insert_benchmark(db_path, name=name, benchmark_type=benchmark_type, version=version, frozen=frozen, path=path)


def register_benchmark_path(db_path: str | Path, benchmark_path: str | Path, manifest_path: str | Path | None = None) -> int:
    manifest = _read_json(manifest_path) or {}
    path = Path(benchmark_path)
    return _insert_benchmark(
        db_path,
        name=str(manifest.get("name") or path.stem),
        benchmark_type=str(manifest.get("benchmark_type") or "router_policy"),
        version=str(manifest.get("version") or "v0"),
        frozen=1 if manifest.get("frozen", True) else 0,
        path=str(path),
    )


def register_opportunities(db_path: str | Path, opportunities: list[Mapping[str, Any]], *, run_id: int | None = None) -> list[int]:
    init_db(db_path)
    ids: list[int] = []
    with connect(db_path) as conn:
        for opportunity in opportunities:
            cursor = conn.execute(
                """
                INSERT INTO evolution_opportunities (
                  run_id, category, observation, hypothesis, proposed_change,
                  expected_metric, risk_level, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(opportunity.get("category") or opportunity.get("id") or "general"),
                    str(opportunity.get("observation") or ""),
                    str(opportunity.get("hypothesis") or ""),
                    str(opportunity.get("proposed_change") or ""),
                    str(opportunity.get("expected_metric") or "quality_score"),
                    str(opportunity.get("risk_level") or "medium"),
                    str(opportunity.get("status") or "proposed"),
                    now_iso(),
                ),
            )
            ids.append(int(cursor.lastrowid))
    return ids


def load_opportunity(db_path: str | Path, opportunity_id: int) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM evolution_opportunities WHERE id = ?", (opportunity_id,)).fetchone()
    if row is None:
        raise ValueError(f"opportunity not found: {opportunity_id}")
    return dict(row)


def register_idea_prd(db_path: str | Path, opportunity_id: int, path: str | Path, *, status: str = "proposed") -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO idea_prds (opportunity_id, path, status, created_at) VALUES (?, ?, ?, ?)",
            (opportunity_id, str(path), status, now_iso()),
        )
        return int(cursor.lastrowid)


def ensure_idea_prd(db_path: str | Path, *, path: str | Path | None = None, opportunity_id: int | None = None) -> int:
    init_db(db_path)
    if opportunity_id is None:
        opportunity_id = register_opportunities(
            db_path,
            [
                {
                    "category": "router_experiment",
                    "observation": "Measured router policy experiment.",
                    "hypothesis": "Variant router policy may improve the primary metric.",
                    "proposed_change": "Evaluate variant policy on a frozen benchmark.",
                    "expected_metric": "cost_per_verified_task",
                    "risk_level": "medium",
                    "status": "proposed",
                }
            ],
        )[0]
    if path is None:
        path = "inline_or_cli_experiment"
    return register_idea_prd(db_path, opportunity_id, path)


def register_experiment_report(
    db_path: str | Path,
    report: Mapping[str, Any],
    *,
    idea_prd_id: int | None = None,
    benchmark_id: int | None = None,
) -> int:
    init_db(db_path)
    idea_prd_id = idea_prd_id or ensure_idea_prd(db_path)
    if benchmark_id is None:
        benchmark_path = report.get("benchmark_path") or "unregistered_benchmark"
        benchmark_id = register_benchmark_path(db_path, benchmark_path, report.get("benchmark_manifest_path"))
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO experiments (
              idea_prd_id, control_version, variant_version, benchmark_id,
              status, started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idea_prd_id,
                str(report.get("control_version") or "control"),
                str(report.get("variant_version") or "variant"),
                benchmark_id,
                str(report.get("status") or "completed"),
                report.get("started_at"),
                report.get("completed_at"),
            ),
        )
        experiment_db_id = int(cursor.lastrowid)
        comparison = report.get("comparison") if isinstance(report.get("comparison"), Mapping) else {}
        for metric, values in comparison.items():
            if not isinstance(values, Mapping):
                continue
            conn.execute(
                """
                INSERT INTO experiment_results (
                  experiment_id, metric, control_value, variant_value, delta, verdict, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_db_id,
                    str(metric),
                    _optional_float(values.get("control")),
                    _optional_float(values.get("variant")),
                    _optional_float(values.get("delta")),
                    "improved" if report.get("primary_metric_improved") else "reject",
                    now_iso(),
                ),
            )
        return experiment_db_id


def find_experiment_db_id(db_path: str | Path, report: Mapping[str, Any]) -> int | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id FROM experiments
            WHERE control_version = ? AND variant_version = ?
            ORDER BY id DESC LIMIT 1
            """,
            (str(report.get("control_version") or "control"), str(report.get("variant_version") or "variant")),
        ).fetchone()
    return int(row["id"]) if row else None


def register_promotion(db_path: str | Path, promotion_path: str | Path, report: Mapping[str, Any]) -> int | None:
    experiment_id = find_experiment_db_id(db_path, report)
    if experiment_id is None:
        return None
    payload = _read_json(promotion_path) or {}
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO promotions (
              experiment_id, promoted_component, old_version, new_version, rollback_path, promoted_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                str(payload.get("promoted_component") or report.get("component") or "router"),
                str(payload.get("old_version") or report.get("control_version") or "control"),
                str(payload.get("new_version") or report.get("variant_version") or "variant"),
                str(payload.get("rollback_path") or report.get("rollback_snapshot") or ""),
                str(payload.get("promoted_at") or now_iso()),
            ),
        )
        return int(cursor.lastrowid)


def _insert_benchmark(
    db_path: str | Path,
    *,
    name: str,
    benchmark_type: str,
    version: str,
    frozen: int,
    path: str,
) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO benchmarks (name, benchmark_type, version, frozen, path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, benchmark_type, version, frozen, path, now_iso()),
        )
        return int(cursor.lastrowid)


def _hydrate_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("tool_calls", "files_changed", "tests_run"):
        row[key] = _json_value(row.get(key), [])
    raw_path = row.get("raw_path")
    if raw_path:
        raw = _read_json(raw_path) or {}
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
        row["metrics"] = dict(metrics)
        row.setdefault("objective", raw.get("objective"))
        row.setdefault("task", raw.get("task"))
    else:
        row["metrics"] = {}
    if row.get("quality_score") is not None:
        row["score"] = row["quality_score"]
    return row


def _selector_limit(selector: str | None) -> int | None:
    if not selector or selector == "all":
        return None
    if selector.startswith("last_"):
        return int(selector.removeprefix("last_"))
    if selector.isdigit():
        return int(selector)
    return None


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
