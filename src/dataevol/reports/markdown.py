from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataevol.storage import connect, init_db


def _rows(db_path: str | Path, table: str) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()]


def list_runs(db_path: str | Path) -> list[dict[str, Any]]:
    return _rows(db_path, "runs")


def list_datasets(db_path: str | Path) -> list[dict[str, Any]]:
    return _rows(db_path, "datasets")


def list_benchmarks(db_path: str | Path) -> list[dict[str, Any]]:
    return _rows(db_path, "benchmarks")


def list_experiments(db_path: str | Path) -> list[dict[str, Any]]:
    return _rows(db_path, "experiments")


def list_opportunities(db_path: str | Path) -> list[dict[str, Any]]:
    return _rows(db_path, "evolution_opportunities")


def list_idea_prds(db_path: str | Path) -> list[dict[str, Any]]:
    return _rows(db_path, "idea_prds")


def list_promotions(db_path: str | Path) -> list[dict[str, Any]]:
    return _rows(db_path, "promotions")


def build_report_payload(db_path: str | Path, artifacts_path: str | Path) -> dict[str, Any]:
    artifacts = Path(artifacts_path)
    rejections = sorted(str(path) for path in artifacts.glob("**/rejected_*.json"))
    return {
        "runs": list_runs(db_path),
        "datasets": list_datasets(db_path),
        "benchmarks": list_benchmarks(db_path),
        "experiments": list_experiments(db_path),
        "opportunities": list_opportunities(db_path),
        "idea_prds": list_idea_prds(db_path),
        "promotions": list_promotions(db_path),
        "rejections": rejections,
        "cost_savings": _metric_files(artifacts, "cost"),
        "quality_improvements": _metric_files(artifacts, "quality"),
        "safety_regression_status": _metric_files(artifacts, "safety"),
    }


def export_markdown_report(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# DataEvol Report", ""]
    for key, value in payload.items():
        lines.extend([f"## {key.replace('_', ' ').title()}", ""])
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, sort_keys=True, default=str))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _metric_files(root: Path, token: str) -> list[str]:
    return sorted(str(path) for path in root.glob("**/*.json") if token in path.read_text(encoding="utf-8", errors="ignore").lower())
