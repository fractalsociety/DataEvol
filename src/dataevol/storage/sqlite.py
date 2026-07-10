from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
MIGRATION_PATH = MIGRATIONS_DIR / "001_init.sql"


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    if db_path.parent:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migration_paths() -> list[Path]:
    """All migration .sql files under the migrations dir, applied in filename order."""
    if not MIGRATIONS_DIR.exists():
        return [MIGRATION_PATH] if MIGRATION_PATH.exists() else []
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def init_db(db_path: str | Path, migration_path: str | Path | None = None) -> None:
    """Apply schema migrations.

    With ``migration_path`` set, only that single file is applied (backward
    compatible). Otherwise every ``migrations/*.sql`` is applied in sorted
    filename order. All shipped migrations are idempotent (CREATE TABLE IF NOT
    EXISTS), so re-running is safe.
    """
    if migration_path is not None:
        paths = [Path(migration_path)]
    else:
        paths = migration_paths()
    with connect(db_path) as conn:
        for path in paths:
            conn.executescript(path.read_text(encoding="utf-8"))

