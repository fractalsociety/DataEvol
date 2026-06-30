from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATION_PATH = Path(__file__).resolve().parents[3] / "migrations" / "001_init.sql"


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    if db_path.parent:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path, migration_path: str | Path | None = None) -> None:
    path = Path(migration_path) if migration_path else MIGRATION_PATH
    sql = path.read_text(encoding="utf-8")
    with connect(db_path) as conn:
        conn.executescript(sql)

