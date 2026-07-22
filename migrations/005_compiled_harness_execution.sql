-- Compiled harness registry and controller-owned execution state.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS compiled_harnesses (
  id INTEGER PRIMARY KEY,
  harness_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  category TEXT NOT NULL,
  status TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  parent_id TEXT,
  source_genome_id TEXT,
  manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (harness_id, version)
);

CREATE INDEX IF NOT EXISTS idx_compiled_harnesses_category_status
  ON compiled_harnesses (category, status);

CREATE TABLE IF NOT EXISTS harness_execution_sessions (
  session_id TEXT PRIMARY KEY,
  harness_id TEXT NOT NULL,
  harness_version INTEGER NOT NULL,
  harness_content_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  task_features TEXT NOT NULL,
  route_decision TEXT NOT NULL,
  state_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (harness_content_hash) REFERENCES compiled_harnesses(content_hash)
);

CREATE INDEX IF NOT EXISTS idx_harness_execution_sessions_status
  ON harness_execution_sessions (status, updated_at);

CREATE TABLE IF NOT EXISTS harness_execution_events (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  event_index INTEGER NOT NULL,
  kind TEXT NOT NULL,
  state_before TEXT NOT NULL,
  proposal TEXT,
  accepted INTEGER,
  violations TEXT,
  expected_action TEXT,
  observation TEXT,
  state_after TEXT NOT NULL,
  teacher_correction TEXT,
  verifier TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (session_id, event_index),
  FOREIGN KEY (session_id) REFERENCES harness_execution_sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_harness_execution_events_session_kind
  ON harness_execution_events (session_id, kind, event_index);
