PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  external_run_id TEXT,
  source_system TEXT NOT NULL,
  objective TEXT,
  status TEXT NOT NULL,
  privacy_mode TEXT NOT NULL,
  created_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS traces (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  trace_type TEXT NOT NULL,
  task_id TEXT,
  agent_id TEXT,
  provider TEXT,
  model TEXT,
  prompt TEXT,
  response TEXT,
  tool_calls TEXT,
  files_changed TEXT,
  tests_run TEXT,
  raw_path TEXT,
  normalized_text TEXT,
  content_hash TEXT NOT NULL UNIQUE,
  duplicate_cluster_id INTEGER,
  privacy_status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS labels (
  id INTEGER PRIMARY KEY,
  trace_id INTEGER NOT NULL,
  label TEXT NOT NULL,
  confidence REAL,
  source TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (trace_id) REFERENCES traces(id)
);

CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY,
  trace_id INTEGER NOT NULL UNIQUE,
  quality_score REAL,
  correctness_score REAL,
  cost_score REAL,
  latency_score REAL,
  novelty_score REAL,
  escalation_rescue_score REAL,
  safety_score REAL,
  training_value_score REAL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (trace_id) REFERENCES traces(id)
);

CREATE TABLE IF NOT EXISTS compressed_traces (
  id INTEGER PRIMARY KEY,
  trace_id INTEGER NOT NULL UNIQUE,
  summary TEXT NOT NULL,
  failure_type TEXT,
  why_useful TEXT,
  corrected_trace_id INTEGER,
  token_reduction_ratio REAL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (trace_id) REFERENCES traces(id),
  FOREIGN KEY (corrected_trace_id) REFERENCES traces(id)
);

CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  dataset_type TEXT NOT NULL,
  version TEXT NOT NULL,
  path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_items (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL,
  trace_id INTEGER,
  item_type TEXT NOT NULL,
  accepted INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id),
  FOREIGN KEY (trace_id) REFERENCES traces(id)
);

CREATE TABLE IF NOT EXISTS benchmarks (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  benchmark_type TEXT NOT NULL,
  version TEXT NOT NULL,
  frozen INTEGER DEFAULT 0,
  path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_opportunities (
  id INTEGER PRIMARY KEY,
  run_id INTEGER,
  category TEXT NOT NULL,
  observation TEXT NOT NULL,
  hypothesis TEXT NOT NULL,
  proposed_change TEXT NOT NULL,
  expected_metric TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS idea_prds (
  id INTEGER PRIMARY KEY,
  opportunity_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (opportunity_id) REFERENCES evolution_opportunities(id)
);

CREATE TABLE IF NOT EXISTS experiments (
  id INTEGER PRIMARY KEY,
  idea_prd_id INTEGER NOT NULL,
  control_version TEXT NOT NULL,
  variant_version TEXT NOT NULL,
  benchmark_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  FOREIGN KEY (idea_prd_id) REFERENCES idea_prds(id),
  FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
);

CREATE TABLE IF NOT EXISTS experiment_results (
  id INTEGER PRIMARY KEY,
  experiment_id INTEGER NOT NULL,
  metric TEXT NOT NULL,
  control_value REAL,
  variant_value REAL,
  delta REAL,
  verdict TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS promotions (
  id INTEGER PRIMARY KEY,
  experiment_id INTEGER NOT NULL,
  promoted_component TEXT NOT NULL,
  old_version TEXT NOT NULL,
  new_version TEXT NOT NULL,
  rollback_path TEXT NOT NULL,
  promoted_at TEXT NOT NULL,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS duplicate_clusters (
  id INTEGER PRIMARY KEY,
  content_hash TEXT NOT NULL UNIQUE,
  canonical_trace_id INTEGER,
  duplicate_count INTEGER NOT NULL DEFAULT 0,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  FOREIGN KEY (canonical_trace_id) REFERENCES traces(id)
);

CREATE TABLE IF NOT EXISTS duplicate_events (
  id INTEGER PRIMARY KEY,
  cluster_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  raw_path TEXT,
  seen_at TEXT NOT NULL,
  FOREIGN KEY (cluster_id) REFERENCES duplicate_clusters(id),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);
