-- Harness Evolver subsystem: structured AI harness genomes, benchmarks,
-- evaluations, lineage, emitted training records, and experiment comparisons.
-- All statements are idempotent so re-running init_db is safe.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS harness_tasks (
  id INTEGER PRIMARY KEY,
  task_type TEXT NOT NULL,
  task_spec TEXT NOT NULL,
  task_spec_hash TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS harness_genomes (
  id INTEGER PRIMARY KEY,
  genome_id TEXT NOT NULL UNIQUE,
  task_id INTEGER NOT NULL,
  version INTEGER NOT NULL,
  parent_genome_id TEXT,
  task_type TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  genome_json TEXT NOT NULL,
  mutation_mode TEXT,
  mutation_target TEXT,
  hypothesis TEXT,
  is_incumbent INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  FOREIGN KEY (task_id) REFERENCES harness_tasks(id)
);

CREATE TABLE IF NOT EXISTS harness_benchmarks (
  id INTEGER PRIMARY KEY,
  task_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  category TEXT NOT NULL,
  path TEXT NOT NULL,
  manifest_path TEXT,
  frozen INTEGER DEFAULT 1,
  sha256 TEXT,
  item_count INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY (task_id) REFERENCES harness_tasks(id)
);

CREATE TABLE IF NOT EXISTS harness_evaluations (
  id INTEGER PRIMARY KEY,
  genome_id TEXT NOT NULL,
  benchmark_id INTEGER,
  quality REAL,
  robustness REAL,
  verifier_agreement REAL,
  cost REAL,
  latency REAL,
  failure_rate REAL,
  score REAL,
  run_count INTEGER NOT NULL,
  per_run_scores TEXT,
  per_category TEXT,
  failure_categories TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (genome_id) REFERENCES harness_genomes(genome_id),
  FOREIGN KEY (benchmark_id) REFERENCES harness_benchmarks(id)
);

CREATE TABLE IF NOT EXISTS harness_lineage (
  id INTEGER PRIMARY KEY,
  genome_id TEXT NOT NULL,
  parent_genome_id TEXT,
  generation INTEGER NOT NULL,
  mutation_mode TEXT,
  mutation_target TEXT,
  hypothesis TEXT,
  benchmark_delta TEXT,
  cost_delta REAL,
  failed_categories_improved TEXT,
  regressions TEXT,
  promoted INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  FOREIGN KEY (genome_id) REFERENCES harness_genomes(genome_id)
);

CREATE TABLE IF NOT EXISTS harness_training_records (
  id INTEGER PRIMARY KEY,
  genome_id TEXT NOT NULL,
  task_features TEXT,
  parent_harness TEXT,
  failure_analysis TEXT,
  proposed_mutation TEXT,
  mutation_hypothesis TEXT,
  candidate_harness TEXT,
  benchmark_results TEXT,
  cost_results TEXT,
  promotion_decision TEXT,
  decision_reason TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (genome_id) REFERENCES harness_genomes(genome_id)
);

CREATE TABLE IF NOT EXISTS harness_experiments (
  id INTEGER PRIMARY KEY,
  incumbent_genome_id TEXT NOT NULL,
  challenger_genome_id TEXT NOT NULL,
  task_id INTEGER NOT NULL,
  benchmark_id INTEGER NOT NULL,
  generation INTEGER NOT NULL,
  paired INTEGER DEFAULT 1,
  matched_seeds TEXT,
  incumbent_score REAL,
  challenger_score REAL,
  bootstrap_ci_low REAL,
  bootstrap_ci_high REAL,
  promoted INTEGER DEFAULT 0,
  decision_reason TEXT,
  status TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  FOREIGN KEY (incumbent_genome_id) REFERENCES harness_genomes(genome_id),
  FOREIGN KEY (challenger_genome_id) REFERENCES harness_genomes(genome_id),
  FOREIGN KEY (task_id) REFERENCES harness_tasks(id),
  FOREIGN KEY (benchmark_id) REFERENCES harness_benchmarks(id)
);
