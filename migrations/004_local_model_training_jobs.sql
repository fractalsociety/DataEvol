PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS local_model_training_jobs (
  job_id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  recoverable INTEGER NOT NULL DEFAULT 0,
  normalized_payload TEXT,
  job_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  completed_at REAL,
  retry_job_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_local_model_training_jobs_latest
  ON local_model_training_jobs(created_at DESC);
