-- Immutable binding evaluation verdicts. Deployment state belongs to FractalWork.

CREATE TABLE IF NOT EXISTS harness_verdicts (
  verdict_id TEXT PRIMARY KEY,
  schema TEXT NOT NULL CHECK (schema = 'dataevol.harness_verdict.v1'),
  verdict TEXT NOT NULL CHECK (verdict IN ('ELIGIBLE', 'REJECTED', 'INCONCLUSIVE')),
  task_type TEXT NOT NULL,
  incumbent_genome_id TEXT NOT NULL,
  candidate_genome_id TEXT NOT NULL,
  candidate_content_hash TEXT NOT NULL CHECK (length(candidate_content_hash) = 64),
  benchmark_hash TEXT NOT NULL CHECK (length(benchmark_hash) = 64),
  evidence_hash TEXT NOT NULL CHECK (length(evidence_hash) = 64),
  executor_kind TEXT NOT NULL,
  reasons TEXT NOT NULL,
  created_at TEXT NOT NULL,
  verdict_hash TEXT NOT NULL UNIQUE CHECK (length(verdict_hash) = 64)
);

CREATE INDEX IF NOT EXISTS idx_harness_verdicts_candidate
  ON harness_verdicts(candidate_genome_id, created_at);
