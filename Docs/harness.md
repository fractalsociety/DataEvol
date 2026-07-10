# Fractal Harness Evolver

Harness Evolver generates, tests, diagnoses, mutates, and selects executable AI
**harnesses** — the execution environment *around* a model (routing, prompts,
tools, workflow graph, agents, memory, verification, recovery). It is the
complement to the DataEvol data/weights pipeline: DataEvol improves the model's
training data; HarnessEvol improves the harness the model runs inside. The two
share infrastructure (storage, promotion primitives, frozen benchmarks, config,
CLI/API) but keep separate tables, artifacts, and loop.

## Architecture

```
Task spec → HarnessArchitect → genome → ReferenceExecutor ↔ frozen Benchmark
  → HarnessEvaluation → FailureAnalyst → HarnessMutator (N candidates)
  → parallel_evaluate (paired, matched seeds, repeated_runs) → select best
  → ExperimentJudge + HarnessPromotionGate (multi-objective S, bootstrap CI)
  → promote(checkpoint + lineage + training record) | discard(training record)
  | switch search strategy on plateau
```

A harness is a structured **genome** (`src/dataevol/harness/genome.py`) —
router, agents, workflow, memory, recovery, output schema — so each component is
independently mutable. The score is multi-objective:
`S = w_q·Q + w_r·R + w_v·V − w_c·Ĉ − w_l·L̂ − w_f·F` (quality, robustness,
verifier-agreement, normalized cost/latency, failure rate), with task-tunable
weights. Promotion requires median quality ≥2%, bootstrap CI >95% with ci_low>0,
no critical benchmark dropping >1%, cost ≤+10% (unless quality >+5%), failure
rate non-increasing, reproducibility, and a rollback snapshot.

## Model backend (required for the 5 specialists)

The architect, benchmark builder, failure analyst, mutator, and judge all require
a **live model**. Configure it under `[model]` in `dataevol.toml`:

```toml
[model]
provider = "openrouter"
endpoint = "https://openrouter.ai/api/v1/chat/completions"
api_key = ""                  # prefer OPENROUTER_API_KEY for real secrets
name = "anthropic/claude-sonnet-4"
judge_name = "openai/gpt-4o"   # different from `name` so the judge is independent
```

Env equivalents: `DATAEVOL_MODEL_ENDPOINT`, `OPENROUTER_API_KEY` (or
`DATAEVOL_MODEL_API_KEY`), `DATAEVOL_MODEL_NAME`, `DATAEVOL_JUDGE_MODEL_NAME`.
Without these, model-driven commands return `status: model_not_configured`.

`evaluate`, `lineage`, `incumbent`, and `training-records` need no model — the
default executor (`ReferenceExecutor`) is a deterministic, in-process capability
model. A real Docker/subprocess sandbox can later implement the same
`HarnessExecutor` Protocol and be swapped in.

## CLI

```bash
dataevol harness evolve --task examples/harness_task.json --max-generations 10
dataevol harness design    --task examples/harness_task.json
dataevol harness benchmark --task examples/harness_task.json
dataevol harness evaluate  --genome ./genome.json --benchmark ./bench.jsonl
dataevol harness failures  --genome ./genome.json --evaluation ./eval.json
dataevol harness mutate    --incumbent ./genome.json --number-of-candidates 8
dataevol harness judge     --incumbent-eval ./i.json --challenger-eval ./c.json
dataevol harness lineage
dataevol harness incumbent
dataevol harness training-records
```

## API

`POST /harness/{design|benchmark|evaluate|failures|mutate|judge|evolve}` (body
`{"payload": {...}}`, token-protected) and `GET /harness/{lineage|incumbent|training_records}`.

## Persistence

Seven tables (migration `002_harness.sql`, applied in sorted order with `001`):
`harness_tasks`, `harness_genomes`, `harness_benchmarks`, `harness_evaluations`,
`harness_lineage`, `harness_training_records`, `harness_experiments`. Every
experiment emits a training record (the JSON shape future fine-tuning will
consume). Out of scope for v1: the real sandbox executor, the Harness Prior
Model, and fine-tuning from records — only their seams are present.
