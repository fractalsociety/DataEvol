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
model. It is simulation-only: its evidence always produces an `INCONCLUSIVE`
binding verdict and can never authorize a production canary. A real runtime can
implement the same `HarnessExecutor` Protocol and submit measured evidence.

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
dataevol harness register-compiled --harness ./compiled-harness.json
dataevol harness route --features ./task-features.json --top-k 3
dataevol harness start-execution --task ./task.json --features ./task-features.json
dataevol harness next-action --session-id hexec_... --proposal ./action.json
dataevol harness observe --session-id hexec_... --success --evidence ./evidence.json
dataevol harness export-next-actions --output .dataevol/harness/next-actions.jsonl
```

## API

`POST /harness/{design|benchmark|evaluate|failures|mutate|judge|evolve}` (body
`{"payload": {...}}`, token-protected) and `GET /harness/{lineage|incumbent|training_records}`.

`POST /harness/verdict` issues and persists a binding evaluation verdict from a
promotion-gate report. `GET /harness/verdicts/{verdict_id}` returns it. Both are
token-protected. Verdicts are `ELIGIBLE`, `REJECTED`, or `INCONCLUSIVE` and use
schema `dataevol.harness_verdict.v1`. `ELIGIBLE` authorizes only a FractalWork
canary; DataEvol does not own `CANARY`, `ACTIVE`, or `ROLLED_BACK` state.

## Compiled harness execution

The external teacher can compile a rich harness with `POST /harness/compile`.
`POST /harness/register_compiled` validates and immutably stores an already
compiled manifest. The compact manifest contains typed triggers, ordered tool or
check steps, state preconditions and effects, branches, retry budgets, tool and
path permissions, evidence requirements, and a terminal complete/escalate step.

`POST /harness/route_compiled` performs deterministic top-1-to-3 selection. It
requires teacher selection for high-risk, unmatched, low-confidence, or
near-boundary routes. `POST /harness/start_execution` pins the chosen content
hash. The 1B worker then submits one local decision at a time to
`POST /harness/execution_action`; observations advance the controller through
`POST /harness/execution_observation`. The controller, not the model, owns order,
tool guards, allowed paths, retry limits, stop conditions, and evidence floors.

Rejected actions and optional teacher corrections are durable events.
`POST /harness/export_next_actions` exports both positive and violation-labelled
examples using `dataevol.next_action_example.v1`. Registry, execution, and
metrics reads are available at `GET /harness/compiled`,
`GET /harness/executions/{session_id}`, and `GET /harness/execution_metrics`.

## Codex routing outcome evidence

FractalWork emits `codex.execution_evidence.v1` only after a real routed
subtask finishes. The receipt pins the executable plan, route decision,
accounting receipt, catalog, pricing, policy, candidate set, full model revision
fingerprint, verifier result, cost, latency, retries, tool failures, and safety
or policy violations.

DataEvol evaluates those receipts with:

```bash
python -m dataevol.experiments.codex_outcome_pipeline \
  --outcomes ./fractalwork-execution-evidence.jsonl \
  --output ./.dataevol/codex-outcomes/epoch-001
```

The output contains normalized immutable outcomes, capability-cell statistics,
randomized-only cheapest-acceptable training targets, and a rollout
recommendation. Synthetic teacher labels never count as capability evidence.
Low/medium-risk cells require at least 100 independent verified task groups,
97% selective precision, a 95% Wilson lower bound, calibrated confidence, zero
serious failures, and no recent regression. High/critical cells remain
conditional and require independent verification even when statistically
qualified. Initial authority is `shadow=0%`; the first eligible stage is a
FractalWork canary capped at 5%.

## Persistence

Harness tables are created by sorted migrations `002`, `003`, and `005`:
`harness_tasks`, `harness_genomes`, `harness_benchmarks`, `harness_evaluations`,
`harness_lineage`, `harness_training_records`, `harness_experiments`, and
`harness_verdicts`, plus `compiled_harnesses`, `harness_execution_sessions`, and
`harness_execution_events`. Every
experiment emits a training record (the JSON shape future fine-tuning will
consume). The compiled controller path also emits next-action distillation rows.
The real sandbox executor, population posterior, and fine-tuning jobs remain
separate gated milestones.
