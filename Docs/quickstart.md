# DataEvol MVP Quickstart

Run these examples from the repository root with `PYTHONPATH=src`.

## Build a Router Dataset

```python
from dataevol.datasets import build_router_dataset
from dataevol.integrations import coordinate_run_trace, router_biolatent_trace

result = build_router_dataset(
    [coordinate_run_trace(), router_biolatent_trace()],
    "out/datasets",
    version="v0",
)
print(result.dataset_path, result.manifest_path)
```

The builder writes `router_dataset.jsonl` plus a manifest containing privacy and provenance fields.

## Freeze a Benchmark

```python
from dataevol.benchmarks import build_frozen_benchmark

benchmark = build_frozen_benchmark(
    [{"id": "case_001", "task": "low-risk routing", "expected": "cheap verified worker"}],
    "out/benchmarks",
)
print(benchmark.manifest_path)
```

Frozen benchmark manifests reject accidental overwrite and can be hash-checked with `FrozenBenchmarkBuilder().assert_immutable(...)`.

## Run a Measured Router Policy Experiment

```python
from dataevol.experiments import run_measured_router_policy_experiment

report = run_measured_router_policy_experiment(
    ".dataevol/dataevol.sqlite3",
    "out/experiments",
    run_id=1,
    variant_provider="openrouter",
)
print(report["benchmark_path"], report["primary_metric_improved"])
```

The experiment path freezes measured traces into a benchmark, replays control and variant router policies over that frozen file, and then applies the promotion gate. Promotion requires primary metric improvement, no non-regression metric decline, safety and verification pass, at least two reproduced primary wins, and a rollback snapshot.

## CLI Loop

```bash
dataevol init
dataevol ingest --jsonl examples/traces/mvp_traces.jsonl --source coordinate
dataevol label --run-id 1
dataevol score --run-id 1
dataevol compress --run-id 1
dataevol dataset build --type router --run-id 1
dataevol dataset router-performance --run-id 1
dataevol dataset candidate-router-policy --run-id 1
dataevol benchmark build --from-runs last_100 --type router
dataevol synthetic generate
dataevol evolve reflect --run-id 1
dataevol evolve idea-prd --opportunity-id 1
dataevol evolve experiment --run-id 1 --variant-provider openrouter
dataevol evolve compare --experiment exp_router_policy_measured
dataevol report inbox
dataevol report opportunities
dataevol report idea-prds
dataevol report promotions
dataevol report markdown
dataevol privacy export-candidates --run-id 1
dataevol prompt variants --pack '{"manager":"plan"}'
```

## Local Expert Adapters

Prepare reproducible MLX-LM LoRA training artifacts for every DataEvol expert:

```bash
dataevol local-model prepare --output .dataevol/local_models --count 24 --iters 2
```

This writes expert datasets, `adapter_training_manifest.json`, and a runnable `train_adapters.py`. Review the planned commands without training:

```bash
dataevol local-model train --output .dataevol/local_models --count 24 --iters 2
python -m dataevol.local_models train --manifest .dataevol/local_models/adapter_training_manifest.json --dry-run
```

Execute real MLX adapter training when `mlx-lm` is installed:

```bash
dataevol local-model train --output .dataevol/local_models --count 24 --iters 2 --execute
```

Promotion is gated on benchmark improvement:

```bash
dataevol local-model evaluate --baseline-quality-score 0.70 --quality-score 0.80
dataevol local-model promote --baseline-quality-score 0.70 --quality-score 0.80
```

## Import Examples

```bash
dataevol ingest --source coordinate --path examples/runs/mock_coordinate_run.json
dataevol ingest --source biolatent --path examples/runs/biolatent_run.json
dataevol ingest --source fractal-router --path examples/runs/router_decisions.json
```
