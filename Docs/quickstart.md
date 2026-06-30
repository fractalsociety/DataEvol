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

## Run the Router Policy Experiment

```python
import json
from pathlib import Path
from dataevol.experiments import run_router_policy_experiment
from dataevol.promotion import PromotionGate

metrics = json.load(open("examples/benchmarks/router_policy_fixture_metrics.json"))
Path("out/rollback").mkdir(parents=True, exist_ok=True)
Path("out/rollback/router_policy_v0.json").write_text('{"version":"v0"}\n')
report = run_router_policy_experiment(
    metrics,
    "out/experiments",
    rollback_snapshot="out/rollback/router_policy_v0.json",
)
decision = PromotionGate().promote(report, "out/promotions")
print(decision.promotion_path)
```

Promotion requires primary metric improvement, no non-regression metric decline, safety and verification pass, at least two reproduced primary wins, and a rollback snapshot.

## CLI Loop

```bash
dataevol init
dataevol ingest --jsonl examples/traces/mvp_traces.jsonl --source coordinate
dataevol label --run-id 1
dataevol score --run-id 1
dataevol compress --run-id 1
dataevol dataset build --type router
dataevol benchmark build --from-runs last_100 --type router
dataevol synthetic generate
dataevol evolve reflect --run-id 1
dataevol report inbox
dataevol report markdown
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
