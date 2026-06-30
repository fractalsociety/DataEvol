# DataEvol Integrations

DataEvol stays outside Coordinate, BioLatent, and the Router API. Those systems should pass compact traces or run artifacts into DataEvol, then consume datasets, benchmark reports, and promotion decisions.

## Mocked Coordinate Run

`examples/runs/mock_coordinate_run.json` shows the MVP shape for a Coordinate run trace:

- `source_system`: `coordinate`
- `trace_type`: usually `router_trace`, `worker_trace`, or `verification_trace`
- `decision`: selected worker/provider/model and routing reason
- `label`, `score`, and `metrics`: downstream labeling and scoring output
- `privacy_status`: defaults to local-only for training candidates

Python example:

```python
from dataevol.datasets import build_router_dataset
from dataevol.integrations import coordinate_run_trace

build_router_dataset([coordinate_run_trace()], "out/datasets")
```

## Router and BioLatent Traces

`examples/runs/mock_router_biolatent_trace.json` shows a scientific workflow routed to a BioLatent verifier. Keep the scientific claim, verifier decision, and privacy fields compact; raw lab notes or private user content should stay outside public datasets unless explicitly opted in.

Python example:

```python
from dataevol.evolve import detect_opportunities
from dataevol.integrations import router_biolatent_trace

opportunities = detect_opportunities([router_biolatent_trace()])
```

## Storage Contract

The modules accept plain mappings and file paths, and the CLI/API paths load SQLite traces, freeze measured benchmark rows, and run the same promotion-gated experiment helpers.
