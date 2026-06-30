# DataEvol

DataEvol is a local CLI/API service for collecting agent traces, labeling and scoring outcomes, building datasets and frozen benchmarks, running measured experiments, and promoting only changes that pass safety and regression gates.

The project includes implemented storage, ingestion, labeling, scoring, compression, dataset, benchmark, experiment, promotion, reporting, and local adapter training paths. Commands and API endpoints return structured errors when required measured data is missing.

## Install

```bash
pip install -e ".[test]"
```

## CLI

```bash
dataevol --help
dataevol init
dataevol ingest --jsonl ./traces.jsonl
dataevol label --run-id run_001
dataevol score --run-id run_001
dataevol compress --run-id run_001
dataevol dataset build --type router
dataevol benchmark build --from-runs last_100
dataevol evolve reflect --run-id run_001
dataevol evolve idea-prd --opportunity opp_001
dataevol evolve experiment --idea ./ideas/opp_001/IDEA_PRD.md --run-id run_001
dataevol evolve compare --experiment exp_001
dataevol evolve promote --experiment exp_001
dataevol evolve reject --experiment exp_001
dataevol local-model prepare
dataevol local-model train --execute
dataevol privacy set private-local-only
dataevol report runs
dataevol serve
```

## Config

`dataevol init` writes `dataevol.toml` by default:

```toml
[paths]
db = ".dataevol/dataevol.sqlite3"
raw = ".dataevol/raw"
artifacts = ".dataevol/artifacts"

[api]
token = "dev-local-token"

[privacy]
mode = "private-local-only"
```

The config path can be overridden with `--config` or `DATAEVOL_CONFIG`.

## API

Start the local API:

```bash
dataevol serve --host 127.0.0.1 --port 8765
```

Health is public:

```bash
curl http://127.0.0.1:8765/health
```

Mutating endpoints require the configured token:

```bash
curl -X POST http://127.0.0.1:8765/ingest_trace \
  -H "Authorization: Bearer dev-local-token" \
  -H "Content-Type: application/json" \
  -d '{"trace":{"type":"router_trace","input":"demo"}}'
```

See [Docs/api.md](Docs/api.md).
