# DataEvol API

The local API is designed for Coordinate, BioLatent, and router services to call DataEvol without importing its internals.

## Authentication

`GET /health` is unauthenticated.

Mutating endpoints require either:

```http
Authorization: Bearer <token>
```

or:

```http
X-DataEvol-Token: <token>
```

The token is read from `dataevol.toml`:

```toml
[api]
token = "dev-local-token"
```

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | No | Service and config smoke check. |
| POST | `/ingest_trace` | Yes | Ingest one trace payload. |
| POST | `/ingest_run` | Yes | Ingest a run payload. |
| POST | `/label` | Yes | Label a run or trace set. |
| POST | `/score` | Yes | Score a run or trace set. |
| POST | `/compress` | Yes | Compress traces. |
| POST | `/build_dataset` | Yes | Build a dataset such as `router`, `critic`, `verifier`, or `compressor`. |
| POST | `/build_benchmark` | Yes | Build benchmarks from selected runs. |
| POST | `/reflect` | Yes | Discover improvement opportunities. |
| POST | `/idea_prd` | Yes | Generate an Idea PRD from an opportunity. |
| POST | `/experiment` | Yes | Run or register a sandbox experiment. |
| POST | `/compare` | Yes | Compare baseline and variant. |
| POST | `/promote` | Yes | Promote a validated experiment. |
| POST | `/reject` | Yes | Reject or archive an experiment. |
| POST | `/local_model/prepare` | Yes | Write local expert adapter datasets, manifest, and training driver. |
| POST | `/local_model/train` | Yes | Prepare and optionally execute MLX-LM LoRA training. |
| POST | `/local_model/evaluate` | Yes | Evaluate local adapter benchmark improvement. |
| POST | `/local_model/promote` | Yes | Promote a local adapter only after improvement. |
| GET | `/runs` | No | List runs when storage core is available. |
| GET | `/datasets` | No | List datasets when storage core is available. |
| GET | `/benchmarks` | No | List benchmarks when storage core is available. |
| GET | `/experiments` | No | List experiments when storage core is available. |

## Payload Shape

Most operation endpoints accept a generic payload wrapper while core modules are still being implemented:

```json
{
  "payload": {
    "run_id": "run_001"
  }
}
```

`/ingest_trace` accepts:

```json
{
  "trace": {
    "type": "router_trace",
    "input": "..."
  },
  "source_system": "coordinate"
}
```

Local model preparation accepts the same wrapper:

```json
{
  "payload": {
    "output": ".dataevol/local_models",
    "experts": ["router", "critic"],
    "count": 24,
    "iters": 2
  }
}
```

Set `"execute": true` on `/local_model/train` to run the generated `mlx_lm lora` jobs.

If a core helper is unavailable, the API returns a structured compatibility response:

```json
{
  "ok": false,
  "status": "not_implemented",
  "operation": "ingest_trace",
  "module": "dataevol.ingest"
}
```
