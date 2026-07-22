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
| POST | `/router_performance` | Yes | Build router performance rows from stored traces. |
| POST | `/candidate_router_policy` | Yes | Generate a candidate router policy from measured trace performance. |
| POST | `/build_benchmark` | Yes | Build benchmarks from selected runs. |
| POST | `/reflect` | Yes | Discover improvement opportunities. |
| POST | `/idea_prd` | Yes | Generate an Idea PRD from an opportunity. |
| POST | `/experiment` | Yes | Run or register a sandbox experiment. |
| POST | `/compare` | Yes | Compare baseline and variant. |
| POST | `/promote` | Yes | Promote a validated experiment. |
| POST | `/reject` | Yes | Reject or archive an experiment. |
| POST | `/privacy/export_training_candidates` | Yes | Export redacted training candidates from stored traces. |
| POST | `/prompts/variants` | Yes | Generate prompt pack variants. |
| POST | `/prompts/version` | Yes | Version a prompt pack artifact. |
| POST | `/prompts/ab_test` | Yes | Compare prompt pack metrics. |
| POST | `/prompts/promote` | Yes | Promote a prompt pack after metric improvement. |
| POST | `/integrations/router_dataset_pull` | Yes | Pull a router dataset manifest from file or HTTP endpoint. |
| POST | `/integrations/post_coordinate_completion` | Yes | POST a Coordinate completion payload to an HTTP endpoint. |
| POST | `/local_model/prepare` | Yes | Write local expert adapter datasets, manifest, and training driver. |
| POST | `/local_model/train` | Yes | Prepare and optionally execute MLX-LM LoRA training. |
| POST | `/local_model/evaluate` | Yes | Evaluate local adapter benchmark improvement. |
| POST | `/local_model/promote` | Yes | Promote a local adapter only after improvement. |
| POST | `/local_model/layerscope/train_layer_specialist` | Yes | Plan or execute pinned single-layer specialist training. |
| POST | `/local_model/training/status` | Yes | Inspect one durable local-model training job. |
| POST | `/local_model/training/latest` | Yes | Inspect the most recent local-model training job. |
| POST | `/local_model/training/cancel` | Yes | Cancel a queued or running local-model training job. |
| POST | `/local_model/training/retry` | Yes | Idempotently retry a recoverable layer-specialist job. |
| POST | `/harness/verdict` | Yes | Issue and persist a binding harness evaluation verdict. |
| GET | `/harness/verdicts/{verdict_id}` | Yes | Load and verify a persisted harness verdict. |
| POST | `/harness/compile` | Yes | Use the configured external teacher to compile a rich harness into a typed state machine. |
| POST | `/harness/register_compiled` | Yes | Validate and immutably register a compiled harness. |
| POST | `/harness/route_compiled` | Yes | Deterministically select at most three applicable harnesses. |
| POST | `/harness/start_execution` | Yes | Pin a routed harness and create a persistent controller session. |
| POST | `/harness/execution_action` | Yes | Validate one worker action against order and hard invariants. |
| POST | `/harness/execution_observation` | Yes | Record a tool observation and advance, retry, or escalate. |
| POST | `/harness/export_next_actions` | Yes | Export positive, violation, and teacher-corrected next-action rows. |
| GET | `/harness/compiled` | Yes | List compiled harness versions. |
| GET | `/harness/executions/{session_id}` | Yes | Inspect pinned state and durable execution events. |
| GET | `/harness/execution_metrics` | Yes | Report adherence, violations, completion, escalation, and corrections. |
| GET | `/runs` | No | List runs when storage core is available. |
| GET | `/datasets` | No | List datasets when storage core is available. |
| GET | `/benchmarks` | No | List benchmarks when storage core is available. |
| GET | `/experiments` | No | List experiments when storage core is available. |
| GET | `/opportunities` | No | List registered evolution opportunities. |
| GET | `/idea_prds` | No | List registered Idea PRDs. |
| GET | `/promotions` | No | List registered promotions. |

## Payload Shape

Most operation endpoints accept a generic payload wrapper:

```json
{
  "payload": {
    "run_id": "run_001"
  }
}
```

Builder endpoints can load real traces from storage instead of requiring inline items:

```json
{
  "payload": {
    "type": "router",
    "run_id": "1"
  }
}
```

Idea PRDs accept either an inline JSON object or a registered opportunity id:

```json
{
  "payload": {
    "opportunity_id": 1,
    "component": "router"
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

## Layer Specialist Contract

Remote base-model IDs require `base_model_revision` containing an immutable commit
hash. Local model fingerprints cover configuration, tokenizer metadata, weight
indexes, and every referenced safetensors shard. HTTP(S) `dataset_uri` values
require `dataset_sha256`; the service downloads them into its bounded artifact
cache only when the host matches `LAYER_SPECIALIST_DATASET_ALLOWED_HOSTS`.
Redirects are revalidated, private addresses are denied by default, and signed URL
queries are not persisted. Size and timeout bounds are configured with
`LAYER_SPECIALIST_DATASET_MAX_BYTES` and `LAYER_SPECIALIST_DATASET_TIMEOUT_S`.

Training jobs are stored in SQLite. Jobs interrupted by restart become failed and
recoverable; authenticated `/local_model/training/retry` is idempotent. Running
jobs can be terminated with `/local_model/training/cancel`, and each request can
set `timeout_seconds`.

Specialist manifests bind `genome_id` and `candidate_content_hash` to the model
fingerprint, revision, layer/task/mode/freeze strategy, tensor hashes, dataset
hash, and contribution profile. The specialist server accepts activation only at
`/routes/activate`, which additionally requires `harness_deployment_id` and
`verdict_hash`. Generation may use only the already authority-bound active route;
it cannot activate a specialist itself.

Single-layer training keeps frozen model layers quantized and casts only the
target layer to FP32. The optimizer is bias-corrected AdamW with an FP32-safe
epsilon, warmup/cosine decay, gradient clipping, and loss/gradient/parameter
finite checks. Rows whose completion is truncated entirely by `max_seq_len` are
excluded before the deterministic split and counted in the manifest.

The offline preference stage uses `training_mode="rl"` with the required
`rl_algorithm="dpo"`. Its JSONL contract is `prompt`, `chosen`, `rejected`, and
an optional `pair_id`. `initial_specialist_manifest` must identify a verified SFT
specialist for the same base fingerprint, revision, and layer. DataEvol applies
that artifact before freezing the DPO reference policy. `beta` and `sft_coef`
configure the preference objective; the manifest records the parent candidate,
prompt-group split, reference log-ratio hash, and baseline/final preference
metrics. Unsupported generic RL algorithms fail synchronously instead of being
queued.

The reproducible DialogSum experiment harness is available at
`python -m dataevol.experiments.layerscope_benchmark`. Install the `benchmark`
extra for the pinned standard ROUGE scorer. Generation runs use immutable
configuration hashes and fail closed on incomplete IDs, mixed candidates,
changed prompts, or stale resume files. Pairwise comparisons require a separate
blinding key and never write the A/B reveal key beside judge inputs.

If a core helper is unavailable, the API returns a structured compatibility response:

```json
{
  "ok": false,
  "status": "not_implemented",
  "operation": "ingest_trace",
  "module": "dataevol.ingest"
}
```
