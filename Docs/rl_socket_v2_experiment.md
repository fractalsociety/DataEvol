# RL Socket Experiment v2

This experiment runs an audited, resumable QLoRA RL comparison on TinyLlama 1.1B Chat using Apple MLX. Training uses a frozen 4-bit base. Precision evaluation is reported separately and is not QAT.

## Correctness Gate

Stage -01 blocks all candidate training until prompt conditioning, completion-only masking, old/reference policy separation, KL behavior, gradient direction, sampling state, a real uniform-LoRA positive control, and exact resume behavior pass.

The real positive control trains a parameter-matched uniform TinyLlama adapter on an exact-response task. It must improve held-out accuracy by at least 50 percentage points. The final preflight improved from 0% to 100% in 150 optimizer updates.

The real-model checkpoint smoke test also compared two uninterrupted batched updates with one update, checkpoint, restart, and one update. Adapter weights, optimizer state, sampler state, Python random state, and reward history matched exactly.

## Search And Confirmation

The configuration in `configs/rl_socket_v2.yaml` runs equal eight-candidate guided and random search arms with two discovery seeds and cumulative 100, 300, and 700-update stages. Promotion uses behavioral confidence bounds and preserves healthy topology families. Discovery uses validation prompts; confirmation uses disjoint test prompts.

Primary methods are uniform LoRA, the discovery-selected single layer, the random-socket distribution, the RL-searched socket, fresh weights in the SFT topology, and continued SFT weights. Mechanistic controls cover MLP-only, attention-only, contiguous layers, zero learning rate, and shuffled rewards.

Every candidate runs in a separate subprocess. A process lock prevents duplicate orchestration. Checkpoints contain adapter, optimizer, accumulator, MLX/sampler/Python/NumPy state, scheduler/training state, locked constructor configuration, checksums, and a final `COMPLETE` sentinel.

Compact handoff files are written to the run directory:

- `resume_packet.json`
- `next_action.md`
- `latest_result.json`
- `failure_summary.json`

The final report issues separate binding verdicts for H1 through H5. No positive conclusion can override a failed trainer audit or unhealthy run.

## Launch

```shell
RUN=.dataevol/experiments/rl_socket_v2
mkdir -p "$RUN/logs"
nohup caffeinate -i uv run python -m dataevol.experiments.rl_socket_pipeline run \
  --config configs/rl_socket_v2.yaml --run "$RUN" --resume \
  >"$RUN/logs/pipeline.log" 2>"$RUN/logs/errors.log" </dev/null &
echo "$!" >"$RUN/pipeline.pid"
```

The pipeline owns the operating-system lock and rejects a duplicate launch even if the PID file is stale.
