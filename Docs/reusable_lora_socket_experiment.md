# Reusable LoRA Socket Experiment

This experiment tests whether a cross-layer LoRA placement discovered on arithmetic and Python repair transfers to held-out JSON extraction on TinyLlama 1.1B Chat. It runs sequentially with MLX on Apple silicon and records immutable socket, adapter, model, timing, memory, and evaluation artifacts.

## Valid Run

The authoritative run is `.dataevol/experiments/reusable_socket_mvp_v3`. Earlier `v1` and `v2` runs are invalid because parameter budgets differed by module mix and NumPy data order was not seeded, respectively.

The confirmed universal socket is candidate 08: `o_proj` at layer 2, `q_proj` and `v_proj` at layer 7, and `down_proj` at layers 9, 12, 17, and 19. Its 2,390,528 trainable parameters are within 0.4% of every compared socket.

The three-seed 4-bit JSON means were:

| Method | Complete-record accuracy | Standard deviation |
| --- | ---: | ---: |
| Best single layer | 100.0% | 0.0% |
| Best sampled random | 99.8% | 0.28% |
| Confirmed universal | 99.6% | 0.57% |
| JSON-specific search | 98.2% | 1.99% |
| Uniform | 97.0% | 4.10% |

The universal socket improved on uniform placement by 2.6 percentage points and had lower variance, but did not beat the best single-layer or best sampled random placement. Its reusability score was 1.014. The search break-even estimate was five experts.

The discovery behavior check found 1.2% arithmetic exact-answer accuracy and 0% restricted Python unit-test pass rate for the confirmed seed-17 adapters. Therefore validation loss was not a valid proxy for acquired capability in this run. `adjudication_report.json` gives the binding `REJECTED` verdict; the strong reusable-socket hypothesis is not established.

The final gate is reproducible with:

```shell
uv run python -m dataevol.experiments.reusable_lora_socket evaluate-discovery --model .dataevol/models/tinyllama-1.1b-chat-4bit --experiment .dataevol/experiments/reusable_socket_mvp_v3
uv run python -m dataevol.experiments.reusable_lora_socket adjudicate --experiment .dataevol/experiments/reusable_socket_mvp_v3
```

The fresh 8-bit seed-17 check produced 100.0% JSON accuracy for the universal socket, 95.6% for uniform, and 100.0% for the best single layer. This rules out a purely 4-bit universal-versus-uniform effect for that seed, but it is not a multi-seed precision study.

## Scope

This is the minimum viable experiment: 12 proxy candidates, 500-step confirmation and final training, three 4-bit final seeds, and one 8-bit precision seed. It does not include the proposed 24-candidate search, 1,500-step final training, learning-rate sweep, gradient probe, or three-seed 8-bit confirmation. The JSON benchmark also has a ceiling effect and must be made harder before another promotion attempt.
