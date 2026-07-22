
## Squad Collaboration

This project uses squad for multi-agent collaboration. Run `squad help` for all commands and usage guide.

## Codex Task Routing

For every substantive user task in this project:

1. Decompose the request into bounded subtasks before execution.
2. Refresh or read the locally available Codex model catalog. Route each subtask
   to the smallest model and lowest reasoning effort that is expected to satisfy
   its acceptance criteria. Do not default every subtask to high effort.
3. Keep hard safety, permission, data-access, and verification requirements as
   eligibility filters. Cost savings must never override correctness or safety.
4. Produce two routing decisions in shadow mode: the deterministic/strong-model
   teacher decision and the Fractal specialized router decision. Execute the
   qualified decision; do not duplicate an expensive task merely to compare
   route labels unless a controlled evaluation calls for both executions.
5. Record the task fingerprint, decomposition, catalog and pricing versions,
   both route decisions, selected model/reasoning effort, actual API-reported
   token usage when available, verification outcome, latency, and fallback.
6. Feed only verified outcomes into router SFT/preference datasets. Retraining
   artifacts remain candidates until DataEvol evaluation authorizes a
   FractalWork canary.
   For Codex model routing, do not use the rejected single-layer generative
   specialist path. Run the tabular classifier only in shadow mode until its
   selected capability cell is trusted and verified selective precision meets
   policy. The deterministic teacher retains authority on abstention.
   Capability trust may use only independently verified FractalWork execution
   evidence. Teacher-imitation accuracy and synthetic/reference executions do
   not count. Cheapest-option targets must come from pinned randomized trials,
   never observational comparisons.
7. At completion, report measured tokens by model when the execution surface
   exposes them. Report Codex credits and API-dollar-equivalent cost against a
   pinned official price table. Clearly label counterfactual baseline savings as
   estimates. Never invent unavailable session usage or present estimates as
   billed savings.

The current Codex thread model cannot be changed mid-turn. Model-specific work
may be dispatched through a bounded `codex exec --json --model ...` subprocess
when isolation, context, permissions, and expected savings justify the extra
call. Collaboration subagents without an explicit model selector do not count as
model-specific routing.
