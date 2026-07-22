# Relational Fractal Coordinate: Cross-Project Implementation Task List

**Status:** Proposed
**Date:** 2026-07-11
**Repositories:** FractalDataevol, fractalwork, fractalchain

## 1. Objective

Replace the assumption of one universal embedding space with a decentralized,
null-calibrated, multi-scale neighborhood graph. A coordinate is defined by
ranked relationships to stable anchors, not by a raw vector shared across
unrelated models.

The system must support heterogeneous LLMs, vision models, CAD models,
scientific models, adapters, and agents while preserving the existing authority
boundary:

```text
DataEvol:    anchor governance, topology computation, statistical evaluation,
             disagreement discovery, ELIGIBLE | REJECTED | INCONCLUSIVE

FractalWork: expert registry, routing, bridges, traffic, CANARY, ACTIVE,
             ROLLED_BACK, runtime verification, visualization

FractalChain: immutable commitments, authorization, finalized lifecycle events,
              marketplace receipts, activation and rollback history
```

Verified task performance remains the primary competence signal. Topology is a
routing, interoperability, portability, and discovery signal. It must never
override a failed benchmark, a DataEvol `REJECTED` verdict, permissions, runtime
health, or FractalWork deployment authority.

## 2. Architectural Decisions

### 2.1 Keep three graph layers separate

| Layer | Meaning | Examples | Authority |
|---|---|---|---|
| Semantic topology | Ranked relationships between anchors, queries, and expert representations | mutual-kNN, cycle-kNN, neighborhood overlap | DataEvol evidence |
| Operational graph | Tasks, agents, tools, models, deployments, proofs, permissions, dependencies | routed-by, verified-by, deployed-on, rollback-to | FractalWork state |
| Display projection | A transient 2D or 3D layout of selected graph facts | x/y/z positions, camera state, clustering layout | UI only |

Display positions are never coordinates used by the router. Operational edges
are not semantic similarity edges. Semantic agreement is not proof of task
correctness.

### 2.2 Do not create a fourth production service initially

No active Fractal Coordinate representation engine was found in the reviewed
repositories. `/Users/jamesstar/FractalCordinate` currently contains planning
material and squad clones, not the proposed coordinate runtime.

Implement the first version as:

- Versioned cross-language topology contracts.
- A DataEvol topology engine and benchmark pipeline.
- FractalWork registry, router, cache, and visualization consumers.
- FractalChain commitments for public artifacts and lifecycle facts.

Reconsider a standalone Coordinate service only after the pipeline has a proven
latency or independent-scaling requirement.

### 2.3 Support three representation evidence modes

| Mode | Source | Use |
|---|---|---|
| Native representation | Hidden state or embedding exposed by the expert | Preferred topology evidence |
| Provider embedding | Stable embedding endpoint associated with the expert | Allowed when identity and revision are bound |
| Behavioral topology | Verified response/outcome signatures over anchors | Fallback for black-box agents |

Fingerprints must record the mode. Scores from different modes cannot be pooled
without a preregistered comparison and null model.

### 2.4 Publish commitments by default, not raw neighborhoods

Top-k IDs and hashed neighbor sets can still leak rare tasks or proprietary
specialization. Public artifacts contain coarse summaries and content
commitments. Full rankings remain private, tenant-scoped, encrypted, or shared
with an authorized evaluator tier.

## 3. Canonical Artifact Set

The following schemas are required before product integration:

| Schema | Required content |
|---|---|
| `AnchorBankManifestV1` | bank ID/version, domain taxonomy, modality, licenses, split hashes, normalization version, anchor count, provenance, visibility |
| `AnchorRecordV1` | stable anchor ID, concept/group ID, modality, task type, content commitment, metadata, benchmark references, privacy class |
| `NeighborhoodCoordinateV1` | subject identity, expert revision, anchor-bank hash, evidence mode, k-scales, ranked anchor IDs, distances only as local optional metadata, reciprocal links, confidence |
| `TopologyFingerprintV1` | coordinate hashes, multi-scale neighbor sketches, mutual-kNN/cycle-kNN summaries, extraction layer/pooling policy, algorithm versions |
| `TopologyComparisonV1` | two fingerprint identities, observed statistics, full-selection null plan, seeds/draw count, effects, intervals, p/q values, support and diagnostics |
| `LocalBridgeManifestV1` | source/target identities, bounded domain, anchor subset, mapping artifact hash, training loss, topology preservation, out-of-region behavior |
| `AdapterPortabilityReportV1` | source/target base identities, adapter identity, in-domain topology preservation, off-domain damage, verified task performance, runtime compatibility |
| `DisagreementCaseV1` | matched anchors/experts, disagreement type, excess-over-null score, uncertainty, evidence hashes, adjudication state |
| `TopologyGraphSketchV1` | sketch algorithm/version, k-scales, seed commitment, quantization/coarsening/privacy policy, sketch blob hash |
| `ModelCapabilityManifestV1` | candidate/base/artifact hashes, fingerprint/sketch/scorecard/privacy hashes, lineage, producer identity and signature |
| `ModelLifecycleEventV1` | deployment/route identity, action, candidate/previous/rollback target, sequence, reason/evidence hashes, attester and finality |

All hashes must declare algorithm and domain tag. Canonical JSON must reject
non-finite numbers. Consensus-facing numeric summaries use bounded fixed-point
integers rather than floats.

## 4. Dependency Order

```text
T0 Research lock and threat model
  -> T1 shared contracts and golden vectors
  -> T2 anchor bank and expert adapters
  -> T3 topology fingerprint engine
  -> T4 null-calibrated comparison
  -> T5 disagreement discovery and bridge training
  -> T6 topology-aware shadow router
  -> T7 adapter portability and canary authority
  -> T8 chain commitments and marketplace binding
  -> T9 visualization
  -> T10 end-to-end qualification and rollout
```

Chain finality and attester authorization remediation in T8 is a production
blocker and may run in parallel with T2-T6. It must finish before any topology
candidate can become `ACTIVE` or enter marketplace settlement.

## 5. Detailed Task List

### T0: Research Lock, Terminology, and Threat Model

- [ ] **T0.1 Identify and pin the source paper and calibration package.**
  Owner: architecture/DataEvol. Record DOI, repository, package name, version,
  license, supported metrics, and reproducibility fixture. Do not implement
  against an inferred package API.
- [ ] **T0.2 Write the metric specification.**
  Owner: DataEvol. Define top-k overlap, mutual-kNN, cycle-kNN, ranking
  correlation, topology damage, neighborhood coverage, and uncertainty.
- [ ] **T0.3 Preregister selection rules.**
  Owner: DataEvol. Fix layer candidates, pooling methods, prompt templates,
  k-scales, checkpoints, and aggregation before evaluating a candidate.
- [ ] **T0.4 Define the full-selection null.**
  Owner: DataEvol. Every permutation must repeat the same layer, prompt,
  checkpoint, k, and maximum-selection process used for the observed result.
- [ ] **T0.5 Complete a topology privacy threat model.**
  Owners: DataEvol/FractalWork/FractalChain. Cover membership inference,
  dictionary attacks on anchor IDs, rare-neighborhood reidentification, model
  extraction, tenant linkage, graph poisoning, Sybil experts, and stale epochs.
- [ ] **T0.6 Establish terminology.**
  Owner: architecture. Use `NeighborhoodCoordinate` for semantic topology,
  `OperationalGraph` for runtime facts, and `DisplayProjection` for UI layout.
- [ ] **T0.7 Define success and abort criteria for the program.**
  Owner: architecture. Require routing utility improvement without verified
  quality, safety, privacy, cost, or latency regression. Define when topology is
  removed from routing rather than continuously retuned.

**T0 acceptance:** the paper/package are pinned; metrics and selection are
preregistered; the threat model is reviewed; no task uses raw cross-model cosine
similarity as a compatibility score.

### T1: Cross-Language Contracts and Identity

- [ ] **T1.1 Add Python topology schemas.**
  Owner: FractalDataevol. Proposed path:
  `src/dataevol/schemas/topology.py` with strict validation and canonical hash
  helpers.
- [ ] **T1.2 Add TypeScript wire types and guards.**
  Owner: fractalwork. Extend `packages/society-schema/src/index.ts` and
  `schemas.ts`; preserve snake_case wire fields.
- [ ] **T1.3 Add Rust canonical artifact types.**
  Owner: fractalchain. Proposed path:
  `crates/fractal-society/src/model_capability/`; do not mutate existing signed
  V1 or RLMF v2 structures.
- [ ] **T1.4 Create golden cross-language fixtures.**
  Owners: all three. Python, TypeScript, and Rust must produce identical hashes
  for the same manifest, ranking, scorecard, privacy policy, and lifecycle event.
- [ ] **T1.5 Bind complete expert identity.**
  Record model weights/revision, tokenizer, architecture, representation layer,
  pooling, quantization, adapter hashes, runtime ABI, prompt template, and
  extraction code version.
- [ ] **T1.6 Define immutable topology epochs.**
  Each fingerprint binds an anchor-bank version and source watermark. Published
  epochs cannot be edited; corrections produce a successor epoch.
- [ ] **T1.7 Define compatibility and version negotiation.**
  Unknown major versions fail closed. Unknown optional attributes round-trip but
  cannot affect a v1 hash or routing score.
- [ ] **T1.8 Add contract size and cardinality limits.**
  Bound anchors, k values, rankings, edges, metadata, and payload bytes to prevent
  memory and denial-of-service failures.

**T1 acceptance:** mutation of any identity-bearing field changes the hash;
invalid hashes, duplicate anchors, dangling IDs, non-finite scores, unsorted
rankings, unsupported k, and unknown major versions are rejected; legacy schema
goldens remain unchanged.

### T2: Anchor Bank and Representation Adapters

- [ ] **T2.1 Define an anchor taxonomy.**
  Owner: DataEvol. Include code, science, CAD, engineering, medicine, law,
  planning, vision, language, tool use, and safety; support hierarchical and
  multi-label domains.
- [ ] **T2.2 Build public calibration and sealed holdout partitions.**
  Deduplicate by semantic content, group related variants, split connected
  concepts together, and prevent test anchors from entering bridge or adapter
  training.
- [ ] **T2.3 Add cross-modal concept groups.**
  Associate text, image, CAD, and structured-task anchors through a stable
  concept ID without claiming their raw representations are comparable.
- [ ] **T2.4 Add anchor provenance and license checks.**
  Every anchor needs source, consent/license, privacy class, content commitment,
  and allowed evaluation/export uses.
- [ ] **T2.5 Add anchor quality review.**
  Detect duplicates, label leakage, ambiguity, culturally narrow coverage,
  unsafe content, and domains with insufficient support.
- [ ] **T2.6 Add `RepresentationProvider` adapters.**
  Owner: DataEvol. Implement native hidden-state, embedding-endpoint, and
  behavioral modes behind one fail-closed interface.
- [ ] **T2.7 Predeclare layer and pooling policies.**
  Support fixed layer, fixed layer set, or externally selected policy. Never
  select the best layer on the final comparison set.
- [ ] **T2.8 Build modality-specific normalization.**
  Token truncation, image transforms, CAD serialization, and structured-input
  normalization become hashed manifest fields.
- [ ] **T2.9 Add resumable fingerprint jobs.**
  Reuse DataEvol durable-job patterns, exact config hashes, subprocess timeouts,
  recovery, row coverage, and content-addressed output artifacts.
- [ ] **T2.10 Establish anchor-bank governance.**
  Define proposal, review, deprecation, successor, emergency quarantine, and
  compatibility-window processes.

**T2 acceptance:** two runs with identical identities produce identical anchor
rows; every expected anchor appears exactly once or the job fails; sealed
anchors are inaccessible to training; black-box experts are labeled behavioral;
missing representation capability returns `INCONCLUSIVE`, not a zero score.

### T3: Multi-Scale Topology Fingerprints

- [ ] **T3.1 Implement exact top-k ranking.**
  Owner: DataEvol. Start with an exact reference backend for k=10,20,50 and use
  deterministic tie-breaking by stable anchor ID.
- [ ] **T3.2 Add approximate-neighbor backend only after parity tests.**
  Record index type, parameters, seed, recall against exact search, and hardware.
- [ ] **T3.3 Compute reciprocal neighborhoods.**
  Persist anchor-to-anchor rankings, mutual-kNN edges, reciprocity rate, and
  cycle-kNN evidence at every scale.
- [ ] **T3.4 Produce compact fingerprints.**
  Separate the private ranked coordinate from public coarse statistics and
  graph-sketch commitments.
- [ ] **T3.5 Add fingerprint stability experiments.**
  Measure seed, batch, quantization, prompt, minor revision, and hardware
  sensitivity before interpreting expert-to-expert differences.
- [ ] **T3.6 Define support and coverage.**
  Report missing anchors, disconnected regions, effective sample size, domain
  balance, and confidence rather than a single aggregate.
- [ ] **T3.7 Store topology artifacts additively.**
  Owner: DataEvol. Add `migrations/005_topology.sql` and repositories for anchor
  banks, fingerprint jobs, fingerprints, comparisons, null plans, disagreement
  cases, and adjudications. Preserve repeated observations.
- [ ] **T3.8 Add privacy-aware export.**
  Public export enforces visibility, minimum cohort size, keyed/pseudonymous
  anchor identifiers where appropriate, and a manifest of omitted fields.
- [ ] **T3.9 Set scale gates.**
  Benchmark representative 100k-anchor and multi-expert loads; keep large
  matrices/sketches in content-addressed artifacts rather than SQLite rows.

**T3 acceptance:** rankings are deterministic; exact and approximate backends
meet declared recall; permutation of storage/order does not change fingerprints;
changes to expert weights, extraction policy, bank, or ranking change identity;
rare private neighborhoods cannot be publicly exported.

### T4: Null-Calibrated Compatibility Evaluation

- [ ] **T4.1 Implement registered null families.**
  Include anchor-label permutation within exchangeability strata,
  degree/type-preserving graph rewiring, matched controls, and the paper/package
  reference method where applicable.
- [ ] **T4.2 Implement calibrated mutual-kNN and cycle-kNN.**
  Store observed effect, null distribution hash, empirical p-value, confidence
  interval, sample count, and diagnostics per k and domain.
- [ ] **T4.3 Correct width and layer multiplicity.**
  Compare the reported best/maximum statistic against a null that repeats the
  complete search. Never compare the selected maximum with a single-test null.
- [ ] **T4.4 Correct prompt/checkpoint multiplicity.**
  Apply the same rule to prompt templates, checkpoints, seeds, pooling methods,
  adapters, and any top-k aggregate selected after inspection.
- [ ] **T4.5 Add multiple-testing control.**
  Report BH q-values or the preregistered alternative across experts, domains,
  layers, k-scales, and bridge candidates.
- [ ] **T4.6 Require practical effect and support thresholds.**
  Statistical significance alone cannot authorize compatibility.
- [ ] **T4.7 Add uncertainty and missingness semantics.**
  Abstention, unavailable anchors, truncated inputs, and failed model calls are
  explicit. Empty cohorts cannot appear perfectly calibrated.
- [ ] **T4.8 Freeze topology benchmarks.**
  Bind feature cutoff, anchor bank, extraction policies, private labels,
  comparison plan, code version, seeds, and artifact hashes. Remove overwrite
  from authoritative benchmark paths.
- [ ] **T4.9 Add anti-cheating tests.**
  Test wide random representations, many-layer selection, many-prompt selection,
  duplicate anchors, memorized public anchors, and cherry-picked checkpoints.
- [ ] **T4.10 Create synthetic power/calibration fixtures.**
  Independent graphs should yield near-uniform p-values; injected local
  structure should be detected at the declared effect and support.

**T4 acceptance:** calibrated scores do not systematically reward width, depth,
or search count; same seed reproduces results; too-small or malformed evidence
returns `INCONCLUSIVE`; every maximum has a full-selection null artifact.

### T5: Disagreement Discovery and Local Bridges

- [ ] **T5.1 Build a matched expert-by-anchor disagreement matrix.**
  Preserve agreement, directional disagreement, abstention, missingness,
  uncertainty, representation mode, and expert independence.
- [ ] **T5.2 Rank excess disagreement over null.**
  Cluster boundary cases by domain neighborhood and require effect/q/support,
  not raw disagreement count.
- [ ] **T5.3 Create immutable disagreement evidence bundles.**
  Include snapshot/fingerprint hashes, matched rows, null report, privacy policy,
  and candidate explanations.
- [ ] **T5.4 Integrate with the Evolver Agent.**
  Owner: DataEvol. Extend `evolve/reflection.py` so validated boundaries create
  investigation opportunities, not automatic mutations or promotions.
- [ ] **T5.5 Add adjudication workflows.**
  Route cases to independent verifiers/experiments and append outcomes without
  rewriting the original disagreement.
- [ ] **T5.6 Define local bridge regions.**
  A region is a versioned anchor subset such as Python debugging, battery design,
  theorem proving, DXF editing, or synthesis.
- [ ] **T5.7 Train neighborhood-local mappings.**
  Use triplet/listwise ranking losses and positive/negative anchor constraints;
  do not optimize global hidden-state reconstruction by default.
- [ ] **T5.8 Add out-of-region detection.**
  Every bridge reports applicability confidence and refuses or falls back when a
  query is outside its trained neighborhood.
- [ ] **T5.9 Evaluate bridge preservation.**
  Measure local rank preservation, reciprocal links, downstream verified tasks,
  latency, and damage outside the target region.
- [ ] **T5.10 Register bridge lineage.**
  Bind source/target expert revisions, anchor bank, training data, loss, mapping
  artifact, runtime, license, and comparison report.

**T5 acceptance:** injected boundary cases rank above random disagreement;
duplicate/non-independent judges do not inflate support; a bridge cannot claim
global compatibility; out-of-region traffic fails safely; verified downstream
behavior remains primary.

### T6: FractalWork Topology Registry and Shadow Router

- [ ] **T6.1 Add topology references to agent-network types.**
  Owner: fractalwork. Extend `packages/agent-network-core/src/types.ts` with
  fingerprint, anchor-bank, epoch, compatibility, uncertainty, and evidence
  references. Preserve v1 route-plan readers.
- [ ] **T6.2 Add topology storage and immutable snapshots.**
  Proposed path: `apps/api/src/topology/` plus additive Postgres migration for
  experts, fingerprint refs, comparisons, bridges, epochs, and snapshot roots.
- [ ] **T6.3 Add guarded topology APIs.**
  Provide publish snapshot, get expert fingerprint summary, query neighbors,
  compare experts, query bridges, and retrieve history. Enforce auth, tenancy,
  payload bounds, freshness, and pagination.
- [ ] **T6.4 Project existing capability evidence.**
  Link CBM cells, eval cards, LayerScope candidates, deployment authority,
  runtime bridges, and marketplace agents to topology identities without
  converting operational edges into semantic neighbors.
- [ ] **T6.5 Implement two-stage query anchorization.**
  Stage one uses a cheap fixed anchorizer to shortlist experts. Stage two asks
  only shortlisted experts/bridges for native local fit when latency permits.
- [ ] **T6.6 Add topology score components.**
  Extend `packages/agent-network-core/src/router.ts` with query-anchor fit,
  reciprocal consistency, compatibility confidence, and bridge applicability.
- [ ] **T6.7 Preserve hard eligibility ordering.**
  Apply verdict, deployment state, permissions, privacy, license, health, and
  verified capability filters before topology ranking.
- [ ] **T6.8 Keep performance primary.**
  Start with topology weight zero, collect shadow evidence, then tune weights in
  a frozen DataEvol router experiment. Do not hard-code recommendation weights
  as production truth.
- [ ] **T6.9 Bind decisions to evidence.**
  Route plans and training examples include topology snapshot hash/epoch,
  shortlist, selected path, score components, uncertainty, fallbacks, and router
  version.
- [ ] **T6.10 Add freshness and cache policy.**
  Cache query-anchor results by privacy-safe fingerprint; stale epochs reduce
  confidence or fail closed for evolved artifacts.
- [ ] **T6.11 Add exploration floors.**
  Prevent feedback loops where current neighborhoods monopolize traffic. Reserve
  bounded, low-risk exploration for qualified experts and disagreement regions.
- [ ] **T6.12 Run shadow routing.**
  Compare baseline and topology rankers on verified success, calibration, cost,
  latency, fallback, safety, and domain coverage without changing live routes.

**T6 acceptance:** identical snapshots replay to the same decision hash;
topology cannot make a rejected/inactive expert eligible; shadow mode never
changes traffic; routing explanations identify anchors/evidence without exposing
private content; lookup meets the declared latency budget.

### T7: Portable Adapters, Runtime Bridges, and Canary Authority

- [ ] **T7.1 Define a complete portable adapter package.**
  Owner: fractalwork/Forge. Include base revision and weights hash, tokenizer,
  architecture, target modules, rank/alpha, tensor index, dtype/quantization,
  framework/ABI, licenses, conversion lineage, and topology report hashes.
- [ ] **T7.2 Add compatibility inspection.**
  Reject base, tokenizer, architecture, target-module, tensor, runtime, license,
  or topology-policy mismatch before loading.
- [ ] **T7.3 Make conversions explicit.**
  MLX/PEFT/cross-base conversion emits a signed conversion receipt and new
  artifact identity; unsupported conversion is an error, never silent.
- [ ] **T7.4 Add topology-ranking training loss.**
  Support preregistered anchor triplets/listwise constraints alongside task loss.
- [ ] **T7.5 Evaluate in-domain preservation and off-domain damage.**
  Compare base, source adapter, target adapter, and no-update transplant control
  on topology and verified behavior.
- [ ] **T7.6 Build a cross-base portability matrix.**
  Record `PORTABLE`, `REJECTED`, or `INCONCLUSIVE` per source/target/runtime;
  never infer transitive portability.
- [ ] **T7.7 Create a unified runtime bridge registry.**
  Runtime adapters advertise formats/ABI, health, placement, privacy, capacity,
  latency, attestation, and supported fingerprint modes.
- [ ] **T7.8 Close specialist activation bypasses.**
  Owner: fractalwork. Generic specialist activation/generation routes must call
  LayerScope/harness authority checks or be removed from public routing.
- [ ] **T7.9 Generalize canary policy by topology region.**
  Add deterministic traffic percentage, cohort, task/domain anchors, privacy,
  runtime node, observation window, minimum sample, and SLO counters.
- [ ] **T7.10 Bind runtime verification.**
  Verification evidence includes deployment identity, candidate/fingerprint,
  topology snapshot, node/runtime attestation, cohort, traffic, SLOs, and
  observed outcomes. A caller-supplied boolean alone is insufficient.
- [ ] **T7.11 Enforce rollback ordering.**
  Stop new selection, drain/disable the runtime, restore the previous candidate,
  verify restoration, then record the finalized rollback.

**T7 acceptance:** adapter import fails on any identity/runtime mismatch;
portability requires behavior plus topology; `REJECTED`/`INCONCLUSIVE` cannot
authorize a canary; deterministic cohorts receive only allowed traffic; rollback
removes the candidate from new routes before final attestation.

### T8: FractalChain Commitments, Finality, and Marketplace Binding

- [ ] **T8.1 Add signed off-chain capability manifests.**
  Owner: fractalchain. Implement `ModelCapabilityManifestV1`, topology/sketch,
  calibrated scorecard, and privacy manifests using domain-separated SHA-256.
- [ ] **T8.2 Add a public-disclosure gate.**
  Reject raw private rankings, rare cohorts, unsafe sketches, missing privacy
  policy, unsalted low-entropy commitments, and unauthorized visibility changes.
- [ ] **T8.3 Introduce a typed model registry protocol.**
  Add new native calls/events for candidate registration and lifecycle; preserve
  RLMF v2 and existing signed marketplace V1 readers unchanged.
- [ ] **T8.4 Enforce attester authorization.**
  Chain state binds subject owner/authorized keys, candidate registration,
  monotonic sequence, previous candidate, deployment, and rollback target.
- [ ] **T8.5 Fix finality semantics before production use.**
  Submission returns pending until a transaction is mined/finalized. Do not
  report the current tip as proof that a newly submitted commitment finalized.
- [ ] **T8.6 Persist registry state across restart.**
  Version consensus snapshots/state and provide legacy migration or fork-height
  checkpoint. Do not rely on memory-only RLMF envelopes.
- [ ] **T8.7 Add RPC v3 model-registry methods.**
  Submit/get/list-active/history with strict filters and cursors. Use signed
  transactions or an authenticated relayer; retain v2 compatibility.
- [ ] **T8.8 Add indexer projections.**
  Store candidate manifests, lifecycle events, and materialized active
  deployment. Reorg/replay must reconstruct the same result.
- [ ] **T8.9 Add marketplace V2 bindings.**
  Intent commits capability constraints; quote binds candidate, capability, and
  deployment; receipt binds the effective lifecycle event/block.
- [ ] **T8.10 Preserve privacy and hash clarity.**
  Put commitments and optional coarse fixed-point summaries on chain, not raw
  embeddings/rankings. Declare SHA-256 versus keccak usage explicitly.
- [ ] **T8.11 Add lifecycle E2E tests.**
  Register -> canary -> activate -> serve/receipt -> rollback -> restore prior;
  test unauthorized, replayed, stale-sequence, mismatched previous/target,
  restart, reorg, and tampered manifest cases.

**T8 acceptance:** no lifecycle event can be spoofed through unauthenticated RPC;
no activation is considered final before actual chain finality; restart and
indexer replay retain exact history; marketplace settlement rejects inactive or
rolled-back deployments; V1/V2 goldens remain unchanged.

### T9: Operational Neighborhood Visualization

- [ ] **T9.1 Define a visualization read model.**
  Owner: fractalwork dashboard. Include anchor nodes, expert-local neighborhoods,
  reciprocal links, bridges, agreement density, uncertainty, disagreement
  boundaries, freshness, and deployment authority.
- [ ] **T9.2 Keep display projection ephemeral.**
  Layout x/y/z, camera, clustering, and animation state never enter semantic
  fingerprints, router evidence, or chain commitments.
- [ ] **T9.3 Build a full-bleed Three.js topology scene.**
  Use stable node IDs, bounded level-of-detail, deterministic colors by entity
  kind, and explicit visual differences between semantic and operational edges.
- [ ] **T9.4 Add inspection and filtering.**
  Filter domain, modality, expert, k, epoch, evidence mode, agreement,
  uncertainty, verdict, deployment state, and privacy-safe detail.
- [ ] **T9.5 Overlay route explanations.**
  Highlight query anchors, shortlist, selected expert, local bridge, fallback,
  and the exact snapshot used by the decision.
- [ ] **T9.6 Add disagreement workflow.**
  Open evidence, comparison/null report, adjudication status, and investigation
  task without exposing sealed/private anchors.
- [ ] **T9.7 Add epoch diff mode.**
  Show gained/lost reciprocal edges, compatibility changes, adapter damage, and
  newly uncertain regions.
- [ ] **T9.8 Verify rendering and accessibility.**
  Playwright desktop/mobile screenshots, canvas pixel checks, nonblank scene,
  stable framing, no UI overlap, keyboard alternatives, reduced motion, and a
  tabular fallback.

**T9 acceptance:** every displayed route references its actual snapshot;
semantic, operational, and display layers are distinguishable; private anchors
are not recoverable from public UI; large fixtures remain interactive within the
declared frame and memory budgets.

### T10: End-to-End Qualification and Rollout

- [ ] **T10.1 Build a cross-project golden protocol suite.**
  Run Python/TS/Rust hash, validation, privacy, lifecycle, and backward-
  compatibility fixtures in CI.
- [ ] **T10.2 Build a heterogeneous expert fixture.**
  Include different dimensions, architectures, modalities, tokenizers, one
  black-box behavioral expert, one intentionally random wide model, and one
  locally aligned specialist.
- [ ] **T10.3 Run anti-cheating qualification.**
  Prove the random wide/deep/many-layer expert does not win after full-selection
  null calibration.
- [ ] **T10.4 Run router replay.**
  Compare current router, metadata-only router, and topology-shadow router on a
  frozen outcome dataset with paired confidence intervals.
- [ ] **T10.5 Run bridge and adapter qualification.**
  Require local topology preservation, verified downstream performance,
  out-of-region rejection, off-domain non-regression, and runtime compatibility.
- [ ] **T10.6 Run privacy red-team tests.**
  Attempt anchor dictionary recovery, rare-neighborhood linking, tenant
  correlation, sketch inversion, unauthorized export, and marketplace manifest
  substitution.
- [ ] **T10.7 Run failure injection.**
  Stale epoch, missing anchors, bridge down, runtime split-brain, invalid verdict,
  canary SLO failure, chain pending/reorg, indexer lag, and rollback restore.
- [ ] **T10.8 Issue a binding DataEvol router verdict.**
  Only real frozen evidence can produce `ELIGIBLE`; incomplete, simulation,
  fixture, or conflicting evidence is `INCONCLUSIVE`.
- [ ] **T10.9 Roll out observe-only snapshots.**
  Publish fingerprints and explanations without influencing routes.
- [ ] **T10.10 Enable shadow scoring.**
  Measure decision deltas and outcomes; keep topology weight zero for traffic.
- [ ] **T10.11 Start a 1-5% deterministic canary.**
  Only public/easy eligible regions with FractalWork authority, minimum samples,
  automatic rollback, and finalized lifecycle evidence.
- [ ] **T10.12 Expand by qualified region.**
  Expand domains/privacy/risk separately. Keep legacy routing behind a kill
  switch until replay and rollback gates pass.
- [ ] **T10.13 Publish operational runbooks.**
  Anchor-bank incident, poisoned expert, privacy leak, stale epoch, bridge
  failure, calibration drift, canary rollback, chain/indexer lag, and key
  rotation.

**T10 acceptance:** topology improves verified routing utility with no protected
regression; rollback works end-to-end; authority and finality cannot be bypassed;
the evidence package is reproducible; the legacy router remains recoverable
until the topology route reaches the declared production gate.

## 6. Repository Work Map

### FractalDataevol

| Area | Proposed files |
|---|---|
| Schemas | `src/dataevol/schemas/topology.py` |
| Anchor bank | `src/dataevol/topology/anchors.py` |
| Providers | `src/dataevol/topology/providers.py` |
| Fingerprints | `src/dataevol/topology/fingerprint.py` |
| Null calibration | `src/dataevol/topology/nulls.py`, `calibration.py` |
| Comparisons | `src/dataevol/topology/comparison.py` |
| Bridges | `src/dataevol/topology/bridges.py` |
| Disagreement | `src/dataevol/discovery/topology_disagreement.py` |
| Persistence | `migrations/005_topology.sql`, `src/dataevol/storage/topology.py` |
| Jobs/API | `src/dataevol/api/topology_jobs.py`, `src/dataevol/api/app.py` |
| Verdict evidence | `src/dataevol/harness/verdicts.py`, `promotion.py` |
| Tests | `tests/test_topology_*.py`, cross-language golden fixtures |

### fractalwork

| Area | Proposed files |
|---|---|
| Shared wire types | `packages/society-schema/src/index.ts`, `schemas.ts` |
| Router core | `packages/agent-network-core/src/relational-topology.ts`, `router.ts`, `types.ts` |
| Registry/API | `apps/api/src/topology/`, additive Postgres migration |
| Runtime bridges | `apps/api/src/runtime-bridges/` |
| Authority | `apps/api/src/harness/`, `apps/api/src/layerscope/`, gateway routes |
| Forge adapters | `packages/forge-core/src/` trainer/export/import modules |
| Dashboard | `apps/dashboard/app/network/topology/`, topology scene components |
| Tests | core router replay, API contract, authority bypass, Playwright/canvas tests |

### fractalchain

| Area | Proposed files |
|---|---|
| Canonical manifests | `crates/fractal-society/src/model_capability/` |
| Privacy verifier | `crates/fractal-society/src/pkgs/` |
| Native registry | `crates/core/src/tx.rs`, state/error/gas modules |
| Snapshot upgrade | `crates/node/src/chain_snapshot.rs` |
| RPC v3 | `crates/rpc/src/model_registry.rs` |
| Indexer | `crates/indexer/src/` DB/decode/sync/query modules |
| Marketplace V2 | `crates/wallet/src/`, Rust SDK, `packages/fractal-provider-ts` |
| Tests | cross-language goldens, authorization/finality/restart/reorg/lifecycle E2E |

## 7. Production Gates

| Gate | Required evidence |
|---|---|
| Contract | Python/TS/Rust hash parity; legacy goldens unchanged |
| Scientific | Full-selection null; q/effect/support; stability and anti-cheating tests |
| Privacy | Approved disclosure manifest; leakage red-team; tenant isolation |
| Routing | Paired verified utility gain; no safety/cost/latency/protected-domain regression |
| Portability | Task performance plus local topology; off-domain damage below threshold |
| Canary | Binding `ELIGIBLE`, deterministic cohort, minimum runtime sample/SLO |
| Chain | Authorized mined/finalized lifecycle event; durable restart/reorg projection |
| Marketplace | Quote/receipt bind active candidate, capability manifest, deployment and block |

## 8. Explicit Non-Goals for V1

- No universal latent vector or canonical foundation-model embedding space.
- No global latent-space translator.
- No raw CKA or uncalibrated representation similarity as a reward signal.
- No best-of-hundreds layer/prompt/checkpoint score without full-selection null.
- No topology-only promotion or marketplace certification.
- No raw private activation publication on chain.
- No assumption that hashed neighbor IDs are anonymous.
- No inference that A-compatible-with-B and B-compatible-with-C implies A-C
  compatibility.
- No mixing UI layout positions with routing coordinates.
- No standalone Coordinate service until evidence justifies another deployment.

## 9. First Executable Milestone

The first vertical slice should be deliberately small:

1. Pin the paper/package and approve schemas.
2. Build a 1,000-anchor text-only bank covering code, CAD, science, engineering,
   law, medicine, and planning with public calibration and sealed holdout splits.
3. Fingerprint three heterogeneous local experts at k=10,20,50 using one fixed
   extraction policy per expert.
4. Compute full-selection-null calibrated mutual-kNN and cycle-kNN.
5. Publish private fingerprints plus public commitments.
6. Add topology scores to FractalWork shadow routing only.
7. Replay at least 1,000 frozen routing outcomes.
8. Issue `ELIGIBLE`, `REJECTED`, or `INCONCLUSIVE` from DataEvol.
9. Do not start a canary until the chain finality/authorization blockers in T8
   are resolved.

This slice proves the scientific and authority contracts before investing in
cross-modal anchors, bridge training, portable adapter losses, or the 3D graph.
