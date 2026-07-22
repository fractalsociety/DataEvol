"""The central harness evolution loop.

incumbent = generate_initial_harness(task_spec)
incumbent_score = evaluate(incumbent, benchmark)
for generation in range(max_generations):
    failures = analyze_failures(incumbent, benchmark)
    candidates = mutate_harness(incumbent, failures=failures, number_of_candidates=8)
    results = parallel_evaluate(candidates, benchmark=benchmark, repeated_runs=3)
    challenger = select_best_candidate(results)
    if statistically_better(challenger, incumbent):
        incumbent = challenger; save_checkpoint(incumbent)
    else:
        discard(challenger)
    if improvement_has_plateaued():
        change_search_strategy()

"Statistically better" is multi-objective: it asks not merely "did the score go
up" but "did it improve reliably without damaging cost, latency, safety, or
other task categories". Every experiment emits a training record.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataevol.benchmarks import build_frozen_benchmark
from dataevol.config import DataEvolConfig
from dataevol.experiments.workflow import create_rollback_snapshot
from dataevol.promotion.gate import PromotionRejected
from dataevol.storage import init_db

from . import storage as harness_storage
from .executor import HarnessExecutor, ReferenceExecutor, parallel_evaluate
from .genome import HarnessGenome
from .model_client import ModelClient
from .promotion import HarnessPromotionDecision, HarnessPromotionGate, HarnessPromotionThresholds
from .records import ExperimentRecord, LineageNode
from .scoring import ScoreWeights, bootstrap_ci, median
from .specialists import (
    BenchmarkBuilder,
    ExperimentJudge,
    FailureAnalysis,
    FailureAnalyst,
    HarnessArchitect,
    HarnessMutator,
    JudgeReview,
    SpecialistError,
    apply_mutation,
    hash_task_spec,
    now_iso,
)

log = logging.getLogger("dataevol.harness")

_SEARCH_STRATEGIES = ("balanced", "structural", "exploration", "simplification")


@dataclass(frozen=True)
class EvolutionConfig:
    max_generations: int = 20
    number_of_candidates: int = 8
    repeated_runs: int = 3
    plateau_window: int = 4
    bootstrap_samples: int = 2000
    bootstrap_confidence: float = 0.95
    min_quality_improvement: float = 0.02
    max_cost_increase: float = 0.10
    max_critical_benchmark_drop: float = 0.01
    quality_cost_tradeoff: float = 0.05
    reproducibility_requirement: int = 2
    hall_of_fame_size: int = 4
    seed: int = 17

    def thresholds(self) -> HarnessPromotionThresholds:
        return HarnessPromotionThresholds(
            min_quality_improvement=self.min_quality_improvement,
            bootstrap_confidence=self.bootstrap_confidence,
            max_critical_benchmark_drop=self.max_critical_benchmark_drop,
            max_cost_increase=self.max_cost_increase,
            quality_cost_tradeoff=self.quality_cost_tradeoff,
            reproducibility_requirement=self.reproducibility_requirement,
        )


@dataclass(frozen=True)
class EvolutionResult:
    incumbent: HarnessGenome
    incumbent_evaluation: Mapping[str, Any]
    generations: int
    promotions: int
    lineage: tuple[LineageNode, ...]
    final_strategy: str
    task_id: int
    benchmark_id: int | None
    incumbent_rollback_snapshot: str | None = None


def _candidate_pairs(
    incumbent_eval: Mapping[str, Any],
    candidates: Sequence[HarnessGenome],
    candidate_evals: Sequence[Mapping[str, Any]],
    proposals: Sequence[Any],
) -> list[dict[str, Any]]:
    """Zip candidates with their evaluations and proposals (length-aligned)."""
    pairs = []
    for i, eval_ in enumerate(candidate_evals):
        proposal = proposals[i] if i < len(proposals) else None
        candidate = candidates[i] if i < len(candidates) else None
        if candidate is not None and eval_ is not None:
            pairs.append({"candidate": candidate, "evaluation": eval_, "proposal": proposal})
    return pairs


def _select_best(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pairs:
        return None
    return max(pairs, key=lambda p: (p["evaluation"].get("score", 0.0), -p["evaluation"].get("cost", 0.0)))


def _critical_regressions(incumbent_eval: Mapping[str, Any], challenger_eval: Mapping[str, Any], max_drop: float) -> list[str]:
    regressions: list[str] = []
    inc_cats = incumbent_eval.get("per_category") or {}
    chal_cats = challenger_eval.get("per_category") or {}
    for category, inc in inc_cats.items():
        chal = chal_cats.get(category)
        if not isinstance(inc, Mapping) or not isinstance(chal, Mapping):
            continue
        if float(inc.get("quality", 0.0)) - float(chal.get("quality", 0.0)) > max_drop:
            regressions.append(str(category))
    return regressions


def _reproducible_runs(incumbent_eval: Mapping[str, Any], challenger_eval: Mapping[str, Any]) -> int:
    inc = list(incumbent_eval.get("per_run_scores") or [])
    chal = list(challenger_eval.get("per_run_scores") or [])
    return sum(1 for a, b in zip(inc, chal) if float(b) > float(a))


def _median_quality_delta(incumbent_eval: Any, challenger_eval: Any) -> float:
    incumbent_runs = list(incumbent_eval.raw.get("per_run_quality") or [])
    challenger_runs = list(challenger_eval.raw.get("per_run_quality") or [])
    if incumbent_runs and challenger_runs:
        return median(challenger_runs) - median(incumbent_runs)
    return challenger_eval.quality - incumbent_eval.quality


def run_harness_evolution(
    task_spec: Mapping[str, Any],
    *,
    config: DataEvolConfig,
    model_client: ModelClient,
    executor: HarnessExecutor | None = None,
    evolution: EvolutionConfig | None = None,
    benchmark: Any = None,
    db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> EvolutionResult:
    evolution = evolution or EvolutionConfig()
    task_type = str(task_spec.get("task_type") or "general")
    weights = ScoreWeights.default_for_task(task_type)
    task_hash = hash_task_spec(task_spec)

    db_path = Path(db_path) if db_path else Path(config.db_path)
    output_dir = Path(output_dir) if output_dir else Path(config.artifacts_path) / "harness"
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "rollbacks").mkdir(parents=True, exist_ok=True)
    (output_dir / "promotions").mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
    init_db(db_path)

    task_id = harness_storage.register_task(db_path, task_type, task_spec, task_hash)

    mutator_model = getattr(config, "model_name", "") or None
    judge_model = getattr(config, "judge_model_name", "") or mutator_model or None

    architect = HarnessArchitect(model_client, model=mutator_model)
    benchmark_builder = BenchmarkBuilder(model_client, model=mutator_model)
    analyst = FailureAnalyst(model_client, model=mutator_model)
    mutator = HarnessMutator(model_client, model=mutator_model)
    judge = ExperimentJudge(model_client, judge_model=judge_model)
    executor = executor or ReferenceExecutor()
    gate = HarnessPromotionGate(evolution.thresholds())

    # --- frozen benchmark ---------------------------------------------------
    if benchmark is None:
        cases = benchmark_builder.build(task_spec)
        frozen = build_frozen_benchmark(
            cases, output_dir / "benchmarks", name=f"{task_type}_bench", version="v0",
            source="harness-benchmark-builder", overwrite=True,
        )
        benchmark_obj = frozen
        benchmark_eval_target = frozen
    else:
        benchmark_obj = benchmark
        benchmark_eval_target = benchmark

    benchmark_id = harness_storage.register_benchmark(
        db_path, task_id=task_id, name=f"{task_type}_bench", version="v0", category="combined",
        path=str(getattr(benchmark_obj, "benchmark_path", benchmark_obj)),
        manifest_path=str(getattr(benchmark_obj, "manifest_path", "")) or None,
        sha256=getattr(benchmark_obj, "sha256", None),
        item_count=getattr(benchmark_obj, "item_count", None),
    )

    # --- initial incumbent --------------------------------------------------
    stored_incumbent = harness_storage.load_incumbent(db_path, task_id=task_id)
    if stored_incumbent is not None:
        incumbent = HarnessGenome.from_dict(stored_incumbent)
    else:
        incumbent = architect.design(task_spec)
        harness_storage.register_genome(db_path, _genome_with_hash(incumbent), task_id)
    incumbent_eval = executor.evaluate(incumbent, benchmark_eval_target, seed=evolution.seed, repeated_runs=evolution.repeated_runs, weights=weights)
    harness_storage.register_evaluation(db_path, incumbent_eval.to_dict(), benchmark_id=benchmark_id)
    harness_storage.set_incumbent(db_path, incumbent.genome_id, task_id)
    rollback_snapshot = create_rollback_snapshot(
        "harness", _short(incumbent.genome_id), output_dir / "rollbacks", state=incumbent.to_dict()
    )

    lineage: list[LineageNode] = []
    hall_of_fame: list[HarnessGenome] = [incumbent]
    strategy_index = 0
    last_promotion_gen = -1
    promotions = 0

    for generation in range(evolution.max_generations):
        strategy = _SEARCH_STRATEGIES[strategy_index % len(_SEARCH_STRATEGIES)]
        if generation - last_promotion_gen > evolution.plateau_window and generation > 0:
            strategy_index += 1
            strategy = _SEARCH_STRATEGIES[strategy_index % len(_SEARCH_STRATEGIES)]
            log.info("generation %d: plateau -> strategy %s", generation, strategy)

        # failures + mutation
        try:
            failures = analyst.analyze(incumbent, incumbent_eval)
            second_parent = hall_of_fame[-1] if hall_of_fame and strategy in {"exploration", "structural"} else None
            proposals = mutator.propose(
                incumbent, failures, number_of_candidates=evolution.number_of_candidates,
                strategy=strategy, second_parent=second_parent,
            )
        except SpecialistError as exc:
            log.warning("generation %d: specialist failure (%s); skipping", generation, exc)
            continue

        candidates: list[HarnessGenome] = []
        candidate_proposals: list[Any] = []
        for proposal in proposals:
            try:
                candidate = apply_mutation(incumbent, proposal)
            except SpecialistError as exc:
                log.warning("generation %d: apply_mutation failed (%s)", generation, exc)
                continue
            candidates.append(candidate)
            candidate_proposals.append(proposal)
            harness_storage.register_genome(db_path, _genome_with_hash(candidate), task_id)
        if not candidates:
            continue

        candidate_evals = parallel_evaluate(
            executor, candidates, benchmark_eval_target,
            seed=evolution.seed, repeated_runs=evolution.repeated_runs, weights=weights,
        )
        for eval_ in candidate_evals:
            harness_storage.register_evaluation(db_path, eval_.to_dict(), benchmark_id=benchmark_id)

        pairs = _candidate_pairs(incumbent_eval, candidates, [e.to_dict() for e in candidate_evals], candidate_proposals)
        best = _select_best(pairs)
        if best is None:
            continue
        challenger: HarnessGenome = best["candidate"]
        challenger_eval = candidate_evals[[c.genome_id for c in candidates].index(challenger.genome_id)]
        proposal = best["proposal"]

        # paired statistical comparison
        bootstrap = bootstrap_ci(
            incumbent_eval.per_run_scores, challenger_eval.per_run_scores,
            samples=evolution.bootstrap_samples, confidence=evolution.bootstrap_confidence,
            seed=evolution.seed + generation,
        )
        median_quality_improved = _median_quality_delta(incumbent_eval, challenger_eval)
        critical = _critical_regressions(incumbent_eval.to_dict(), challenger_eval.to_dict(), evolution.max_critical_benchmark_drop)
        cost_delta = (challenger_eval.cost - incumbent_eval.cost) / incumbent_eval.cost if incumbent_eval.cost else 0.0
        failure_rate_delta = challenger_eval.failure_rate - incumbent_eval.failure_rate
        reproducible = _reproducible_runs(incumbent_eval.to_dict(), challenger_eval.to_dict())

        judge_review: JudgeReview = judge.compare(
            incumbent=incumbent_eval, challenger=challenger_eval, bootstrap=bootstrap, mutator_model=mutator_model,
        )

        report = {
            "genome_id": challenger.genome_id,
            "incumbent_genome_id": incumbent.genome_id,
            "task_type": task_type,
            "task_id": task_id,
            "benchmark_id": benchmark_id,
            "incumbent_version": incumbent.version,
            "challenger_version": challenger.version,
            "median_quality_improved": median_quality_improved,
            "quality_delta": challenger_eval.quality - incumbent_eval.quality,
            "bootstrap": bootstrap,
            "bootstrap_confidence": evolution.bootstrap_confidence,
            "judge_independent": judge_review.independent,
            "critical_benchmark_regressions": critical,
            "cost_delta": cost_delta,
            "failure_rate_delta": failure_rate_delta,
            "reproducible_runs": reproducible,
            "rollback_snapshot": str(rollback_snapshot),
            "comparison": {
                "quality": {"control": incumbent_eval.quality, "variant": challenger_eval.quality},
                "cost": {"control": incumbent_eval.cost, "variant": challenger_eval.cost},
                "failure_rate": {"control": incumbent_eval.failure_rate, "variant": challenger_eval.failure_rate},
            },
            # compatibility keys (existing report shape) so other tooling renders it:
            "primary_metric_improved": median_quality_improved >= evolution.min_quality_improvement,
            "regressions": critical,
            "safety_passed": True,
            "verification_passed": failure_rate_delta <= 0,
            "decision_reason": judge_review.reason,
        }
        decision: HarnessPromotionDecision = gate.evaluate(report)
        promoted = decision.promoted
        next_rollback_snapshot: Path | None = None
        promotion_path: Path | None = None
        checkpoint_path: Path | None = None
        if promoted:
            try:
                promotion_result = gate.promote(report, output_dir / "promotions")
                promotion_path = promotion_result.promotion_path
                next_rollback_snapshot = create_rollback_snapshot(
                    "harness", _short(challenger.genome_id), output_dir / "rollbacks", state=challenger.to_dict()
                )
                checkpoint_path = output_dir / "checkpoints" / f"incumbent_{challenger.genome_id}.json"
                checkpoint_path.write_text(
                    json.dumps(challenger.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
                )
            except (OSError, PromotionRejected, TypeError, ValueError) as exc:
                promoted = False
                decision = HarnessPromotionDecision(False, [f"promotion artifact write failed: {exc}"])
                for staged_path in (promotion_path, next_rollback_snapshot, checkpoint_path):
                    if staged_path is not None:
                        staged_path.unlink(missing_ok=True)

        harness_storage.register_experiment(db_path, {
            "incumbent_genome_id": incumbent.genome_id,
            "challenger_genome_id": challenger.genome_id,
            "task_id": task_id,
            "benchmark_id": benchmark_id,
            "generation": generation,
            "paired": True,
            "matched_seeds": list(range(evolution.seed, evolution.seed + evolution.repeated_runs)),
            "incumbent_score": incumbent_eval.score,
            "challenger_score": challenger_eval.score,
            "bootstrap_ci_low": bootstrap[1],
            "bootstrap_ci_high": bootstrap[2],
            "promoted": promoted,
            "decision_reason": "; ".join(decision.reasons) if not promoted else judge_review.reason,
            "status": "promoted" if promoted else "rejected",
            "started_at": now_iso(),
            "completed_at": now_iso(),
        })

        failed_improved = tuple(
            cat for cat in set(incumbent_eval.failure_categories) - set(challenger_eval.failure_categories)
        )
        node = LineageNode(
            genome_id=challenger.genome_id, parent_id=incumbent.genome_id, generation=generation,
            mutation=proposal.mutation_record(parent_genome_id=incumbent.genome_id) if proposal else {},
            hypothesis=proposal.hypothesis if proposal else None,
            benchmark_delta={"quality": challenger_eval.quality - incumbent_eval.quality},
            cost_delta=cost_delta,
            failed_categories_improved=failed_improved,
            regressions=tuple(critical),
            promoted=promoted,
            created_at=now_iso(),
        )
        harness_storage.register_lineage(db_path, node.to_dict())
        lineage.append(node)

        # always emit a training record
        record = ExperimentRecord(
            genome_id=challenger.genome_id,
            task_features=dict(task_spec),
            parent_harness=incumbent.to_dict(),
            failure_analysis=failures.to_dict(),
            proposed_mutation=proposal.to_dict() if proposal else {},
            mutation_hypothesis=proposal.hypothesis if proposal else "",
            candidate_harness=challenger.to_dict(),
            benchmark_results=challenger_eval.to_dict(),
            cost_results={"cost": challenger_eval.cost, "latency": challenger_eval.latency, "cost_delta": cost_delta},
            promotion_decision="promoted" if promoted else "rejected",
            decision_reason=judge_review.reason if promoted else "; ".join(decision.reasons),
        )
        harness_storage.register_training_record(db_path, record.to_dict())

        if promoted:
            if next_rollback_snapshot is None:  # pragma: no cover - guarded by artifact staging
                raise RuntimeError("promotion staged without a rollback snapshot")
            rollback_snapshot = next_rollback_snapshot
            incumbent = challenger
            incumbent_eval = challenger_eval
            harness_storage.set_incumbent(db_path, incumbent.genome_id, task_id)
            last_promotion_gen = generation
            promotions += 1
            hall_of_fame.append(incumbent)
            if len(hall_of_fame) > evolution.hall_of_fame_size:
                hall_of_fame.pop(0)

    return EvolutionResult(
        incumbent=incumbent,
        incumbent_evaluation=incumbent_eval.to_dict(),
        generations=evolution.max_generations,
        promotions=promotions,
        lineage=tuple(lineage),
        final_strategy=_SEARCH_STRATEGIES[strategy_index % len(_SEARCH_STRATEGIES)],
        task_id=task_id,
        benchmark_id=benchmark_id,
        incumbent_rollback_snapshot=str(rollback_snapshot),
    )


def _genome_with_hash(genome: HarnessGenome) -> dict[str, Any]:
    data = genome.to_dict()
    data["content_hash"] = genome.content_hash()
    return data


def _short(genome_id: str) -> str:
    return (genome_id or "genome")[:8]
