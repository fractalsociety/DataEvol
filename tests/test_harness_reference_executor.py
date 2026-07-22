from __future__ import annotations

from dataclasses import replace

import pytest

from dataevol.harness.executor import ReferenceExecutor
from dataevol.harness.genome import (
    AgentSpec,
    HarnessGenome,
    MemorySpec,
    OutputSchemaSpec,
    RecoverySpec,
    RouterSpec,
    WorkflowStep,
    new_genome_id,
)
from dataevol.harness.scoring import ScoreWeights

BENCHMARK = [
    {"id": "n1", "category": "normal"},
    {"id": "a1", "category": "adversarial"},
    {"id": "t1", "category": "tool_failure"},
    {"id": "g1", "category": "ambiguous"},
    {"id": "l1", "category": "long_context"},
    {"id": "e1", "category": "edge"},
    {"id": "h1", "category": "hidden_holdout"},
]


def _genome(
    *,
    retries=0,
    memory="none",
    verifier=False,
    output_strict=True,
    threshold=0.5,
    model="local-9b-router",
) -> HarnessGenome:
    agents = [
        AgentSpec(role="drawing_inventory", model=model, prompt_ref="prompts/drawing.md"),
        AgentSpec(role="code_compliance", model=model, prompt_ref="prompts/code.md", tools=("building_code_search",)),
    ]
    workflow = [
        WorkflowStep(step_id="extract", agent_role="drawing_inventory"),
        WorkflowStep(step_id="classify", agent_role="code_compliance", depends_on=("extract",)),
    ]
    if verifier:
        agents.append(AgentSpec(role="verifier", model=model, prompt_ref="prompts/verify.md", cannot_view=("previous_agent_confidence",)))
        workflow.append(WorkflowStep(step_id="verify", agent_role="verifier", depends_on=("classify",)))
    return HarnessGenome(
        genome_id=new_genome_id(),
        version=1,
        parent_id=None,
        task_type="permit_set_review",
        task_spec_hash="h",
        router=RouterSpec(model=model, confidence_threshold=threshold),
        agents=tuple(agents),
        workflow=tuple(workflow),
        memory=MemorySpec(type=memory),
        recovery=RecoverySpec(max_retries=retries, retry_on=("malformed_output", "verifier_disagreement", "TOOL_ARGUMENT_ERROR")),
        output=OutputSchemaSpec(schema={"report": "v2"}, validation="strict" if output_strict else "lenient"),
    )


def test_evaluation_is_deterministic_for_same_seed():
    ex = ReferenceExecutor()
    g = _genome()
    e1 = ex.evaluate(g, BENCHMARK, seed=42, repeated_runs=3, weights=ScoreWeights())
    e2 = ex.evaluate(g, BENCHMARK, seed=42, repeated_runs=3, weights=ScoreWeights())
    assert e1.quality == e2.quality
    assert e1.per_run_scores == e2.per_run_scores
    assert e1.score == e2.score


def test_repeated_runs_produces_per_run_scores():
    e = ReferenceExecutor().evaluate(_genome(), BENCHMARK, seed=7, repeated_runs=3)
    assert e.run_count == 3
    assert len(e.per_run_scores) == 3


def test_verifier_and_retries_lower_failure_rate_and_raise_robustness():
    ex = ReferenceExecutor()
    weak = ex.evaluate(_genome(retries=0, verifier=False), BENCHMARK, seed=3, repeated_runs=1)
    strong = ex.evaluate(_genome(retries=2, verifier=True), BENCHMARK, seed=3, repeated_runs=1)
    assert strong.failure_rate < weak.failure_rate
    assert strong.robustness > weak.robustness
    assert strong.verifier_agreement > weak.verifier_agreement


def test_cheaper_model_lowers_cost():
    ex = ReferenceExecutor()
    expensive = ex.evaluate(_genome(model="frontier-opus"), BENCHMARK, seed=3, repeated_runs=1)
    cheap = ex.evaluate(_genome(model="local-mlx-9b"), BENCHMARK, seed=3, repeated_runs=1)
    assert cheap.cost < expensive.cost


def test_router_threshold_changes_ambiguous_case_outcome():
    ex = ReferenceExecutor()
    # threshold 0.95 is out-of-band for the "ambiguous" category → it fails.
    bad = ex.evaluate(_genome(threshold=0.95), BENCHMARK, seed=3, repeated_runs=1)
    good = ex.evaluate(_genome(threshold=0.7), BENCHMARK, seed=3, repeated_runs=1)
    assert good.failure_rate <= bad.failure_rate


def test_long_context_needs_memory():
    ex = ReferenceExecutor()
    no_mem = ex.evaluate(_genome(memory="none"), BENCHMARK, seed=3, repeated_runs=1)
    with_mem = ex.evaluate(_genome(memory="summary_buffer"), BENCHMARK, seed=3, repeated_runs=1)
    assert with_mem.failure_rate < no_mem.failure_rate


def test_failure_categories_populated_when_cases_fail():
    e = ReferenceExecutor().evaluate(_genome(retries=0, verifier=False, threshold=0.95, memory="none"), BENCHMARK, seed=3, repeated_runs=1)
    assert e.failure_categories  # earliest-causal labels present
    # Evaluation has a to_dict with metrics
    assert "quality" in e.metrics()


def test_paired_runs_use_the_same_noise_for_equivalent_capabilities():
    baseline = _genome()
    changed_metadata = baseline.with_component(
        genome_id=new_genome_id(),
        router=replace(baseline.router, fallback_model="unused-fallback"),
    )
    executor = ReferenceExecutor()
    control = executor.evaluate(baseline, BENCHMARK, seed=11, repeated_runs=5)
    candidate = executor.evaluate(changed_metadata, BENCHMARK, seed=11, repeated_runs=5)
    assert candidate.per_run_scores == control.per_run_scores


def test_per_category_quality_aggregates_all_repeated_runs():
    evaluation = ReferenceExecutor().evaluate(_genome(), BENCHMARK, seed=13, repeated_runs=5)
    total_cases = sum(int(metrics["count"]) for metrics in evaluation.per_category.values())
    weighted_quality = sum(
        metrics["quality"] * metrics["count"] for metrics in evaluation.per_category.values()
    ) / total_cases
    assert weighted_quality == pytest.approx(evaluation.quality)


def test_empty_or_missing_benchmark_fails_closed(tmp_path):
    executor = ReferenceExecutor()
    with pytest.raises(ValueError, match="no cases"):
        executor.evaluate(_genome(), [])
    with pytest.raises(ValueError, match="no cases"):
        executor.evaluate(_genome(), tmp_path / "missing.jsonl")
