from __future__ import annotations

import pytest

from dataevol.harness.genome import HarnessGenome
from dataevol.harness.model_client import FakeModelClient
from dataevol.harness.scoring import HarnessEvaluation
from dataevol.harness.specialists import (
    FAILURE_TAXONOMY,
    BenchmarkBuilder,
    ExperimentJudge,
    FailureAnalyst,
    HarnessArchitect,
    HarnessMutator,
    MutationProposal,
    SpecialistError,
    apply_mutation,
)
from tests.test_harness_reference_executor import _genome  # reuse fixture builder


def _eval(genome_id="g1", quality=0.5) -> HarnessEvaluation:
    return HarnessEvaluation(
        genome_id=genome_id, quality=quality, robustness=0.5, verifier_agreement=0.5,
        cost=0.2, latency=800.0, failure_rate=0.4, score=quality,
        per_category={"adversarial": {"quality": 0.2, "failure_rate": 1.0, "count": 1}},
        failure_categories=("VERIFICATION_FAILURE",), run_count=3, per_run_scores=(quality,) * 3,
    )


def test_architect_design_returns_genome():
    client = FakeModelClient()
    genome = HarnessArchitect(client).design({"task_type": "permit_set_review", "tools": ["building_code_search"]})
    assert isinstance(genome, HarnessGenome)
    assert genome.task_type == "permit_set_review"
    assert genome.version == 1
    assert genome.parent_id is None


def test_benchmark_builder_and_holdout_partition():
    client = FakeModelClient(scripts=[("Benchmark Builder", [
        {"id": "n1", "category": "normal"},
        {"id": "h1", "category": "hidden_holdout"},
    ])])
    cases = BenchmarkBuilder(client).build({"task_type": "permit_set_review"})
    assert {c["category"] for c in cases} >= {"normal", "hidden_holdout"}
    train, holdout = BenchmarkBuilder.partition_holdout(cases)
    assert all(c["category"] != "hidden_holdout" for c in train)
    assert all(c["category"] == "hidden_holdout" for c in holdout)


def test_failure_analyst_returns_earliest_cause():
    client = FakeModelClient(scripts=[("Failure Analyst", {
        "failures": [{"category": "VERIFICATION_FAILURE", "earliest_cause": "no verifier agent", "evidence": "e"}],
        "summary": "s",
    })])
    analysis = FailureAnalyst(client).analyze(_genome(), _eval())
    assert analysis.failures[0].category == "VERIFICATION_FAILURE"
    assert "VERIFICATION_FAILURE" in FAILURE_TAXONOMY


def test_failure_analyst_does_not_expose_hidden_holdout_metrics():
    client = FakeModelClient()
    evaluation = _eval()
    evaluation = HarnessEvaluation(
        **{
            **evaluation.__dict__,
            "per_category": {
                **evaluation.per_category,
                "hidden_holdout": {"quality": 0.0, "failure_rate": 1.0, "count": 1},
            },
        }
    )
    FailureAnalyst(client).analyze(_genome(), evaluation)
    assert "hidden_holdout" not in client.calls[-1]["user"]


def test_mutator_proposes_and_apply_mutation_merges_patch():
    patch = {
        "agents": [
            {"role": "worker", "model": "local-7b", "prompt_ref": "prompts/worker.md", "tools": []},
            {"role": "verifier", "model": "local-7b", "prompt_ref": "prompts/verify.md", "cannot_view": ["previous_agent_confidence"]},
        ],
        "recovery": {"max_retries": 2, "retry_on": ["TOOL_ARGUMENT_ERROR"], "backoff": "exponential"},
    }
    client = FakeModelClient(scripts=[("Mutator", [{
        "hypothesis": "add independent verifier + retries",
        "mutation": {"mode": "component", "target": "agents", "description": "add verifier"},
        "patch": patch,
        "expected_effect": {"quality": 0.1},
        "affected_tests": ["adversarial"],
    }])])
    proposals = HarnessMutator(client).propose(
        _genome(),
        FailureAnalyst(FakeModelClient()).analyze(_genome(), _eval()),
        number_of_candidates=1,
    )
    assert proposals and proposals[0].mode == "component"
    parent = _genome()
    candidate = apply_mutation(parent, proposals[0])
    assert candidate.parent_id == parent.genome_id
    assert candidate.version == parent.version + 1
    assert candidate.genome_id != parent.genome_id
    assert any(a.role == "verifier" for a in candidate.agents)
    assert candidate.recovery.max_retries == 2


def test_judge_review_and_independence_flag():
    client = FakeModelClient(scripts=[("Judge", {"verdict": "promotable", "reason": "reliably better", "confidence": 0.97})])
    review = ExperimentJudge(client, judge_model="judge-model").compare(
        incumbent=_eval("inc", 0.5), challenger=_eval("chal", 0.8), bootstrap=(0.2, 0.1, 0.3), mutator_model="mutator-model",
    )
    assert review.verdict == "promotable"
    assert review.independent is True


def test_malformed_model_output_raises_specialist_error():
    client = FakeModelClient(scripts=[("Architect", "not valid json {{{")])
    with pytest.raises(SpecialistError):
        HarnessArchitect(client).design({"task_type": "x"})


def test_apply_mutation_deep_merges_router_threshold():
    parent = _genome()
    proposal = MutationProposal(
        hypothesis="h", mode="local", target="router.confidence_threshold", description="d",
        patch={"router": {"confidence_threshold": 0.9}},
    )
    candidate = apply_mutation(parent, proposal)
    assert candidate.router.model == parent.router.model  # deep-merged, model preserved
    assert candidate.router.confidence_threshold == 0.9   # threshold updated
