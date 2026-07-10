from __future__ import annotations

from dataevol.harness.genome import (
    AgentSpec,
    HarnessGenome,
    MemorySpec,
    MutationRecord,
    OutputSchemaSpec,
    RecoverySpec,
    RouterSpec,
    WorkflowStep,
    new_genome_id,
)


def _genome(**overrides) -> HarnessGenome:
    base = dict(
        genome_id=new_genome_id(),
        version=1,
        parent_id=None,
        task_type="permit_set_review",
        task_spec_hash="abc",
        router=RouterSpec(model="local-9b-router", confidence_threshold=0.78),
        agents=(
            AgentSpec(role="drawing_inventory", model="vision-model", prompt_ref="prompts/drawing_inventory.md"),
            AgentSpec(role="code_compliance", model="reasoning-model", prompt_ref="prompts/code.md", tools=("building_code_search",)),
            AgentSpec(role="verifier", model="reasoning-model", prompt_ref="prompts/verify.md", cannot_view=("previous_agent_confidence",)),
        ),
        workflow=(
            WorkflowStep(step_id="extract", agent_role="drawing_inventory"),
            WorkflowStep(step_id="classify", agent_role="code_compliance", depends_on=("extract",)),
            WorkflowStep(step_id="verify", agent_role="verifier", depends_on=("classify",)),
        ),
        memory=MemorySpec(type="structured_state", schema={"v": 3}),
        recovery=RecoverySpec(max_retries=2, retry_on=("malformed_output", "verifier_disagreement")),
        output=OutputSchemaSpec(schema={"report": "permit_review_report_v2"}, validation="strict"),
    )
    base.update(overrides)
    return HarnessGenome(**base)


def test_round_trip_preserves_structure():
    g = _genome()
    restored = HarnessGenome.from_dict(g.to_dict())
    assert restored.router.model == g.router.model
    assert restored.router.confidence_threshold == g.router.confidence_threshold
    assert tuple(a.role for a in restored.agents) == tuple(a.role for a in g.agents)
    assert restored.recovery.max_retries == g.recovery.max_retries
    assert restored.content_hash() == g.content_hash()


def test_content_hash_ignores_lineage_fields():
    a = _genome()
    b = _genome(
        genome_id=new_genome_id(),
        version=7,
        parent_id="ancestor",
        mutation=MutationRecord(mode="local", target="router.confidence_threshold", description="tweak"),
        hypothesis="maybe cheaper",
        created_at="2099-01-01T00:00:00+00:00",
    )
    assert a.content_hash() == b.content_hash(), "ancestry/metadata must not affect dedup hash"


def test_content_hash_changes_when_component_changes():
    a = _genome()
    b = _genome(router=RouterSpec(model="frontier-opus", confidence_threshold=0.78))
    assert a.content_hash() != b.content_hash()


def test_with_component_returns_copy_with_replacement():
    a = _genome()
    b = a.with_component(recovery=RecoverySpec(max_retries=5))
    assert a.recovery.max_retries == 2
    assert b.recovery.max_retries == 5
    assert b.genome_id == a.genome_id
