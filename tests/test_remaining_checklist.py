from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dataevol.benchmarks import BENCHMARK_TYPES, build_benchmark
from dataevol.compat import call_core
from dataevol.compress import ExtractiveCompressionModel, compress_trace, key_fact_retention
from dataevol.config import DataEvolConfig
from dataevol.datasets import (
    DATASET_TYPES,
    build_dataset,
    build_router_performance_dataset,
    cost_normalized_quality,
    escalation_rescue_rate,
    export_local_training_datasets,
    generate_candidate_router_policy,
    provider_success_rate,
)
from dataevol.dedupe import find_near_duplicates, near_duplicate_score
from dataevol.evolve import (
    detect_opportunities,
    generate_component_idea_prd,
    reject_weak_opportunities,
    save_idea_prd,
    save_learning_opportunities,
    validate_idea_prd,
)
from dataevol.experiments import (
    compare_experiment,
    create_rollback_snapshot,
    freeze_benchmark_for_experiment,
    promote_experiment,
    reject_experiment,
    run_router_policy_experiment,
)
from dataevol.ingest import (
    import_biolatent_run,
    import_coordinate_run,
    import_fractal_router_decisions,
    parse_openrouter_metadata,
)
from dataevol.integrations import (
    LOCAL_MODEL_METADATA,
    OPENROUTER_MODEL_METADATA,
    biolatent_verification_payload,
    coordinate_completion_payload,
    router_dataset_pull,
)
from dataevol.label import KeywordLocalModelLabeler, label_trace, load_human_overrides
from dataevol.local_models import evaluate_local_adapter, prepare_local_adapter_training, promote_local_adapter
from dataevol.prompts import (
    ab_test_prompt_packs,
    generate_candidate_prompt_pack,
    generate_prompt_variants,
    promote_prompt_pack,
    version_prompt_pack,
)
from dataevol.reports import build_report_payload, export_markdown_report
from dataevol.schemas import normalize_outcome_label, normalize_task_type, validate_trace
from dataevol.storage import init_db
from dataevol.synthetic import generate_synthetic_data


def test_importers_normalization_and_openrouter_metadata(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"traces": [{"type": "router_trace", "input": "Route docs", "output": "ok"}]}), encoding="utf-8")
    assert import_coordinate_run(run)[0]["trace_type"] == "router_trace"

    bio = tmp_path / "biolatent_run.json"
    bio.write_text(json.dumps({"verification_traces": [{"prompt": "Verify protocol", "response": "passed"}]}), encoding="utf-8")
    assert import_biolatent_run(tmp_path)[0]["trace_type"] == "scientific_trace"

    router = tmp_path / "router_decisions.json"
    router.write_text(json.dumps({"decisions": [{"prompt": "route task", "provider": "OpenRouter", "model": "free"}]}), encoding="utf-8")
    assert import_fractal_router_decisions(tmp_path)[0]["metadata"]["router_decision"]["model"] == "free"

    meta = parse_openrouter_metadata({"provider": "OpenRouter", "prompt_cost_usd": 0, "completion_cost_usd": 0, "duration_ms": 12})
    assert meta["free_or_cheap"] is True
    assert normalize_task_type("literature review") == "documentation"
    assert normalize_outcome_label("passed") == "accepted"
    assert validate_trace({"type": "router_trace", "input": "demo"}).trace_type == "router_trace"


def test_near_duplicate_and_low_value_repeat_compaction() -> None:
    left = {"id": "a", "prompt": "Fix router timeout test", "response": "pytest passed", "task": "router timeout"}
    right = {"id": "b", "prompt": "Fix the router timeout test", "response": "pytest passed", "task": "router timeout"}
    assert near_duplicate_score(left, right) > 0.8
    matches = find_near_duplicates([left, right])
    assert matches and matches[0]["prompt_similarity"] >= 0.8


def test_label_compress_and_scoring_extension_points(tmp_path: Path) -> None:
    trace = {"trace_type": "worker_trace", "prompt": "rescued by stronger model", "response": "training candidate"}
    assert KeywordLocalModelLabeler().label(trace)[0] == "rescued_by_stronger_model"
    assert label_trace({"trace_type": "worker_trace", "prompt": "unsafe output"})[0] == "unsafe_or_policy_blocked"

    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"task-1": "accepted"}), encoding="utf-8")
    assert load_human_overrides(overrides)["task-1"] == "accepted"

    compressed = compress_trace(
        {"trace_type": "coding_trace", "task_id": "task-1", "prompt": "bad", "response": "Verification failed"},
        "failed_verification",
        model=ExtractiveCompressionModel(),
    )
    assert compressed["failure_type"] == "weak_evidence"
    assert compressed["key_fact_retention"] > 0
    assert key_fact_retention(compressed["summary"], {"task_id": "task-1"}) == 1.0


def test_generic_datasets_benchmarks_router_loop_and_local_exports(tmp_path: Path) -> None:
    traces = [{"id": "t1", "run_id": "r1", "prompt": "route", "label": "accepted", "provider": "openrouter", "score": 0.9, "cost_usd": 0.01}]
    for dataset_type in DATASET_TYPES - {"local-router", "local-compressor", "local-duplicate-detector", "local-evaluator"}:
        result = build_dataset(dataset_type, traces, tmp_path / "datasets")
        assert result.dataset_path.exists()
    for benchmark_type in BENCHMARK_TYPES:
        result = build_benchmark(benchmark_type, [{"task": "case"}], tmp_path / f"bench_{benchmark_type}", overwrite=True)
        assert result.manifest_path.exists()

    perf = build_router_performance_dataset(traces)
    assert provider_success_rate(perf)["openrouter"] == 1.0
    assert cost_normalized_quality(perf[0]) > 1
    assert escalation_rescue_rate(perf) == 0
    assert generate_candidate_router_policy(perf)["preferred_providers"] == ["openrouter"]

    local = export_local_training_datasets(traces, tmp_path / "local", opt_in=True)
    assert set(local) == {"local-router", "local-compressor", "local-duplicate-detector", "local-evaluator"}


def test_synthetic_engine_provenance_and_filters() -> None:
    items = generate_synthetic_data([{"id": "bad-1", "trace_type": "router_trace", "failure_type": "bad_router_assignment", "prompt": "route"}])
    assert items
    assert all(item["synthetic"] is True and item["measured"] is False for item in items)
    assert all("generation_method" in item and "provenance" in item for item in items)


def test_evolution_idea_experiment_promotion_rejection_and_prompt_local_loops(tmp_path: Path) -> None:
    opportunities = detect_opportunities([
        {"id": "f1", "label": "failed_tests", "failure_type": "failed_code_tests"},
        {"id": "f2", "label": "failed_tests", "failure_type": "failed_code_tests"},
    ])
    assert any(item["category"] == "missing_benchmark" for item in opportunities)
    assert reject_weak_opportunities([{"observation": ""}])[0]["status"] == "NO_IDEA"
    opp_path = save_learning_opportunities(opportunities, tmp_path / "evolution")
    assert opp_path.exists()

    for component in ("prompt", "verifier", "local_model", "benchmark"):
        prd = generate_component_idea_prd(opportunities[0], component)
        assert validate_idea_prd(prd)[0]
        assert save_idea_prd(prd, tmp_path / "ideas", slug=component).exists()

    bench = tmp_path / "benchmark.jsonl"
    bench.write_text('{"id":"case"}\n', encoding="utf-8")
    assert freeze_benchmark_for_experiment(bench, tmp_path / "exp").exists()
    rollback = create_rollback_snapshot("router", "v0", tmp_path / "rollbacks")
    report = run_router_policy_experiment(
        {
            "control": [{"cost_per_verified_task": 1, "correctness": 1, "verification_pass_rate": 1, "safety_score": 1}] * 2,
            "variant": [{"cost_per_verified_task": 0.5, "correctness": 1, "verification_pass_rate": 1, "safety_score": 1}] * 2,
        },
        tmp_path / "exp",
        rollback_snapshot=str(rollback),
    )
    assert compare_experiment(report, tmp_path / "exp")["verdict"] == "promotable"
    assert promote_experiment(report, tmp_path / "promote")["promoted"] is True
    assert reject_experiment({"experiment_id": "bad"}, tmp_path / "reject")["rejected"] is True

    pack_path = version_prompt_pack({"manager": "plan", "worker": "do", "critic": "review", "verifier": "check"}, tmp_path / "prompts")
    assert pack_path.exists()
    variants = generate_prompt_variants({"manager": "plan"})
    assert generate_candidate_prompt_pack(variants)["manager"].count("verifiable") >= 1
    prompt_test = ab_test_prompt_packs({"success_rate": 0.8, "hallucination_rate": 0.1}, {"success_rate": 0.9, "hallucination_rate": 0.1})
    assert promote_prompt_pack(prompt_test, tmp_path / "prompts").exists()

    plan = prepare_local_adapter_training(tmp_path / "local", python_bin="python", experts=("router",), count=4, iters=1)
    script = plan.script_path.read_text(encoding="utf-8")
    assert plan.manifest_path.exists()
    assert "run_local_adapter_training_from_manifest" in script
    assert "placeholder" not in script.lower()
    evaluation = evaluate_local_adapter({"baseline_quality_score": 0.7, "quality_score": 0.8})
    assert promote_local_adapter(evaluation, tmp_path / "local").exists()


def test_integrations_reports_migrations_and_compat(tmp_path: Path) -> None:
    db = tmp_path / "dataevol.sqlite"
    init_db(db)
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "traces", "scores", "compressed_traces", "duplicate_clusters"}.issubset(tables)

    cfg = DataEvolConfig(tmp_path / "dataevol.toml", db, tmp_path / "raw", tmp_path / "artifacts", "secret")
    report = call_core("reports", "markdown", {}, config=cfg)
    assert report["markdown_path"]
    payload = build_report_payload(db, tmp_path / "artifacts")
    assert export_markdown_report(payload, tmp_path / "artifacts/report.md").exists()

    assert coordinate_completion_payload({"traces": []})["source_system"] == "coordinate"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"name": "router_dataset"}), encoding="utf-8")
    assert router_dataset_pull(manifest)["consumer"] == "fractal-router-api"
    assert biolatent_verification_payload({"prompt": "verify"})["trace"]["trace_type"] == "verification_trace"
    assert OPENROUTER_MODEL_METADATA["provider"] == "openrouter"
    assert LOCAL_MODEL_METADATA["provider"] == "local"
