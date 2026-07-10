from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from dataevol.api.app import create_app
from dataevol.config import (
    VALID_PRIVACY_MODES,
    config_text,
    ensure_project_dirs,
    load_config,
)
from dataevol.compat import call_core

app = typer.Typer(help="DataEvol trace evolution CLI.")
dataset_app = typer.Typer(help="Dataset commands.")
benchmark_app = typer.Typer(help="Benchmark commands.")
evolve_app = typer.Typer(help="Reflection, experiment, and promotion commands.")
synthetic_app = typer.Typer(help="Synthetic data commands.")
privacy_app = typer.Typer(help="Privacy commands.")
report_app = typer.Typer(help="Report commands.")
local_model_app = typer.Typer(help="Local model adapter commands.")
prompt_app = typer.Typer(help="Prompt pack commands.")
integration_app = typer.Typer(help="Integration client commands.")
harness_app = typer.Typer(help="Harness evolver commands.")

app.add_typer(dataset_app, name="dataset")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(evolve_app, name="evolve")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(privacy_app, name="privacy")
app.add_typer(report_app, name="report")
app.add_typer(local_model_app, name="local-model")
app.add_typer(prompt_app, name="prompt")
app.add_typer(integration_app, name="integration")
app.add_typer(harness_app, name="harness")


def _print_result(result: dict[str, Any]) -> None:
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


def _expert_list(expert: str | None) -> list[str] | None:
    if not expert:
        return None
    return [item.strip() for item in expert.split(",") if item.strip()]


def _json_option(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


@app.command()
def init(
    config: Annotated[Path, typer.Option("--config", help="Config file path.")] = Path(
        "dataevol.toml"
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing config.")] = False,
) -> None:
    """Initialize local DataEvol config and directories."""
    if config.exists() and not force:
        typer.echo(f"Config already exists: {config}")
    else:
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(config_text(), encoding="utf-8")
        typer.echo(f"Wrote config: {config}")
    loaded = load_config(config)
    ensure_project_dirs(loaded)
    typer.echo(f"DataEvol directories ready under {loaded.db_path.parent}")


@app.command()
def ingest(
    jsonl: Annotated[Path | None, typer.Option("--jsonl", help="JSONL trace file.")] = None,
    source: Annotated[str | None, typer.Option("--source", help="Source system.")] = None,
    path: Annotated[Path | None, typer.Option("--path", help="Run artifact path.")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Config file path.")] = None,
) -> None:
    """Ingest traces or run artifacts."""
    cfg = load_config(config)
    if jsonl is None and path is None:
        raise typer.BadParameter("Provide --jsonl or --path.")
    payload = {"jsonl": str(jsonl) if jsonl else None, "source": source, "path": str(path) if path else None}
    _print_result(call_core("ingest", "ingest", payload, config=cfg))


@app.command()
def label(run_id: Annotated[str, typer.Option("--run-id")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """Label traces for a run."""
    _print_result(call_core("labeling", "label_run", {"run_id": run_id}, config=load_config(config)))


@app.command()
def score(run_id: Annotated[str, typer.Option("--run-id")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """Score traces for a run."""
    _print_result(call_core("scoring", "score_run", {"run_id": run_id}, config=load_config(config)))


@app.command()
def compress(run_id: Annotated[str, typer.Option("--run-id")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """Compress traces for a run."""
    _print_result(call_core("compression", "compress_run", {"run_id": run_id}, config=load_config(config)))


@dataset_app.command("build")
def dataset_build(
    type: Annotated[str, typer.Option("--type", help="router, critic, verifier, or compressor.")],
    run_id: Annotated[str | None, typer.Option("--run-id", help="Build from a specific run.")] = None,
    from_runs: Annotated[str | None, typer.Option("--from-runs", help="Run selector such as last_100 or all.")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Build a dataset."""
    _print_result(call_core("datasets", "build_dataset", {"type": type, "run_id": run_id, "from_runs": from_runs}, config=load_config(config)))


@dataset_app.command("router-performance")
def dataset_router_performance(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    from_runs: Annotated[str | None, typer.Option("--from-runs")] = "last_100",
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    _print_result(call_core("datasets", "router_performance", {"run_id": run_id, "from_runs": from_runs}, config=load_config(config)))


@dataset_app.command("candidate-router-policy")
def dataset_candidate_router_policy(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    from_runs: Annotated[str | None, typer.Option("--from-runs")] = "last_100",
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    _print_result(call_core("datasets", "candidate_router_policy", {"run_id": run_id, "from_runs": from_runs}, config=load_config(config)))


@benchmark_app.command("build")
def benchmark_build(
    from_runs: Annotated[str | None, typer.Option("--from-runs", help="Run selector such as last_100 or all.")] = None,
    type: Annotated[str | None, typer.Option("--type", help="router, prompt, verifier, critic, or compressor.")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Build from a specific run.")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Build a benchmark."""
    _print_result(call_core("benchmarks", "build_benchmark", {"from_runs": from_runs or "last_100", "run_id": run_id, "type": type}, config=load_config(config)))


@evolve_app.command("reflect")
def reflect(run_id: Annotated[str, typer.Option("--run-id")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("evolve", "reflect", {"run_id": run_id}, config=load_config(config)))


@evolve_app.command("idea-prd")
def idea_prd(
    opportunity: Annotated[str | None, typer.Option("--opportunity", help="Opportunity JSON object or JSON file path.")] = None,
    opportunity_id: Annotated[int | None, typer.Option("--opportunity-id", help="Load opportunity from SQLite by id.")] = None,
    component: Annotated[str, typer.Option("--component", help="router, prompt, verifier, local_model, benchmark, or generic.")] = "router",
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    _print_result(call_core("evolve", "idea_prd", {"opportunity": opportunity, "opportunity_id": opportunity_id, "component": component}, config=load_config(config)))


@evolve_app.command("experiment")
def experiment(
    idea: Annotated[Path | None, typer.Option("--idea")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    variant_provider: Annotated[str, typer.Option("--variant-provider")] = "openrouter",
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {
        "idea": str(idea) if idea else None,
        "run_id": run_id,
        "variant_provider": variant_provider,
    }
    _print_result(call_core("evolve", "experiment", payload, config=load_config(config)))


@evolve_app.command("compare")
def compare(experiment: Annotated[str, typer.Option("--experiment")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("evolve", "compare", {"experiment": experiment}, config=load_config(config)))


@evolve_app.command("promote")
def promote(experiment: Annotated[str, typer.Option("--experiment")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("evolve", "promote", {"experiment": experiment}, config=load_config(config)))


@evolve_app.command("reject")
def reject(experiment: Annotated[str, typer.Option("--experiment")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("evolve", "reject", {"experiment": experiment}, config=load_config(config)))


@synthetic_app.command("generate")
def synthetic_generate(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """Generate deterministic synthetic variants from supplied or fixture traces."""
    _print_result(call_core("synthetic", "synthetic_generate", {"traces": []}, config=load_config(config)))


@privacy_app.command("set")
def privacy_set(
    mode: Annotated[str, typer.Argument(help="Privacy mode.")],
    config: Annotated[Path, typer.Option("--config")] = Path("dataevol.toml"),
) -> None:
    """Set privacy mode in the local config."""
    if mode not in VALID_PRIVACY_MODES:
        raise typer.BadParameter(f"Expected one of: {', '.join(sorted(VALID_PRIVACY_MODES))}")
    cfg = load_config(config)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        config_text(
            token=cfg.api_token,
            privacy_mode=mode,
            db_path=cfg.db_path,
            raw_path=cfg.raw_path,
            artifacts_path=cfg.artifacts_path,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Privacy mode set to {mode}")


@privacy_app.command("export-candidates")
def privacy_export_candidates(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    from_runs: Annotated[str | None, typer.Option("--from-runs")] = "last_100",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    public: Annotated[bool, typer.Option("--public", help="Enforce public benchmark export policy.")] = False,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {"run_id": run_id, "from_runs": from_runs, "output": str(output) if output else None, "public": public}
    _print_result(call_core("privacy", "export_training_candidates", payload, config=load_config(config)))


@report_app.command("runs")
def report_runs(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "runs", {}, config=load_config(config)))


@report_app.command("datasets")
def report_datasets(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "datasets", {}, config=load_config(config)))


@report_app.command("benchmarks")
def report_benchmarks(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "benchmarks", {}, config=load_config(config)))


@report_app.command("experiments")
def report_experiments(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "experiments", {}, config=load_config(config)))


@report_app.command("opportunities")
def report_opportunities(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "opportunities", {}, config=load_config(config)))


@report_app.command("idea-prds")
def report_idea_prds(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "idea_prds", {}, config=load_config(config)))


@report_app.command("promotions")
def report_promotions(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "promotions", {}, config=load_config(config)))


@report_app.command("inbox")
def report_inbox(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "inbox", {}, config=load_config(config)))


@report_app.command("markdown")
def report_markdown(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("reports", "markdown", {}, config=load_config(config)))


@local_model_app.command("prepare")
def local_model_prepare(
    output: Annotated[Path | None, typer.Option("--output", help="Training artifact directory.")] = None,
    base_model: Annotated[str | None, typer.Option("--base-model", help="MLX model id.")] = None,
    expert: Annotated[str | None, typer.Option("--expert", help="Expert name or comma-separated expert list.")] = None,
    count: Annotated[int, typer.Option("--count", help="Examples per expert.")] = 24,
    iters: Annotated[int, typer.Option("--iters", help="MLX LoRA iterations.")] = 2,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Write datasets, manifest, and real MLX training driver."""
    payload = {
        "output": str(output) if output else None,
        "base_model": base_model,
        "experts": _expert_list(expert),
        "count": count,
        "iters": iters,
    }
    _print_result(call_core("local_models", "prepare", payload, config=load_config(config)))


@local_model_app.command("train")
def local_model_train(
    output: Annotated[Path | None, typer.Option("--output", help="Training artifact directory.")] = None,
    base_model: Annotated[str | None, typer.Option("--base-model", help="MLX model id.")] = None,
    expert: Annotated[str | None, typer.Option("--expert", help="Expert name or comma-separated expert list.")] = None,
    count: Annotated[int, typer.Option("--count", help="Examples per expert.")] = 24,
    iters: Annotated[int, typer.Option("--iters", help="MLX LoRA iterations.")] = 2,
    execute: Annotated[bool, typer.Option("--execute", help="Actually run mlx_lm lora for each expert.")] = False,
    timeout: Annotated[int, typer.Option("--timeout", help="Per-expert training timeout in seconds.")] = 1800,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Prepare and optionally execute local expert adapter training."""
    payload = {
        "output": str(output) if output else None,
        "base_model": base_model,
        "experts": _expert_list(expert),
        "count": count,
        "iters": iters,
        "execute": execute,
        "timeout": timeout,
    }
    _print_result(call_core("local_models", "train", payload, config=load_config(config)))


@local_model_app.command("evaluate")
def local_model_evaluate(
    baseline_quality_score: Annotated[float, typer.Option("--baseline-quality-score")],
    quality_score: Annotated[float, typer.Option("--quality-score")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Evaluate a local adapter candidate against a baseline score."""
    payload = {"metrics": {"baseline_quality_score": baseline_quality_score, "quality_score": quality_score}}
    _print_result(call_core("local_models", "evaluate", payload, config=load_config(config)))


@local_model_app.command("promote")
def local_model_promote(
    baseline_quality_score: Annotated[float, typer.Option("--baseline-quality-score")],
    quality_score: Annotated[float, typer.Option("--quality-score")],
    output: Annotated[Path | None, typer.Option("--output", help="Promotion artifact directory.")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Promote a local adapter only if its benchmark score improves."""
    evaluation = {"baseline_quality_score": baseline_quality_score, "quality_score": quality_score}
    evaluated = call_core("local_models", "evaluate", {"metrics": evaluation}, config=load_config(config))
    payload = {"output": str(output) if output else None, "evaluation": evaluated}
    _print_result(call_core("local_models", "promote", payload, config=load_config(config)))


@prompt_app.command("variants")
def prompt_variants(
    pack: Annotated[str | None, typer.Option("--pack", help="Prompt pack JSON object.")] = None,
    pack_path: Annotated[Path | None, typer.Option("--pack-path", help="Prompt pack JSON file.")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {"pack": _json_option(pack), "pack_path": str(pack_path) if pack_path else None}
    _print_result(call_core("prompts", "variants", payload, config=load_config(config)))


@prompt_app.command("version")
def prompt_version(
    pack: Annotated[str | None, typer.Option("--pack", help="Prompt pack JSON object.")] = None,
    pack_path: Annotated[Path | None, typer.Option("--pack-path", help="Prompt pack JSON file.")] = None,
    version: Annotated[str, typer.Option("--version")] = "v1",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {"pack": _json_option(pack), "pack_path": str(pack_path) if pack_path else None, "version": version, "output": str(output) if output else None}
    _print_result(call_core("prompts", "version", payload, config=load_config(config)))


@prompt_app.command("ab-test")
def prompt_ab_test(
    control_metrics: Annotated[str, typer.Option("--control-metrics", help="JSON metrics object or file.")],
    variant_metrics: Annotated[str, typer.Option("--variant-metrics", help="JSON metrics object or file.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {"control_metrics": _json_option(control_metrics), "variant_metrics": _json_option(variant_metrics)}
    _print_result(call_core("prompts", "ab_test", payload, config=load_config(config)))


@prompt_app.command("promote")
def prompt_promote(
    test_result: Annotated[str, typer.Option("--test-result", help="JSON test result object or file.")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {"test_result": _json_option(test_result), "output": str(output) if output else None}
    _print_result(call_core("prompts", "promote", payload, config=load_config(config)))


@integration_app.command("router-dataset-pull")
def integration_router_dataset_pull(
    manifest: Annotated[Path | None, typer.Option("--manifest", help="Local manifest path.")] = None,
    endpoint: Annotated[str | None, typer.Option("--endpoint", help="Remote router/DataEvol endpoint.")] = None,
    token: Annotated[str | None, typer.Option("--token")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {"manifest": str(manifest) if manifest else None, "endpoint": endpoint, "token": token}
    _print_result(call_core("integrations", "router_dataset_pull", payload, config=load_config(config)))


@integration_app.command("post-coordinate-completion")
def integration_post_coordinate_completion(
    endpoint: Annotated[str, typer.Option("--endpoint")],
    run: Annotated[str, typer.Option("--run", help="Run JSON object or file.")],
    token: Annotated[str | None, typer.Option("--token")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    payload = {"endpoint": endpoint, "run": _json_option(run), "token": token}
    _print_result(call_core("integrations", "post_coordinate_completion", payload, config=load_config(config)))


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8765,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run the local DataEvol API."""
    import uvicorn

    uvicorn.run(create_app(load_config(config)), host=host, port=port)


@harness_app.command("evolve")
def harness_evolve(
    task: Annotated[str, typer.Option("--task", help="Task spec: JSON object, JSON string, or path to a .json file.")],
    max_generations: Annotated[int, typer.Option("--max-generations")] = 20,
    number_of_candidates: Annotated[int, typer.Option("--number-of-candidates")] = 8,
    repeated_runs: Annotated[int, typer.Option("--repeated-runs")] = 3,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run the harness evolution loop (requires a configured [model] backend)."""
    _print_result(call_core("harness", "evolve", {
        "task": task, "max_generations": max_generations,
        "number_of_candidates": number_of_candidates, "repeated_runs": repeated_runs,
    }, config=load_config(config)))


@harness_app.command("design")
def harness_design(
    task: Annotated[str, typer.Option("--task", help="Task spec JSON / path.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Design an initial harness genome for a task (requires [model])."""
    _print_result(call_core("harness", "design", {"task": task}, config=load_config(config)))


@harness_app.command("benchmark")
def harness_benchmark(
    task: Annotated[str, typer.Option("--task", help="Task spec JSON / path.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Build a frozen harness benchmark suite (requires [model])."""
    _print_result(call_core("harness", "benchmark", {"task": task}, config=load_config(config)))


@harness_app.command("evaluate")
def harness_evaluate(
    genome: Annotated[str, typer.Option("--genome", help="Genome JSON / path.")],
    benchmark: Annotated[str | None, typer.Option("--benchmark", help="Benchmark path or cases JSON.")] = None,
    repeated_runs: Annotated[int, typer.Option("--repeated-runs")] = 3,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Evaluate a genome against a benchmark with the deterministic executor."""
    _print_result(call_core("harness", "evaluate", {"genome": genome, "benchmark": benchmark, "repeated_runs": repeated_runs}, config=load_config(config)))


@harness_app.command("failures")
def harness_failures(
    genome: Annotated[str, typer.Option("--genome", help="Genome JSON / path.")],
    evaluation: Annotated[str, typer.Option("--evaluation", help="Evaluation JSON / path.")] = "{}",
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Classify earliest-causal failures (requires [model])."""
    _print_result(call_core("harness", "failures", {"genome": genome, "evaluation": evaluation}, config=load_config(config)))


@harness_app.command("mutate")
def harness_mutate(
    incumbent: Annotated[str, typer.Option("--incumbent", help="Incumbent genome JSON / path.")],
    number_of_candidates: Annotated[int, typer.Option("--number-of-candidates")] = 8,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Propose targeted harness mutations (requires [model])."""
    _print_result(call_core("harness", "mutate", {"incumbent": incumbent, "number_of_candidates": number_of_candidates}, config=load_config(config)))


@harness_app.command("judge")
def harness_judge(
    incumbent_eval: Annotated[str, typer.Option("--incumbent-eval", help="Incumbent evaluation JSON / path.")],
    challenger_eval: Annotated[str, typer.Option("--challenger-eval", help="Challenger evaluation JSON / path.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run the qualitative experiment judge (requires [model])."""
    _print_result(call_core("harness", "judge", {"incumbent_eval": incumbent_eval, "challenger_eval": challenger_eval}, config=load_config(config)))


@harness_app.command("lineage")
def harness_lineage(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """List harness lineage records."""
    _print_result(call_core("harness", "lineage", {}, config=load_config(config)))


@harness_app.command("incumbent")
def harness_incumbent(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """Show the current incumbent harness genome."""
    _print_result(call_core("harness", "incumbent", {}, config=load_config(config)))


@harness_app.command("training-records")
def harness_training_records(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """List emitted harness training records."""
    _print_result(call_core("harness", "training_records", {}, config=load_config(config)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
