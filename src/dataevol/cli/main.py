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

app.add_typer(dataset_app, name="dataset")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(evolve_app, name="evolve")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(privacy_app, name="privacy")
app.add_typer(report_app, name="report")
app.add_typer(local_model_app, name="local-model")


def _print_result(result: dict[str, Any]) -> None:
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


def _expert_list(expert: str | None) -> list[str] | None:
    if not expert:
        return None
    return [item.strip() for item in expert.split(",") if item.strip()]


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
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Build a dataset."""
    _print_result(call_core("datasets", "build_dataset", {"type": type}, config=load_config(config)))


@benchmark_app.command("build")
def benchmark_build(
    from_runs: Annotated[str, typer.Option("--from-runs", help="Run selector such as last_100.")],
    type: Annotated[str | None, typer.Option("--type", help="router, prompt, verifier, critic, or compressor.")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Build a benchmark."""
    _print_result(call_core("benchmarks", "build_benchmark", {"from_runs": from_runs, "type": type}, config=load_config(config)))


@evolve_app.command("reflect")
def reflect(run_id: Annotated[str, typer.Option("--run-id")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("evolve", "reflect", {"run_id": run_id}, config=load_config(config)))


@evolve_app.command("idea-prd")
def idea_prd(opportunity: Annotated[str, typer.Option("--opportunity")], config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    _print_result(call_core("evolve", "idea_prd", {"opportunity": opportunity}, config=load_config(config)))


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


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8765,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run the local DataEvol API."""
    import uvicorn

    uvicorn.run(create_app(load_config(config)), host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
