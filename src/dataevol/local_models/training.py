from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .adapters import BASE_MODEL, EXPERTS, AdapterJob, build_adapter_jobs, run_adapter_job, write_expert_datasets


@dataclass(frozen=True)
class LocalAdapterTrainingPlan:
    base_model: str
    data_root: Path
    adapter_root: Path
    manifest_path: Path
    script_path: Path
    jobs: tuple[AdapterJob, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "data_root": str(self.data_root),
            "adapter_root": str(self.adapter_root),
            "manifest_path": str(self.manifest_path),
            "script_path": str(self.script_path),
            "experts": [job.expert for job in self.jobs],
            "jobs": [_job_to_dict(job) for job in self.jobs],
        }


def prepare_local_adapter_training(
    output_dir: str | Path,
    *,
    python_bin: str | Path = sys.executable,
    base_model: str = BASE_MODEL,
    experts: Iterable[str] = EXPERTS,
    count: int = 24,
    iters: int = 2,
) -> LocalAdapterTrainingPlan:
    """Create reproducible datasets, manifest, and driver for MLX LoRA training."""
    selected_experts = tuple(experts)
    if not selected_experts:
        raise ValueError("at least one expert is required")
    if count < 4:
        raise ValueError("count must be at least 4 so train/valid/test splits are populated")
    if iters < 1:
        raise ValueError("iters must be at least 1")

    root = Path(output_dir)
    data_root = root / "adapter_data"
    adapter_root = root / "adapters"
    root.mkdir(parents=True, exist_ok=True)
    write_expert_datasets(data_root, selected_experts, count=count)
    jobs = tuple(
        build_adapter_jobs(
            python_bin,
            data_root,
            adapter_root,
            base_model=base_model,
            experts=selected_experts,
            iters=iters,
        )
    )

    manifest_path = root / "adapter_training_manifest.json"
    script_path = root / "train_adapters.py"
    manifest = {
        "schema_version": 1,
        "base_model": base_model,
        "data_root": str(data_root),
        "adapter_root": str(adapter_root),
        "count": count,
        "iters": iters,
        "experts": list(selected_experts),
        "jobs": [_job_to_dict(job) for job in jobs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    script_path.write_text(_driver_script(manifest_path.name), encoding="utf-8")
    return LocalAdapterTrainingPlan(base_model, data_root, adapter_root, manifest_path, script_path, jobs)


def run_local_adapter_training(
    output_dir: str | Path,
    *,
    python_bin: str | Path = sys.executable,
    base_model: str = BASE_MODEL,
    experts: Iterable[str] = EXPERTS,
    count: int = 24,
    iters: int = 2,
    execute: bool = False,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Prepare and optionally execute MLX LoRA training for each DataEvol expert."""
    plan = prepare_local_adapter_training(
        output_dir,
        python_bin=python_bin,
        base_model=base_model,
        experts=experts,
        count=count,
        iters=iters,
    )
    if not execute:
        result = plan.to_dict()
        result.update({"ok": True, "status": "planned", "executed": False, "results": []})
        return result

    results = [run_adapter_job(job, timeout=timeout) for job in plan.jobs]
    ok = all(result.get("returncode") == 0 for result in results)
    return {
        **plan.to_dict(),
        "ok": ok,
        "status": "completed" if ok else "failed",
        "executed": True,
        "results": results,
    }


def run_local_adapter_training_from_manifest(
    manifest_path: str | Path,
    *,
    execute: bool = True,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Run adapter training from a committed manifest written by prepare."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    jobs = tuple(_job_from_dict(job) for job in manifest.get("jobs", []))
    if not jobs:
        raise ValueError("manifest contains no adapter jobs")

    plan = {
        "base_model": manifest.get("base_model"),
        "data_root": manifest.get("data_root"),
        "adapter_root": manifest.get("adapter_root"),
        "manifest_path": str(manifest_file),
        "script_path": str(manifest_file.with_name("train_adapters.py")),
        "experts": [job.expert for job in jobs],
        "jobs": [_job_to_dict(job) for job in jobs],
    }
    if not execute:
        return {**plan, "ok": True, "status": "planned", "executed": False, "results": []}

    results = [run_adapter_job(job, timeout=timeout) for job in jobs]
    ok = all(result.get("returncode") == 0 for result in results)
    return {**plan, "ok": ok, "status": "completed" if ok else "failed", "executed": True, "results": results}


def evaluate_local_adapter(metrics: Mapping[str, float]) -> dict[str, Any]:
    baseline = float(metrics.get("baseline_quality_score", 0.0))
    candidate = float(metrics.get("quality_score", 0.0))
    improvement = candidate - baseline
    return {"baseline": baseline, "candidate": candidate, "improvement": improvement, "promotable": improvement > 0.0}


def promote_local_adapter(evaluation: Mapping[str, Any], output_dir: str | Path) -> Path:
    if not evaluation.get("promotable"):
        raise ValueError("local adapter cannot promote without benchmark improvement")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "local_adapter_promotion.json"
    path.write_text(json.dumps(dict(evaluation), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _job_to_dict(job: AdapterJob) -> dict[str, Any]:
    return {
        "expert": job.expert,
        "data_dir": str(job.data_dir),
        "adapter_dir": str(job.adapter_dir),
        "command": list(job.command),
    }


def _job_from_dict(value: Mapping[str, Any]) -> AdapterJob:
    return AdapterJob(
        expert=str(value["expert"]),
        data_dir=Path(value["data_dir"]),
        adapter_dir=Path(value["adapter_dir"]),
        command=[str(part) for part in value["command"]],
    )


def _driver_script(manifest_name: str) -> str:
    return f'''"""Run DataEvol local expert adapter training from a reproducible manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataevol.local_models import run_local_adapter_training_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DataEvol MLX LoRA adapters for local experts.")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("{manifest_name}")))
    parser.add_argument("--dry-run", action="store_true", help="Print planned mlx_lm commands without executing them.")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    result = run_local_adapter_training_from_manifest(args.manifest, execute=not args.dry_run, timeout=args.timeout)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''
