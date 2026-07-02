from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

DATA_PIPELINE_EXPERTS = (
    "ingestor",
    "deduper",
    "cleaner",
    "classifier",
    "difficulty_scorer",
    "quality_scorer",
    "trace_compressor",
    "early_failure_detector",
    "verifier",
    "critic",
    "error_taxonomist",
    "synthetic_generator",
    "mutation_evolver",
    "recipe_generator",
    "recipe_verifier",
    "curriculum_builder",
    "dataset_mixer",
    "benchmark_generator",
    "regression_tester",
    "promotion_gatekeeper",
)

ECOSYSTEM_EXPERTS = (
    "router",
    "planner",
    "worker",
    "manager",
    "coordinator",
    "inspector",
    "compressor",
    "duplicate_detector",
    "failure_classifier",
    "prompt_improver",
    "prompt_pack_generator",
    "local_model_trainer",
    "local_evaluator",
    "model_mix_optimizer",
    "scientific_method",
    "coding_agent",
    "tool_trace_analyzer",
    "correction_linker",
    "benchmark_task",
    "privacy_redactor",
    "report_builder",
    "integration_bridge",
)

EXPERTS = DATA_PIPELINE_EXPERTS + ECOSYSTEM_EXPERTS


@dataclass(frozen=True)
class AdapterJob:
    expert: str
    data_dir: Path
    adapter_dir: Path
    command: list[str]


def expert_examples(expert: str, count: int = 24) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for index in range(count):
        trace = {
            "task_id": f"{expert}_{index:03d}",
            "trace_type": f"{expert}_trace",
            "objective": f"Improve DataEvol {expert} behavior on case {index}.",
            "label": "accepted" if index % 4 else "failed_verification",
            "cost_usd": round((index % 5) * 0.003, 4),
            "latency_ms": 600 + (index * 37),
            "failure_type": "bad_router_assignment" if expert in {"router", "classifier", "error_taxonomist"} and index % 4 == 0 else None,
        }
        prompt = f"You are the DataEvol {expert} expert. Analyze this trace and return compact JSON:\n{json.dumps(trace, sort_keys=True)}"
        completion = json.dumps(_expert_completion(expert, trace, index), sort_keys=True)
        examples.append({"prompt": prompt, "completion": completion})
    return examples


def write_expert_datasets(root: str | Path, experts: Iterable[str] = EXPERTS, *, count: int = 24) -> dict[str, Path]:
    out = Path(root)
    paths: dict[str, Path] = {}
    for expert in experts:
        data_dir = out / expert
        data_dir.mkdir(parents=True, exist_ok=True)
        rows = expert_examples(expert, count=count)
        train, valid, test = rows[: max(1, count - 4)], rows[max(1, count - 4) : max(2, count - 2)], rows[max(2, count - 2) :]
        _write_jsonl(data_dir / "train.jsonl", train)
        _write_jsonl(data_dir / "valid.jsonl", valid)
        _write_jsonl(data_dir / "test.jsonl", test)
        paths[expert] = data_dir
    return paths


def build_adapter_jobs(
    python_bin: str | Path,
    data_root: str | Path,
    adapter_root: str | Path,
    *,
    base_model: str = BASE_MODEL,
    experts: Iterable[str] = EXPERTS,
    iters: int = 2,
) -> list[AdapterJob]:
    jobs: list[AdapterJob] = []
    for expert in experts:
        data_dir = Path(data_root) / expert
        adapter_dir = Path(adapter_root) / expert
        command = [
            str(python_bin),
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            base_model,
            "--train",
            "--data",
            str(data_dir),
            "--adapter-path",
            str(adapter_dir),
            "--iters",
            str(iters),
            "--batch-size",
            "1",
            "--num-layers",
            "4",
            "--max-seq-length",
            "512",
            "--steps-per-report",
            "1",
            "--steps-per-eval",
            str(max(1, iters)),
            "--val-batches",
            "1",
            "--save-every",
            str(max(1, iters)),
            "--learning-rate",
            "1e-5",
        ]
        jobs.append(AdapterJob(expert, data_dir, adapter_dir, command))
    return jobs


def run_adapter_job(job: AdapterJob, *, timeout: int = 1800) -> dict[str, object]:
    job.adapter_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(job.command, text=True, capture_output=True, timeout=timeout)
    return {
        "expert": job.expert,
        "returncode": proc.returncode,
        "adapter_dir": str(job.adapter_dir),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "files": sorted(path.name for path in job.adapter_dir.glob("*")),
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _expert_completion(expert: str, trace: dict[str, object], index: int) -> dict[str, object]:
    if expert == "router":
        return {"route": "cheap_verified_model" if index % 3 else "strong_model", "reason": "cost-risk balanced"}
    if expert == "planner":
        return {"steps": ["classify task", "choose expert", "verify output"]}
    if expert == "worker":
        return {"execution_plan": ["read task", "perform work", "return evidence"], "needs_review": index % 4 == 0}
    if expert == "manager":
        return {"delegation": ["split work", "assign owner", "collect review"], "acceptance_criteria": ["tests", "summary"]}
    if expert == "coordinator":
        return {"coordination_plan": ["map dependencies", "sequence edits", "verify integration"], "blocked": False}
    if expert == "inspector":
        return {"review_verdict": "pass" if trace["label"] == "accepted" else "fail", "review_focus": ["correctness", "security", "tests"]}
    if expert == "ingestor":
        return {"accepted": True, "normalized_trace_type": trace["trace_type"], "required_fields": ["task_id", "objective", "label"]}
    if expert == "deduper":
        return {"duplicate": index % 5 == 0, "cluster_key": f"task:{trace['task_id']}", "similarity_threshold": 0.82}
    if expert == "cleaner":
        return {"cleaned": True, "removed_fields": ["raw_private_context"], "retained_fields": ["task_id", "objective", "label"]}
    if expert == "classifier":
        return {"class": "failure" if trace["label"] != "accepted" else "success", "confidence": 0.86}
    if expert == "difficulty_scorer":
        return {"difficulty": "hard" if index % 7 == 0 else "medium", "score": round(0.35 + ((index % 6) * 0.1), 2)}
    if expert == "quality_scorer":
        return {"quality_score": 0.82 if trace["label"] == "accepted" else 0.41, "rubric": ["correctness", "evidence", "cost"]}
    if expert == "trace_compressor":
        return {"summary": f"{trace['task_id']} compact outcome {trace['label']}", "retain": ["task_id", "label", "failure_type"]}
    if expert == "compressor":
        return {"summary": f"{trace['task_id']} compact outcome {trace['label']}", "retain": ["task_id", "label", "failure_type"]}
    if expert == "early_failure_detector":
        return {"early_failure": index % 4 == 0, "signal": "failed_verification" if trace["label"] != "accepted" else "none"}
    if expert == "critic":
        return {"critique": "check evidence, cost, and failure labels", "severity": "medium" if index % 4 == 0 else "low"}
    if expert == "verifier":
        return {"verdict": "pass" if trace["label"] == "accepted" else "fail", "required_evidence": ["tests", "trace"]}
    if expert == "error_taxonomist":
        return {"failure_type": trace.get("failure_type") or "weak_evidence", "taxonomy_path": ["data_pipeline", "quality"]}
    if expert == "duplicate_detector":
        return {"duplicate": index % 5 == 0, "similarity_threshold": 0.82}
    if expert == "failure_classifier":
        return {"failure_type": trace.get("failure_type") or "weak_evidence", "use_as_negative": trace["label"] != "accepted"}
    if expert == "synthetic_generator":
        return {"synthetic_trace": True, "generation_method": "counterfactual_variant", "source_task_id": trace["task_id"]}
    if expert == "mutation_evolver":
        return {"mutation": "increase_edge_case_coverage", "expected_gain": "benchmark_coverage"}
    if expert == "recipe_generator":
        return {"recipe": ["filter traces", "score quality", "mix curriculum"], "output_artifact": "dataset_recipe.json"}
    if expert == "recipe_verifier":
        return {"recipe_valid": trace["label"] == "accepted", "checks": ["schema", "privacy", "reproducibility"]}
    if expert == "curriculum_builder":
        return {"curriculum_stage": "foundation" if index % 3 else "advanced", "sampling_weight": round(1.0 + (index % 5) * 0.15, 2)}
    if expert == "dataset_mixer":
        return {"mix": {"accepted": 0.55, "failures": 0.3, "synthetic": 0.15}, "dedupe_required": True}
    if expert == "benchmark_generator":
        return {"benchmark_case": f"heldout_{trace['task_id']}", "metric": "quality_score", "acceptance_threshold": 0.8}
    if expert == "benchmark_task":
        return {"benchmark_case": f"heldout_{trace['task_id']}", "metric": "quality_score", "acceptance_threshold": 0.8}
    if expert == "regression_tester":
        return {"regression_risk": "high" if index % 6 == 0 else "low", "required_tests": ["quality", "safety", "cost"]}
    if expert == "promotion_gatekeeper":
        return {"promote": trace["label"] == "accepted", "gates": ["improvement", "non_regression", "rollback"]}
    if expert == "prompt_improver":
        return {"prompt_patch": "Add explicit output contract and verification criteria."}
    if expert == "prompt_pack_generator":
        return {"prompt_pack": {"manager": "plan with evidence", "worker": "return verifiable output"}, "version": "candidate"}
    if expert == "local_model_trainer":
        return {"training_plan": ["prepare dataset", "train adapter", "evaluate heldout"], "promote_only_after_benchmark": True}
    if expert == "local_evaluator":
        return {"quality_score": 0.82 if trace["label"] == "accepted" else 0.41, "rubric": ["correctness", "evidence", "cost"]}
    if expert == "model_mix_optimizer":
        return {"model_mix": {"cheap_verified": 0.55, "strong_model": 0.35, "local_adapter": 0.1}, "constraint": "no_quality_regression"}
    if expert == "scientific_method":
        return {"hypothesis": "trace outcome is testable", "controls": ["baseline", "heldout"], "evidence_required": ["measurements", "reproducibility"]}
    if expert == "coding_agent":
        return {"workflow": ["inspect code", "patch minimal surface", "run focused tests"], "risk": "medium" if index % 4 == 0 else "low"}
    if expert == "tool_trace_analyzer":
        return {"tool_signal": "useful" if trace["label"] == "accepted" else "needs_retry", "fields": ["tool", "args", "result"]}
    if expert == "correction_linker":
        return {"correction_required": trace["label"] != "accepted", "link_type": "failed_trace_to_fixed_trace"}
    if expert == "privacy_redactor":
        return {"privacy_status": "local_only", "redactions": ["private_user_content"], "public_export_allowed": False}
    if expert == "report_builder":
        return {"report_sections": ["summary", "metrics", "artifacts"], "audience": "operator"}
    if expert == "integration_bridge":
        return {"payload_target": "external_service", "contract": ["trace", "dataset_manifest", "status"]}
    return {"result": "accepted"}
