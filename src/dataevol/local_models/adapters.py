from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

EXPERTS = (
    "router",
    "planner",
    "critic",
    "verifier",
    "compressor",
    "duplicate_detector",
    "failure_classifier",
    "prompt_improver",
)


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
            "failure_type": "bad_router_assignment" if expert == "router" and index % 4 == 0 else None,
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
    if expert == "critic":
        return {"critique": "check evidence, cost, and failure labels", "severity": "medium" if index % 4 == 0 else "low"}
    if expert == "verifier":
        return {"verdict": "pass" if trace["label"] == "accepted" else "fail", "required_evidence": ["tests", "trace"]}
    if expert == "compressor":
        return {"summary": f"{trace['task_id']} compact outcome {trace['label']}", "retain": ["task_id", "label", "failure_type"]}
    if expert == "duplicate_detector":
        return {"duplicate": index % 5 == 0, "similarity_threshold": 0.82}
    if expert == "failure_classifier":
        return {"failure_type": trace.get("failure_type") or "weak_evidence", "use_as_negative": trace["label"] != "accepted"}
    if expert == "prompt_improver":
        return {"prompt_patch": "Add explicit output contract and verification criteria."}
    return {"result": "accepted"}
