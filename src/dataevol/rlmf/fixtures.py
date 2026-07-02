from __future__ import annotations

import json
import math
import re
import urllib.request
from collections.abc import Iterable as AbcIterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


DEFAULT_RLMF_BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_QWEN_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_RLMF_EXPERTS = ("failure_classifier", "verifier", "local_model_trainer")
RLMF_SCHEMA_VERSION = 1
PUBLIC_RLMF_PRIVACY_STATUSES = frozenset({"fixture-public", "public_benchmark", "redacted_public"})
RLMF_TRAINER_PROFILES: dict[str, dict[str, Any]] = {
    "mac_mlx": {"profile": "mac_mlx", "trainer": "mlx-lora", "device": "apple_silicon", "launcher": "python -m mlx_lm lora", "supports_execute": True},
    "single_cuda": {"profile": "single_cuda", "trainer": "hf-trl-qlora", "device": "cuda", "launcher": "accelerate launch -m trl.scripts.sft", "supports_execute": True},
    "multi_gpu_accelerate": {"profile": "multi_gpu_accelerate", "trainer": "hf-trl-qlora", "device": "multi_cuda", "launcher": "accelerate launch --multi_gpu -m trl.scripts.sft", "supports_execute": True},
    "remote_worker": {"profile": "remote_worker", "trainer": "hf-trl-qlora", "device": "remote", "launcher": "remote-worker qlora", "supports_execute": False},
}
VLLM_JUDGE_SERVER_PROFILES: dict[str, dict[str, Any]] = {
    "local_cuda": {
        "profile": "local_cuda",
        "mode": "vllm-local",
        "base_url": "http://127.0.0.1:8000/v1",
        "device": "cuda",
        "model": DEFAULT_QWEN_JUDGE_MODEL,
        "max_context": 32768,
        "batch_capacity": 8,
        "generation_parameters": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
    },
    "remote_host": {
        "profile": "remote_host",
        "mode": "vllm-remote",
        "base_url": "http://judge.internal:8000/v1",
        "device": "remote_cuda",
        "model": DEFAULT_QWEN_JUDGE_MODEL,
        "max_context": 32768,
        "batch_capacity": 16,
        "generation_parameters": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
    },
    "openai_compatible": {
        "profile": "openai_compatible",
        "mode": "openai-compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "device": "provider",
        "model": DEFAULT_QWEN_JUDGE_MODEL,
        "max_context": 32768,
        "batch_capacity": 4,
        "generation_parameters": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
    },
}
QWEN_JUDGE_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen_high_quality": {
        "profile": "qwen_high_quality",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "quality_tier": "high",
        "recommended_server_profile": "remote_host",
        "max_context": 32768,
    },
    "qwen_dev": {
        "profile": "qwen_dev",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "quality_tier": "dev",
        "recommended_server_profile": "local_cuda",
        "max_context": 32768,
    },
}
QWEN_JUDGE_PROMPT_TEMPLATES: dict[str, str] = {
    "calibration": "Assess whether the answer confidence is calibrated to the evidence. Return strict JSON with verdict, scores, and rationale.",
    "confidence_faithfulness": "Judge whether the stated confidence faithfully reflects uncertainty, evidence limits, and verification results. Return strict JSON.",
    "hidden_trap_detection": "Identify hidden trap failures, overconfident mistakes, and missed escalation signals. Return strict JSON.",
    "acceptance_criteria_grading": "Grade each acceptance criterion as pass, partial, or fail, then return strict JSON with the aggregate verdict.",
}


@dataclass(frozen=True)
class RlmfDatasetArtifact:
    output: Path
    manifest_path: Path
    train_path: Path
    valid_path: Path
    test_path: Path
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "completed",
            "schema_version": RLMF_SCHEMA_VERSION,
            "output": str(self.output),
            "manifest_path": str(self.manifest_path),
            "train_path": str(self.train_path),
            "valid_path": str(self.valid_path),
            "test_path": str(self.test_path),
            "row_count": len(self.rows),
            "manifest": self.manifest,
        }


def build_rlmf_fixture_dataset(output: str | Path, *, count: int = 8) -> RlmfDatasetArtifact:
    if count < 6:
        raise ValueError("RLMF fixture count must be at least 6 so train/valid/test splits are populated")
    root = Path(output)
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = tuple(_fixture_row(index) for index in range(count))
    train_rows, valid_rows, test_rows = rows[:-4], rows[-4:-2], rows[-2:]
    train_path = dataset_dir / "train.jsonl"
    valid_path = dataset_dir / "valid.jsonl"
    test_path = dataset_dir / "test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)
    _write_jsonl(test_path, test_rows)
    manifest = {
        "schema_version": RLMF_SCHEMA_VERSION,
        "dataset_type": "rlmf.metacog.fixture",
        "row_count": len(rows),
        "splits": {
            "train": _split_manifest(train_path, train_rows),
            "valid": _split_manifest(valid_path, valid_rows),
            "test": _split_manifest(test_path, test_rows),
        },
        "privacy": {
            "mode": "fixture-public",
            "contains_private_user_data": False,
            "consent_required": False,
        },
        "fields": [
            "task_id",
            "prompt",
            "answer",
            "acceptance_criteria",
            "confidence",
            "self_judged_success_probability",
            "verification_result",
            "failure_type",
            "should_escalate",
            "privacy_status",
            "source_trace_hash",
        ],
    }
    manifest["dataset_manifest_hash"] = _hash_json(_dataset_hash_payload(manifest))
    manifest_path = root / "rlmf_dataset_manifest.json"
    _write_json(manifest_path, manifest)
    return RlmfDatasetArtifact(root, manifest_path, train_path, valid_path, test_path, rows, manifest)


def build_metacognitive_sft_rows(traces: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows = []
    for index, trace in enumerate(traces):
        prompt = str(trace.get("prompt") or trace.get("task") or trace.get("input") or "")
        answer = str(trace.get("answer") or trace.get("response") or trace.get("output") or "")
        source_id = str(trace.get("id") or trace.get("trace_id") or f"trace-{index:04d}")
        row = {
            "task_id": str(trace.get("task_id") or source_id),
            "task": prompt,
            "answer": answer,
            "reasoning_summary": str(trace.get("reasoning_summary") or trace.get("summary") or ""),
            "confidence": _unit(trace.get("confidence"), 0.5),
            "self_judged_success_probability": _unit(
                trace.get("self_judged_success_probability", trace.get("self_judged_success")),
                _unit(trace.get("confidence"), 0.5),
            ),
            "verification_result": str(trace.get("verification_result") or trace.get("label") or "unknown"),
            "failure_type": trace.get("failure_type"),
            "acceptance_criteria": extract_acceptance_criteria(trace),
            "privacy_status": str(trace.get("privacy_status") or "local_only"),
            "redaction_status": str(trace.get("redaction_status") or "unknown"),
            "consent": str(trace.get("consent") or "unknown"),
            "provenance": {
                "source": str(trace.get("source") or "trace"),
                "source_run_id": trace.get("run_id"),
                "source_trace_id": source_id,
            },
        }
        row["source_trace_hash"] = _hash_json(
            {
                "source_trace_id": source_id,
                "task": row["task"],
                "answer": row["answer"],
                "verification_result": row["verification_result"],
            }
        )
        rows.append(row)
    return tuple(rows)


def export_metacognitive_sft_dataset(
    traces: Iterable[Mapping[str, Any]],
    output: str | Path,
    *,
    dataset_type: str = "rlmf.metacog.sft",
) -> RlmfDatasetArtifact:
    rows = build_metacognitive_sft_rows(traces)
    validate_public_rlmf_rows(rows)
    root = Path(output)
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_rows, valid_rows, test_rows = _split_rows(rows)
    train_path = dataset_dir / "train.jsonl"
    valid_path = dataset_dir / "valid.jsonl"
    test_path = dataset_dir / "test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)
    _write_jsonl(test_path, test_rows)
    manifest = {
        "schema_version": RLMF_SCHEMA_VERSION,
        "dataset_type": dataset_type,
        "row_count": len(rows),
        "splits": {
            "train": _split_manifest(train_path, train_rows),
            "valid": _split_manifest(valid_path, valid_rows),
            "test": _split_manifest(test_path, test_rows),
        },
        "privacy": _privacy_manifest(rows),
        "fields": [
            "task_id",
            "task",
            "answer",
            "reasoning_summary",
            "confidence",
            "self_judged_success_probability",
            "verification_result",
            "failure_type",
            "acceptance_criteria",
            "privacy_status",
            "source_trace_hash",
        ],
    }
    manifest["dataset_manifest_hash"] = _hash_json(_dataset_hash_payload(manifest))
    manifest_path = root / "rlmf_dataset_manifest.json"
    _write_json(manifest_path, manifest)
    return RlmfDatasetArtifact(root, manifest_path, train_path, valid_path, test_path, rows, manifest)


def extract_acceptance_criteria(trace: Mapping[str, Any]) -> list[str]:
    raw = trace.get("acceptance_criteria")
    if isinstance(raw, str):
        criteria = [raw]
    elif isinstance(raw, AbcIterable) and not isinstance(raw, (str, bytes, Mapping)):
        criteria = [str(item) for item in raw]
    else:
        criteria = []
    if not criteria:
        text = "\n".join(str(trace.get(key) or "") for key in ("prompt", "task", "description"))
        match = re.search(r"(?is)acceptance criteria\s*:\s*(.+)", text)
        if match:
            criteria = [
                re.sub(r"^[-*\d.\s]+", "", line).strip()
                for line in match.group(1).splitlines()
                if line.strip()
            ]
    cleaned = [item.strip() for item in criteria if item and item.strip()]
    return cleaned or ["Satisfy the task request and pass verification."]


def export_failure_assets(rows: Iterable[Mapping[str, Any]], output: str | Path) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    assets = []
    for row in rows:
        asset_types = _failure_asset_types(row)
        for asset_type in asset_types:
            asset = {
                "asset_type": asset_type,
                "task_id": row.get("task_id"),
                "source_trace_hash": row.get("source_trace_hash") or _hash_json({"row": dict(row)}),
                "confidence": float(row.get("confidence") or 0.0),
                "verification_result": row.get("verification_result"),
                "failure_type": row.get("failure_type"),
            }
            asset["evidence_hash"] = _hash_json(asset)
            assets.append(asset)
    report = {
        "ok": True,
        "status": "completed",
        "schema_version": RLMF_SCHEMA_VERSION,
        "asset_count": len(assets),
        "asset_types": sorted({asset["asset_type"] for asset in assets}),
        "assets": assets,
    }
    report["failure_asset_report_hash"] = _hash_json({k: v for k, v in report.items() if k != "failure_asset_report_hash"})
    path = root / "failure_assets.json"
    _write_json(path, report)
    return {**report, "path": str(path)}


def generate_intrinsic_confidence_dataset(samples: Iterable[Mapping[str, Any]], output: str | Path) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        key = str(sample.get("task_id") or sample.get("cell_id") or "unknown")
        groups.setdefault(key, []).append(sample)
    rows = []
    for key, group in sorted(groups.items()):
        confidences = [_unit(sample.get("confidence"), 0.5) for sample in group]
        accepted = [str(sample.get("verification_result") or sample.get("label") or "").lower() in {"accepted", "pass", "passed", "correct"} for sample in group]
        mean = sum(confidences) / len(confidences)
        variance = sum((item - mean) ** 2 for item in confidences) / len(confidences)
        row = {
            "task_id": key,
            "sample_count": len(group),
            "mean_confidence": round(mean, 6),
            "confidence_variance": round(variance, 6),
            "acceptance_rate": round(sum(1 for item in accepted if item) / len(accepted), 6),
            "intrinsic_confidence_score": round(max(0.0, 1.0 - variance), 6),
            "sample_hashes": [
                sample.get("source_trace_hash") or _hash_json({"sample": dict(sample)})
                for sample in group
            ],
        }
        row["mds_row_hash"] = _hash_json(row)
        rows.append(row)
    path = root / "intrinsic_confidence_mds.jsonl"
    _write_jsonl(path, rows)
    manifest = {
        "ok": True,
        "status": "completed",
        "schema_version": RLMF_SCHEMA_VERSION,
        "dataset_type": "rlmf.intrinsic_confidence.mds",
        "row_count": len(rows),
        "path": str(path),
        "sha256": _file_hash(path),
    }
    manifest["dataset_manifest_hash"] = _hash_json({k: v for k, v in manifest.items() if k != "dataset_manifest_hash"})
    _write_json(root / "intrinsic_confidence_manifest.json", manifest)
    return {**manifest, "rows": rows}


def validate_public_rlmf_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        privacy = str(row.get("privacy_status") or "")
        if privacy not in PUBLIC_RLMF_PRIVACY_STATUSES:
            raise PermissionError(f"RLMF row {row.get('task_id') or '<unknown>'} is not public-exportable: {privacy}")
        if bool(row.get("contains_private_user_data")):
            raise PermissionError(f"RLMF row {row.get('task_id') or '<unknown>'} contains private user data")


def validate_rlmf_dataset_for_training(
    dataset_manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    privacy = dataset_manifest.get("privacy") if isinstance(dataset_manifest.get("privacy"), Mapping) else {}
    statuses = [str(status) for status in privacy.get("statuses", [])]
    if statuses:
        forbidden = sorted(status for status in statuses if status not in PUBLIC_RLMF_PRIVACY_STATUSES)
        if forbidden:
            raise PermissionError(f"RLMF dataset has non-exportable privacy statuses: {', '.join(forbidden)}")
    if bool(privacy.get("contains_private_user_data")):
        raise PermissionError("RLMF dataset contains private user data")
    if bool(privacy.get("consent_required")):
        raise PermissionError("RLMF dataset requires consent before training")
    if not dataset_manifest.get("dataset_manifest_hash"):
        raise ValueError("RLMF dataset manifest is missing dataset_manifest_hash")
    if rows is not None:
        validate_public_rlmf_rows(rows)
    report = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "status": "passed",
        "dataset_manifest_hash": str(dataset_manifest["dataset_manifest_hash"]),
        "privacy_mode": str(privacy.get("mode") or "unknown"),
        "statuses": statuses or [str(privacy.get("mode") or "unknown")],
        "contains_private_user_data": False,
        "consent_required": False,
    }
    report["privacy_gate_hash"] = _hash_json(report)
    return report


def trainer_profile(profile: str = "mac_mlx") -> dict[str, Any]:
    try:
        return dict(RLMF_TRAINER_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown RLMF trainer profile: {profile}") from exc


def build_mlx_lora_manifest(
    output: str | Path,
    dataset_manifest: Mapping[str, Any],
    *,
    python_bin: str = "python",
    base_model: str = DEFAULT_RLMF_BASE_MODEL,
    experts: Iterable[str] = DEFAULT_RLMF_EXPERTS,
    iters: int = 2,
    profile: str = "mac_mlx",
    execute: bool = False,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    selected_profile = trainer_profile(profile)
    privacy_gate = validate_rlmf_dataset_for_training(dataset_manifest)
    dataset_hash = str(dataset_manifest["dataset_manifest_hash"])
    expert_list = _experts(experts)
    jobs = []
    for expert in expert_list:
        data_dir = root / "dataset"
        adapter_dir = root / "adapters" / expert
        command = [
            python_bin,
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
            str(max(1, int(iters))),
            "--batch-size",
            "1",
            "--max-seq-length",
            "1024",
        ]
        jobs.append(
            {
                "expert": expert,
                "job_type": f"rlmf.{expert}.mlx_lora",
                "adapter_dir": str(adapter_dir),
                "checkpoint_path": str(adapter_dir / "adapters.safetensors"),
                "command": command,
                "mode": "execute" if execute else "dry_run",
            }
        )
    manifest = {
        "ok": True,
        "status": "ready_to_execute" if execute else "planned",
        "schema_version": RLMF_SCHEMA_VERSION,
        "trainer": "mlx-lora",
        "profile": selected_profile,
        "dry_run": not execute,
        "execute": execute,
        "base_model": base_model,
        "dataset_manifest_hash": dataset_hash,
        "privacy_gate": privacy_gate,
        "jobs": jobs,
    }
    manifest["manifest_hash"] = _hash_json(manifest)
    path = root / "mlx_lora_manifest.json"
    _write_json(path, manifest)
    return {**manifest, "manifest_path": str(path)}


def build_qlora_manifest(
    output: str | Path,
    dataset_manifest: Mapping[str, Any],
    *,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_adapter: str = "rlmf-qwen-failure-classifier",
    available_memory_gb: float = 24.0,
    sequence_length: int = 1024,
    profile: str = "single_cuda",
    execute: bool = False,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    selected_profile = trainer_profile(profile)
    privacy_gate = validate_rlmf_dataset_for_training(dataset_manifest)
    dataset_hash = str(dataset_manifest["dataset_manifest_hash"])
    memory_profile = _estimate_qlora_memory_profile(
        base_model,
        available_memory_gb=float(available_memory_gb),
        sequence_length=max(256, int(sequence_length)),
    )
    command = [
        "accelerate",
        "launch",
        *(["--multi_gpu"] if profile == "multi_gpu_accelerate" else []),
        "-m",
        "trl.scripts.sft",
        "--model_name",
        base_model,
        "--dataset_text_field",
        "prompt",
        "--load_in_4bit",
        "true",
        "--use_peft",
        "true",
        "--lora_r",
        "16",
        "--per_device_train_batch_size",
        str(memory_profile["batch_size"]),
        "--gradient_accumulation_steps",
        str(memory_profile["gradient_accumulation_steps"]),
        "--max_seq_length",
        str(memory_profile["sequence_length"]),
        "--output_dir",
        str(root / "qlora" / output_adapter),
    ]
    manifest = {
        "ok": True,
        "status": "ready_to_execute" if execute else "planned",
        "schema_version": RLMF_SCHEMA_VERSION,
        "trainer": "hf-trl-qlora",
        "profile": selected_profile,
        "dry_run": not execute,
        "execute": execute,
        "base_model": base_model,
        "framework_stack": ["accelerate", "bitsandbytes", "peft", "transformers", "trl"],
        "dataset_manifest_hash": dataset_hash,
        "privacy_gate": privacy_gate,
        "memory_profile": {
            "quantization": "4bit",
            "lora_r": 16,
            "available_memory_gb": memory_profile["available_memory_gb"],
            "estimated_model_memory_gb": memory_profile["estimated_model_memory_gb"],
            "estimated_step_memory_gb": memory_profile["estimated_step_memory_gb"],
            "batch_size": memory_profile["batch_size"],
            "gradient_accumulation_steps": memory_profile["gradient_accumulation_steps"],
            "sequence_length": memory_profile["sequence_length"],
            "target": "single_cuda_or_remote_worker",
        },
        "command": command,
        "checkpoint_path": str(root / "qlora" / output_adapter / "adapter_model.safetensors"),
    }
    manifest["manifest_hash"] = _hash_json(manifest)
    path = root / "qlora_manifest.json"
    _write_json(path, manifest)
    return {**manifest, "manifest_path": str(path)}


def vllm_judge_server_profile(profile: str = "local_cuda") -> dict[str, Any]:
    try:
        return json.loads(json.dumps(VLLM_JUDGE_SERVER_PROFILES[profile]))
    except KeyError as exc:
        raise ValueError(f"unknown vLLM judge server profile: {profile}") from exc


def build_vllm_judge_health(
    profile: Mapping[str, Any] | str = "local_cuda",
    *,
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = vllm_judge_server_profile(profile) if isinstance(profile, str) else dict(profile)
    value = dict(observed or {})
    model_loaded = bool(value.get("model_loaded", True))
    ready = bool(value.get("ready", model_loaded))
    max_context = int(value.get("max_context", selected.get("max_context", 0)))
    batch_capacity = int(value.get("batch_capacity", selected.get("batch_capacity", 0)))
    generation_parameters = dict(selected.get("generation_parameters") or {})
    generation_parameters.update(dict(value.get("generation_parameters") or {}))
    required_generation_keys = {"temperature", "top_p", "max_tokens"}
    healthy = ready and model_loaded and max_context >= 4096 and batch_capacity >= 1 and required_generation_keys <= set(generation_parameters)
    health = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "profile": selected["profile"],
        "mode": selected["mode"],
        "base_url": selected["base_url"],
        "model": str(value.get("model") or selected["model"]),
        "ready": ready,
        "model_loaded": model_loaded,
        "max_context": max_context,
        "batch_capacity": batch_capacity,
        "generation_parameters": generation_parameters,
        "status": "ready" if healthy else "unavailable",
    }
    health["server_config_hash"] = _hash_json(
        {
            "profile": selected["profile"],
            "mode": selected["mode"],
            "base_url": selected["base_url"],
            "model": health["model"],
            "generation_parameters": generation_parameters,
        }
    )
    health["health_hash"] = _hash_json(health)
    return health


def qwen_judge_model_profile(profile: str = "qwen_high_quality") -> dict[str, Any]:
    try:
        return dict(QWEN_JUDGE_MODEL_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown Qwen judge model profile: {profile}") from exc


def qwen_judge_prompt_template(kind: str) -> dict[str, Any]:
    try:
        template = QWEN_JUDGE_PROMPT_TEMPLATES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown Qwen judge prompt template: {kind}") from exc
    return {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "kind": kind,
        "template": template,
        "prompt_hash": _hash_json({"kind": kind, "template": template}),
    }


def build_judge_request(
    task: str,
    answer: str,
    *,
    acceptance_criteria: Iterable[str] = (),
    rubric: str = "confidence_faithfulness",
    model_profile: str = "qwen_high_quality",
) -> dict[str, Any]:
    profile = qwen_judge_model_profile(model_profile)
    criteria = [str(item) for item in acceptance_criteria if str(item).strip()]
    prompt = {
        "task": str(task),
        "answer": str(answer),
        "acceptance_criteria": criteria,
        "rubric": rubric,
        "instructions": qwen_judge_prompt_template(rubric if rubric in QWEN_JUDGE_PROMPT_TEMPLATES else "confidence_faithfulness")["template"],
    }
    request = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "request_type": "rlmf.judge",
        "model_profile": profile,
        "task": prompt["task"],
        "answer": prompt["answer"],
        "acceptance_criteria": criteria,
        "rubric": rubric,
        "required_scores": [
            "confidence_faithfulness",
            "acceptance_criteria_score",
            "escalation_correctness",
            "failure_classification",
        ],
    }
    request["prompt_hash"] = _hash_json(prompt)
    request["request_hash"] = _hash_json(request)
    return request


def parse_structured_qwen_judge_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    verdict = str(value.get("verdict") or "").lower()
    if verdict not in {"pass", "fail", "review"}:
        raise ValueError("Qwen judge verdict must be one of pass, fail, or review")
    scores = {
        "confidence_faithfulness": _unit(value.get("confidence_faithfulness"), 0.0),
        "acceptance_criteria_score": _unit(value.get("acceptance_criteria_score"), 0.0),
        "escalation_correctness": _unit(value.get("escalation_correctness"), 0.0),
        "failure_classification": _unit(value.get("failure_classification"), 0.0),
    }
    parsed = {
        "ok": True,
        "status": "completed",
        "schema_version": RLMF_SCHEMA_VERSION,
        "judge_model": str(value.get("judge_model") or DEFAULT_QWEN_JUDGE_MODEL),
        "provider": str(value.get("provider") or "qwen-vllm"),
        "verdict": verdict,
        "scores": scores,
        "rationale": str(value.get("rationale") or ""),
        "structured_output_valid": True,
    }
    parsed["response_hash"] = _hash_json(parsed)
    return parsed


def persist_judge_log(
    output: str | Path,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    server_health: Mapping[str, Any],
    *,
    latency_ms: float,
    token_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    root = Path(output) / "judge_logs"
    root.mkdir(parents=True, exist_ok=True)
    log = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "prompt_hash": str(request["prompt_hash"]),
        "response_hash": str(response.get("response_hash") or _hash_json(dict(response))),
        "model_id": str(response.get("judge_model") or server_health.get("model")),
        "server_config_hash": str(server_health["server_config_hash"]),
        "latency_ms": round(float(latency_ms), 3),
        "token_counts": dict(token_counts or {"prompt": 0, "completion": 0, "total": 0}),
        "verdict": str(response.get("verdict")),
    }
    log["decision_hash"] = _hash_json(log)
    path = root / f"{log['decision_hash']}.json"
    _write_json(path, log)
    return {**log, "path": str(path)}


def select_judge_server_with_fallback(
    profiles: Iterable[str | Mapping[str, Any]],
    observed_health: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    observed = observed_health or {}
    attempts = []
    for profile in profiles:
        name = profile if isinstance(profile, str) else str(profile.get("profile"))
        health = build_vllm_judge_health(profile, observed=observed.get(str(name), {}))
        attempts.append(health)
        if health["status"] == "ready":
            return {
                "ok": True,
                "schema_version": RLMF_SCHEMA_VERSION,
                "status": "ready",
                "selected_profile": health["profile"],
                "attempts": attempts,
                "fallback_used": len(attempts) > 1,
            }
    return {
        "ok": False,
        "schema_version": RLMF_SCHEMA_VERSION,
        "status": "fallback_to_fixture",
        "selected_profile": None,
        "attempts": attempts,
        "fallback_used": True,
    }


def build_judge_dashboard_panel(
    server_health: Iterable[Mapping[str, Any]],
    recent_decisions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    servers = [dict(item) for item in server_health]
    decisions = [dict(item) for item in recent_decisions]
    panel = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "panel": "rlmf_judge_servers",
        "ready_servers": sum(1 for server in servers if server.get("status") == "ready"),
        "server_count": len(servers),
        "servers": servers,
        "recent_decisions": decisions[:20],
    }
    panel["panel_hash"] = _hash_json(panel)
    return panel


def load_qwen_judge_golden_output(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    parsed = parse_structured_qwen_judge_output(payload)
    parsed["golden_fixture_path"] = str(Path(path))
    parsed["golden_fixture_hash"] = _hash_json(payload)
    return parsed


def call_qwen_judge_openai_compatible(
    request_payload: Mapping[str, Any],
    server_profile: Mapping[str, Any] | str = "openai_compatible",
    *,
    api_key: str | None = None,
    transport: Callable[[str, Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    profile = vllm_judge_server_profile(server_profile) if isinstance(server_profile, str) else dict(server_profile)
    endpoint = str(profile["base_url"]).rstrip("/") + "/chat/completions"
    body = {
        "model": str(request_payload.get("model_profile", {}).get("model") or profile["model"]),
        "messages": [
            {
                "role": "system",
                "content": "You are a strict RLMF judge. Return only JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": request_payload.get("task"),
                        "answer": request_payload.get("answer"),
                        "acceptance_criteria": request_payload.get("acceptance_criteria", []),
                        "rubric": request_payload.get("rubric"),
                    },
                    sort_keys=True,
                ),
            },
        ],
        **dict(profile.get("generation_parameters") or {}),
    }
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    raw = transport(endpoint, headers, body) if transport else _post_json(endpoint, headers, body)
    message = ((raw.get("choices") or [{}])[0].get("message") or {}) if isinstance(raw.get("choices"), list) else {}
    content = str(message.get("content") or "{}")
    parsed = parse_structured_qwen_judge_output(json.loads(content))
    return {
        **parsed,
        "adapter": "openai-compatible-chat-completions",
        "endpoint": endpoint,
        "request_hash": request_payload.get("request_hash"),
        "raw_response_hash": _hash_json(raw),
    }


def build_qwen_benchmark_receipt(
    dataset_manifest: Mapping[str, Any],
    judge_report: Mapping[str, Any],
    *,
    network_id: str = "fractalwork-mcl",
) -> dict[str, Any]:
    benchmark = build_benchmark_fixture(dataset_manifest, judge_report)
    receipt = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "receipt_type": "fractalwork.mcl.rlmf_judge",
        "network_id": network_id,
        "dataset_manifest_hash": str(dataset_manifest["dataset_manifest_hash"]),
        "judge_report_hash": str(judge_report.get("judge_report_hash") or judge_report.get("response_hash")),
        "benchmark_report_hash": benchmark["benchmark_report_hash"],
        "verdict": judge_report.get("verdict"),
        "passed": benchmark["passed"],
    }
    receipt["receipt_hash"] = _hash_json(receipt)
    return {**receipt, "benchmark": benchmark}


def build_judge_disagreement_report(
    qwen_judge: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    qwen_passed = str(qwen_judge.get("verdict") or "").lower() == "pass"
    buckets = {"verifier": [], "adjudication": [], "trap": []}
    for sample in samples:
        source = str(sample.get("source") or "")
        if source in buckets:
            buckets[source].append(sample)
    source_summaries = {}
    disagreements = []
    for source, rows in buckets.items():
        if not rows:
            source_summaries[source] = {"count": 0, "pass_rate": None}
            continue
        passed = [_sample_passed(row) for row in rows]
        pass_rate = sum(1 for item in passed if item) / len(passed)
        source_summaries[source] = {"count": len(rows), "pass_rate": round(pass_rate, 6)}
        source_passed = pass_rate >= 0.5
        if source_passed != qwen_passed:
            disagreements.append(
                {
                    "source": source,
                    "qwen_verdict": qwen_judge.get("verdict"),
                    "source_pass_rate": round(pass_rate, 6),
                    "sample_hashes": [str(row.get("source_trace_hash") or _hash_json(dict(row))) for row in rows],
                }
            )
    report = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "qwen_verdict": qwen_judge.get("verdict"),
        "source_summaries": source_summaries,
        "disagreements": disagreements,
        "requires_adjudication": bool(disagreements),
    }
    report["disagreement_hash"] = _hash_json(report)
    return report


def build_rlmf_reward_signal(
    sample: Mapping[str, Any],
    judge_report: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    selected_weights = {
        "correctness": 0.35,
        "calibration": 0.2,
        "confidence_faithfulness": 0.2,
        "overconfidence_penalty": 0.15,
        "escalation_recall": 0.1,
    }
    selected_weights.update(dict(weights or {}))
    scores = dict(judge_report.get("scores") or {})
    confidence = _unit(sample.get("confidence"), 0.5)
    passed = _sample_passed(sample)
    should_escalate = bool(sample.get("should_escalate"))
    escalated = bool(sample.get("escalated") or str(sample.get("answer") or "").lower().startswith("escalate"))
    components = {
        "correctness": 1.0 if passed else 0.0,
        "calibration": max(0.0, 1.0 - abs(confidence - (1.0 if passed else 0.0))),
        "confidence_faithfulness": _unit(scores.get("confidence_faithfulness"), 0.5),
        "overconfidence_penalty": max(0.0, confidence - 0.7) if not passed else 0.0,
        "escalation_recall": 1.0 if not should_escalate or escalated else 0.0,
    }
    reward = (
        components["correctness"] * selected_weights["correctness"]
        + components["calibration"] * selected_weights["calibration"]
        + components["confidence_faithfulness"] * selected_weights["confidence_faithfulness"]
        - components["overconfidence_penalty"] * selected_weights["overconfidence_penalty"]
        + components["escalation_recall"] * selected_weights["escalation_recall"]
    )
    signal = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "reward": round(max(-1.0, min(1.0, reward)), 6),
        "components": components,
        "weights": selected_weights,
        "sample_hash": str(sample.get("source_trace_hash") or _hash_json(dict(sample))),
        "judge_hash": str(judge_report.get("judge_report_hash") or judge_report.get("response_hash") or _hash_json(dict(judge_report))),
    }
    signal["reward_hash"] = _hash_json(signal)
    return signal


def generate_failure_classifier_benchmark(
    dataset_manifest: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
    *,
    benchmark_id: str = "bench.calib.failure_classifier.v1",
) -> dict[str, Any]:
    rows = [dict(sample) for sample in samples]
    benchmark = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "dataset_manifest_hash": str(dataset_manifest["dataset_manifest_hash"]),
        "row_count": len(rows),
        "sample_hashes": [str(row.get("source_trace_hash") or _hash_json(row)) for row in rows],
        "tasks": [
            {
                "task_id": str(row.get("task_id") or f"bench-row-{index:04d}"),
                "expected_failure_type": row.get("failure_type"),
                "expected_escalation": bool(row.get("should_escalate")),
                "label": "fail" if not _sample_passed(row) else "pass",
                "confidence": _unit(row.get("confidence"), 0.5),
            }
            for index, row in enumerate(rows)
        ],
    }
    benchmark["benchmark_hash"] = _hash_json(benchmark)
    return benchmark


def build_calibration_benchmark_report(
    benchmark: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(sample) for sample in samples]
    trap_rows = [row for row in rows if row.get("source") == "trap"]
    verifier_rows = [row for row in rows if row.get("source") in {"verifier", "adjudication"}]
    metrics = {
        "ece": _expected_calibration_error(rows, bins=10),
        "adaptive_ece": _adaptive_calibration_error(rows),
        "high_confidence_failure_rate": _high_confidence_failure_rate(rows),
        "trap_verifier_gap": abs(_adaptive_calibration_error(trap_rows) - _adaptive_calibration_error(verifier_rows)),
        "escalation_precision": _escalation_precision(rows),
        "escalation_recall": _escalation_recall(rows),
        "confidence_faithfulness": round(sum(float(row.get("confidence_faithfulness", 0.8)) for row in rows) / len(rows), 6) if rows else 0.0,
    }
    report = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "benchmark_id": str(benchmark.get("benchmark_id") or "bench.calib.failure_classifier.v1"),
        "benchmark_hash": benchmark.get("benchmark_hash"),
        "dataset_manifest_hash": benchmark.get("dataset_manifest_hash"),
        "metrics": metrics,
        "blind_spot_leads": _blind_spot_leads(trap_rows),
        "passed": metrics["adaptive_ece"] <= 0.25 and metrics["high_confidence_failure_rate"] <= 0.2 and metrics["escalation_recall"] >= 0.7,
    }
    report["benchmark_report_hash"] = _hash_json(report)
    return report


def build_mds_selection_certificate(
    training_pool: Iterable[Mapping[str, Any]],
    *,
    selection_id: str = "rlmf-mds-selection",
    min_intrinsic_confidence_score: float = 0.5,
) -> dict[str, Any]:
    rows = [dict(row) for row in training_pool]
    selected = [row for row in rows if float(row.get("intrinsic_confidence_score", 0.0)) >= min_intrinsic_confidence_score]
    certificate = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "selection_id": selection_id,
        "pool_size": len(rows),
        "selected_count": len(selected),
        "min_intrinsic_confidence_score": min_intrinsic_confidence_score,
        "selected_hashes": [str(row.get("mds_row_hash") or row.get("source_trace_hash") or _hash_json(row)) for row in selected],
        "rejected_hashes": [str(row.get("mds_row_hash") or row.get("source_trace_hash") or _hash_json(row)) for row in rows if row not in selected],
    }
    certificate["selection_certificate_hash"] = _hash_json(certificate)
    return certificate


def build_model_comparison_report(
    baseline_report: Mapping[str, Any],
    sft_report: Mapping[str, Any],
    rlmf_report: Mapping[str, Any],
) -> dict[str, Any]:
    reports = {"baseline": dict(baseline_report), "sft": dict(sft_report), "rlmf": dict(rlmf_report)}
    adaptive_ece_delta = float(reports["baseline"].get("metrics", {}).get("adaptive_ece", 1.0)) - float(reports["rlmf"].get("metrics", {}).get("adaptive_ece", 1.0))
    faithfulness_delta = float(reports["rlmf"].get("metrics", {}).get("confidence_faithfulness", 0.0)) - float(reports["baseline"].get("metrics", {}).get("confidence_faithfulness", 0.0))
    report = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "comparison_id": "rlmf.model_comparison.v1",
        "reports": reports,
        "deltas": {
            "adaptive_ece_improvement": round(adaptive_ece_delta, 6),
            "confidence_faithfulness_improvement": round(faithfulness_delta, 6),
        },
        "winner": "rlmf" if adaptive_ece_delta >= 0 and faithfulness_delta >= 0 else "review",
    }
    report["model_comparison_hash"] = _hash_json(report)
    return report


def enforce_g16_promotion_gate(benchmark_report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(benchmark_report.get("metrics") or {})
    required = {
        "adaptive_ece": float(metrics.get("adaptive_ece", 1.0)) <= 0.25,
        "high_confidence_failure_rate": float(metrics.get("high_confidence_failure_rate", 1.0)) <= 0.2,
        "confidence_faithfulness": float(metrics.get("confidence_faithfulness", 0.0)) >= 0.75,
        "escalation_recall": float(metrics.get("escalation_recall", 0.0)) >= 0.7,
        "regression_passed": bool(benchmark_report.get("regression_passed", True)),
        "benchmark_passed": bool(benchmark_report.get("passed")),
    }
    reasons = [name for name, passed in required.items() if not passed]
    gate = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "promotion_gate": "G16",
        "source": "real_benchmark_report",
        "benchmark_report_hash": benchmark_report.get("benchmark_report_hash"),
        "checks": required,
        "promoted": not reasons,
        "status": "promoted" if not reasons else "blocked",
        "reasons": reasons,
        "rollback_metadata": None if not reasons else {"required": True, "blocked_checks": reasons},
    }
    gate["promotion_gate_hash"] = _hash_json(gate)
    return gate


def emit_model_promotion_record(
    model_artifact: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    previous_model_hash: str | None = None,
) -> dict[str, Any]:
    promoted = bool(gate.get("promoted"))
    record = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "record_type": "rlmf.model_promotion",
        "status": "promoted" if promoted else "rejected",
        "model_artifact_hash": str(model_artifact.get("model_artifact_hash") or model_artifact.get("manifest_hash") or _hash_json(dict(model_artifact))),
        "benchmark_report_hash": gate.get("benchmark_report_hash"),
        "promotion_gate_hash": gate.get("promotion_gate_hash"),
        "rollback_metadata": {
            "previous_model_hash": previous_model_hash,
            "rollback_required": not promoted,
            "reason": ",".join(str(reason) for reason in gate.get("reasons", [])),
        },
    }
    record["promotion_record_hash"] = _hash_json(record)
    return record


def parse_qwen_judge_fixture(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = dict(payload or {})
    scores = {
        "correctness": _unit(value.get("correctness"), 0.82),
        "calibration": _unit(value.get("calibration"), 0.76),
        "confidence_faithfulness": _unit(value.get("confidence_faithfulness"), 0.8),
        "escalation_recall": _unit(value.get("escalation_recall"), 0.75),
    }
    report = {
        "ok": True,
        "status": "completed",
        "schema_version": RLMF_SCHEMA_VERSION,
        "judge_model": str(value.get("judge_model") or DEFAULT_QWEN_JUDGE_MODEL),
        "provider": str(value.get("provider") or "qwen-vllm"),
        "verdict": str(value.get("verdict") or "pass"),
        "scores": scores,
        "structured_output_valid": True,
    }
    report["judge_report_hash"] = _hash_json(report)
    return report


def build_benchmark_fixture(dataset_manifest: Mapping[str, Any], judge_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    judge = dict(judge_report or parse_qwen_judge_fixture())
    scores = dict(judge.get("scores") or {})
    metrics = {
        "adaptive_ece": round(1.0 - float(scores.get("calibration", 0.75)), 4),
        "high_confidence_failure_rate": 0.125,
        "escalation_recall": float(scores.get("escalation_recall", 0.75)),
        "confidence_faithfulness": float(scores.get("confidence_faithfulness", 0.8)),
    }
    report = {
        "ok": True,
        "status": "completed",
        "schema_version": RLMF_SCHEMA_VERSION,
        "benchmark_id": "bench.calib.failure_classifier.v1.fixture",
        "dataset_manifest_hash": str(dataset_manifest["dataset_manifest_hash"]),
        "judge_report_hash": judge.get("judge_report_hash"),
        "metrics": metrics,
        "passed": metrics["adaptive_ece"] <= 0.25 and metrics["confidence_faithfulness"] >= 0.75,
    }
    report["benchmark_report_hash"] = _hash_json(report)
    return report


def build_promotion_fixture(benchmark_report: Mapping[str, Any]) -> dict[str, Any]:
    passed = bool(benchmark_report.get("passed"))
    decision = {
        "ok": True,
        "status": "promoted" if passed else "blocked",
        "schema_version": RLMF_SCHEMA_VERSION,
        "promotion_gate": "G16",
        "benchmark_report_hash": benchmark_report.get("benchmark_report_hash"),
        "promoted": passed,
        "rollback_required": not passed,
    }
    decision["promotion_hash"] = _hash_json(decision)
    return decision


def build_rlmf_job_state(
    job_id: str,
    *,
    status: str = "planned",
    process_id: int | None = None,
    checkpoint_path: str | None = None,
    last_metric: Mapping[str, Any] | None = None,
    logs: Iterable[str] = (),
    failure_reason: str | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    state = {
        "ok": status not in {"failed", "error"},
        "schema_version": RLMF_SCHEMA_VERSION,
        "job_id": job_id,
        "status": status,
        "process_id": process_id,
        "checkpoint_path": checkpoint_path,
        "last_metric": dict(last_metric or {}),
        "logs": [str(line) for line in logs],
        "failure_reason": failure_reason,
        "artifact_hashes": dict(artifact_hashes or {}),
    }
    state["job_state_hash"] = _hash_json(state)
    return state


def build_training_command_record(
    manifest: Mapping[str, Any],
    *,
    execute: bool = False,
    job_id: str | None = None,
) -> dict[str, Any]:
    jobs = manifest.get("jobs") if isinstance(manifest.get("jobs"), list) else []
    first_job = jobs[0] if jobs and isinstance(jobs[0], Mapping) else {}
    command = list(manifest.get("command") or first_job.get("command") or [])
    record = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "job_id": job_id or f"rlmf_job_{str(manifest.get('manifest_hash') or _hash_json(dict(manifest)))[:16]}",
        "mode": "execute" if execute else "dry_run",
        "execute": execute,
        "trainer": manifest.get("trainer"),
        "dataset_manifest_hash": manifest.get("dataset_manifest_hash"),
        "command": command,
        "would_execute": not execute,
    }
    record["command_hash"] = _hash_json(record)
    return record


def enforce_rlmf_promotion_gate(benchmark_report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(benchmark_report.get("metrics") or {})
    adaptive_ece = float(metrics.get("adaptive_ece", 1.0))
    confidence_faithfulness = float(metrics.get("confidence_faithfulness", 0.0))
    high_confidence_failure_rate = float(metrics.get("high_confidence_failure_rate", 1.0))
    regression_passed = bool(benchmark_report.get("regression_passed", True))
    passed = bool(benchmark_report.get("passed")) and adaptive_ece <= 0.25 and confidence_faithfulness >= 0.75 and high_confidence_failure_rate <= 0.2 and regression_passed
    reasons = []
    if not benchmark_report.get("passed"):
        reasons.append("benchmark_failed")
    if adaptive_ece > 0.25:
        reasons.append("adaptive_ece_regression")
    if confidence_faithfulness < 0.75:
        reasons.append("confidence_faithfulness_regression")
    if high_confidence_failure_rate > 0.2:
        reasons.append("high_confidence_failure_rate_regression")
    if not regression_passed:
        reasons.append("regression_suite_failed")
    gate = {
        "ok": True,
        "schema_version": RLMF_SCHEMA_VERSION,
        "promotion_gate": "G16",
        "status": "promoted" if passed else "blocked",
        "promoted": passed,
        "benchmark_report_hash": benchmark_report.get("benchmark_report_hash"),
        "reasons": reasons,
        "rollback_metadata": None if passed else {"required": True, "reason": ",".join(reasons) or "promotion gate failed"},
    }
    gate["promotion_gate_hash"] = _hash_json(gate)
    return gate


def run_rlmf_training_fixture(
    output: str | Path,
    *,
    count: int = 8,
    execute: bool = False,
    benchmark_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = prepare_rlmf_fixture(output, count=count)
    mlx_command = build_training_command_record(prepared["mlx_lora"], execute=execute)
    qlora_command = build_training_command_record(prepared["qlora"], execute=execute)
    benchmark = dict(benchmark_report or prepared["benchmark"])
    gate = enforce_rlmf_promotion_gate(benchmark)
    state = build_rlmf_job_state(
        mlx_command["job_id"],
        status="completed" if execute else "planned",
        checkpoint_path=prepared["qlora"]["checkpoint_path"],
        last_metric={"loss": 0.0, "adaptive_ece": benchmark["metrics"]["adaptive_ece"]},
        logs=["deterministic fixture mode", f"execute={execute}"],
        artifact_hashes={
            "dataset_manifest_hash": prepared["dataset"]["manifest"]["dataset_manifest_hash"],
            "mlx_lora_manifest_hash": prepared["mlx_lora"]["manifest_hash"],
            "qlora_manifest_hash": prepared["qlora"]["manifest_hash"],
            "benchmark_report_hash": benchmark["benchmark_report_hash"],
        },
    )
    result = {
        "ok": True,
        "status": "completed" if execute else "planned",
        "schema_version": RLMF_SCHEMA_VERSION,
        "fixture_mode": True,
        "execute": execute,
        "prepared": prepared,
        "commands": {"mlx_lora": mlx_command, "qlora": qlora_command},
        "job_state": state,
        "promotion_gate": gate,
    }
    result["training_run_hash"] = _hash_json(
        {
            "job_state_hash": state["job_state_hash"],
            "promotion_gate_hash": gate["promotion_gate_hash"],
            "dataset_manifest_hash": state["artifact_hashes"]["dataset_manifest_hash"],
        }
    )
    return result


def prepare_rlmf_fixture(output: str | Path, *, count: int = 8) -> dict[str, Any]:
    dataset = build_rlmf_fixture_dataset(output, count=count)
    mlx = build_mlx_lora_manifest(output, dataset.manifest)
    qlora = build_qlora_manifest(output, dataset.manifest)
    judge = parse_qwen_judge_fixture()
    benchmark = build_benchmark_fixture(dataset.manifest, judge)
    promotion = build_promotion_fixture(benchmark)
    return {
        "ok": True,
        "status": "completed",
        "schema_version": RLMF_SCHEMA_VERSION,
        "output": str(Path(output)),
        "dataset": dataset.to_dict(),
        "mlx_lora": mlx,
        "qlora": qlora,
        "judge": judge,
        "benchmark": benchmark,
        "promotion": promotion,
    }


def _fixture_row(index: int) -> dict[str, Any]:
    failed = index % 4 == 0
    should_escalate = failed or index % 5 == 0
    prompt = f"Classify confidence and escalation for fixture task {index}."
    answer = "Escalate to verifier" if should_escalate else "Proceed with bounded confidence"
    row = {
        "task_id": f"rlmf-fixture-{index:03d}",
        "prompt": prompt,
        "answer": answer,
        "acceptance_criteria": [
            "Return calibrated confidence",
            "Identify whether escalation is required",
            "Name the failure type when verification fails",
        ],
        "confidence": 0.92 if failed else round(0.62 + ((index % 3) * 0.08), 2),
        "self_judged_success_probability": 0.9 if failed else 0.72,
        "verification_result": "failed" if failed else "accepted",
        "failure_type": "overconfident_failure" if failed else None,
        "should_escalate": should_escalate,
        "privacy_status": "fixture-public",
    }
    row["source_trace_hash"] = _hash_json({"task_id": row["task_id"], "prompt": prompt, "answer": answer})
    return row


def _split_rows(rows: tuple[dict[str, Any], ...]) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if len(rows) < 3:
        raise ValueError("RLMF datasets require at least 3 rows so train/valid/test splits are populated")
    valid_count = max(1, min(2, len(rows) // 5))
    test_count = max(1, min(2, len(rows) // 5))
    train_count = len(rows) - valid_count - test_count
    if train_count < 1:
        raise ValueError("RLMF datasets require at least one train row")
    return rows[:train_count], rows[train_count : train_count + valid_count], rows[train_count + valid_count :]


def _split_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = tuple(rows)
    return {
        "path": str(path),
        "rows": len(materialized),
        "sha256": _file_hash(path),
        "source_trace_hashes": [str(row.get("source_trace_hash")) for row in materialized if row.get("source_trace_hash")],
        "redaction_statuses": sorted({str(row.get("redaction_status") or "not_required") for row in materialized}),
        "consent": sorted({str(row.get("consent") or "not_required") for row in materialized}),
        "provenance": {
            "source_count": len({str(row.get("source_trace_hash")) for row in materialized if row.get("source_trace_hash")}),
            "contains_private_user_data": any(bool(row.get("contains_private_user_data")) for row in materialized),
        },
    }


def _privacy_manifest(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = tuple(rows)
    statuses = sorted({str(row.get("privacy_status") or "unknown") for row in materialized})
    return {
        "mode": "public-rlmf",
        "statuses": statuses,
        "contains_private_user_data": any(bool(row.get("contains_private_user_data")) for row in materialized),
        "consent_required": any(str(row.get("consent") or "unknown") not in {"granted", "fixture", "not_required"} for row in materialized),
        "redaction_statuses": sorted({str(row.get("redaction_status") or "unknown") for row in materialized}),
    }


def _failure_asset_types(row: Mapping[str, Any]) -> list[str]:
    failure_type = str(row.get("failure_type") or "")
    verification = str(row.get("verification_result") or "").lower()
    confidence = float(row.get("confidence") or 0.0)
    should_escalate = bool(row.get("should_escalate"))
    escalated = bool(row.get("escalated") or str(row.get("answer") or "").lower().startswith("escalate"))
    types = []
    if failure_type == "overconfident_failure" or (verification in {"failed", "fail", "rejected", "incorrect"} and confidence >= 0.8):
        types.append("overconfident_failure")
    if failure_type == "unfaithful_hedge" or bool(row.get("unfaithful_hedge")):
        types.append("unfaithful_hedge")
    if failure_type == "hidden_trap_failure" or bool(row.get("hidden_trap_failure")):
        types.append("hidden_trap_failure")
    if failure_type == "missed_escalation" or (should_escalate and not escalated):
        types.append("missed_escalation")
    return sorted(set(types))


def _sample_passed(sample: Mapping[str, Any]) -> bool:
    value = str(sample.get("verification_result") or sample.get("label") or sample.get("verdict") or "").lower()
    return value in {"accepted", "pass", "passed", "correct", "success"}


def _expected_calibration_error(samples: Iterable[Mapping[str, Any]], *, bins: int) -> float:
    rows = list(samples)
    if not rows:
        return 0.0
    total = len(rows)
    ece = 0.0
    for bucket_index in range(bins):
        low = bucket_index / bins
        high = (bucket_index + 1) / bins
        bucket = []
        for row in rows:
            confidence = _unit(row.get("confidence"), 0.5)
            if confidence >= low and (confidence < high or bucket_index == bins - 1):
                bucket.append(row)
        if not bucket:
            continue
        avg_confidence = sum(_unit(row.get("confidence"), 0.5) for row in bucket) / len(bucket)
        accuracy = sum(1 for row in bucket if _sample_passed(row)) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_confidence - accuracy)
    return round(ece, 6)


def _adaptive_calibration_error(samples: Iterable[Mapping[str, Any]]) -> float:
    rows = sorted(list(samples), key=lambda row: _unit(row.get("confidence"), 0.5))
    if not rows:
        return 0.0
    bucket_count = min(4, len(rows))
    bucket_size = math.ceil(len(rows) / bucket_count)
    buckets = [rows[index : index + bucket_size] for index in range(0, len(rows), bucket_size)]
    total = len(rows)
    error = 0.0
    for bucket in buckets:
        avg_confidence = sum(_unit(row.get("confidence"), 0.5) for row in bucket) / len(bucket)
        accuracy = sum(1 for row in bucket if _sample_passed(row)) / len(bucket)
        error += (len(bucket) / total) * abs(avg_confidence - accuracy)
    return round(error, 6)


def _high_confidence_failure_rate(samples: Iterable[Mapping[str, Any]]) -> float:
    rows = [row for row in samples if _unit(row.get("confidence"), 0.5) >= 0.8]
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if not _sample_passed(row)) / len(rows), 6)


def _escalation_precision(samples: Iterable[Mapping[str, Any]]) -> float:
    escalated = [row for row in samples if bool(row.get("escalated") or str(row.get("answer") or "").lower().startswith("escalate"))]
    if not escalated:
        return 1.0
    return round(sum(1 for row in escalated if bool(row.get("should_escalate"))) / len(escalated), 6)


def _escalation_recall(samples: Iterable[Mapping[str, Any]]) -> float:
    required = [row for row in samples if bool(row.get("should_escalate"))]
    if not required:
        return 1.0
    return round(sum(1 for row in required if bool(row.get("escalated") or str(row.get("answer") or "").lower().startswith("escalate"))) / len(required), 6)


def _blind_spot_leads(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in samples:
        groups.setdefault(str(row.get("cell_id") or row.get("cellId") or "unknown"), []).append(row)
    leads = []
    for cell_id, rows in groups.items():
        failures = [row for row in rows if not _sample_passed(row)]
        if not failures:
            continue
        leads.append(
            {
                "cell_id": cell_id,
                "trap_failure_rate": round(len(failures) / len(rows), 6),
                "evidence_hashes": [str(row.get("source_trace_hash") or _hash_json(dict(row))) for row in failures],
            }
        )
    return sorted(leads, key=lambda row: (-row["trap_failure_rate"], row["cell_id"]))


def _post_json(endpoint: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _hash_json(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _dataset_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    splits = manifest.get("splits") if isinstance(manifest.get("splits"), Mapping) else {}
    return {
        "schema_version": manifest.get("schema_version"),
        "dataset_type": manifest.get("dataset_type"),
        "row_count": manifest.get("row_count"),
        "privacy": manifest.get("privacy"),
        "fields": manifest.get("fields"),
        "splits": {
            name: {
                "rows": value.get("rows"),
                "sha256": value.get("sha256"),
                "source_trace_hashes": value.get("source_trace_hashes"),
                "redaction_statuses": value.get("redaction_statuses"),
                "consent": value.get("consent"),
                "provenance": value.get("provenance"),
            }
            for name, value in sorted(splits.items())
            if isinstance(value, Mapping)
        },
    }


def _unit(value: Any, fallback: float) -> float:
    parsed = fallback if value is None else float(value)
    if parsed < 0 or parsed > 1:
        raise ValueError("judge fixture scores must be between 0 and 1")
    return parsed


def _estimate_qlora_memory_profile(
    base_model: str,
    *,
    available_memory_gb: float,
    sequence_length: int,
) -> dict[str, Any]:
    model_b = _model_size_in_billions(base_model)
    estimated_model_memory_gb = round((model_b * 0.48) + 1.8, 2)
    estimated_step_memory_gb = round(estimated_model_memory_gb + (sequence_length / 1024.0) * 2.2, 2)
    if estimated_step_memory_gb <= available_memory_gb * 0.35:
        batch_size = 4
    elif estimated_step_memory_gb <= available_memory_gb * 0.65:
        batch_size = 2
    else:
        batch_size = 1
    return {
        "available_memory_gb": round(float(available_memory_gb), 2),
        "estimated_model_memory_gb": estimated_model_memory_gb,
        "estimated_step_memory_gb": estimated_step_memory_gb,
        "batch_size": batch_size,
        "gradient_accumulation_steps": max(1, math.ceil(4 / batch_size)),
        "sequence_length": int(sequence_length),
    }


def _model_size_in_billions(base_model: str) -> float:
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*b\b", base_model)
    if match:
        return float(match.group(1))
    if "1.5B" in base_model:
        return 1.5
    if "0.5B" in base_model or "500m" in base_model.lower():
        return 0.5
    return 7.0


def _experts(experts: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(experts, str):
        return (experts,)
    return tuple(str(expert) for expert in experts if str(expert).strip())
