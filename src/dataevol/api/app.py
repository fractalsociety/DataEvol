from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from dataevol import __version__
from dataevol.compat import call_core
from dataevol.api.auth import require_token
from dataevol.api.training_job_store import initialize_training_jobs, jobs_for_db, persist_training_job
from dataevol.config import DataEvolConfig, load_config
from dataevol.local_models import EXPERTS, prepare_local_adapter_training
from dataevol.local_models.layer_specialist import SCHEMA as LAYER_SPECIALIST_SCHEMA
from dataevol.local_models.layer_specialist import build_manifest as build_layer_specialist_manifest
from dataevol.local_models.layer_specialist import model_fingerprint
from dataevol.local_models.layer_specialist import validate_initial_specialist_manifest
from dataevol.local_models.remote_dataset import materialize_layer_dataset
from dataevol.rlmf import (
    build_benchmark_fixture,
    build_mlx_lora_manifest,
    build_promotion_fixture,
    build_qlora_manifest,
    build_rlmf_fixture_dataset,
    parse_qwen_judge_fixture,
    prepare_rlmf_fixture,
)


DEFAULT_TRAINABLE_LAYER_MODEL = "mlx-community/Qwen3-0.6B-4bit"
ORNITH_9B_MODEL_PATH = ".dataevol/models/Ornith-1.0-9B-8bit"
DEFAULT_DASHBOARD_OUTPUT = ".dataevol/qwen3_0_6b_experts"
TRAINING_JOBS: dict[str, dict[str, Any]] = {}
TRAINING_JOBS_LOCK = threading.Lock()
TRAINING_PROCESSES: dict[str, subprocess.Popen[str]] = {}
TRAINING_CANCEL_EVENTS: dict[str, threading.Event] = {}
ITERATION_RE = re.compile(r"\bIter\s+(\d+)\s*:")
MAX_ADAPTER_EXPORT_FILE_BYTES = 25 * 1024 * 1024
ADAPTER_EXPORT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapters.safetensors",
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
SPECIALIST_MANIFEST_RE = re.compile(r".+\.manifest\.json$")


class OperationRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestTraceRequest(BaseModel):
    trace: dict[str, Any]
    source_system: str | None = None


class IngestRunRequest(BaseModel):
    run: dict[str, Any] = Field(default_factory=dict)
    source_system: str | None = None


class IngestWorkerReportRequest(BaseModel):
    report: dict[str, Any] | None = None
    reports: list[dict[str, Any]] | None = None
    source_system: str | None = None
    external_run_id: str | None = None
    objective: str | None = None


def _dashboard_training_experts(payload: dict[str, Any]) -> tuple[str, ...]:
    requested = payload.get("experts")
    if requested is None:
        expert = payload.get("expert")
        requested = [expert] if expert else list(EXPERTS)
    experts = tuple(str(expert) for expert in requested if expert)
    return experts or tuple(EXPERTS)


def _training_job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    started_at = job.get("started_at") or job.get("created_at") or now
    completed_at = job.get("completed_at")
    end_time = completed_at or now
    elapsed = max(0.0, end_time - started_at)
    progress = max(0.0, min(1.0, float(job.get("progress") or 0.0)))
    eta = None
    if progress > 0 and progress < 1 and not completed_at:
        eta = max(0.0, (elapsed / progress) - elapsed)
    snapshot = {
        key: value
        for key, value in job.items()
        if key != "logs" and not key.startswith("_")
    }
    snapshot["logs"] = list(job.get("logs") or [])
    snapshot["elapsed_seconds"] = elapsed
    snapshot["gpu_seconds"] = round(elapsed * _gpu_device_factor(job), 6)
    snapshot["eta_seconds"] = eta
    snapshot["percent"] = round(progress * 100, 1)
    return snapshot


def _gpu_device_factor(job: dict[str, Any]) -> float:
    raw = job.get("gpu_device_factor") or os.environ.get("DATAEVOL_GPU_DEVICE_FACTOR") or 1.0
    try:
        factor = float(raw)
    except Exception:
        return 1.0
    if factor < 0 or factor == float("inf") or factor == float("-inf"):
        return 1.0
    return factor


def _model_available(model: str) -> bool:
    if Path(model).expanduser().exists():
        return True
    return "/" in model and not model.startswith((".", "/", "~"))


def _export_local_model_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    output = Path(payload.get("output") or DEFAULT_DASHBOARD_OUTPUT)
    requested = payload.get("experts")
    experts = tuple(str(expert) for expert in requested if expert) if isinstance(requested, list) else tuple(EXPERTS)
    exported_files: list[dict[str, Any]] = []
    adapters: dict[str, Any] = {}
    manifest_path = output / "adapter_training_manifest.json"

    for expert in experts:
        if expert not in EXPERTS:
            continue
        adapter_dir = output / "adapters" / expert
        expert_files: list[dict[str, Any]] = []
        for path in sorted(adapter_dir.glob("*")) if adapter_dir.exists() else []:
            if not path.is_file() or path.name not in ADAPTER_EXPORT_FILES:
                continue
            size = path.stat().st_size
            if size > MAX_ADAPTER_EXPORT_FILE_BYTES:
                expert_files.append({
                    "relative_path": f"adapters/{expert}/{path.name}",
                    "path": str(path),
                    "size_bytes": size,
                    "skipped": True,
                    "reason": "file_too_large",
                })
                continue
            data = path.read_bytes()
            record = {
                "relative_path": f"adapters/{expert}/{path.name}",
                "path": str(path),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "content_base64": base64.b64encode(data).decode("ascii"),
            }
            expert_files.append(record)
            exported_files.append(record)
        adapters[expert] = {
            "path": str(adapter_dir),
            "exists": (adapter_dir / "adapters.safetensors").exists() or (adapter_dir / "adapter_model.safetensors").exists(),
            "files": expert_files,
        }

    manifest = None
    if manifest_path.exists() and manifest_path.stat().st_size <= MAX_ADAPTER_EXPORT_FILE_BYTES:
        data = manifest_path.read_bytes()
        manifest = {
            "relative_path": "adapter_training_manifest.json",
            "path": str(manifest_path),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "content_base64": base64.b64encode(data).decode("ascii"),
        }

    specialists, specialist_count, specialist_hashes = _export_layer_specialists(output)
    payload_hash = hashlib.sha256(
        "|".join(sorted([file["sha256"] for file in exported_files if "sha256" in file] + specialist_hashes)).encode("utf-8")
    ).hexdigest()
    return {
        "ok": True,
        "schema": "dataevol.local_model_artifact_export.v1",
        "output": str(output),
        "experts": list(experts),
        "manifest": manifest,
        "adapters": adapters,
        "specialists": specialists,
        "file_count": len(exported_files) + specialist_count + (1 if manifest else 0),
        "payload_hash": payload_hash,
    }


def _export_layer_specialists(output: Path) -> tuple[dict[str, Any], int, list[str]]:
    root = output / "layerscope"
    specialists: dict[str, Any] = {}
    count = 0
    hashes: list[str] = []
    if not root.exists():
        return specialists, count, hashes
    for layer_dir in sorted(path for path in root.glob("layer_*") if path.is_dir()):
        manifest_records: list[dict[str, Any]] = []
        tensor_records: list[dict[str, Any]] = []
        for path in sorted(layer_dir.glob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output).as_posix()
            if path.suffix == ".json" and SPECIALIST_MANIFEST_RE.match(path.name):
                data = path.read_bytes()
                manifest_record = {
                    "relative_path": relative,
                    "path": str(path),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "content_base64": base64.b64encode(data).decode("ascii"),
                }
                hashes.append(manifest_record["sha256"])
                manifest_records.append(manifest_record)
                count += 1
            elif path.suffix == ".safetensors":
                size = path.stat().st_size
                if size > MAX_ADAPTER_EXPORT_FILE_BYTES:
                    record = {
                        "relative_path": relative,
                        "path": str(path),
                        "size_bytes": size,
                        "sha256": _sha256_path(path),
                        "content_omitted": True,
                        "reason": "use_artifact_file_endpoint",
                    }
                    tensor_records.append(record)
                    hashes.append(record["sha256"])
                    count += 1
                    continue
                data = path.read_bytes()
                record = {
                    "relative_path": relative,
                    "path": str(path),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "content_base64": base64.b64encode(data).decode("ascii"),
                }
                tensor_records.append(record)
                hashes.append(record["sha256"])
                count += 1
        specialists[layer_dir.name] = {
            "path": str(layer_dir),
            "exists": bool(manifest_records) or bool(tensor_records),
            "manifest": manifest_records[0] if manifest_records else None,
            "manifests": manifest_records,
            "tensors": tensor_records,
        }
    return specialists, count, hashes


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_train_layer_specialist(payload: dict[str, Any], cfg: DataEvolConfig) -> dict[str, Any]:
    layer_index = _non_negative_int(payload.get("layer_index"), "layer_index")
    training_mode = str(payload.get("training_mode") or "").strip()
    if training_mode not in {"sft", "rl"}:
        raise HTTPException(status_code=422, detail="training_mode must be sft or rl")
    rl_algorithm = _optional_string(payload.get("rl_algorithm"))
    if training_mode == "rl" and rl_algorithm != "dpo":
        raise HTTPException(status_code=422, detail="training_mode=rl requires supported rl_algorithm=dpo")
    if training_mode == "sft" and rl_algorithm is not None:
        raise HTTPException(status_code=422, detail="rl_algorithm is only valid when training_mode=rl")
    base_model = _required_string(payload.get("base_model") or payload.get("model"), "base_model")
    base_model_revision = _optional_string(payload.get("base_model_revision"))
    output = _required_string(payload.get("output"), "output")
    task_type = _required_string(payload.get("task_type"), "task_type")
    dataset_uri = _required_string(payload.get("dataset_uri") or payload.get("dataset"), "dataset_uri")
    try:
        dataset = materialize_layer_dataset(
            dataset_uri,
            expected_sha256=_optional_string(payload.get("dataset_sha256")),
            artifacts_root=cfg.artifacts_path,
        )
        fingerprint = model_fingerprint(base_model, base_model_revision=base_model_revision)
        initial_specialist_manifest = _optional_string(payload.get("initial_specialist_manifest"))
        initial_specialist = None
        if initial_specialist_manifest is not None:
            if training_mode != "rl":
                raise ValueError("initial_specialist_manifest is only valid when training_mode=rl")
            initial_specialist = validate_initial_specialist_manifest(
                initial_specialist_manifest,
                base_model=base_model,
                base_model_revision=base_model_revision,
                layer_index=layer_index,
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    output_path = _safe_output_path(output)
    contribution = _optional_float(payload.get("contribution"), "contribution")
    min_contribution = _positive_float(payload.get("min_contribution", 0.5), "min_contribution")
    if contribution is not None:
        if contribution <= 0:
            raise HTTPException(status_code=422, detail="non-positive contribution is not allowed for layer specialist training")
        if contribution < min_contribution:
            raise HTTPException(status_code=422, detail=f"contribution {contribution} below min_contribution {min_contribution}")
    normalized = {
        "base_model": base_model,
        "base_model_revision": fingerprint.get("resolved_revision"),
        "base_model_hash": fingerprint["sha256"],
        "output": str(output_path),
        "task_type": task_type,
        "training_mode": training_mode,
        "rl_algorithm": rl_algorithm,
        "layer_index": layer_index,
        "dataset_uri": str(dataset.path),
        "dataset_source_uri": dataset.source_uri,
        "dataset_sha256": dataset.sha256,
        "execute": bool(payload.get("execute", True)),
        "learning_rate": _positive_float(payload.get("learning_rate", 1e-5), "learning_rate"),
        "batch_size": max(1, _positive_int(payload.get("batch_size", 1), "batch_size")),
        "max_steps": _positive_int(payload.get("max_steps", 100), "max_steps"),
        "seed": _int_value(payload.get("seed", 17), "seed"),
        "eval_split": _eval_split(payload.get("eval_split", 0.1)),
        "beta": _positive_float(payload.get("beta", 0.1), "beta"),
        "sft_coef": _non_negative_float(payload.get("sft_coef", 0.0), "sft_coef"),
        "max_seq_len": _positive_int(
            payload.get("max_seq_len", payload.get("max_seq_length", payload.get("max_sequence_length", 512))),
            "max_seq_len",
        ),
        "min_contribution": min_contribution,
        "contribution": contribution,
        "contribution_profile_id": _optional_string(payload.get("contribution_profile_id")),
        "contribution_profile_hash": _optional_string(payload.get("contribution_profile_hash")),
        "genome_id": _optional_string(payload.get("genome_id")),
        "initial_specialist_manifest": initial_specialist["path"] if initial_specialist else None,
        "initial_specialist_candidate_content_hash": (
            initial_specialist["manifest"]["candidate_content_hash"] if initial_specialist else None
        ),
        "timeout_seconds": _positive_float(payload.get("timeout_seconds", payload.get("timeout", 7200)), "timeout_seconds"),
    }
    return normalized


def _plan_train_layer_specialist(payload: dict[str, Any]) -> dict[str, Any]:
    specialist_output = Path(payload["output"]) / "layerscope" / f"layer_{payload['layer_index']}"
    command = _layer_specialist_command({**payload, "output": str(specialist_output)})
    return {
        "ok": True,
        "status": "planned",
        "dry_run": True,
        "schema": LAYER_SPECIALIST_SCHEMA,
        "base_model": payload["base_model"],
        "base_model_revision": payload.get("base_model_revision"),
        "base_model_hash": payload["base_model_hash"],
        "output": payload["output"],
        "layer_index": payload["layer_index"],
        "task_type": payload["task_type"],
        "training_mode": payload["training_mode"],
        "rl_algorithm": payload.get("rl_algorithm"),
        "beta": payload["beta"],
        "sft_coef": payload["sft_coef"],
        "initial_specialist_manifest": payload.get("initial_specialist_manifest"),
        "initial_specialist_candidate_content_hash": payload.get("initial_specialist_candidate_content_hash"),
        "dataset_uri": payload["dataset_uri"],
        "dataset_source_uri": payload["dataset_source_uri"],
        "dataset_sha256": payload["dataset_sha256"],
        "timeout_seconds": payload["timeout_seconds"],
        "freeze_strategy": "mlx_full_layer",
        "planned_command": command,
        "acceptance_criteria": {
            "trainable_layers": [payload["layer_index"]],
            "freeze_strategy": "mlx_full_layer",
            "frozen_layer_count": "N - 1, resolved at train time",
            "required_manifest_fields": [
                "schema",
                "base_model_id",
                "base_model_hash",
                "base_model_revision",
                "base_model_fingerprint",
                "genome_id",
                "candidate_content_hash",
                "layer_index",
                "task_type",
                "training_mode",
                *(
                    ["rl_algorithm", "objective", "initial_policy", "parent_candidate_content_hash"]
                    if payload["training_mode"] == "rl"
                    else []
                ),
                "freeze_strategy",
                "dataset_uri",
                "dataset_hash",
                "trainable_param_names",
                "frozen_param_count",
                "trainable_param_count",
                "param_shapes",
                "tensor_files",
                "sha256",
                "eval_metric",
                "baseline_metric",
                "created_at",
            ],
        },
    }


def _build_mlx_specialist_manifest(**kwargs: Any) -> dict[str, Any]:
    return build_layer_specialist_manifest(**kwargs)


def _run_layer_specialist_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        output = Path(payload["output"]) / "layerscope" / f"layer_{payload['layer_index']}"
        output.mkdir(parents=True, exist_ok=True)
        command = _layer_specialist_command({**payload, "output": str(output)})
        _update_training_job(
            job_id,
            status="running",
            started_at=time.time(),
            current_command=" ".join(command),
            output=str(output),
            progress=0.0,
        )
        _append_training_log(job_id, "Starting MLX full-layer specialist training.")
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        final_json: dict[str, Any] | None = None

        def handle_line(line: str) -> None:
            nonlocal final_json
            _append_training_log(job_id, line)
            parsed = _maybe_json(line)
            if parsed:
                final_json = parsed
            match = re.search(r"\bstep\s+(\d+)\b", line)
            if match:
                step = int(match.group(1))
                progress = min(0.95, step / max(1, int(payload["max_steps"])))
                _update_training_job(job_id, current_step=step, progress=progress)

        returncode = _stream_training_process(
            job_id,
            proc,
            timeout_seconds=float(payload["timeout_seconds"]),
            on_line=handle_line,
        )
        if returncode != 0:
            raise RuntimeError(f"layer specialist training exited with return code {returncode}")
        if not final_json or final_json.get("status") != "completed" or not final_json.get("manifest_path"):
            raise RuntimeError("layer specialist training exited without a completed manifest")
        manifest_path = str(final_json["manifest_path"])
        manifest_file = Path(manifest_path).resolve()
        if not manifest_file.is_file() or Path(output).resolve() not in manifest_file.parents:
            raise RuntimeError("layer specialist training returned an invalid manifest path")
        manifest = _read_json_file(manifest_file)
        if not isinstance(manifest, dict) or manifest.get("schema") != LAYER_SPECIALIST_SCHEMA:
            raise RuntimeError("layer specialist training returned an invalid manifest")
        if manifest.get("training_mode") != payload["training_mode"]:
            raise RuntimeError("layer specialist training returned the wrong training_mode")
        if manifest.get("rl_algorithm") != payload.get("rl_algorithm"):
            raise RuntimeError("layer specialist training returned the wrong rl_algorithm")
        if manifest.get("parent_candidate_content_hash") != payload.get("initial_specialist_candidate_content_hash"):
            raise RuntimeError("layer specialist training returned the wrong initial policy identity")
        for tensor_file in manifest.get("tensor_files") or []:
            tensor_path = (manifest_file.parent / str(tensor_file)).resolve()
            expected_hash = (manifest.get("sha256") or {}).get(str(tensor_file))
            if manifest_file.parent not in tensor_path.parents or not tensor_path.is_file():
                raise RuntimeError(f"layer specialist tensor is missing or unsafe: {tensor_file}")
            if not expected_hash or _sha256_path(tensor_path) != expected_hash:
                raise RuntimeError(f"layer specialist tensor hash mismatch: {tensor_file}")
        _update_training_job(
            job_id,
            status="completed",
            completed_at=time.time(),
            progress=1.0,
            ok=True,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        _append_training_log(job_id, "Layer specialist training job completed.")
    except _TrainingCancelled as exc:
        _update_training_job(job_id, status="cancelled", completed_at=time.time(), progress=1.0, ok=False, recoverable=True, error=str(exc))
        _append_training_log(job_id, f"Cancelled: {exc}")
    except Exception as exc:  # pragma: no cover - exercised by host runner runtime.
        _update_training_job(job_id, status="failed", completed_at=time.time(), progress=1.0, ok=False, recoverable=True, error=str(exc))
        _append_training_log(job_id, f"Failed: {exc}")


def _layer_specialist_command(payload: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dataevol.local_models.layer_specialist",
        "train",
        "--model",
        str(payload["base_model"]),
        "--layer-index",
        str(payload["layer_index"]),
        "--data",
        str(payload["dataset_uri"]),
        "--output",
        str(payload["output"]),
        "--task-type",
        str(payload["task_type"]),
        "--training-mode",
        str(payload["training_mode"]),
        *(_optional_arg("--rl-algorithm", payload.get("rl_algorithm"))),
        "--beta",
        str(payload["beta"]),
        "--sft-coef",
        str(payload["sft_coef"]),
        "--learning-rate",
        str(payload["learning_rate"]),
        "--batch-size",
        str(payload["batch_size"]),
        "--max-steps",
        str(payload["max_steps"]),
        "--max-seq-length",
        str(payload["max_seq_len"]),
        "--eval-split",
        str(payload["eval_split"]),
        "--seed",
        str(payload["seed"]),
        *(_optional_arg("--contribution-profile-id", payload.get("contribution_profile_id"))),
        *(_optional_arg("--contribution-profile-hash", payload.get("contribution_profile_hash"))),
        *(_optional_arg("--contribution", payload.get("contribution"))),
        *(_optional_arg("--dataset-source-uri", payload.get("dataset_source_uri"))),
        *(_optional_arg("--base-model-revision", payload.get("base_model_revision"))),
        *(_optional_arg("--genome-id", payload.get("genome_id"))),
        *(_optional_arg("--initial-specialist-manifest", payload.get("initial_specialist_manifest"))),
    ]


def _optional_arg(name: str, value: Any) -> list[str]:
    return [name, str(value)] if value else []


def _maybe_json(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _read_json_file(path: str | Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _resolve_local_dataset_uri(raw: str) -> Path:
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        raise HTTPException(status_code=422, detail=f"unsupported dataset uri scheme: {parsed.scheme}")
    path = Path(parsed.path if parsed.scheme == "file" else raw)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=422, detail=f"dataset not found: {path}")
    return path


def _resolve_artifact_file_path(raw: str, cfg: DataEvolConfig) -> Path:
    path = Path(raw or "").expanduser()
    if not str(path):
        raise HTTPException(status_code=422, detail="path is required")
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    roots = [Path.cwd().joinpath(".dataevol").resolve(), cfg.artifacts_path.resolve()]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise HTTPException(status_code=422, detail="artifact path is outside allowed roots")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="artifact file not found")
    return resolved


def _safe_output_path(raw: str) -> Path:
    path = Path(raw)
    if ".." in path.parts:
        raise HTTPException(status_code=422, detail="output path traversal is not allowed")
    return path


def _required_string(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field} is required")
    return text


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _non_negative_int(value: Any, field: str) -> int:
    n = _int_value(value, field)
    if n < 0:
        raise HTTPException(status_code=422, detail=f"{field} must be a non-negative integer")
    return n


def _positive_int(value: Any, field: str) -> int:
    n = _int_value(value, field)
    if n < 1:
        raise HTTPException(status_code=422, detail=f"{field} must be at least 1")
    return n


def _int_value(value: Any, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be an integer") from exc


def _optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _float_value(value, field)


def _positive_float(value: Any, field: str) -> float:
    n = _float_value(value, field)
    if n <= 0:
        raise HTTPException(status_code=422, detail=f"{field} must be positive")
    return n


def _non_negative_float(value: Any, field: str) -> float:
    n = _float_value(value, field)
    if n < 0:
        raise HTTPException(status_code=422, detail=f"{field} must be non-negative")
    return n


def _float_value(value: Any, field: str) -> float:
    try:
        n = float(value)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be a number") from exc
    if not (n < float("inf") and n > float("-inf")):
        raise HTTPException(status_code=422, detail=f"{field} must be finite")
    return n


def _eval_split(value: Any) -> float:
    n = _float_value(value, "eval_split")
    if n <= 0 or n >= 1:
        raise HTTPException(status_code=422, detail="eval_split must be between 0 and 1")
    return n


def _update_training_job(job_id: str, **updates: Any) -> dict[str, Any] | None:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if job is None:
            return None
        job.update(updates)
        persist_training_job(job)
        return job


def _append_training_log(job_id: str, line: str) -> None:
    text = line.rstrip()
    if not text:
        return
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if job is not None:
            job["logs"].append(text)
            persist_training_job(job)


class _TrainingCancelled(RuntimeError):
    pass


def _stream_training_process(
    job_id: str,
    proc: subprocess.Popen[str],
    *,
    timeout_seconds: float,
    on_line,
) -> int:
    if proc.stdout is None:
        raise RuntimeError("training subprocess stdout is unavailable")
    output: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in proc.stdout:
                output.put(line)
        finally:
            output.put(None)

    reader = threading.Thread(target=read_output, name=f"training-output-{job_id}", daemon=True)
    reader.start()
    cancel_event = TRAINING_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    with TRAINING_JOBS_LOCK:
        TRAINING_PROCESSES[job_id] = proc
    deadline = time.monotonic() + timeout_seconds
    try:
        output_closed = False
        while True:
            if cancel_event.is_set():
                _terminate_process(proc)
                raise _TrainingCancelled("training job was cancelled")
            if time.monotonic() >= deadline:
                _terminate_process(proc)
                raise TimeoutError(f"training job exceeded timeout of {timeout_seconds:g} seconds")
            if output_closed:
                returncode = proc.poll()
                if returncode is not None:
                    return returncode
                time.sleep(min(0.05, max(0.01, deadline - time.monotonic())))
                continue
            try:
                line = output.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if proc.poll() is not None and not reader.is_alive():
                    return int(proc.returncode or 0)
                continue
            if line is None:
                output_closed = True
            else:
                on_line(line)
    finally:
        with TRAINING_JOBS_LOCK:
            TRAINING_PROCESSES.pop(job_id, None)
        reader.join(timeout=1.0)


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=max(0.1, float(os.environ.get("LAYER_SPECIALIST_TERMINATE_GRACE_S", "5"))))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _layer_training_job(job_id: str, payload: dict[str, Any], cfg: DataEvolConfig, *, retry_of: str | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "job_id": job_id,
        "job_type": "layerscope_train_layer_specialist",
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "percent": 0.0,
        "layer_index": payload["layer_index"],
        "task_type": payload["task_type"],
        "training_mode": payload["training_mode"],
        "rl_algorithm": payload.get("rl_algorithm"),
        "manifest_path": None,
        "error": None,
        "recoverable": False,
        "retry_of": retry_of,
        "normalized_payload": dict(payload),
        "logs": deque(maxlen=160),
        "_db_path": str(Path(cfg.db_path).resolve()),
    }


def _dashboard_training_job(
    job_id: str,
    payload: dict[str, Any],
    cfg: DataEvolConfig,
    *,
    retry_of: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "job_id": job_id,
        "job_type": "local_model_training",
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "percent": 0.0,
        "current_expert": None,
        "current_iter": 0,
        "completed_experts": 0,
        "total_experts": 0,
        "total_iters": 0,
        "gpu_device_factor": float(payload.get("gpu_device_factor") or os.environ.get("DATAEVOL_GPU_DEVICE_FACTOR") or 1.0),
        "manifest_path": None,
        "script_path": None,
        "error": None,
        "recoverable": False,
        "retry_of": retry_of,
        "normalized_payload": dict(payload),
        "logs": deque(maxlen=160),
        "_db_path": str(Path(cfg.db_path).resolve()),
    }


def _start_layer_training_job(job: dict[str, Any]) -> None:
    job_id = str(job["job_id"])
    payload = dict(job["normalized_payload"])
    TRAINING_CANCEL_EVENTS[job_id] = threading.Event()
    thread = threading.Thread(
        target=_run_layer_specialist_job,
        args=(job_id, payload),
        name=f"dataevol-specialist-{job_id}",
        daemon=True,
    )
    thread.start()


def _start_dashboard_training_job(job: dict[str, Any]) -> None:
    job_id = str(job["job_id"])
    payload = dict(job["normalized_payload"])
    TRAINING_CANCEL_EVENTS[job_id] = threading.Event()
    thread = threading.Thread(
        target=_run_dashboard_training_job,
        args=(job_id, payload),
        name=f"dataevol-training-{job_id}",
        daemon=True,
    )
    thread.start()


def _run_dashboard_training_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        output = Path(payload.get("output") or DEFAULT_DASHBOARD_OUTPUT)
        base_model = str(payload.get("base_model") or payload.get("model") or ORNITH_9B_MODEL_PATH)
        experts = _dashboard_training_experts(payload)
        count = max(4, int(payload.get("count") or 24))
        iters = max(1, int(payload.get("iters") or 2))
        timeout = max(60, int(payload.get("timeout") or 7200))
        plan = prepare_local_adapter_training(
            output,
            python_bin=sys.executable,
            base_model=base_model,
            experts=experts,
            count=count,
            iters=iters,
        )
        jobs = list(plan.jobs)
        total_iters = max(1, len(jobs) * iters)
        _update_training_job(
            job_id,
            status="running",
            started_at=time.time(),
            total_experts=len(jobs),
            total_iters=total_iters,
            manifest_path=str(plan.manifest_path),
            script_path=str(plan.script_path),
        )
        _append_training_log(job_id, f"Prepared {len(jobs)} expert job(s) in {output}.")

        completed_experts = 0
        for job in jobs:
            job.adapter_dir.mkdir(parents=True, exist_ok=True)
            _update_training_job(
                job_id,
                status="running",
                current_expert=job.expert,
                current_iter=0,
                current_command=" ".join(job.command),
                completed_experts=completed_experts,
                progress=completed_experts * iters / total_iters,
            )
            _append_training_log(job_id, f"Starting {job.expert}: {' '.join(job.command)}")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                job.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            def handle_line(line: str) -> None:
                _append_training_log(job_id, line)
                match = ITERATION_RE.search(line)
                if match:
                    current_iter = min(iters, max(0, int(match.group(1))))
                    progress = ((completed_experts * iters) + current_iter) / total_iters
                    _update_training_job(job_id, current_iter=current_iter, progress=progress)

            returncode = _stream_training_process(
                job_id,
                proc,
                timeout_seconds=float(timeout),
                on_line=handle_line,
            )
            if returncode != 0:
                raise RuntimeError(f"{job.expert} training exited with return code {returncode}")
            completed_experts += 1
            _update_training_job(
                job_id,
                current_iter=iters,
                completed_experts=completed_experts,
                progress=completed_experts * iters / total_iters,
            )
            _append_training_log(job_id, f"Completed {job.expert}.")

        _update_training_job(
            job_id,
            status="completed",
            completed_at=time.time(),
            current_expert=None,
            progress=1.0,
            ok=True,
        )
        _append_training_log(job_id, "Training job completed.")
    except _TrainingCancelled as exc:
        _update_training_job(
            job_id,
            status="cancelled",
            completed_at=time.time(),
            ok=False,
            recoverable=True,
            error=str(exc),
        )
        _append_training_log(job_id, f"Cancelled: {exc}")
    except Exception as exc:  # pragma: no cover - exercised through dashboard runtime
        _update_training_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            ok=False,
            recoverable=True,
            error=str(exc),
        )
        _append_training_log(job_id, f"Failed: {exc}")


def create_app(config: DataEvolConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    recovered_jobs = initialize_training_jobs(cfg.db_path)
    with TRAINING_JOBS_LOCK:
        for recovered_job in recovered_jobs:
            TRAINING_JOBS[str(recovered_job["job_id"])] = recovered_job
    app = FastAPI(title="DataEvol API", version=__version__)
    protected = Depends(require_token(cfg))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "dataevol",
            "version": __version__,
            "privacy_mode": cfg.privacy_mode,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return _dashboard_html()

    @app.post("/ingest_trace", dependencies=[protected])
    def ingest_trace(request: IngestTraceRequest) -> dict[str, Any]:
        return call_core("ingest", "ingest_trace", request.model_dump(), config=cfg)

    @app.post("/ingest_run", dependencies=[protected])
    def ingest_run(request: IngestRunRequest) -> dict[str, Any]:
        return call_core("ingest", "ingest_run", request.model_dump(), config=cfg)

    @app.post("/ingest_worker_report", dependencies=[protected])
    def ingest_worker_report(request: IngestWorkerReportRequest) -> dict[str, Any]:
        return call_core("ingest", "ingest_worker_report", request.model_dump(), config=cfg)

    @app.post("/label", dependencies=[protected])
    def label(request: OperationRequest) -> dict[str, Any]:
        return call_core("labeling", "label_run", request.payload, config=cfg)

    @app.post("/score", dependencies=[protected])
    def score(request: OperationRequest) -> dict[str, Any]:
        return call_core("scoring", "score_run", request.payload, config=cfg)

    @app.post("/compress", dependencies=[protected])
    def compress(request: OperationRequest) -> dict[str, Any]:
        return call_core("compression", "compress_run", request.payload, config=cfg)

    @app.post("/build_dataset", dependencies=[protected])
    def build_dataset(request: OperationRequest) -> dict[str, Any]:
        return call_core("datasets", "build_dataset", request.payload, config=cfg)

    @app.post("/router_performance", dependencies=[protected])
    def router_performance(request: OperationRequest) -> dict[str, Any]:
        return call_core("datasets", "router_performance", request.payload, config=cfg)

    @app.post("/candidate_router_policy", dependencies=[protected])
    def candidate_router_policy(request: OperationRequest) -> dict[str, Any]:
        return call_core("datasets", "candidate_router_policy", request.payload, config=cfg)

    @app.post("/build_benchmark", dependencies=[protected])
    def build_benchmark(request: OperationRequest) -> dict[str, Any]:
        return call_core("benchmarks", "build_benchmark", request.payload, config=cfg)

    @app.post("/reflect", dependencies=[protected])
    def reflect(request: OperationRequest) -> dict[str, Any]:
        return call_core("evolve", "reflect", request.payload, config=cfg)

    @app.post("/idea_prd", dependencies=[protected])
    def idea_prd(request: OperationRequest) -> dict[str, Any]:
        return call_core("evolve", "idea_prd", request.payload, config=cfg)

    @app.post("/experiment", dependencies=[protected])
    def experiment(request: OperationRequest) -> dict[str, Any]:
        return call_core("evolve", "experiment", request.payload, config=cfg)

    @app.post("/compare", dependencies=[protected])
    def compare(request: OperationRequest) -> dict[str, Any]:
        return call_core("evolve", "compare", request.payload, config=cfg)

    @app.post("/promote", dependencies=[protected])
    def promote(request: OperationRequest) -> dict[str, Any]:
        return call_core("evolve", "promote", request.payload, config=cfg)

    @app.post("/reject", dependencies=[protected])
    def reject(request: OperationRequest) -> dict[str, Any]:
        return call_core("evolve", "reject", request.payload, config=cfg)

    @app.post("/privacy/export_training_candidates", dependencies=[protected])
    def export_training_candidates(request: OperationRequest) -> dict[str, Any]:
        return call_core("privacy", "export_training_candidates", request.payload, config=cfg)

    @app.post("/prompts/variants", dependencies=[protected])
    def prompt_variants(request: OperationRequest) -> dict[str, Any]:
        return call_core("prompts", "variants", request.payload, config=cfg)

    @app.post("/prompts/version", dependencies=[protected])
    def prompt_version(request: OperationRequest) -> dict[str, Any]:
        return call_core("prompts", "version", request.payload, config=cfg)

    @app.post("/prompts/ab_test", dependencies=[protected])
    def prompt_ab_test(request: OperationRequest) -> dict[str, Any]:
        return call_core("prompts", "ab_test", request.payload, config=cfg)

    @app.post("/prompts/promote", dependencies=[protected])
    def prompt_promote(request: OperationRequest) -> dict[str, Any]:
        return call_core("prompts", "promote", request.payload, config=cfg)

    @app.post("/integrations/router_dataset_pull", dependencies=[protected])
    def router_dataset_pull(request: OperationRequest) -> dict[str, Any]:
        return call_core("integrations", "router_dataset_pull", request.payload, config=cfg)

    @app.post("/integrations/post_coordinate_completion", dependencies=[protected])
    def post_coordinate_completion(request: OperationRequest) -> dict[str, Any]:
        return call_core("integrations", "post_coordinate_completion", request.payload, config=cfg)

    @app.post("/local_model/prepare", dependencies=[protected])
    def local_model_prepare(request: OperationRequest) -> dict[str, Any]:
        return call_core("local_models", "prepare", request.payload, config=cfg)

    @app.post("/local_model/train", dependencies=[protected])
    def local_model_train(request: OperationRequest) -> dict[str, Any]:
        return call_core("local_models", "train", request.payload, config=cfg)

    @app.post("/local_model/evaluate", dependencies=[protected])
    def local_model_evaluate(request: OperationRequest) -> dict[str, Any]:
        return call_core("local_models", "evaluate", request.payload, config=cfg)

    @app.post("/local_model/promote", dependencies=[protected])
    def local_model_promote(request: OperationRequest) -> dict[str, Any]:
        return call_core("local_models", "promote", request.payload, config=cfg)

    @app.post("/local_model/status", dependencies=[protected])
    def local_model_status(request: OperationRequest) -> dict[str, Any]:
        output = Path(request.payload.get("output") or DEFAULT_DASHBOARD_OUTPUT)
        selected_model = str(request.payload.get("model") or DEFAULT_TRAINABLE_LAYER_MODEL)
        adapters = {}
        datasets = {}
        for expert in EXPERTS:
            adapter_dir = output / "adapters" / expert
            data_dir = output / "adapter_data" / expert
            adapters[expert] = {
                "path": str(adapter_dir),
                "exists": (adapter_dir / "adapters.safetensors").exists(),
                "files": sorted(path.name for path in adapter_dir.glob("*")) if adapter_dir.exists() else [],
            }
            datasets[expert] = {
                "path": str(data_dir),
                "exists": (data_dir / "train.jsonl").exists(),
                "files": sorted(path.name for path in data_dir.glob("*")) if data_dir.exists() else [],
            }
        return {
            "ok": True,
            "experts": list(EXPERTS),
            "models": [
                {
                    "id": DEFAULT_TRAINABLE_LAYER_MODEL,
                    "label": "Qwen3 0.6B MLX 4-bit",
                    "exists": _model_available(DEFAULT_TRAINABLE_LAYER_MODEL),
                    "recommended_for": "layer-specialist training",
                },
                {
                    "id": ORNITH_9B_MODEL_PATH,
                    "label": "Ornith 1.0 9B MLX 8-bit",
                    "exists": _model_available(ORNITH_9B_MODEL_PATH),
                    "recommended_for": "inference/serving experiments; middle layers hit MLX custom-kernel VJP limits",
                },
            ],
            "selected_model": selected_model,
            "model_exists": _model_available(selected_model),
            "output": str(output),
            "manifest_exists": (output / "adapter_training_manifest.json").exists(),
            "datasets": datasets,
            "adapters": adapters,
        }

    @app.post("/local_model/artifacts/export", dependencies=[protected])
    def local_model_artifacts_export(request: OperationRequest) -> dict[str, Any]:
        return _export_local_model_artifacts(request.payload)

    @app.get("/local_model/artifacts/file", dependencies=[protected])
    def local_model_artifacts_file(path: Annotated[str, Query()]):
        resolved = _resolve_artifact_file_path(path, cfg)
        return FileResponse(resolved)

    @app.post("/local_model/training/start", dependencies=[protected])
    def local_model_training_start(request: OperationRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = _dashboard_training_job(job_id, dict(request.payload), cfg)
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[job_id] = job
            persist_training_job(job)
        _start_dashboard_training_job(job)
        return _training_job_snapshot(job)

    @app.post("/local_model/training/status", dependencies=[protected])
    def local_model_training_status(request: OperationRequest) -> dict[str, Any]:
        job_id = request.payload.get("job_id")
        if not job_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Missing training job id.")
        with TRAINING_JOBS_LOCK:
            job = TRAINING_JOBS.get(str(job_id))
            if job is None or str(Path(str(job.get("_db_path") or "")).resolve()) != str(Path(cfg.db_path).resolve()):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown training job id.")
            return _training_job_snapshot(job)

    @app.post("/local_model/training/latest", dependencies=[protected])
    def local_model_training_latest(request: OperationRequest) -> dict[str, Any]:
        with TRAINING_JOBS_LOCK:
            configured_jobs = jobs_for_db(TRAINING_JOBS.values(), cfg.db_path)
            if not configured_jobs:
                return {"ok": True, "status": "idle", "job_id": None, "logs": []}
            job = max(configured_jobs, key=lambda item: item.get("created_at") or 0)
            return _training_job_snapshot(job)

    @app.post("/local_model/training/cancel", dependencies=[protected])
    def local_model_training_cancel(request: OperationRequest) -> dict[str, Any]:
        job_id = str(request.payload.get("job_id") or "")
        with TRAINING_JOBS_LOCK:
            job = TRAINING_JOBS.get(job_id)
            if job is None or str(Path(str(job.get("_db_path") or "")).resolve()) != str(Path(cfg.db_path).resolve()):
                raise HTTPException(status_code=404, detail="Unknown training job id.")
            if job.get("status") not in {"queued", "running"}:
                return _training_job_snapshot(job)
            event = TRAINING_CANCEL_EVENTS.setdefault(job_id, threading.Event())
            event.set()
            proc = TRAINING_PROCESSES.get(job_id)
        if proc is not None:
            _terminate_process(proc)
        updated = _update_training_job(
            job_id,
            status="cancelled",
            completed_at=time.time(),
            ok=False,
            recoverable=True,
            error="training job was cancelled",
        )
        return _training_job_snapshot(updated or job)

    @app.post("/local_model/training/retry", dependencies=[protected])
    def local_model_training_retry(request: OperationRequest) -> dict[str, Any]:
        source_id = str(request.payload.get("job_id") or "")
        with TRAINING_JOBS_LOCK:
            source = TRAINING_JOBS.get(source_id)
            if source is None or str(Path(str(source.get("_db_path") or "")).resolve()) != str(Path(cfg.db_path).resolve()):
                raise HTTPException(status_code=404, detail="Unknown training job id.")
            if source.get("job_type") not in {"layerscope_train_layer_specialist", "local_model_training"} or not source.get("recoverable"):
                raise HTTPException(status_code=409, detail="Training job is not recoverable.")
            existing_retry_id = source.get("retry_job_id")
            if existing_retry_id and existing_retry_id in TRAINING_JOBS:
                return _training_job_snapshot(TRAINING_JOBS[existing_retry_id])
            normalized_payload = source.get("normalized_payload")
            if not isinstance(normalized_payload, dict):
                raise HTTPException(status_code=409, detail="Recoverable job has no normalized payload.")
            retry_id = uuid.uuid4().hex[:12]
            if source.get("job_type") == "layerscope_train_layer_specialist":
                retry_job = _layer_training_job(retry_id, dict(normalized_payload), cfg, retry_of=source_id)
            else:
                retry_job = _dashboard_training_job(retry_id, dict(normalized_payload), cfg, retry_of=source_id)
            TRAINING_JOBS[retry_id] = retry_job
            source["retry_job_id"] = retry_id
            persist_training_job(source)
            persist_training_job(retry_job)
        if retry_job["job_type"] == "layerscope_train_layer_specialist":
            _start_layer_training_job(retry_job)
        else:
            _start_dashboard_training_job(retry_job)
        return _training_job_snapshot(retry_job)

    @app.post("/local_model/layerscope/train_layer_specialist", dependencies=[protected])
    def local_model_layerscope_train_layer_specialist(request: OperationRequest) -> dict[str, Any]:
        payload = _validate_train_layer_specialist(request.payload, cfg)
        if not payload.get("execute", True):
            return _plan_train_layer_specialist(payload)
        job_id = uuid.uuid4().hex[:12]
        job = _layer_training_job(job_id, payload, cfg)
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[job_id] = job
            persist_training_job(job)
        _start_layer_training_job(job)
        return _training_job_snapshot(job)

    @app.post("/rlmf/prepare", dependencies=[protected])
    def rlmf_prepare(request: OperationRequest) -> dict[str, Any]:
        output = Path(request.payload.get("output") or cfg.artifacts_path / "rlmf_fixture")
        count = max(6, int(request.payload.get("count") or 8))
        return prepare_rlmf_fixture(output, count=count)

    @app.post("/rlmf/train_dry_run", dependencies=[protected])
    def rlmf_train_dry_run(request: OperationRequest) -> dict[str, Any]:
        output = Path(request.payload.get("output") or cfg.artifacts_path / "rlmf_fixture")
        dataset = build_rlmf_fixture_dataset(output, count=max(6, int(request.payload.get("count") or 8)))
        mlx = build_mlx_lora_manifest(
            output,
            dataset.manifest,
            python_bin=str(request.payload.get("python_bin") or "python"),
            base_model=str(request.payload.get("base_model") or request.payload.get("model") or "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
            experts=request.payload.get("experts") or ("failure_classifier", "verifier", "local_model_trainer"),
            iters=max(1, int(request.payload.get("iters") or 2)),
        )
        qlora = build_qlora_manifest(
            output,
            dataset.manifest,
            base_model=str(request.payload.get("qlora_base_model") or "Qwen/Qwen2.5-7B-Instruct"),
        )
        return {
            "ok": True,
            "status": "planned",
            "dry_run": True,
            "dataset_manifest_hash": dataset.manifest["dataset_manifest_hash"],
            "mlx_lora": mlx,
            "qlora": qlora,
        }

    @app.post("/rlmf/judge_fixture", dependencies=[protected])
    def rlmf_judge_fixture(request: OperationRequest) -> dict[str, Any]:
        return parse_qwen_judge_fixture(request.payload)

    @app.post("/rlmf/benchmark_fixture", dependencies=[protected])
    def rlmf_benchmark_fixture(request: OperationRequest) -> dict[str, Any]:
        output = Path(request.payload.get("output") or cfg.artifacts_path / "rlmf_fixture")
        dataset = build_rlmf_fixture_dataset(output, count=max(6, int(request.payload.get("count") or 8)))
        judge = parse_qwen_judge_fixture(request.payload.get("judge") if isinstance(request.payload.get("judge"), dict) else None)
        return build_benchmark_fixture(dataset.manifest, judge)

    @app.post("/rlmf/promotion_fixture", dependencies=[protected])
    def rlmf_promotion_fixture(request: OperationRequest) -> dict[str, Any]:
        output = Path(request.payload.get("output") or cfg.artifacts_path / "rlmf_fixture")
        dataset = build_rlmf_fixture_dataset(output, count=max(6, int(request.payload.get("count") or 8)))
        judge = parse_qwen_judge_fixture(request.payload.get("judge") if isinstance(request.payload.get("judge"), dict) else None)
        benchmark = build_benchmark_fixture(dataset.manifest, judge)
        return build_promotion_fixture(benchmark)

    @app.get("/runs")
    def runs() -> dict[str, Any]:
        return call_core("reports", "runs", {}, config=cfg)

    @app.get("/datasets")
    def datasets() -> dict[str, Any]:
        return call_core("reports", "datasets", {}, config=cfg)

    @app.get("/benchmarks")
    def benchmarks() -> dict[str, Any]:
        return call_core("reports", "benchmarks", {}, config=cfg)

    @app.get("/experiments")
    def experiments() -> dict[str, Any]:
        return call_core("reports", "experiments", {}, config=cfg)

    @app.get("/opportunities")
    def opportunities() -> dict[str, Any]:
        return call_core("reports", "opportunities", {}, config=cfg)

    @app.get("/idea_prds")
    def idea_prds() -> dict[str, Any]:
        return call_core("reports", "idea_prds", {}, config=cfg)

    @app.get("/promotions")
    def promotions() -> dict[str, Any]:
        return call_core("reports", "promotions", {}, config=cfg)

    # --- Harness Evolver ----------------------------------------------------
    for _harness_op in (
        "design", "benchmark", "evaluate", "failures", "mutate", "judge", "evolve",
        "compile", "register_compiled", "route_compiled", "start_execution",
        "execution_action", "execution_observation", "export_next_actions",
    ):

        @app.post(f"/harness/{_harness_op}", dependencies=[protected], name=f"harness_{_harness_op}")
        def _harness_endpoint(request: OperationRequest, _op: str = _harness_op) -> dict[str, Any]:
            return call_core("harness", _op, request.payload, config=cfg)

    @app.post("/harness/verdict", dependencies=[protected])
    def harness_verdict(request: dict[str, Any]) -> dict[str, Any]:
        payload = request.get("payload") if isinstance(request.get("payload"), dict) else request
        return call_core("harness", "verdict", payload, config=cfg)

    @app.get("/harness/verdicts/{verdict_id}", dependencies=[protected])
    def harness_get_verdict(verdict_id: str) -> dict[str, Any]:
        result = call_core("harness", "get_verdict", {"verdict_id": verdict_id}, config=cfg)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"harness verdict not found: {verdict_id}")
        return result

    @app.get("/harness/lineage")
    def harness_lineage() -> dict[str, Any]:
        return call_core("harness", "lineage", {}, config=cfg)

    @app.get("/harness/incumbent")
    def harness_incumbent() -> dict[str, Any]:
        return call_core("harness", "incumbent", {}, config=cfg)

    @app.get("/harness/training_records")
    def harness_training_records() -> dict[str, Any]:
        return call_core("harness", "training_records", {}, config=cfg)

    @app.get("/harness/compiled", dependencies=[protected])
    def harness_compiled_registry(
        status_filter: str | None = Query(default=None, alias="status"),
        category: str | None = None,
    ) -> dict[str, Any]:
        return call_core(
            "harness", "compiled_registry", {"status": status_filter, "category": category}, config=cfg
        )

    @app.get("/harness/executions/{session_id}", dependencies=[protected])
    def harness_execution(session_id: str) -> dict[str, Any]:
        result = call_core("harness", "get_execution", {"session_id": session_id}, config=cfg)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"harness execution not found: {session_id}")
        return result

    @app.get("/harness/execution_metrics", dependencies=[protected])
    def harness_execution_metrics(session_id: str | None = None) -> dict[str, Any]:
        return call_core("harness", "execution_metrics", {"session_id": session_id}, config=cfg)

    return app


app = create_app()


def _dashboard_html() -> str:
    experts = "\n".join(f'<option value="{expert}">{expert}</option>' for expert in EXPERTS)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataEvol Local Training</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #17202a;
      --muted: #667085;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #9a3412;
      --ok: #166534;
      --shadow: 0 1px 2px rgba(16, 24, 40, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    main.wrap {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
      align-items: start;
    }}
    section, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .controls {{
      padding: 16px;
    }}
    .field {{
      margin-bottom: 14px;
    }}
    label {{
      display: block;
      margin-bottom: 6px;
      font-weight: 650;
      color: #273444;
    }}
    input, select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
      font: inherit;
    }}
    input[readonly] {{
      background: #f8fafc;
      color: #475569;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 16px;
    }}
    button {{
      min-height: 40px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }}
    button.secondary {{
      background: #fff;
      color: var(--accent-dark);
    }}
    button:disabled {{
      opacity: .55;
      cursor: wait;
    }}
    .status {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .progressPanel {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }}
    .progressTitle {{
      margin: 0;
      font-size: 14px;
      letter-spacing: 0;
    }}
    .progressTrack {{
      width: 100%;
      height: 12px;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      overflow: hidden;
      background: #eef2f7;
    }}
    .progressFill {{
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width .2s ease;
    }}
    .progressStats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
      min-width: 0;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .stat strong {{
      display: block;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: #475569;
      font-size: 12px;
      font-weight: 700;
    }}
    .pill.ok {{ color: var(--ok); border-color: #bbf7d0; background: #f0fdf4; }}
    .pill.warn {{ color: var(--warn); border-color: #fed7aa; background: #fff7ed; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 16px;
    }}
    .expert {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 96px;
      background: #fff;
    }}
    .expert h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-word;
    }}
    pre {{
      margin: 0;
      padding: 16px;
      max-height: 340px;
      overflow: auto;
      border-top: 1px solid var(--line);
      background: #0b1020;
      color: #dbeafe;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    @media (max-width: 860px) {{
      main.wrap {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: 1fr; }}
      .progressStats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>DataEvol Local Expert Training</h1>
    </div>
  </header>
  <main class="wrap">
    <section class="controls">
      <div class="field">
        <label for="token">API token</label>
        <input id="token" type="password" value="dev-local-token" autocomplete="off">
      </div>
      <div class="field">
        <label for="model">Model</label>
        <select id="model">
          <option value="{DEFAULT_TRAINABLE_LAYER_MODEL}" selected>Qwen3 0.6B MLX 4-bit</option>
          <option value="{ORNITH_9B_MODEL_PATH}">Ornith 1.0 9B MLX 8-bit</option>
        </select>
      </div>
      <div class="field">
        <label for="expert">Expert</label>
        <select id="expert">
          <option value="">All experts</option>
          {experts}
        </select>
      </div>
      <div class="field">
        <label for="output">Output directory</label>
        <input id="output" value="{DEFAULT_DASHBOARD_OUTPUT}">
      </div>
      <div class="row">
        <div class="field">
          <label for="count">Examples</label>
          <input id="count" type="number" min="4" step="1" value="24">
        </div>
        <div class="field">
          <label for="iters">Iterations</label>
          <input id="iters" type="number" min="1" step="1" value="2">
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="timeout">Timeout sec</label>
          <input id="timeout" type="number" min="60" step="60" value="7200">
        </div>
        <div class="field">
          <label for="execute">Mode</label>
          <input id="execute" value="execute training" readonly>
        </div>
      </div>
      <div class="actions">
        <button class="secondary" id="prepare">Create Dataset</button>
        <button id="train">Train Adapter</button>
      </div>
    </section>
    <section>
      <div class="status">
        <span id="modelStatus" class="pill">model unknown</span>
        <span id="runStatus" class="pill">idle</span>
      </div>
      <div class="progressPanel">
        <h2 class="progressTitle">Training Progress</h2>
        <div class="progressTrack"><div id="progressFill" class="progressFill"></div></div>
        <div class="progressStats">
          <div class="stat"><span>Progress</span><strong id="progressText">0%</strong></div>
          <div class="stat"><span>Expert</span><strong id="currentExpert">idle</strong></div>
          <div class="stat"><span>Iteration</span><strong id="currentIter">0 / 0</strong></div>
          <div class="stat"><span>ETA</span><strong id="etaText">--</strong></div>
        </div>
        <div class="progressStats">
          <div class="stat"><span>Elapsed</span><strong id="elapsedText">0s</strong></div>
          <div class="stat"><span>Completed experts</span><strong id="completedExperts">0 / 0</strong></div>
          <div class="stat"><span>Job</span><strong id="jobId">none</strong></div>
          <div class="stat"><span>Output</span><strong id="jobOutput">{DEFAULT_DASHBOARD_OUTPUT}</strong></div>
        </div>
      </div>
      <div id="experts" class="grid"></div>
      <pre id="outputLog">{{}}</pre>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let trainingPoll = null;
    let activeTrainingJob = null;
    const log = (value) => {{
      $("outputLog").textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }};
    const formatDuration = (seconds) => {{
      if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "--";
      const total = Math.max(0, Math.round(Number(seconds)));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const secs = total % 60;
      if (hours) return `${{hours}}h ${{minutes}}m`;
      if (minutes) return `${{minutes}}m ${{secs}}s`;
      return `${{secs}}s`;
    }};
    const payload = () => {{
      const expert = $("expert").value;
      return {{
        output: $("output").value,
        base_model: $("model").value,
        model: $("model").value,
        experts: expert ? [expert] : null,
        count: Number($("count").value),
        iters: Number($("iters").value),
        timeout: Number($("timeout").value)
      }};
    }};
    async function post(path, body) {{
      const response = await fetch(path, {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
          "Authorization": `Bearer ${{$("token").value}}`
        }},
        body: JSON.stringify({{ payload: body }})
      }});
      const text = await response.text();
      let data;
      try {{ data = JSON.parse(text); }} catch {{ data = {{ ok: false, raw: text }}; }}
      if (!response.ok) throw data;
      return data;
    }}
    function renderStatus(data) {{
      $("modelStatus").textContent = data.model_exists ? "Model ready/configured" : "Local model path missing";
      $("modelStatus").className = `pill ${{data.model_exists ? "ok" : "warn"}}`;
      $("jobOutput").textContent = data.output || $("output").value;
      const names = data.experts || [];
      $("experts").innerHTML = names.map((name) => {{
        const dataset = data.datasets[name] || {{}};
        const adapter = data.adapters[name] || {{}};
        const datasetClass = dataset.exists ? "ok" : "warn";
        const adapterClass = adapter.exists ? "ok" : "warn";
        return `<div class="expert">
          <h2>${{name}}</h2>
          <div><span class="pill ${{datasetClass}}">${{dataset.exists ? "dataset ready" : "no dataset"}}</span></div>
          <div style="height:6px"></div>
          <div><span class="pill ${{adapterClass}}">${{adapter.exists ? "adapter ready" : "no adapter"}}</span></div>
          <div class="meta">${{adapter.path || ""}}</div>
        </div>`;
      }}).join("");
    }}
    function renderTraining(job) {{
      const percent = Number(job.percent || 0);
      $("progressFill").style.width = `${{Math.max(0, Math.min(100, percent))}}%`;
      $("progressText").textContent = `${{percent.toFixed(1)}}%`;
      $("currentExpert").textContent = job.current_expert || job.status || "idle";
      $("currentIter").textContent = `${{job.current_iter || 0}} / ${{job.total_iters ? Math.max(1, Math.round(job.total_iters / Math.max(1, job.total_experts || 1))) : 0}}`;
      $("etaText").textContent = job.status === "completed" ? "done" : formatDuration(job.eta_seconds);
      $("elapsedText").textContent = formatDuration(job.elapsed_seconds);
      $("completedExperts").textContent = `${{job.completed_experts || 0}} / ${{job.total_experts || 0}}`;
      $("jobId").textContent = job.job_id || "none";
      if (job.logs && job.logs.length) log(job.logs.join("\\n"));
      if (job.status === "failed") {{
        $("runStatus").textContent = job.error || "failed";
        $("runStatus").className = "pill warn";
      }} else if (job.status === "completed") {{
        $("runStatus").textContent = "training completed";
        $("runStatus").className = "pill ok";
      }} else {{
        $("runStatus").textContent = job.status || "training";
        $("runStatus").className = "pill warn";
      }}
    }}
    async function refresh() {{
      try {{
        const data = await post("/local_model/status", payload());
        renderStatus(data);
      }} catch (err) {{
        $("runStatus").textContent = "status failed";
        $("runStatus").className = "pill warn";
        log(err);
      }}
    }}
    async function run(path, label, extra) {{
      $("prepare").disabled = true;
      $("train").disabled = true;
      $("runStatus").textContent = label;
      $("runStatus").className = "pill warn";
      try {{
        const body = Object.assign(payload(), extra || {{}});
        const data = await post(path, body);
        $("runStatus").textContent = data.status || "done";
        $("runStatus").className = `pill ${{data.ok === false ? "warn" : "ok"}}`;
        log(data);
        await refresh();
      }} catch (err) {{
        $("runStatus").textContent = "failed";
        $("runStatus").className = "pill warn";
        log(err);
      }} finally {{
        $("prepare").disabled = false;
        $("train").disabled = false;
      }}
    }}
    async function pollTraining(jobId) {{
      try {{
        const job = await post("/local_model/training/status", {{ job_id: jobId }});
        renderTraining(job);
        if (job.status === "completed" || job.status === "failed") {{
          clearInterval(trainingPoll);
          trainingPoll = null;
          activeTrainingJob = null;
          $("prepare").disabled = false;
          $("train").disabled = false;
          await refresh();
        }}
      }} catch (err) {{
        clearInterval(trainingPoll);
        trainingPoll = null;
        activeTrainingJob = null;
        $("prepare").disabled = false;
        $("train").disabled = false;
        $("runStatus").textContent = "progress failed";
        $("runStatus").className = "pill warn";
        log(err);
      }}
    }}
    async function startTraining() {{
      if (activeTrainingJob) return;
      $("prepare").disabled = true;
      $("train").disabled = true;
      $("runStatus").textContent = "starting training";
      $("runStatus").className = "pill warn";
      try {{
        const job = await post("/local_model/training/start", payload());
        activeTrainingJob = job.job_id;
        renderTraining(job);
        await pollTraining(activeTrainingJob);
        if (activeTrainingJob) {{
          trainingPoll = setInterval(() => pollTraining(activeTrainingJob), 2000);
        }}
      }} catch (err) {{
        activeTrainingJob = null;
        $("prepare").disabled = false;
        $("train").disabled = false;
        $("runStatus").textContent = "failed";
        $("runStatus").className = "pill warn";
        log(err);
      }}
    }}
    async function resumeLatestTraining() {{
      try {{
        const job = await post("/local_model/training/latest", {{}});
        if (!job.job_id) return;
        renderTraining(job);
        if (job.status !== "completed" && job.status !== "failed") {{
          activeTrainingJob = job.job_id;
          $("prepare").disabled = true;
          $("train").disabled = true;
          trainingPoll = setInterval(() => pollTraining(activeTrainingJob), 2000);
        }}
      }} catch (err) {{
        log(err);
      }}
    }}
    $("prepare").addEventListener("click", () => run("/local_model/prepare", "creating dataset"));
    $("train").addEventListener("click", startTraining);
    $("expert").addEventListener("change", refresh);
    $("output").addEventListener("change", refresh);
    refresh();
    resumeLatestTraining();
  </script>
</body>
</html>"""
