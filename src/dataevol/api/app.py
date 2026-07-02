from __future__ import annotations

import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from dataevol import __version__
from dataevol.compat import call_core
from dataevol.config import DataEvolConfig, load_config
from dataevol.local_models import EXPERTS, prepare_local_adapter_training
from dataevol.rlmf import (
    build_benchmark_fixture,
    build_mlx_lora_manifest,
    build_promotion_fixture,
    build_qlora_manifest,
    build_rlmf_fixture_dataset,
    parse_qwen_judge_fixture,
    prepare_rlmf_fixture,
)


ORNITH_9B_MODEL_PATH = ".dataevol/models/Ornith-1.0-9B-8bit"
DEFAULT_DASHBOARD_OUTPUT = ".dataevol/ornith_9b_experts"
TRAINING_JOBS: dict[str, dict[str, Any]] = {}
TRAINING_JOBS_LOCK = threading.Lock()
ITERATION_RE = re.compile(r"\bIter\s+(\d+)\s*:")


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


def _extract_token(authorization: str | None, x_dataevol_token: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return x_dataevol_token


def _require_token(config: DataEvolConfig):
    def dependency(
        authorization: Annotated[str | None, Header()] = None,
        x_dataevol_token: Annotated[str | None, Header(alias="X-DataEvol-Token")] = None,
    ) -> None:
        supplied = _extract_token(authorization, x_dataevol_token)
        if not supplied or not secrets.compare_digest(supplied, config.api_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid DataEvol API token.",
            )

    return dependency


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
        if key not in {"logs"}
    }
    snapshot["logs"] = list(job.get("logs") or [])
    snapshot["elapsed_seconds"] = elapsed
    snapshot["eta_seconds"] = eta
    snapshot["percent"] = round(progress * 100, 1)
    return snapshot


def _update_training_job(job_id: str, **updates: Any) -> dict[str, Any] | None:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if job is None:
            return None
        job.update(updates)
        return job


def _append_training_log(job_id: str, line: str) -> None:
    text = line.rstrip()
    if not text:
        return
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if job is not None:
            job["logs"].append(text)


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
            started_at = time.time()
            proc = subprocess.Popen(
                job.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _append_training_log(job_id, line)
                match = ITERATION_RE.search(line)
                if match:
                    current_iter = min(iters, max(0, int(match.group(1))))
                    progress = ((completed_experts * iters) + current_iter) / total_iters
                    _update_training_job(job_id, current_iter=current_iter, progress=progress)
                if time.time() - started_at > timeout:
                    proc.kill()
                    raise TimeoutError(f"{job.expert} exceeded timeout of {timeout} seconds")
            returncode = proc.wait()
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
    except Exception as exc:  # pragma: no cover - exercised through dashboard runtime
        _update_training_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            ok=False,
            error=str(exc),
        )
        _append_training_log(job_id, f"Failed: {exc}")


def create_app(config: DataEvolConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="DataEvol API", version=__version__)
    protected = Depends(_require_token(cfg))

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
        model_path = Path(request.payload.get("model") or ORNITH_9B_MODEL_PATH)
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
                    "id": ORNITH_9B_MODEL_PATH,
                    "label": "Ornith 1.0 9B MLX 8-bit",
                    "exists": Path(ORNITH_9B_MODEL_PATH).exists(),
                }
            ],
            "selected_model": str(model_path),
            "model_exists": model_path.exists(),
            "output": str(output),
            "manifest_exists": (output / "adapter_training_manifest.json").exists(),
            "datasets": datasets,
            "adapters": adapters,
        }

    @app.post("/local_model/training/start", dependencies=[protected])
    def local_model_training_start(request: OperationRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "ok": True,
            "job_id": job_id,
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
            "manifest_path": None,
            "script_path": None,
            "error": None,
            "logs": deque(maxlen=160),
        }
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[job_id] = job
        thread = threading.Thread(
            target=_run_dashboard_training_job,
            args=(job_id, dict(request.payload)),
            name=f"dataevol-training-{job_id}",
            daemon=True,
        )
        thread.start()
        return _training_job_snapshot(job)

    @app.post("/local_model/training/status", dependencies=[protected])
    def local_model_training_status(request: OperationRequest) -> dict[str, Any]:
        job_id = request.payload.get("job_id")
        if not job_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Missing training job id.")
        with TRAINING_JOBS_LOCK:
            job = TRAINING_JOBS.get(str(job_id))
            if job is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown training job id.")
            return _training_job_snapshot(job)

    @app.post("/local_model/training/latest", dependencies=[protected])
    def local_model_training_latest(request: OperationRequest) -> dict[str, Any]:
        with TRAINING_JOBS_LOCK:
            if not TRAINING_JOBS:
                return {"ok": True, "status": "idle", "job_id": None, "logs": []}
            job = max(TRAINING_JOBS.values(), key=lambda item: item.get("created_at") or 0)
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
          <option value="{ORNITH_9B_MODEL_PATH}" selected>Ornith 1.0 9B MLX 8-bit</option>
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
      $("modelStatus").textContent = data.model_exists ? "Ornith model ready" : "Ornith model missing";
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
