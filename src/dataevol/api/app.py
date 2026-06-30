from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from dataevol import __version__
from dataevol.compat import call_core
from dataevol.config import DataEvolConfig, load_config


class OperationRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestTraceRequest(BaseModel):
    trace: dict[str, Any]
    source_system: str | None = None


class IngestRunRequest(BaseModel):
    run: dict[str, Any] = Field(default_factory=dict)
    source_system: str | None = None


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

    @app.post("/ingest_trace", dependencies=[protected])
    def ingest_trace(request: IngestTraceRequest) -> dict[str, Any]:
        return call_core("ingest", "ingest_trace", request.model_dump(), config=cfg)

    @app.post("/ingest_run", dependencies=[protected])
    def ingest_run(request: IngestRunRequest) -> dict[str, Any]:
        return call_core("ingest", "ingest_run", request.model_dump(), config=cfg)

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

    return app


app = create_app()
