from __future__ import annotations

import os
import secrets
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("dataevol.toml")
DEFAULT_API_TOKEN = "dev-local-token"
VALID_PRIVACY_MODES = {
    "private-local-only",
    "shared-anonymous-learning",
    "public-benchmark-contribution",
}


@dataclass(frozen=True)
class DataEvolConfig:
    path: Path
    db_path: Path
    raw_path: Path
    artifacts_path: Path
    api_token: str
    privacy_mode: str = "private-local-only"
    # Harness Evolver model backend (live model required for the 5 specialists).
    # Empty by default; resolve_model_client raises ModelNotConfiguredError if unset.
    model_provider: str = ""
    model_endpoint: str = ""
    model_api_key: str = ""
    model_name: str = ""
    judge_model_name: str = ""


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    value = config_path or os.environ.get("DATAEVOL_CONFIG") or DEFAULT_CONFIG_PATH
    return Path(value)


def default_config(path: str | Path | None = None) -> DataEvolConfig:
    config_path = resolve_config_path(path)
    return DataEvolConfig(
        path=config_path,
        db_path=Path(".dataevol/dataevol.sqlite3"),
        raw_path=Path(".dataevol/raw"),
        artifacts_path=Path(".dataevol/artifacts"),
        api_token=os.environ.get("DATAEVOL_API_TOKEN", DEFAULT_API_TOKEN),
        model_provider=os.environ.get("DATAEVOL_MODEL_PROVIDER", ""),
        model_endpoint=os.environ.get("DATAEVOL_MODEL_ENDPOINT", ""),
        model_api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("DATAEVOL_MODEL_API_KEY", ""),
        model_name=os.environ.get("DATAEVOL_MODEL_NAME", ""),
        judge_model_name=os.environ.get("DATAEVOL_JUDGE_MODEL_NAME", ""),
    )


def _read_table(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_config(config_path: str | Path | None = None) -> DataEvolConfig:
    path = resolve_config_path(config_path)
    base = default_config(path)
    table = _read_table(path)
    paths = table.get("paths", {})
    api = table.get("api", {})
    privacy = table.get("privacy", {})
    model = table.get("model", {})
    mode = str(privacy.get("mode", base.privacy_mode))
    if mode not in VALID_PRIVACY_MODES:
        raise ValueError(
            f"Invalid privacy mode {mode!r}. Expected one of: "
            f"{', '.join(sorted(VALID_PRIVACY_MODES))}"
        )
    return DataEvolConfig(
        path=path,
        db_path=_resolve_path(paths.get("db", base.db_path), path.parent),
        raw_path=_resolve_path(paths.get("raw", base.raw_path), path.parent),
        artifacts_path=_resolve_path(paths.get("artifacts", base.artifacts_path), path.parent),
        api_token=str(api.get("token") or base.api_token),
        privacy_mode=mode,
        model_provider=str(model.get("provider") or base.model_provider),
        model_endpoint=str(model.get("endpoint") or base.model_endpoint),
        model_api_key=str(model.get("api_key") or base.model_api_key),
        model_name=str(model.get("name") or base.model_name),
        judge_model_name=str(model.get("judge_name") or base.judge_model_name),
    )


def config_text(
    token: str | None = None,
    privacy_mode: str = "private-local-only",
    *,
    db_path: str | Path = ".dataevol/dataevol.sqlite3",
    raw_path: str | Path = ".dataevol/raw",
    artifacts_path: str | Path = ".dataevol/artifacts",
    model_endpoint: str = "",
    model_name: str = "",
    model_api_key: str = "",
    model_provider: str = "",
    judge_model_name: str = "",
) -> str:
    api_token = token or secrets.token_urlsafe(24)
    parts = [
        "[paths]\n"
        f'db = "{db_path}"\n'
        f'raw = "{raw_path}"\n'
        f'artifacts = "{artifacts_path}"\n'
        "\n"
        "[api]\n"
        f'token = "{api_token}"\n'
        "\n"
        "[privacy]\n"
        f'mode = "{privacy_mode}"\n'
    ]
    # The [model] section is only emitted when a harness backend is configured.
    if model_endpoint or model_name:
        provider = model_provider or "openrouter"
        parts.append(
            "\n[model]\n"
            f'provider = "{provider}"\n'
            f'endpoint = "{model_endpoint}"\n'
            f'name = "{model_name}"\n'
        )
        if judge_model_name:
            parts.append(f'judge_name = "{judge_model_name}"\n')
    return "".join(parts)


def ensure_project_dirs(config: DataEvolConfig) -> None:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    config.raw_path.mkdir(parents=True, exist_ok=True)
    config.artifacts_path.mkdir(parents=True, exist_ok=True)
