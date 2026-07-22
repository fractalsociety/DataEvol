from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataevol.datasets.codex_routing import build_codex_routing_datasets, freeze_model_catalog


API_PRICING = {
    "gpt-5.6-sol": {"input": 5.0, "cached_input": 0.5, "output": 30.0},
    "gpt-5.5": {"input": 5.0, "cached_input": 0.5, "output": 30.0},
    "gpt-5.4": {"input": 2.5, "cached_input": 0.25, "output": 15.0},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.5},
}

CODEX_CREDIT_PRICING = {
    "gpt-5.6-sol": {"input": 125.0, "cached_input": 12.5, "output": 750.0},
    "gpt-5.5": {"input": 125.0, "cached_input": 12.5, "output": 750.0},
    "gpt-5.4": {"input": 62.5, "cached_input": 6.25, "output": 375.0},
    "gpt-5.4-mini": {"input": 18.75, "cached_input": 1.875, "output": 113.0},
}

CAPABILITIES = {
    "gpt-5.6-sol": ["architecture", "code", "docs", "integration", "migration", "reasoning", "research", "review", "search", "security", "tests", "verification"],
    "gpt-5.5": ["architecture", "code", "docs", "integration", "migration", "reasoning", "research", "review", "search", "security", "tests", "verification"],
    "gpt-5.4": ["architecture", "code", "docs", "integration", "migration", "reasoning", "research", "review", "search", "security", "tests", "verification"],
    "gpt-5.4-mini": ["code", "docs", "reasoning", "review", "search", "tests", "verification"],
    "gpt-5.3-codex-spark": ["code", "search", "tests"],
}

RISK_TIERS = {
    "gpt-5.6-sol": ["low", "medium", "high", "critical"],
    "gpt-5.5": ["low", "medium", "high", "critical"],
    "gpt-5.4": ["low", "medium", "high", "critical"],
    "gpt-5.4-mini": ["low", "medium"],
    "gpt-5.3-codex-spark": ["low"],
}

ARCHETYPES = (
    ("format", "Format a focused file and report the diff", "docs", "low"),
    ("lookup", "Locate a symbol and explain its callers", "search", "low"),
    ("docs", "Update concise API documentation", "docs", "low"),
    ("tests", "Add focused unit tests for an established behavior", "tests", "low"),
    ("bug", "Repair a localized parser bug with a reproduction", "code", "medium"),
    ("feature", "Implement a bounded feature across a few files", "code", "medium"),
    ("frontend", "Implement and visually verify a responsive workflow", "integration", "medium"),
    ("api", "Change an API contract and preserve compatibility", "integration", "medium"),
    ("review", "Review a cross-module change for regressions", "review", "medium"),
    ("performance", "Diagnose a performance regression and benchmark the fix", "reasoning", "medium"),
    ("architecture", "Design a cross-repository architecture and migration", "architecture", "high"),
    ("security", "Audit an authorization boundary and implement remediation", "security", "high"),
    ("migration", "Plan and execute a durable data migration with rollback", "migration", "critical"),
    ("research", "Investigate an unfamiliar technical claim and validate sources", "research", "high"),
    ("model", "Train and evaluate a specialist model with blind holdouts", "research", "high"),
    ("incident", "Diagnose a production incident across services", "architecture", "critical"),
)


def snapshot_current_catalog(cache_path: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    captured_at = str(cache.get("fetched_at") or datetime.now(timezone.utc).isoformat())
    if output_path.is_file():
        captured_at = str(json.loads(output_path.read_text(encoding="utf-8")).get("captured_at") or captured_at)
    configured = str(config.get("model") or "")
    visible = {
        str(item["slug"]): item
        for item in cache.get("models", [])
        if item.get("visibility") == "list"
    }
    model_ids = sorted(set(visible) | ({configured} if configured else set()))
    models = []
    for model_id in model_ids:
        if model_id not in CAPABILITIES:
            continue
        cached = visible.get(model_id, {})
        raw_efforts = cached.get("supported_reasoning_levels") or [
            {"effort": effort} for effort in ("low", "medium", "high", "xhigh")
        ]
        efforts = [str(item["effort"]) for item in raw_efforts if str(item.get("effort")) in {"none", "minimal", "low", "medium", "high", "xhigh"}]
        models.append(
            {
                "model_id": model_id,
                "provider": "openai-codex",
                "availability": "listed" if model_id in visible else "configured_verified",
                "revision": cache.get("fetched_at"),
                "enabled": True,
                "capabilities": CAPABILITIES[model_id],
                "supported_reasoning_efforts": efforts,
                "default_reasoning_effort": str(cached.get("default_reasoning_level") or "medium"),
                "context_window": int(cached.get("context_window") or 272_000),
                "latency_p50_ms": 0,
                "risk_tiers": RISK_TIERS[model_id],
                "pricing": {
                    "codex_credit": CODEX_CREDIT_PRICING.get(model_id),
                    "api_usd_equivalent": API_PRICING.get(model_id),
                },
                "metadata": {
                    "description": cached.get("description"),
                    "visibility": cached.get("visibility"),
                    "unpriced": model_id not in API_PRICING,
                },
            }
        )
    raw = {
        "source": "local Codex cache/config plus pinned official OpenAI pricing",
        "source_revision": "2026-07-11",
        "captured_at": captured_at,
        "configured_model_id": configured,
        "models": models,
        "metadata": {
            "cache_sha256": _sha256(cache_path),
            "config_sha256": _sha256(config_path),
            "api_pricing_source": "https://developers.openai.com/api/docs/pricing",
            "codex_pricing_source": "https://learn.chatgpt.com/docs/pricing#what-are-tokens-and-credits",
        },
    }
    return freeze_model_catalog(raw, output_path)


def build_teacher_examples(repetitions: int = 16) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    for archetype, objective, capability, risk in ARCHETYPES:
        for iteration in range(repetitions):
            task_id = f"codex-{archetype}-{iteration:03d}"
            subtasks = _subtasks(archetype, objective, capability, risk, iteration)
            chosen = _teacher_route(subtasks)
            rejected = _rejected_route(chosen, subtasks)
            task = {
                "task_id": task_id,
                "task_group": task_id,
                "task": f"{objective}. Scenario {iteration + 1}; preserve evidence and minimize unnecessary model cost.",
                "subtasks": subtasks,
                "route": chosen,
                "source": "deterministic_teacher_v1",
            }
            tasks.append(task)
            preferences.append(
                {
                    **{key: task[key] for key in ("task_id", "task_group", "task", "subtasks")},
                    "chosen_route": chosen,
                    "rejected_route": rejected,
                    "provenance": {"teacher": "deterministic_teacher_v1", "preference": "smallest_sufficient_verified_route"},
                }
            )
    return tasks, preferences


def prepare(output_dir: Path, cache_path: Path, config_path: Path, repetitions: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = snapshot_current_catalog(cache_path, config_path, output_dir / "codex_model_catalog.json")
    tasks, preferences = build_teacher_examples(repetitions)
    _write_jsonl(output_dir / "teacher_tasks.jsonl", tasks)
    _write_jsonl(output_dir / "teacher_preferences.jsonl", preferences)
    result = build_codex_routing_datasets(catalog, tasks, preferences, output_dir / "datasets")
    summary = {
        "schema": "dataevol.codex_router_training_plan.v1",
        "catalog_hash": catalog["catalog_hash"],
        "configured_model_id": catalog.get("configured_model_id"),
        "task_rows": len(tasks),
        "preference_rows": len(preferences),
        "dataset_content_hash": result.dataset_content_hash,
        "dataset_manifest": str(result.manifest_path),
        "teacher_policy": "smallest sufficient model and effort; verified performance remains the promotion gate",
    }
    (output_dir / "training_plan.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _subtasks(archetype: str, objective: str, capability: str, risk: str, iteration: int) -> list[dict[str, Any]]:
    estimated = 1_000 + (iteration % 8) * 1_500
    if risk in {"high", "critical"}:
        return [
            {"subtask_id": "inspect", "objective": f"Inspect evidence for {objective.lower()}", "required_capabilities": ["reasoning", "search"], "risk": "medium", "estimated_input_tokens": estimated},
            {"subtask_id": "execute", "objective": objective, "depends_on": ["inspect"], "required_capabilities": [capability], "risk": risk, "estimated_input_tokens": estimated * 2},
            {"subtask_id": "verify", "objective": "Verify correctness, safety, and rollback evidence", "depends_on": ["execute"], "required_capabilities": ["verification"], "risk": risk, "estimated_input_tokens": estimated},
        ]
    if archetype in {"feature", "frontend", "api", "performance", "review", "bug"}:
        return [
            {"subtask_id": "inspect", "objective": "Inspect the relevant implementation and constraints", "required_capabilities": ["search"], "risk": "low", "estimated_input_tokens": estimated},
            {"subtask_id": "execute", "objective": objective, "depends_on": ["inspect"], "required_capabilities": [capability], "risk": risk, "estimated_input_tokens": estimated * 2},
            {"subtask_id": "verify", "objective": "Run focused verification and review the result", "depends_on": ["execute"], "required_capabilities": ["verification"], "risk": risk, "estimated_input_tokens": estimated},
        ]
    return [{"subtask_id": "execute", "objective": objective, "required_capabilities": [capability], "risk": risk, "estimated_input_tokens": estimated}]


def _teacher_route(subtasks: list[dict[str, Any]]) -> dict[str, Any]:
    assignments = []
    for subtask in subtasks:
        risk = subtask["risk"]
        capability = subtask["required_capabilities"][0]
        if risk in {"high", "critical"}:
            model, effort = "gpt-5.6-sol", "xhigh" if risk == "critical" else "high"
        elif risk == "medium" or capability in {"architecture", "integration", "migration", "research", "security"}:
            model, effort = "gpt-5.4", "medium"
        else:
            model = "gpt-5.4-mini"
            effort = "medium" if capability in {"code", "tests", "verification"} else "low"
        assignments.append({"subtask_id": subtask["subtask_id"], "model_id": model, "reasoning_effort": effort, "reason": "smallest sufficient route"})
    return {"assignments": assignments, "fallback_model_id": "gpt-5.6-sol", "requires_verification": any(row["risk"] != "low" for row in subtasks)}


def _rejected_route(chosen: dict[str, Any], subtasks: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = []
    risks = {row["subtask_id"]: row["risk"] for row in subtasks}
    for assignment in chosen["assignments"]:
        if assignment["model_id"] == "gpt-5.4-mini":
            model, effort, reason = "gpt-5.4", "high", "unnecessarily expensive"
        elif assignment["model_id"] == "gpt-5.4":
            model, effort, reason = "gpt-5.6-sol", "high", "unnecessarily expensive"
        else:
            model, effort, reason = "gpt-5.4", "low" if risks[assignment["subtask_id"]] == "high" else "medium", "insufficient reasoning margin"
        rejected.append({"subtask_id": assignment["subtask_id"], "model_id": model, "reasoning_effort": effort, "reason": reason})
    return {"assignments": rejected, "fallback_model_id": "gpt-5.6-sol", "requires_verification": chosen["requires_verification"]}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Codex router SFT/DPO datasets")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models-cache", type=Path, default=Path.home() / ".codex/models_cache.json")
    parser.add_argument("--codex-config", type=Path, default=Path.home() / ".codex/config.toml")
    parser.add_argument("--repetitions", type=int, default=16)
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    print(json.dumps(prepare(args.output, args.models_cache, args.codex_config, args.repetitions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
