from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROLES = ("manager", "worker", "critic", "verifier")


def version_prompt_pack(pack: Mapping[str, str], output_dir: str | Path, *, version: str = "v1") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"prompt_pack_{version}.json"
    path.write_text(json.dumps({"version": version, "prompts": dict(pack), "created_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def generate_prompt_variants(pack: Mapping[str, str]) -> dict[str, str]:
    return {role: f"{pack.get(role, '')}\nReturn verifiable, concise output with explicit evidence." for role in ROLES}


def generate_candidate_prompt_pack(pack: Mapping[str, str]) -> dict[str, str]:
    return generate_prompt_variants(pack)


def ab_test_prompt_packs(control_metrics: Mapping[str, float], variant_metrics: Mapping[str, float]) -> dict[str, Any]:
    success_delta = float(variant_metrics.get("success_rate", 0)) - float(control_metrics.get("success_rate", 0))
    hallucination_delta = float(variant_metrics.get("hallucination_rate", 0)) - float(control_metrics.get("hallucination_rate", 0))
    return {
        "success_delta": success_delta,
        "hallucination_delta": hallucination_delta,
        "promotable": success_delta > 0 and hallucination_delta <= 0,
    }


def promote_prompt_pack(test_result: Mapping[str, Any], output_dir: str | Path) -> Path:
    if not test_result.get("promotable"):
        raise ValueError("prompt pack cannot promote without success improvement and hallucination non-regression")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prompt_pack_promotion.json"
    path.write_text(json.dumps(dict(test_result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
