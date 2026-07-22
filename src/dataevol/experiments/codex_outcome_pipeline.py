from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dataevol.datasets.codex_execution_outcomes import (
    build_capability_cell_report,
    build_cheapest_acceptable_rows,
    evaluate_rollout,
    normalize_execution_outcomes,
)


PIPELINE_SCHEMA = "dataevol.codex_outcome_pipeline.v1"


def run_outcome_pipeline(
    raw_outcomes: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    current_stage: str = "shadow",
    successful_evidence_epochs: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outcomes = normalize_execution_outcomes(raw_outcomes)
    cells = build_capability_cell_report(outcomes)
    targets = build_cheapest_acceptable_rows(outcomes)
    rollout = evaluate_rollout(cells, current_stage=current_stage, successful_evidence_epochs=successful_evidence_epochs)
    payloads = {
        "execution_outcomes.jsonl": _jsonl(outcomes),
        "cheapest_acceptable_targets.jsonl": _jsonl(targets),
        "capability_cells.json": _json(cells),
        "rollout.json": _json(rollout),
    }
    files = {}
    for name, payload in payloads.items():
        path = output / name
        _write_immutable(path, payload)
        files[name] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    body = {
        "schema": PIPELINE_SCHEMA,
        "outcome_count": len(outcomes),
        "causal_target_count": len(targets),
        "trusted_cell_count": cells["trusted_cell_count"],
        "authorized_stage": rollout["authorized_stage"],
        "maximum_classifier_coverage": rollout["maximum_classifier_coverage"],
        "requires_binding_dataevol_verdict": True,
        "requires_fractalwork_canary": rollout["authorized_stage"] != "shadow",
        "files": files,
    }
    manifest = {**body, "pipeline_hash": _hash(body)}
    _write_immutable(output / "manifest.json", _json(manifest))
    return manifest


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"outcome pipeline artifact already exists with different content: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FractalWork Codex execution evidence")
    parser.add_argument("--outcomes", required=True, type=Path, help="FractalWork execution evidence JSONL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--current-stage", default="shadow", choices=("shadow", "canary", "limited", "expanded", "broad"))
    parser.add_argument("--successful-evidence-epochs", default=0, type=int)
    args = parser.parse_args()
    raw = [json.loads(line) for line in args.outcomes.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = run_outcome_pipeline(
        raw,
        args.output,
        current_stage=args.current_stage,
        successful_evidence_epochs=args.successful_evidence_epochs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
