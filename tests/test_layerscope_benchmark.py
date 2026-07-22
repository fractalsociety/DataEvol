from __future__ import annotations

import json
import hashlib
import stat
from pathlib import Path

import pytest

from dataevol.experiments.layerscope_benchmark import (
    _blind_assignment,
    _generation_prompt,
    _parse_exact_verdict,
    _score_summary,
    compare_outputs,
    prepare_dialogsum,
    score_blind_judgments,
)


def test_generation_prompt_matches_training_chat_boundary() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "route this"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<user>route this</user><assistant>"

    assert _generation_prompt(Tokenizer(), "route this") == "<user>route this</user><assistant>"


def test_generation_prompt_supports_plain_tokenizers() -> None:
    assert _generation_prompt(object(), "route this") == "route this"


def test_prepare_dialogsum_freezes_disjoint_blind_partition(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "dialogsum.train.jsonl", [
        {"fname": f"train_{index}", "dialogue": f"dialogue {index}", "summary": f"summary {index}"}
        for index in range(8)
    ])
    _write(source / "dialogsum.dev.jsonl", [
        {"fname": "dev_0", "dialogue": "dev dialogue", "summary": "dev summary"}
    ])
    _write(source / "dialogsum.test.jsonl", [
        {
            "fname": "test_0",
            "dialogue": "blind dialogue",
            "summary1": "first reference",
            "summary2": "second reference",
            "summary3": "third reference",
        }
    ])

    manifest = prepare_dialogsum(
        source,
        tmp_path / "benchmark",
        selection_count=2,
        train_count=4,
        preference_count=2,
    )

    blind = _rows(tmp_path / "benchmark/blind_inputs.jsonl")
    references = _rows(tmp_path / "benchmark/blind_references.sealed.jsonl")
    assert set(blind[0]) == {"id", "prompt"}
    assert references[0]["references"] == ["first reference", "second reference", "third reference"]
    assert manifest["partitions"]["blind_inputs"]["sha256"] != manifest["partitions"]["blind_references"]["sha256"]
    assert stat.S_IMODE((tmp_path / "benchmark/blind_references.sealed.jsonl").stat().st_mode) == 0o400


def test_compare_outputs_uses_paired_scores_and_separate_blind_key(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    references = tmp_path / "references.jsonl"
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(inputs, [{"id": "a", "prompt": "prompt a"}, {"id": "b", "prompt": "prompt b"}])
    _write(references, [
        {"id": "a", "references": ["the exact concise summary"]},
        {"id": "b", "references": ["another correct summary"]},
    ])
    _write_generation(base, "base", "a" * 64, [
        {"id": "a", "output": "unrelated output", "latency_ms": 10, "prompt_sha256": _hash("prompt a")},
        {"id": "b", "output": "wrong", "latency_ms": 10, "prompt_sha256": _hash("prompt b")},
    ])
    _write_generation(candidate, "candidate", "b" * 64, [
        {"id": "a", "output": "the exact concise summary", "latency_ms": 12, "prompt_sha256": _hash("prompt a")},
        {"id": "b", "output": "another correct summary", "latency_ms": 12, "prompt_sha256": _hash("prompt b")},
    ])
    key = tmp_path / "blinding.key"
    key.write_bytes(b"blind-key" * 8)

    result = compare_outputs(
        {"base": base, "candidate": candidate},
        inputs,
        references,
        tmp_path / "comparison",
        blinding_key_file=key,
        bootstrap_samples=100,
    )

    delta = result["pairwise"]["candidate-minus-base"]
    assert delta["mean_delta_rougeL"] > 0
    assert delta["wins"] == 2
    blind = _rows(tmp_path / "comparison/blind_pairs.jsonl")
    assert "base" not in json.dumps(blind)
    assert "example_id" not in blind[0]
    assert not (tmp_path / "comparison/blind_pairs.key.jsonl").exists()


def test_rouge_like_metrics_reward_exact_reference() -> None:
    exact = _score_summary("a concise useful summary", ["a concise useful summary"])
    unrelated = _score_summary("different words entirely", ["a concise useful summary"])
    assert exact["rouge1"] == 1.0
    assert exact["rouge2"] == 1.0
    assert exact["rougeL"] == 1.0
    assert unrelated["rougeL"] == 0.0


def test_blind_judge_accepts_only_an_exact_verdict() -> None:
    assert _parse_exact_verdict("A") == "A"
    assert _parse_exact_verdict(" tie.\n") == "TIE"
    assert _parse_exact_verdict("A because it is more accurate") == "INVALID"
    assert _parse_exact_verdict("A\nB\nTIE") == "INVALID"


def test_score_judgments_requires_exact_pair_coverage(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    key = tmp_path / "key"
    key_bytes = b"k" * 32
    _write(inputs, [{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "two"}])
    pair_id, _, _ = _blind_assignment("a", "base", "candidate", key_bytes, 1701)
    _write(judgments, [{"pair_id": pair_id, "verdict": "A"}])
    key.write_bytes(key_bytes)

    with pytest.raises(ValueError, match="exactly cover"):
        score_blind_judgments(
            variants={"base": tmp_path / "base.jsonl", "candidate": tmp_path / "candidate.jsonl"},
            input_path=inputs,
            judgment_path=judgments,
            blinding_key_file=key,
        )


def test_compare_fails_closed_when_candidate_omits_a_benchmark_row(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    references = tmp_path / "references.jsonl"
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(inputs, [{"id": "a", "prompt": "prompt a"}, {"id": "b", "prompt": "prompt b"}])
    _write(references, [
        {"id": "a", "references": ["good"]},
        {"id": "b", "references": ["hard"]},
    ])
    _write_generation(base, "base", "a" * 64, [
        {"id": "a", "output": "bad", "latency_ms": 1, "prompt_sha256": _hash("prompt a")},
        {"id": "b", "output": "bad", "latency_ms": 1, "prompt_sha256": _hash("prompt b")},
    ])
    _write_generation(candidate, "candidate", "b" * 64, [
        {"id": "a", "output": "good", "latency_ms": 1, "prompt_sha256": _hash("prompt a")},
    ])
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)

    with pytest.raises(ValueError, match="exactly cover"):
        compare_outputs(
            {"base": base, "candidate": candidate},
            inputs,
            references,
            tmp_path / "comparison",
            blinding_key_file=key,
            bootstrap_samples=100,
        )


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_generation(path: Path, variant: str, candidate_hash: str, rows: list[dict]) -> None:
    _write(path, rows)
    output_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
        "schema": "dataevol.layerscope_generation_run.v1",
        "status": "completed",
        "run_config": {
            "variant": variant,
            "candidate_hash": candidate_hash,
            "input_rows_hash": "f" * 64,
        },
        "output_sha256": output_hash,
    }), encoding="utf-8")
