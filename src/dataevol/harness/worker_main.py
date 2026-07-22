"""JSONL subprocess worker for fixture and MLX-backed execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dataevol.harness.worker_main")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--scripted", type=Path)
    modes.add_argument("--mlx", action="store_true")
    parser.add_argument("--model")
    return parser


def _prompt(case: Mapping[str, Any]) -> str:
    for key in ("prompt", "input", "question"):
        if key in case:
            value = case[key]
            return value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"))
    return json.dumps(dict(case), sort_keys=True, separators=(",", ":"))


def _scripted_handler(path: Path) -> Callable[[Mapping[str, Any], int], tuple[str, int, int]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load scripted responses from {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("scripted response file must contain a JSON object")
    responses = loaded.get("responses", loaded)
    if not isinstance(responses, Mapping):
        raise ValueError("scripted responses must be a JSON object")

    def handle(case: Mapping[str, Any], sequence: int) -> tuple[str, int, int]:
        prompt = _prompt(case)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        keys = []
        if case.get("id") is not None:
            keys.append(str(case["id"]))
        keys.extend((digest, f"sha256:{digest}"))
        value: Any = ""
        for key in keys:
            if key in responses:
                value = responses[key]
                break
        if isinstance(value, list):
            value = value[min(sequence, len(value) - 1)] if value else ""
        sleep_s = 0.0
        if isinstance(value, Mapping):
            sleep_s = float(value.get("sleep_s", 0.0))
            value = value.get("output", "")
        if sleep_s < 0:
            raise ValueError("sleep_s must be non-negative")
        if not isinstance(value, str):
            raise ValueError("scripted output must be a string or directive object")
        if sleep_s:
            time.sleep(sleep_s)
        return value, len(prompt.split()), len(value.split())

    return handle


def _mlx_handler(model_name: str) -> Callable[[Mapping[str, Any], int], tuple[str, int, int]]:
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        print("mlx-lm is required for --mlx mode; install the 'train' optional dependency", file=sys.stderr)
        raise SystemExit(3) from exc
    model, tokenizer = load(model_name)
    sampler = make_sampler(temp=0.0)

    def handle(case: Mapping[str, Any], sequence: int) -> tuple[str, int, int]:
        del sequence
        prompt = _prompt(case)
        requested = case.get("max_tokens", 256)
        max_tokens = requested if isinstance(requested, int) and not isinstance(requested, bool) else 256
        max_tokens = max(1, max_tokens)
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return output, len(tokenizer.encode(prompt)), len(tokenizer.encode(output))

    return handle


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mlx and not args.model:
        print("--mlx requires --model", file=sys.stderr)
        return 2
    try:
        handler = _scripted_handler(args.scripted) if args.scripted else _mlx_handler(args.model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("request must be a JSON object")
            case = request.get("case")
            seed = request.get("seed")
            sequence = request.get("sequence")
            if not isinstance(case, Mapping):
                raise ValueError("request case must be an object")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("request seed must be an integer")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise ValueError("request sequence must be a non-negative integer")
            output, tokens_in, tokens_out = handler(case, sequence)
            response = {"output": output, "tokens_in": tokens_in, "tokens_out": tokens_out}
            sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"invalid request: {exc}", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
