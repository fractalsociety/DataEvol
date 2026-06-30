from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .frozen import FrozenBenchmark, FrozenBenchmarkBuilder


BENCHMARK_TYPES = {"router", "prompt", "verifier", "critic", "compressor"}


def build_benchmark(
    benchmark_type: str,
    items: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    version: str = "v0",
    overwrite: bool = False,
) -> FrozenBenchmark:
    if benchmark_type not in BENCHMARK_TYPES:
        raise ValueError(f"unknown benchmark type: {benchmark_type}")
    rows = [_benchmark_item(benchmark_type, dict(item), index) for index, item in enumerate(items, start=1)]
    return FrozenBenchmarkBuilder().build(
        rows,
        output_dir,
        name=f"{benchmark_type}_benchmark",
        version=version,
        source=f"{benchmark_type}-builder",
        overwrite=overwrite,
    )


def _benchmark_item(benchmark_type: str, item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": item.get("id") or f"{benchmark_type}_case_{index:03d}",
        "benchmark_type": benchmark_type,
        "task": item.get("task") or item.get("objective") or item.get("prompt") or "benchmark task",
        "expected": item.get("expected") or item.get("label") or item.get("outcome") or "accepted",
        "non_regression_metrics": item.get("non_regression_metrics", ["safety_score", "verification_pass_rate"]),
    }
