from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


class FrozenBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenBenchmark:
    benchmark_path: Path
    manifest_path: Path
    item_count: int
    sha256: str


class FrozenBenchmarkBuilder:
    def build(
        self,
        items: Iterable[Mapping[str, Any]],
        output_dir: str | Path,
        *,
        name: str = "router_policy_benchmark",
        version: str = "v0",
        source: str = "prior-runs-or-fixtures",
        overwrite: bool = False,
    ) -> FrozenBenchmark:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        benchmark_path = out / f"{name}.jsonl"
        manifest_path = out / f"{name}.manifest.json"
        if manifest_path.exists() and not overwrite:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("frozen") is True:
                raise FrozenBenchmarkError(f"benchmark is frozen and cannot be overwritten: {manifest_path}")

        rows = [dict(item) for item in items]
        with benchmark_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        sha = _sha256_file(benchmark_path)
        manifest = {
            "name": name,
            "benchmark_type": "router_policy",
            "version": version,
            "frozen": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path": str(benchmark_path),
            "item_count": len(rows),
            "sha256": sha,
            "source": source,
            "immutability": "writes are rejected while frozen=true unless overwrite=True",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return FrozenBenchmark(benchmark_path, manifest_path, len(rows), sha)

    def assert_immutable(self, manifest_path: str | Path) -> None:
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("frozen") is not True:
            raise FrozenBenchmarkError("benchmark manifest is not frozen")
        current = _sha256_file(Path(manifest["path"]))
        if current != manifest.get("sha256"):
            raise FrozenBenchmarkError("frozen benchmark content hash changed")


def build_frozen_benchmark(items: Iterable[Mapping[str, Any]], output_dir: str | Path, **kwargs: Any) -> FrozenBenchmark:
    return FrozenBenchmarkBuilder().build(items, output_dir, **kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
