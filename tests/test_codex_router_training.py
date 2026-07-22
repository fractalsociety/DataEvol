from __future__ import annotations

import json
from pathlib import Path

from dataevol.experiments.codex_router_training import snapshot_current_catalog


def test_catalog_snapshot_is_repeatable(tmp_path: Path) -> None:
    cache = tmp_path / "models.json"
    config = tmp_path / "config.toml"
    output = tmp_path / "catalog.json"
    cache.write_text(json.dumps({
        "fetched_at": "2026-07-11T12:09:28Z",
        "models": [{
            "slug": "gpt-5.4-mini",
            "visibility": "list",
            "context_window": 272000,
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}],
        }],
    }), encoding="utf-8")
    config.write_text('model = "gpt-5.4-mini"\n', encoding="utf-8")

    first = snapshot_current_catalog(cache, config, output)
    second = snapshot_current_catalog(cache, config, output)

    assert first == second
    assert first["captured_at"] == "2026-07-11T12:09:28Z"
    assert first["configured_model_id"] == "gpt-5.4-mini"
