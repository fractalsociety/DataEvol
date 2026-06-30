from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class LocalModelLabeler(Protocol):
    def label(self, trace: dict[str, Any]) -> tuple[str, float, str]:
        ...


class KeywordLocalModelLabeler:
    """Deterministic local-labeler stand-in for adapter/model integration."""

    def label(self, trace: dict[str, Any]) -> tuple[str, float, str]:
        text = json.dumps(trace, sort_keys=True).lower()
        if "rescued" in text or "stronger model" in text:
            return "rescued_by_stronger_model", 0.82, "local keyword model matched rescue language"
        if "training candidate" in text:
            return "good_training_candidate", 0.78, "local keyword model matched training value"
        return "inconclusive", 0.51, "local keyword model had no stronger match"


def load_human_overrides(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in data.items()}
