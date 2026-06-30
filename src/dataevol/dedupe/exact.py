from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from dataevol.schemas.traces import CanonicalTrace


def normalized_text(trace: CanonicalTrace) -> str:
    parts = [
        trace.trace_type,
        trace.task_id or "",
        trace.provider or "",
        trace.model or "",
        trace.objective or "",
        trace.prompt or "",
        trace.response or "",
        json.dumps(trace.tool_calls, sort_keys=True, separators=(",", ":")),
        json.dumps(trace.files_changed, sort_keys=True, separators=(",", ":")),
        json.dumps(trace.tests_run, sort_keys=True, separators=(",", ":")),
    ]
    return re.sub(r"\s+", " ", "\n".join(parts)).strip().lower()


def canonical_hash(trace: CanonicalTrace) -> str:
    return hashlib.sha256(normalized_text(trace).encode("utf-8")).hexdigest()


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

