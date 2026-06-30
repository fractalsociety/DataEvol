from __future__ import annotations

import re
from typing import Any

from dataevol.schemas.traces import PRIVACY_STATUS_BY_MODE

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
API_KEY_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|[A-Za-z0-9_-]{32,})\b"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization|bearer)\b\s*[:=]\s*['\"]?([^'\"\s,;]+)"
)
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")


def redact_text(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = BEARER_RE.sub("Bearer [REDACTED_TOKEN]", text)
    text = SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[REDACTED_SECRET]", text)
    text = API_KEY_RE.sub("[REDACTED_KEY]", text)
    return text


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if re.search(r"(?i)(api[_-]?key|token|secret|password|passwd|authorization)", str(key)):
                redacted[key] = "[REDACTED_SECRET]"
            else:
                redacted[key] = redact_value(item)
        return redacted
    return value


def privacy_status_for_mode(mode: str) -> str:
    return PRIVACY_STATUS_BY_MODE.get(mode, "local_only")


def can_export_publicly(privacy_status: str) -> bool:
    return privacy_status == "public_benchmark"

