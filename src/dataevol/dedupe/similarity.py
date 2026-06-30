from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def token_set(text: str | None) -> set[str]:
    return {part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or "")).split() if len(part) > 2}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def prompt_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return jaccard(token_set(str(left.get("prompt") or "")), token_set(str(right.get("prompt") or "")))


def response_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return jaccard(token_set(str(left.get("response") or "")), token_set(str(right.get("response") or "")))


def task_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_task = left.get("task_id") or left.get("objective") or left.get("task") or left.get("prompt")
    right_task = right.get("task_id") or right.get("objective") or right.get("task") or right.get("prompt")
    return jaccard(token_set(str(left_task or "")), token_set(str(right_task or "")))


def near_duplicate_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return round(
        (prompt_similarity(left, right) * 0.45)
        + (response_similarity(left, right) * 0.25)
        + (task_similarity(left, right) * 0.30),
        4,
    )


def find_near_duplicates(
    traces: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 0.82,
) -> list[dict[str, Any]]:
    rows = [dict(trace) for trace in traces]
    matches: list[dict[str, Any]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            score = near_duplicate_score(left, right)
            if score >= threshold:
                matches.append(
                    {
                        "left_id": left.get("id") or left.get("trace_id") or i,
                        "right_id": right.get("id") or right.get("trace_id"),
                        "score": score,
                        "prompt_similarity": prompt_similarity(left, right),
                        "response_similarity": response_similarity(left, right),
                        "task_similarity": task_similarity(left, right),
                    }
                )
    return matches
