"""Model client for the harness specialists.

The five specialists (architect, benchmark builder, failure analyst, mutator,
judge) all require a live model. The only production backend is
OpenRouterModelClient (an OpenAI/OpenRouter-compatible HTTP client over urllib,
mirroring integrations/clients.py). There is deliberately NO rules-based
production fallback. Tests inject FakeModelClient.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request


class ModelNotConfiguredError(ValueError):
    """Raised when no model endpoint/key/name is available."""


class ModelClientError(RuntimeError):
    """Raised on transport/HTTP/parse failures talking to the model."""


class ModelClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        response_format: str = "json",  # "json" | "text"
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        seed: int | None = None,
        timeout: int = 60,
    ) -> Mapping[str, Any]:
        """Return {"content": str, "model": str, "usage": {...}}."""
        ...


def _extract_content(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("error"), Mapping):
        err = payload["error"]
        raise ModelClientError(str(err.get("message") or err))
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping) and message.get("content") is not None:
                return str(message["content"])
            if first.get("text") is not None:
                return str(first["text"])
    if payload.get("content") is not None:
        return str(payload["content"])
    raise ModelClientError(f"unexpected model response shape: {str(payload)[:300]}")


class OpenRouterModelClient:
    """OpenAI/OpenRouter-compatible chat completion client over stdlib urllib."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        default_model: str,
        *,
        provider: str = "openrouter",
        timeout: int = 60,
    ) -> None:
        if not endpoint:
            raise ModelNotConfiguredError("model endpoint is required")
        if not api_key:
            raise ModelNotConfiguredError("model api_key is required")
        if not default_model:
            raise ModelNotConfiguredError("model name is required")
        self.endpoint = endpoint
        self.api_key = api_key
        self.default_model = default_model
        self.provider = provider
        self.timeout = timeout

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_format: str = "json",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        seed: int | None = None,
        timeout: int = 60,
    ) -> Mapping[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if seed is not None:
            body["seed"] = seed
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/fractal/dataevol"
            headers["X-Title"] = "DataEvol Harness Evolver"
        req = request.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout or self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500] if hasattr(exc, "read") else ""
            raise ModelClientError(f"model HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise ModelClientError(f"model request failed: {exc.reason}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelClientError(f"model returned non-JSON: {raw[:300]}") from exc
        content = _extract_content(payload)
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        return {"content": content, "model": body["model"], "usage": dict(usage)}


class FakeModelClient:
    """Deterministic in-process client for tests.

    Call with a ``responder`` callable ``(system, user, **kw) -> Mapping`` or a
    list of ``(system_substring, response)`` scripts (first match wins). Default
    behavior returns deterministic JSON keyed off the system prompt so it can
    stand in for all five specialists at once. Honors ``model`` and ``seed`` and
    records every call in ``calls``.
    """

    def __init__(
        self,
        responder: Callable[..., Mapping[str, Any]] | None = None,
        *,
        scripts: list[tuple[str, Mapping[str, Any] | str]] | None = None,
        default_model: str = "fake-model",
    ) -> None:
        self._responder = responder
        self._scripts = scripts or []
        self.default_model = default_model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_format: str = "json",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        seed: int | None = None,
        timeout: int = 60,
    ) -> Mapping[str, Any]:
        used_model = model or self.default_model
        self.calls.append({
            "system": system,
            "user": user,
            "response_format": response_format,
            "model": used_model,
            "seed": seed,
        })
        for needle, response in self._scripts:
            if needle in system:
                return self._normalize(response, used_model)
        if self._responder is not None:
            return self._normalize(self._responder(system=system, user=user, model=used_model, seed=seed), used_model)
        return self._normalize(self._default(system, user), used_model)

    @staticmethod
    def _normalize(response: Any, model: str) -> Mapping[str, Any]:
        if isinstance(response, str):
            return {"content": response, "model": model, "usage": {"total_tokens": 0}}
        if isinstance(response, Mapping):
            content = response.get("content")
            if content is None:
                content = json.dumps(dict(response), sort_keys=True)
            return {
                "content": content,
                "model": str(response.get("model") or model),
                "usage": dict(response.get("usage") or {"total_tokens": 0}),
            }
        # list or scalar: JSON-encode as the model's textual content
        return {"content": json.dumps(response, sort_keys=True), "model": model, "usage": {"total_tokens": 0}}

    def _default(self, system: str, user: str) -> Mapping[str, Any]:
        s = (system or "").lower()
        if "architect" in s:
            content = json.dumps({
                "task_type": "permit_set_review",
                "router": {"model": "local-9b-router", "confidence_threshold": 0.5},
                "agents": [
                    {"role": "worker", "model": "local-7b", "prompt_ref": "prompts/worker.md", "tools": []},
                ],
                "workflow": [{"step_id": "do", "agent_role": "worker"}],
                "memory": {"type": "none"},
                "recovery": {"max_retries": 0, "retry_on": []},
                "output": {"schema": {}, "validation": "strict"},
            })
        elif "judge" in s:
            content = json.dumps({"verdict": "promotable", "reason": "candidate is reliably better"})
        elif "failure" in s or "analyst" in s:
            content = json.dumps({"failures": [{"category": "VERIFICATION_FAILURE", "earliest_cause": "no verifier"}]})
        elif "mutat" in s:
            content = json.dumps({
                "hypothesis": "Add an independent verifier.",
                "mutation": {"mode": "component", "target": "agents", "description": "add verifier"},
            })
        elif "benchmark" in s:
            content = json.dumps([
                {"id": "n1", "category": "normal"},
                {"id": "a1", "category": "adversarial"},
            ])
        else:
            content = json.dumps({"ok": True})
        return {"content": content, "usage": {"total_tokens": 0}}


def resolve_model_client(config: Any) -> ModelClient:
    """Build the production model client from config (endpoint/key/name).

    Raises ModelNotConfiguredError if any required field is missing. No fallback.
    """
    endpoint = getattr(config, "model_endpoint", "") or ""
    api_key = getattr(config, "model_api_key", "") or ""
    default_model = getattr(config, "model_name", "") or ""
    provider = getattr(config, "model_provider", "") or "openrouter"
    missing = [name for name, value in (("endpoint", endpoint), ("api_key", api_key), ("name", default_model)) if not value]
    if missing:
        raise ModelNotConfiguredError(
            "Harness Evolver requires a live model. Missing: "
            + ", ".join(missing)
            + ". Set [model] in dataevol.toml (endpoint/api_key/name) or env "
              "(DATAEVOL_MODEL_ENDPOINT / OPENROUTER_API_KEY / DATAEVOL_MODEL_NAME)."
        )
    return OpenRouterModelClient(endpoint, api_key, default_model, provider=provider or "openrouter")


# urllib_parse imported to keep the module importable even if unused downstream.
_ = urllib_parse
