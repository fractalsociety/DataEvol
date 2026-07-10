from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from dataevol.config import DataEvolConfig
from dataevol.harness.model_client import (
    FakeModelClient,
    ModelClientError,
    ModelNotConfiguredError,
    OpenRouterModelClient,
    resolve_model_client,
)


class _Handler(BaseHTTPRequestHandler):
    last_request: dict = {}

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        _Handler.last_request = {
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "content_type": self.headers.get("Content-Type"),
            "json": json.loads(body) if body else {},
        }
        payload = {
            "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
            "usage": {"total_tokens": 12},
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def http_server():
    _Handler.last_request = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://127.0.0.1:{port}/chat/completions"
    server.shutdown()
    server.server_close()


def test_openrouter_client_posts_bearer_and_parses_content(http_server):
    client = OpenRouterModelClient(http_server, "secret-key", "frontier-model", provider="openrouter")
    result = client.complete(system="be brief", user="hi", response_format="json", seed=7)
    assert result["content"] == '{"ok": true}'
    assert result["model"] == "frontier-model"
    assert result["usage"]["total_tokens"] == 12
    req = _Handler.last_request
    assert req["authorization"] == "Bearer secret-key"
    assert req["content_type"] == "application/json"
    assert req["json"]["model"] == "frontier-model"
    assert req["json"]["messages"][0]["role"] == "system"
    assert req["json"]["response_format"] == {"type": "json_object"}
    assert req["json"]["seed"] == 7


def test_openrouter_client_raises_on_http_error(http_server):
    # Point at a path the handler still answers 200 to, but force a transport
    # error by using a bad host.
    client = OpenRouterModelClient("http://127.0.0.1:1/chat/completions", "k", "m")
    with pytest.raises(ModelClientError):
        client.complete(system="x", user="y", timeout=2)


def test_resolve_model_client_raises_when_unconfigured(tmp_path: Path):
    cfg = DataEvolConfig(
        path=tmp_path / "c.toml",
        db_path=tmp_path / "db",
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="t",
    )
    with pytest.raises(ModelNotConfiguredError):
        resolve_model_client(cfg)


def test_resolve_model_client_builds_real_client(tmp_path: Path):
    cfg = DataEvolConfig(
        path=tmp_path / "c.toml",
        db_path=tmp_path / "db",
        raw_path=tmp_path / "raw",
        artifacts_path=tmp_path / "artifacts",
        api_token="t",
        model_endpoint="https://example.org/api/v1/chat/completions",
        model_api_key="key",
        model_name="frontier-model",
    )
    client = resolve_model_client(cfg)
    assert isinstance(client, OpenRouterModelClient)
    assert client.default_model == "frontier-model"


def test_fake_model_client_deterministic_and_records_calls():
    fake = FakeModelClient(default_model="fake-9b")
    a = fake.complete(system="You are the harness architect.", user="design", seed=3, model="fake-9b")
    b = fake.complete(system="You are the harness architect.", user="design", seed=3, model="fake-9b")
    assert a["content"] == b["content"]
    assert a["model"] == "fake-9b"
    assert len(fake.calls) == 2
    assert fake.calls[0]["seed"] == 3


def test_fake_model_client_scripts_override_default():
    fake = FakeModelClient(scripts=[("judge", {"verdict": "reject", "reason": "r"})])
    result = fake.complete(system="experiment judge review", user="u")
    assert json.loads(result["content"])["verdict"] == "reject"


def test_fake_model_client_judge_can_differ_from_mutator_model():
    fake = FakeModelClient(default_model="mutator-model")
    fake.complete(system="mutate the harness", user="u", model="mutator-model")
    judge = fake.complete(system="experiment judge review", user="u", model="judge-model")
    assert fake.calls[0]["model"] == "mutator-model"
    assert judge["model"] == "judge-model"
