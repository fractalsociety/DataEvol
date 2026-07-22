from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dataevol.local_models.remote_dataset import materialize_layer_dataset


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.offset = 0
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        return None

    def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class _Opener:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests = []

    def open(self, request, timeout):  # noqa: ANN001
        self.requests.append((request.full_url, timeout))
        return _Response(self.body)


def test_remote_dataset_is_allowlisted_verified_redacted_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'{"prompt":"a","completion":"b"}\n'
    digest = hashlib.sha256(body).hexdigest()
    opener = _Opener(body)
    monkeypatch.setenv("LAYER_SPECIALIST_DATASET_ALLOWED_HOSTS", "objects.example.test")
    monkeypatch.setattr("dataevol.local_models.remote_dataset.socket.getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))])
    monkeypatch.setattr("dataevol.local_models.remote_dataset.build_opener", lambda *handlers: opener)

    result = materialize_layer_dataset(
        "https://objects.example.test/signed/train.jsonl?X-Amz-Signature=secret",
        expected_sha256=digest,
        artifacts_root=tmp_path,
    )

    assert result.path.read_bytes() == body
    assert result.sha256 == digest
    assert result.source_uri == "https://objects.example.test/signed/train.jsonl"
    assert "secret" not in result.source_uri
    assert result.path.parent.name == digest
    second = materialize_layer_dataset(
        "https://objects.example.test/other?token=changed",
        expected_sha256=digest,
        artifacts_root=tmp_path,
    )
    assert second.path == result.path
    assert len(opener.requests) == 1


def test_remote_dataset_rejects_missing_hash_unlisted_and_private_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAYER_SPECIALIST_DATASET_ALLOWED_HOSTS", "allowed.example")
    with pytest.raises(ValueError, match="dataset_sha256"):
        materialize_layer_dataset("https://allowed.example/data", expected_sha256=None, artifacts_root=tmp_path)
    with pytest.raises(ValueError, match="not allowlisted"):
        materialize_layer_dataset("https://other.example/data", expected_sha256="a" * 64, artifacts_root=tmp_path)

    monkeypatch.setattr("dataevol.local_models.remote_dataset.socket.getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="non-public"):
        materialize_layer_dataset("https://allowed.example/data", expected_sha256="a" * 64, artifacts_root=tmp_path)


def test_local_dataset_retains_path_and_optionally_verifies_hash(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"a"}\n', encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    result = materialize_layer_dataset(str(dataset), expected_sha256=digest, artifacts_root=tmp_path / "artifacts")
    assert result.path == dataset.resolve()
    assert result.remote is False
    with pytest.raises(ValueError, match="mismatch"):
        materialize_layer_dataset(str(dataset), expected_sha256="0" * 64, artifacts_root=tmp_path / "artifacts")
