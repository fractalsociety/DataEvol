from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class MaterializedDataset:
    path: Path
    sha256: str
    source_uri: str
    remote: bool


def materialize_layer_dataset(
    raw_uri: str,
    *,
    expected_sha256: str | None,
    artifacts_root: str | Path,
) -> MaterializedDataset:
    parsed = urlparse(raw_uri)
    if parsed.scheme in {"http", "https"}:
        digest = _required_sha256(expected_sha256)
        allowed_hosts = _host_patterns(os.environ.get("LAYER_SPECIALIST_DATASET_ALLOWED_HOSTS", ""))
        if not allowed_hosts:
            raise ValueError("remote dataset hosts are disabled; set LAYER_SPECIALIST_DATASET_ALLOWED_HOSTS")
        private_hosts = _host_patterns(os.environ.get("LAYER_SPECIALIST_DATASET_ALLOWED_PRIVATE_HOSTS", ""))
        _validate_remote_url(raw_uri, allowed_hosts=allowed_hosts, private_hosts=private_hosts)
        cache_dir = Path(artifacts_root).resolve() / "layerscope_dataset_cache" / digest
        destination = cache_dir / "dataset.jsonl"
        if destination.is_file() and sha256_file(destination) == digest:
            return MaterializedDataset(destination, digest, _redacted_url(raw_uri), True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _download_verified(
            raw_uri,
            destination,
            expected_sha256=digest,
            allowed_hosts=allowed_hosts,
            private_hosts=private_hosts,
        )
        return MaterializedDataset(destination, digest, _redacted_url(raw_uri), True)

    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"unsupported dataset uri scheme: {parsed.scheme}")
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise ValueError("file dataset URI must refer to the local host")
    path = Path(parsed.path if parsed.scheme == "file" else raw_uri).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != _required_sha256(expected_sha256):
        raise ValueError("local dataset SHA-256 mismatch")
    return MaterializedDataset(path, digest, str(path), False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    allowed_hosts: tuple[str, ...],
    private_hosts: tuple[str, ...],
) -> None:
    max_bytes = _positive_int_env("LAYER_SPECIALIST_DATASET_MAX_BYTES", DEFAULT_MAX_BYTES)
    timeout = _positive_float_env("LAYER_SPECIALIST_DATASET_TIMEOUT_S", DEFAULT_TIMEOUT_SECONDS)
    opener = build_opener(_ValidatedRedirectHandler(allowed_hosts, private_hosts))
    request = Request(url, headers={"User-Agent": "dataevol-layer-dataset/1", "Accept": "application/jsonl, application/x-ndjson, application/octet-stream"})
    temp_path: Path | None = None
    try:
        with opener.open(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("remote layer dataset exceeds size limit")
            with tempfile.NamedTemporaryFile(prefix="dataset-", suffix=".tmp", dir=destination.parent, delete=False) as handle:
                temp_path = Path(handle.name)
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = response.read(min(1024 * 1024, max_bytes + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("remote layer dataset exceeds size limit")
                    digest.update(chunk)
                    handle.write(chunk)
        if total == 0:
            raise ValueError("remote layer dataset is empty")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("remote layer dataset SHA-256 mismatch")
        _validate_jsonl(temp_path)
        os.chmod(temp_path, 0o600)
        temp_path.replace(destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: tuple[str, ...], private_hosts: tuple[str, ...]) -> None:
        self.allowed_hosts = allowed_hosts
        self.private_hosts = private_hosts
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        target = urljoin(req.full_url, newurl)
        _validate_remote_url(target, allowed_hosts=self.allowed_hosts, private_hosts=self.private_hosts)
        return super().redirect_request(req, fp, code, msg, headers, target)


def _validate_remote_url(url: str, *, allowed_hosts: tuple[str, ...], private_hosts: tuple[str, ...]) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("remote dataset URI must be HTTP(S)")
    if parsed.username or parsed.password:
        raise ValueError("remote dataset URI must not contain userinfo")
    hostname = parsed.hostname.rstrip(".").lower()
    if not _host_allowed(hostname, allowed_hosts):
        raise ValueError(f"remote dataset host is not allowlisted: {hostname}")
    allow_private = _host_allowed(hostname, private_hosts, wildcards=False)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError(f"remote dataset host cannot be resolved: {hostname}") from exc
    if not addresses:
        raise ValueError(f"remote dataset host cannot be resolved: {hostname}")
    for raw_address in addresses:
        address = ipaddress.ip_address(raw_address.split("%", 1)[0])
        if not address.is_global and not allow_private:
            raise ValueError(f"remote dataset host resolves to a non-public address: {hostname}")


def _validate_jsonl(path: Path) -> None:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"remote layer dataset has invalid JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"remote layer dataset line {line_number} must be an object")
            count += 1
    if count == 0:
        raise ValueError("remote layer dataset contains no JSON objects")


def _redacted_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _required_sha256(value: str | None) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("dataset_sha256 must be a 64-character hexadecimal digest for remote datasets")
    return digest


def _host_patterns(raw: str) -> tuple[str, ...]:
    return tuple(item.strip().lower().rstrip(".") for item in raw.split(",") if item.strip())


def _host_allowed(hostname: str, patterns: Iterable[str], *, wildcards: bool = True) -> bool:
    for pattern in patterns:
        if hostname == pattern:
            return True
        if wildcards and pattern.startswith("*.") and hostname.endswith(pattern[1:]) and hostname != pattern[2:]:
            return True
    return False


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
