from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataevol.local_models.layer_specialist import (
    FREEZE_STRATEGY,
    SCHEMA,
    _decoder_layers,
    _dequantize_quantized_linears,
    _flatten_params,
    _layer_parameters,
    candidate_content_hash,
)


@dataclass
class LoadedSpecialist:
    name: str
    manifest_path: str
    manifest_hash: str
    layer_index: int
    task_type: str
    base_model_id: str
    genome_id: str
    candidate_content_hash: str
    tensor_bytes: int
    verified: bool
    loaded_at: str
    active: bool = False
    dtype_cast: bool = False


class MlxLayerSwapper:
    def __init__(self, model: Any, *, base_model_id: str, base_model_hash: str | None, num_layers: int, base_model_revision: str | None = None, max_specialists: int = 8) -> None:
        self.model = model
        self.base_model_id = base_model_id
        self.base_model_hash = base_model_hash
        self.base_model_revision = base_model_revision
        self.num_layers = num_layers
        self.max_specialists = max(1, int(max_specialists))
        self.registry: dict[str, LoadedSpecialist] = {}
        self.tensors: dict[str, dict[str, Any]] = {}
        self.base_snapshots: dict[int, dict[str, Any]] = {}
        self.dequantized_layers: set[int] = set()
        self.dequantized_module_count = 0
        self.active: str | None = None
        self.active_binding: dict[str, Any] | None = None

    def register(self, manifest_path: str | Path, tensors: dict[str, Any] | None = None) -> LoadedSpecialist:
        _ensure_mlx_thread_stream()
        path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self._validate_manifest(manifest)
        name = str(manifest.get("name") or path.stem.replace(".manifest", ""))
        manifest_hash = sha256_file(path)
        existing = self.registry.get(name)
        if existing:
            if existing.manifest_hash == manifest_hash:
                return existing
            raise ValueError(f"specialist name already registered with different contents: {name}")
        if len(self.registry) >= self.max_specialists:
            raise ValueError("max specialists reached")
        self._prepare_target_layer(int(manifest["layer_index"]))
        loaded_tensors = tensors or self._load_tensor_files(path, manifest)
        dtype_cast = self._validate_tensors(manifest, loaded_tensors)
        record = LoadedSpecialist(
            name=name,
            manifest_path=str(path),
            manifest_hash=manifest_hash,
            layer_index=int(manifest["layer_index"]),
            task_type=str(manifest["task_type"]),
            base_model_id=str(manifest["base_model_id"]),
            genome_id=str(manifest["genome_id"]),
            candidate_content_hash=str(manifest["candidate_content_hash"]),
            tensor_bytes=sum(int(getattr(t, "nbytes", 0)) for t in loaded_tensors.values()),
            verified=True,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            dtype_cast=dtype_cast,
        )
        self.registry[name] = record
        self.tensors[name] = loaded_tensors
        return record

    def list(self) -> list[dict[str, Any]]:
        return [asdict(item) | {"active": item.name == self.active} for item in self.registry.values()]

    def activate(self, name: str) -> LoadedSpecialist:
        _ensure_mlx_thread_stream()
        if name not in self.registry:
            raise KeyError(f"unknown specialist: {name}")
        if self.active == name:
            record = self.registry[name]
            record.active = True
            return record
        if self.active:
            self.restore_base()
        record = self.registry[name]
        self._snapshot_layer(record.layer_index)
        try:
            self.model.update(_tree_from_flat(self.tensors[name], self.model.parameters()))
            self._materialize()
        except Exception:
            snapshot = self.base_snapshots[record.layer_index]
            self.model.update(_tree_from_flat(snapshot, self.model.parameters()))
            self._materialize()
            raise
        self.active = name
        for item in self.registry.values():
            item.active = item.name == name
        return record

    def activate_for_task(self, task_type: str) -> LoadedSpecialist | None:
        for record in self.registry.values():
            if record.task_type == task_type:
                return self.activate(record.name)
        self.restore_base()
        return None

    def restore_base(self) -> None:
        _ensure_mlx_thread_stream()
        if not self.active:
            return
        record = self.registry[self.active]
        snapshot = self.base_snapshots.get(record.layer_index)
        if snapshot:
            self.model.update(_tree_from_flat(snapshot, self.model.parameters()))
            self._materialize()
        for item in self.registry.values():
            item.active = False
        self.active = None
        self.active_binding = None

    def unload(self, name: str) -> None:
        if name not in self.registry:
            raise KeyError(f"unknown specialist: {name}")
        if self.active == name:
            self.restore_base()
        self.registry.pop(name, None)
        self.tensors.pop(name, None)

    def _validate_manifest(self, manifest: dict[str, Any]) -> None:
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported schema")
        if manifest.get("freeze_strategy") != FREEZE_STRATEGY:
            raise ValueError("unsupported freeze_strategy")
        manifest_model_id = manifest.get("base_model_id")
        manifest_model_hash = manifest.get("base_model_hash")
        if not self.base_model_hash or manifest_model_hash != self.base_model_hash:
            raise ValueError("base model fingerprint mismatch")
        if manifest_model_id != self.base_model_id:
            if not manifest_model_hash or not self.base_model_hash or manifest_model_hash != self.base_model_hash:
                raise ValueError("base model mismatch")
        if manifest.get("base_model_revision") != self.base_model_revision:
            raise ValueError("base model revision mismatch")
        genome_id = manifest.get("genome_id")
        content_hash = manifest.get("candidate_content_hash")
        if not isinstance(genome_id, str) or not genome_id:
            raise ValueError("genome_id is required")
        if not isinstance(content_hash, str) or len(content_hash) != 64:
            raise ValueError("candidate_content_hash must be a 64-character digest")
        if candidate_content_hash(manifest) != content_hash.lower():
            raise ValueError("candidate_content_hash mismatch")
        layer_index = int(manifest.get("layer_index", -1))
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError("layer index out of range")
        tensor_files = manifest.get("tensor_files")
        hashes = manifest.get("sha256")
        if not isinstance(tensor_files, list) or not tensor_files:
            raise ValueError("tensor_files must contain at least one artifact")
        if not isinstance(hashes, dict):
            raise ValueError("sha256 tensor hashes are required")
        for file_name in tensor_files:
            file_path = Path(str(file_name))
            expected = hashes.get(str(file_name))
            if file_path.is_absolute() or ".." in file_path.parts:
                raise ValueError(f"unsafe tensor file path: {file_name}")
            if not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected):
                raise ValueError(f"valid SHA-256 is required for {file_name}")

    def _load_tensor_files(self, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        import mlx.core as mx

        tensors: dict[str, Any] = {}
        for file_name in manifest.get("tensor_files") or []:
            tensor_path = (manifest_path.parent / str(file_name)).resolve()
            expected = (manifest.get("sha256") or {}).get(str(file_name))
            if sha256_file(tensor_path) != str(expected).lower():
                raise ValueError(f"hash mismatch for {file_name}")
            tensors.update(mx.load(str(tensor_path)))
        return tensors

    def _validate_tensors(self, manifest: dict[str, Any], tensors: dict[str, Any]) -> bool:
        import mlx.core as mx

        layer_params = _layer_parameters(self.model, int(manifest["layer_index"]))
        missing = sorted(set(layer_params) - set(tensors))
        extra = sorted(set(tensors) - set(layer_params))
        if missing or extra:
            raise ValueError(f"tensor names incompatible missing={missing[:5]} extra={extra[:5]}")
        dtype_cast = False
        offenders: list[str] = []
        for name, value in tensors.items():
            wanted = layer_params[name]
            if tuple(getattr(value, "shape", ())) != tuple(getattr(wanted, "shape", ())):
                offenders.append(name)
                continue
            if str(getattr(value, "dtype", "")) != str(getattr(wanted, "dtype", "")):
                if _float_dtype(value) and _float_dtype(wanted):
                    dtype_cast = True
                    value = value.astype(wanted.dtype)
                    tensors[name] = value
                else:
                    offenders.append(name)
                    continue
            finite = mx.all(mx.isfinite(value))
            mx.eval(finite)
            if not bool(finite.item()):
                offenders.append(name)
        if offenders:
            raise ValueError(f"tensor shapes/dtypes incompatible: {offenders[:5]}")
        return dtype_cast

    def _snapshot_layer(self, layer_index: int) -> None:
        if layer_index in self.base_snapshots:
            return
        import mlx.core as mx

        self.base_snapshots[layer_index] = {name: mx.array(value) for name, value in _layer_parameters(self.model, layer_index).items()}

    def _prepare_target_layer(self, layer_index: int) -> None:
        if layer_index in self.dequantized_layers:
            return
        layers = _decoder_layers(self.model)
        if layers is None:
            raise RuntimeError("loaded model does not expose decoder layers")
        count = _dequantize_quantized_linears(layers[layer_index])
        if count:
            self.dequantized_layers.add(layer_index)
            self.dequantized_module_count += count
            self._materialize()

    def _materialize(self) -> None:
        import mlx.core as mx

        mx.eval(self.model.parameters())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float_dtype(value: Any) -> bool:
    return "float" in str(getattr(value, "dtype", "")) or "bfloat" in str(getattr(value, "dtype", ""))


def _ensure_mlx_thread_stream() -> None:
    """Ensure FastAPI worker threads have an MLX stream before model operations."""
    import mlx.core as mx

    try:
        mx.default_stream(mx.default_device())
    except RuntimeError:
        mx.set_default_stream(mx.new_stream(mx.default_device()))


def _tree_from_flat(flat: dict[str, Any], template: Any) -> Any:
    root = _empty_like(template)
    for name, value in flat.items():
        _insert(root, template, name.split("."), value)
    return root


def _empty_like(template: Any) -> Any:
    if isinstance(template, list):
        return [_empty_like(item) for item in template]
    return {}


def _insert(node: Any, template: Any, parts: list[str], value: Any) -> None:
    part = parts[0]
    if isinstance(template, list):
        idx = int(part)
        if len(parts) == 1:
            node[idx] = value
        else:
            _insert(node[idx], template[idx], parts[1:], value)
        return
    if len(parts) == 1:
        node[part] = value
        return
    node.setdefault(part, _empty_like(template[part]))
    _insert(node[part], template[part], parts[1:], value)
