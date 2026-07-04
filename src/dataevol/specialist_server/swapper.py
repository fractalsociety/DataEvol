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
    _flatten_params,
    _layer_parameters,
)


@dataclass
class LoadedSpecialist:
    name: str
    manifest_path: str
    manifest_hash: str
    layer_index: int
    task_type: str
    base_model_id: str
    tensor_bytes: int
    verified: bool
    loaded_at: str
    active: bool = False
    dtype_cast: bool = False


class MlxLayerSwapper:
    def __init__(self, model: Any, *, base_model_id: str, base_model_hash: str | None, num_layers: int, max_specialists: int = 8) -> None:
        self.model = model
        self.base_model_id = base_model_id
        self.base_model_hash = base_model_hash
        self.num_layers = num_layers
        self.max_specialists = max(1, int(max_specialists))
        self.registry: dict[str, LoadedSpecialist] = {}
        self.tensors: dict[str, dict[str, Any]] = {}
        self.base_snapshots: dict[int, dict[str, Any]] = {}
        self.active: str | None = None

    def register(self, manifest_path: str | Path, tensors: dict[str, Any] | None = None) -> LoadedSpecialist:
        if len(self.registry) >= self.max_specialists:
            raise ValueError("max specialists reached")
        path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self._validate_manifest(manifest)
        loaded_tensors = tensors or self._load_tensor_files(path, manifest)
        dtype_cast = self._validate_tensors(manifest, loaded_tensors)
        name = str(manifest.get("name") or path.stem.replace(".manifest", ""))
        record = LoadedSpecialist(
            name=name,
            manifest_path=str(path),
            manifest_hash=sha256_file(path),
            layer_index=int(manifest["layer_index"]),
            task_type=str(manifest["task_type"]),
            base_model_id=str(manifest["base_model_id"]),
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
        self.model.update(_tree_from_flat(self.tensors[name], self.model.parameters()))
        self._materialize()
        self.active = name
        for item in self.registry.values():
            item.active = item.name == name
        return record

    def activate_for_task(self, task_type: str) -> LoadedSpecialist | None:
        for record in self.registry.values():
            if record.task_type == task_type:
                return self.activate(record.name)
        return None

    def restore_base(self) -> None:
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
        if manifest.get("base_model_id") != self.base_model_id and manifest.get("base_model_hash") != self.base_model_hash:
            raise ValueError("base model mismatch")
        layer_index = int(manifest.get("layer_index", -1))
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError("layer index out of range")

    def _load_tensor_files(self, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        import mlx.core as mx

        tensors: dict[str, Any] = {}
        for file_name in manifest.get("tensor_files") or []:
            tensor_path = (manifest_path.parent / str(file_name)).resolve()
            expected = (manifest.get("sha256") or {}).get(str(file_name))
            if expected and sha256_file(tensor_path) != expected:
                raise ValueError(f"hash mismatch for {file_name}")
            tensors.update(mx.load(str(tensor_path)))
        return tensors

    def _validate_tensors(self, manifest: dict[str, Any], tensors: dict[str, Any]) -> bool:
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
                else:
                    offenders.append(name)
        if offenders:
            raise ValueError(f"tensor shapes/dtypes incompatible: {offenders[:5]}")
        return dtype_cast

    def _snapshot_layer(self, layer_index: int) -> None:
        if layer_index in self.base_snapshots:
            return
        import mlx.core as mx

        self.base_snapshots[layer_index] = {name: mx.array(value) for name, value in _layer_parameters(self.model, layer_index).items()}

    def _materialize(self) -> None:
        try:
            import mlx.core as mx

            mx.eval(self.model.parameters())
        except Exception:
            return


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float_dtype(value: Any) -> bool:
    return "float" in str(getattr(value, "dtype", "")) or "bfloat" in str(getattr(value, "dtype", ""))


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
