# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic large-payload reference helpers.

Payload refs are an SDK-facing envelope for values that are too large or too
expensive to move inline through Weaver HTTP responses. The storage location is
treated as implementation detail by normal SDK flows; callers only need these
helpers when they explicitly want to inspect the referenced bytes/content.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class PayloadRef:
    """Reference to a large payload stored outside the HTTP response body."""

    storage: str
    format: str
    schema: str | None = None
    size_bytes: int | None = None
    uri: str | None = None
    relative_path: str | None = None
    path: str | None = None
    dtype: str | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PayloadRef":
        known = {
            "storage",
            "format",
            "schema",
            "size_bytes",
            "uri",
            "relative_path",
            "path",
            "dtype",
        }
        metadata = {key: value for key, value in payload.items() if key not in known}
        return cls(
            storage=str(payload.get("storage", "")),
            format=str(payload.get("format", "")),
            schema=_optional_str(payload.get("schema")),
            size_bytes=_optional_int(payload.get("size_bytes")),
            uri=_optional_str(payload.get("uri")),
            relative_path=_optional_str(payload.get("relative_path")),
            path=_optional_str(payload.get("path")),
            dtype=_optional_str(payload.get("dtype")),
            metadata=metadata,
        )

    def to_payload(self, *, include_private: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "storage": self.storage,
            "format": self.format,
        }
        if self.schema is not None:
            payload["schema"] = self.schema
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.uri is not None:
            payload["uri"] = self.uri
        if self.relative_path is not None:
            payload["relative_path"] = self.relative_path
        if include_private and self.path is not None:
            payload["path"] = self.path
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        payload.update(self.metadata)
        return payload


class PayloadRefMaterializationError(RuntimeError):
    """Raised when a payload ref cannot be materialized by this SDK process."""


def materialize_payload_ref(
    ref: PayloadRef | Mapping[str, Any],
    *,
    field: str | None = None,
) -> Any:
    """Load the content referenced by ``ref`` for explicit inspection.

    The first implementation supports local/GPFS refs because R2 RECORD stores
    top-k indices as ``torch.save`` files on a shared filesystem. S3 refs are
    intentionally represented by the same schema, but should be resolved via a
    server-backed resolver once that backend is wired in.
    """

    payload_ref = ref if isinstance(ref, PayloadRef) else PayloadRef.from_payload(ref)
    storage = payload_ref.storage.lower()
    if storage not in {"gpfs", "filesystem", "local"}:
        raise PayloadRefMaterializationError(
            f"Cannot materialize storage={payload_ref.storage!r} locally."
        )

    local_path = _resolve_local_ref_path(payload_ref)
    if local_path is None:
        raise PayloadRefMaterializationError("Payload ref does not include a readable path.")
    if not local_path.exists():
        raise PayloadRefMaterializationError(f"Payload ref path does not exist: {local_path}")

    fmt = payload_ref.format.lower()
    if fmt == "torch.save":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional torch install
            raise PayloadRefMaterializationError(
                "Materializing torch.save payload refs requires torch."
            ) from exc
        value = torch.load(local_path, map_location="cpu", weights_only=False)
    elif fmt == "json":
        value = json.loads(local_path.read_text(encoding="utf-8"))
    else:
        raise PayloadRefMaterializationError(
            f"Unsupported payload ref format for local materialization: {payload_ref.format!r}"
        )

    if field is not None:
        if not isinstance(value, Mapping) or field not in value:
            raise PayloadRefMaterializationError(
                f"Materialized payload does not contain field {field!r}."
            )
        return value[field]
    return value


def _resolve_local_ref_path(payload_ref: PayloadRef) -> Path | None:
    """Resolve local/GPFS refs without requiring callers to handle storage paths."""

    legacy_path = _existing_path(payload_ref.path)
    if legacy_path is not None:
        return legacy_path
    if payload_ref.uri and payload_ref.uri.startswith("weaver://"):
        resolved = _resolve_weaver_uri(payload_ref.uri)
        if resolved is not None:
            return resolved
    if payload_ref.relative_path:
        return _resolve_relative_ref_path(payload_ref.relative_path)
    if payload_ref.path:
        return Path(payload_ref.path)
    return None


def _existing_path(path: str | None) -> Path | None:
    if not path:
        return None
    resolved = Path(path)
    return resolved if resolved.exists() else None


def _resolve_relative_ref_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    for env_key in ("WEAVER_PAYLOAD_REF_ROOT", "WEAVER_ROUTER_REPLAY_REF_ROOT"):
        root = os.environ.get(env_key)
        if root:
            return Path(root) / relative
    return relative


def _resolve_weaver_uri(uri: str) -> Path | None:
    relative = uri.removeprefix("weaver://").lstrip("/")
    if not relative:
        return None
    for env_key in ("WEAVER_PAYLOAD_REF_ROOT", "WEAVER_ROUTER_REPLAY_REF_ROOT"):
        root = os.environ.get(env_key)
        if root:
            return Path(root) / relative
    return Path(relative)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
