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

import copy
import json
import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # composite_ref imports this module; import it lazily at runtime.
    from .composite_ref import CompositeRef

TOKEN_ALIGNMENT_TARGET_ALIGNED = "target_aligned"
MOE_TOPK_INDICES_REF_KEY = "moe_topk_indices_ref"

_META_REQUIRED_KEYS = ("token_alignment", "num_tokens", "prompt_tokens", "num_rows")


@dataclass(frozen=True, slots=True)
class PayloadRefMeta:
    """Server-attached description of the token sequence behind a leaf ref.

    Newer servers attach this to each R3 router-replay leaf ref
    (``sequence["moe_topk_indices_ref"]["meta"]``); older servers omit it::

        {"token_alignment": "target_aligned", "num_tokens": P + R,
         "prompt_tokens": P, "num_rows": P + R - 1}

    ``P``/``R`` are the prompt/response token counts of that sampling request.

    Target alignment: row ``i`` holds the routing used when input token ``i``
    was forwarded, and that forward pass predicts token ``i + 1`` (its
    *target*). The last sampled token is never forwarded, so ``N`` tokens have
    ``N - 1`` rows. Hence the first response token (index ``P``) is the target
    of row ``P - 1``, which is :attr:`response_row_start`.

    Keys a newer server adds are kept in :attr:`extra` and written back by
    :meth:`to_payload`, so parsing never drops information.

    Raises:
        ValueError: on construction, if a count is not a non-negative int,
            ``prompt_tokens > num_tokens``, or a target-aligned meta has
            ``num_rows != num_tokens - 1``.
    """

    token_alignment: str
    num_tokens: int
    prompt_tokens: int
    num_rows: int
    extra: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.token_alignment, str) or not self.token_alignment:
            raise ValueError(
                f"meta.token_alignment must be a non-empty string, got {self.token_alignment!r}"
            )
        for key in ("num_tokens", "prompt_tokens", "num_rows"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"meta.{key} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"meta.{key} must be >= 0, got {value}")
        if self.prompt_tokens > self.num_tokens:
            raise ValueError(
                f"meta.prompt_tokens={self.prompt_tokens} exceeds "
                f"meta.num_tokens={self.num_tokens}"
            )
        if self.is_target_aligned and self.num_rows != self.num_tokens - 1:
            raise ValueError(
                f"target_aligned meta requires num_rows == num_tokens - 1, got "
                f"num_rows={self.num_rows}, num_tokens={self.num_tokens}"
            )
        if not isinstance(self.extra, dict):
            raise ValueError(f"meta.extra must be a dict, got {type(self.extra).__name__}")
        overlap = set(self.extra) & set(_META_REQUIRED_KEYS)
        if overlap:
            raise ValueError(f"meta.extra must not repeat typed keys: {sorted(overlap)}")

    @classmethod
    def from_payload(cls, payload: Any) -> PayloadRefMeta:
        """Parse and validate a wire ``meta`` object; unknown keys go to ``extra``."""

        if isinstance(payload, PayloadRefMeta):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError(f"meta must be an object, got {type(payload).__name__}")
        missing = [key for key in _META_REQUIRED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"meta is missing required keys: {missing}")
        return cls(
            token_alignment=payload["token_alignment"],
            num_tokens=payload["num_tokens"],
            prompt_tokens=payload["prompt_tokens"],
            num_rows=payload["num_rows"],
            extra={
                key: copy.deepcopy(value)
                for key, value in payload.items()
                if key not in _META_REQUIRED_KEYS
            },
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the wire ``meta`` object (typed keys plus ``extra``)."""

        return {
            "token_alignment": self.token_alignment,
            "num_tokens": self.num_tokens,
            "prompt_tokens": self.prompt_tokens,
            "num_rows": self.num_rows,
            **copy.deepcopy(self.extra),
        }

    @property
    def is_target_aligned(self) -> bool:
        return self.token_alignment == TOKEN_ALIGNMENT_TARGET_ALIGNED

    @property
    def response_tokens(self) -> int:
        """Number of response tokens: ``num_tokens - prompt_tokens``."""

        return self.num_tokens - self.prompt_tokens

    @property
    def response_row_start(self) -> int | None:
        """First row whose target is a response token, or None if not target-aligned.

        Equals ``prompt_tokens - 1``: the row of the last prompt token, whose
        forward pass produced the first response token. Rows
        ``[response_row_start, num_rows)`` are the routing that generated the
        response. With ``prompt_tokens == 0`` the first response token has no
        row and this is 0.
        """

        if not self.is_target_aligned:
            return None
        return max(self.prompt_tokens - 1, 0)


@dataclass(slots=True)
class PayloadRef:
    """Reference to a large payload stored outside the HTTP response body.

    ``meta`` is the typed server-attached :class:`PayloadRefMeta` (``None`` for
    refs from older servers); other unknown keys are kept in ``metadata``.
    """

    storage: str
    format: str
    schema: str | None = None
    size_bytes: int | None = None
    uri: str | None = None
    relative_path: str | None = None
    path: str | None = None
    dtype: str | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)
    meta: PayloadRefMeta | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PayloadRef":
        """Parse a wire leaf ref; ``payload["meta"]`` is validated into :attr:`meta`.

        Raises:
            ValueError: if ``meta`` is present but malformed, or its ``num_rows``
                disagrees with ``shape[0]``.
        """

        raw_meta = payload.get("meta")
        meta = None if raw_meta is None else PayloadRefMeta.from_payload(raw_meta)
        shape_rows = _shape_rows(payload.get("shape"))
        if meta is not None and shape_rows is not None and shape_rows != meta.num_rows:
            raise ValueError(
                f"inconsistent leaf ref: meta.num_rows={meta.num_rows} but shape[0]={shape_rows}"
            )
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
        if meta is not None:
            known.add("meta")
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
            meta=meta,
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
        if self.meta is not None:
            payload["meta"] = self.meta.to_payload()
        return payload


def parse_moe_topk_indices_ref(sequence: Mapping[str, Any]) -> PayloadRef | None:
    """Return a sample-result sequence's R3 routing ref as a typed :class:`PayloadRef`.

    ``sequence`` is one entry of ``sample(...)["sequences"]``. Returns None when
    the sequence has no ``moe_topk_indices_ref`` (R3 not requested, or the
    indices came back inline). The raw dict stays available as
    ``sequence["moe_topk_indices_ref"]``::

        ref = parse_moe_topk_indices_ref(seq)
        if ref is not None and ref.meta is not None:
            ref.meta.prompt_tokens, ref.meta.num_rows, ref.meta.response_row_start

    Raises:
        ValueError: if the ref is not an object or its ``meta`` is malformed.
    """

    raw = sequence.get(MOE_TOPK_INDICES_REF_KEY)
    if raw is None:
        return None
    if isinstance(raw, PayloadRef):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"{MOE_TOPK_INDICES_REF_KEY} must be an object, got {type(raw).__name__}")
    return PayloadRef.from_payload(raw)


class PayloadRefMaterializationError(RuntimeError):
    """Raised when a payload ref cannot be materialized by this SDK process."""


def materialize_payload_ref(
    ref: PayloadRef | CompositeRef | Mapping[str, Any],
    *,
    field: str | None = None,
) -> Any:
    """Load the content referenced by ``ref`` for explicit inspection.

    The first implementation supports local/GPFS refs because R2 RECORD stores
    top-k indices as ``torch.save`` files on a shared filesystem. S3 refs are
    intentionally represented by the same schema, but should be resolved via a
    server-backed resolver once that backend is wired in.

    Supported leaf formats are ``torch.save``, ``json`` and ``safetensors``
    (the latter returns a ``{name: tensor}`` dict, e.g. ``{"indices": ...}``).

    Composite refs (see :mod:`weaver.types.composite_ref`) materialize to a
    single tensor: each leaf is loaded (``field`` selects the tensor inside it,
    defaulting to ``"indices"`` or the leaf's only tensor), its segment rows
    are sliced, and the slices are concatenated along dim 0. Trailing dims must
    match; integer dtypes are upcast to the widest one.
    """

    from .composite_ref import CompositeRef, is_composite_ref

    if isinstance(ref, CompositeRef) or is_composite_ref(ref):
        return _materialize_composite(CompositeRef.from_payload(ref), field=field)
    if isinstance(ref, PayloadRef):
        payload_ref = ref
    else:
        payload_ref = PayloadRef.from_payload(ref)
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
    elif fmt == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise PayloadRefMaterializationError(
                "Materializing safetensors payload refs requires the 'safetensors' and "
                "'torch' packages (pip install safetensors torch)."
            ) from exc
        value = load_file(str(local_path), device="cpu")
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


def _materialize_composite(composite: CompositeRef, *, field: str | None) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional torch install
        raise PayloadRefMaterializationError(
            "Materializing composite payload refs requires torch."
        ) from exc

    segments = composite.segments
    if not segments:
        raise PayloadRefMaterializationError("Cannot materialize an empty composite ref.")

    parts: list[Any] = []
    for index, segment in enumerate(segments):
        leaf_value = materialize_payload_ref(segment.ref)
        tensor = _leaf_tensor(leaf_value, field=field, torch=torch, where=f"segment {index}")
        rows = int(tensor.shape[0]) if tensor.dim() > 0 else 0
        if segment.stop > rows:
            raise PayloadRefMaterializationError(
                f"segment {index}: rows [{segment.start}, {segment.stop}) exceed the "
                f"materialized leaf's {rows} rows"
            )
        parts.append(tensor[segment.start : segment.stop])

    trailing = tuple(parts[0].shape[1:])
    for index, part in enumerate(parts[1:], start=1):
        if tuple(part.shape[1:]) != trailing:
            raise PayloadRefMaterializationError(
                f"segment {index}: trailing dims {tuple(part.shape[1:])} do not match "
                f"segment 0 trailing dims {trailing}"
            )
    dtypes = {part.dtype for part in parts}
    if len(dtypes) > 1:
        if any(
            dtype.is_floating_point or dtype.is_complex or dtype == torch.bool for dtype in dtypes
        ):
            raise PayloadRefMaterializationError(
                f"composite segments have incompatible dtypes: {sorted(map(str, dtypes))}"
            )
        target = parts[0].dtype
        for dtype in dtypes:
            target = torch.promote_types(target, dtype)
        parts = [part.to(target) for part in parts]
    return torch.cat(parts, dim=0)


def _leaf_tensor(value: Any, *, field: str | None, torch: Any, where: str) -> Any:
    if isinstance(value, Mapping):
        if field is not None:
            key = field
        elif "indices" in value:
            key = "indices"
        elif len(value) == 1:
            key = next(iter(value))
        else:
            raise PayloadRefMaterializationError(
                f"{where}: leaf holds {sorted(value)}; pass field= to select one."
            )
        if key not in value:
            raise PayloadRefMaterializationError(f"{where}: leaf does not contain field {key!r}.")
        value = value[key]
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise PayloadRefMaterializationError(
                f"{where}: leaf content is not tensor-like ({type(value).__name__})."
            ) from exc
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


def _shape_rows(shape: Any) -> int | None:
    if isinstance(shape, list | tuple) and shape:
        first = shape[0]
        if isinstance(first, int) and not isinstance(first, bool):
            return first
    return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
