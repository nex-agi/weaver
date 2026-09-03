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

"""Public, non-sensitive types for Weaver-managed datasets."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, overload

import torch

from .tensor import TensorData

WEAVER_REDACTED_TOKEN_ID = -8
MAX_DATUM_ID_LENGTH = 255
MAX_DATASET_NAME_LENGTH = 160
MAX_DATASET_VERSION_LENGTH = 128
MAX_SAMPLE_REF_LENGTH_REQUEST_ITEMS = 4096


def _required_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _datum_id(value: Any) -> str:
    normalized = _required_name(value, "datum_id")
    if len(normalized) > MAX_DATUM_ID_LENGTH:
        raise ValueError(f"datum_id must be at most {MAX_DATUM_ID_LENGTH} characters")
    return normalized


def _dataset_path_segment(value: Any, field_name: str, max_length: int) -> str:
    """Validate a managed-dataset identifier as one URL path segment."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc
    if len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must be a single safe path segment")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _dataset_name(value: Any, field_name: str = "dataset") -> str:
    return _dataset_path_segment(value, field_name, MAX_DATASET_NAME_LENGTH)


def _dataset_version(value: Any) -> str:
    return _dataset_path_segment(value, "version", MAX_DATASET_VERSION_LENGTH)


_TOKEN_IDENTITY_FIELDS = frozenset(
    {
        "token",
        "token_id",
        "tokens",
        "token_ids",
        "input_token",
        "input_id",
        "input_tokens",
        "input_ids",
        "input_token_ids",
        "output_token",
        "output_id",
        "output_tokens",
        "output_ids",
        "output_token_ids",
        "target_token",
        "target_id",
        "target_tokens",
        "target_ids",
        "target_token_ids",
        "prompt_token",
        "prompt_id",
        "prompt_tokens",
        "prompt_ids",
        "prompt_token_ids",
        "completion_token",
        "completion_tokens",
        "completion_token_ids",
        "generated_token",
        "generated_id",
        "generated_tokens",
        "generated_ids",
        "generated_token_ids",
        "sampled_token",
        "sampled_tokens",
        "sampled_token_ids",
        "label",
        "labels",
        "top_k_tokens",
        "topk_tokens",
        "top_k_token_ids",
        "topk_token_ids",
    }
)

_MANAGED_ALIGNED_OUTPUT_FIELDS = frozenset(
    {
        "logprobs",
        "elementwise_loss",
        "teacher_logprobs",
        "detached_kl_advantages",
        "token_losses",
        "per_token_kl",
    }
)
_MANAGED_OUTPUT_DTYPES = frozenset({"float16", "float32", "float64", "bfloat16", "int32", "int64"})
_MANAGED_TOKEN_OUTPUT_DTYPES = frozenset({"int32", "int64"})


def _is_token_identity_field(field_name: str) -> bool:
    """Return whether a field can reveal token identities or vocabulary logits.

    Names such as ``per_token_kl`` and ``token_losses`` describe safe aligned
    numeric values and are intentionally not rejected merely for containing
    the word "token".
    """

    normalized = _canonical_output_field(field_name)
    return normalized in _TOKEN_IDENTITY_FIELDS or normalized.endswith(
        ("_token_ids", "_tokens", "_labels", "_token", "_label")
    )


def _is_forbidden_managed_output_field(field_name: str) -> bool:
    """Reject sensitive fields that are not aligned token-identity arrays.

    The server drops these fields and the trainer rejects them. Seeing one in a
    public managed response therefore means the service violated the wire
    contract; silently promoting a numeric ``*_ids`` array would expose an
    identifier sequence whose semantics the SDK cannot prove safe.
    """

    normalized = _canonical_output_field(field_name)
    if (
        "logit" in normalized
        or "top_k" in normalized
        or "topk" in normalized
        or "text" in normalized
        or "message" in normalized
    ):
        return True
    return (
        normalized in {"content", "id", "model_input"}
        or normalized.endswith("_id")
        or normalized.endswith("_ids")
    )


def _canonical_output_field(field_name: str) -> str:
    return field_name.strip().lower().replace("-", "_")


def _is_token_count_field(field_name: str) -> bool:
    normalized = _canonical_output_field(field_name)
    if normalized in {
        "tokens",
        "prompt_tokens",
        "training_tokens",
        "generated_tokens",
        "completion_tokens",
        "total_tokens",
        "token_count",
        "tokens_count",
    }:
        return True
    return normalized.endswith(("_token_count", "_tokens_count"))


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return normalized


def _positive_int(value: Any, field_name: str) -> int:
    normalized = _nonnegative_int(value, field_name)
    if normalized == 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


@dataclass(frozen=True, slots=True)
class SampleRef:
    """A public reference to one immutable managed-dataset sample."""

    dataset: str
    version: str
    sample_idx: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _dataset_name(self.dataset))
        object.__setattr__(self, "version", _dataset_version(self.version))
        object.__setattr__(self, "sample_idx", _nonnegative_int(self.sample_idx, "sample_idx"))

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "sample_idx": self.sample_idx,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SampleRef:
        return cls(
            dataset=_dataset_name(payload.get("dataset")),
            version=_dataset_version(payload.get("version")),
            sample_idx=_nonnegative_int(payload.get("sample_idx"), "sample_idx"),
        )


@dataclass(frozen=True, slots=True)
class SampleRefLength:
    """Model-bound effective input length for one :class:`SampleRef`."""

    sample_ref: SampleRef
    input_token_count: int
    model_data_revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_token_count",
            _positive_int(self.input_token_count, "input_token_count"),
        )
        if self.model_data_revision is not None:
            if (
                not isinstance(self.model_data_revision, str)
                or not self.model_data_revision.strip()
                or self.model_data_revision != self.model_data_revision.strip()
            ):
                raise ValueError(
                    "model_data_revision must be a non-empty string without boundary "
                    "whitespace, or null"
                )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SampleRefLength:
        return cls(
            sample_ref=SampleRef.from_payload(payload),
            input_token_count=_positive_int(payload.get("input_token_count"), "input_token_count"),
            model_data_revision=payload.get("model_data_revision"),
        )


def parse_sample_ref_lengths(requested: Sequence[SampleRef], payload: Any) -> list[SampleRefLength]:
    """Parse an order-preserving model-bound length response."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
        raise ValueError("sample length response must contain an items array")
    model_data_revision = payload.get("model_data_revision")
    if model_data_revision is not None and (
        not isinstance(model_data_revision, str)
        or not model_data_revision.strip()
        or model_data_revision != model_data_revision.strip()
    ):
        raise ValueError(
            "model_data_revision must be a non-empty string without boundary whitespace, or null"
        )
    raw_items = payload["items"]
    if len(raw_items) != len(requested):
        raise ValueError(f"Expected {len(requested)} sample lengths, got {len(raw_items)}")

    resolved: list[SampleRefLength] = []
    known_counts: dict[SampleRef, int] = {}
    for index, (reference, item) in enumerate(zip(requested, raw_items, strict=False)):
        if not isinstance(item, Mapping):
            raise ValueError(f"sample length item {index} must be an object")
        length = SampleRefLength.from_payload({**item, "model_data_revision": model_data_revision})
        if length.sample_ref != reference:
            raise ValueError(f"sample length item {index} does not match request order")
        previous = known_counts.setdefault(reference, length.input_token_count)
        if previous != length.input_token_count:
            raise ValueError("duplicate SampleRef entries returned inconsistent lengths")
        resolved.append(length)
    return resolved


@dataclass(frozen=True, slots=True)
class ManagedDatasetInfo:
    """Safe metadata for one authorized managed-dataset version."""

    name: str
    version: str
    description: str
    sample_count: int
    recommended_ratio: float | None
    compatible_models: tuple[str, ...]
    status: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ManagedDatasetInfo:
        raw_models = payload.get("compatible_models", [])
        if not isinstance(raw_models, list) or not all(
            isinstance(model, str) for model in raw_models
        ):
            raise ValueError("compatible_models must be a list of strings")
        raw_ratio = payload.get("recommended_ratio")
        if raw_ratio is not None and (
            isinstance(raw_ratio, bool) or not isinstance(raw_ratio, Real)
        ):
            raise ValueError("recommended_ratio must be numeric or null")
        return cls(
            name=_dataset_name(payload.get("name"), "name"),
            version=_dataset_version(payload.get("version")),
            description=str(payload.get("description") or ""),
            sample_count=_nonnegative_int(payload.get("sample_count"), "sample_count"),
            recommended_ratio=float(raw_ratio) if raw_ratio is not None else None,
            compatible_models=tuple(raw_models),
            status=str(payload.get("status") or ""),
        )


@dataclass(frozen=True, slots=True)
class ManagedDatasetPage(Sequence[ManagedDatasetInfo]):
    """One explicit page of authorized managed-dataset versions."""

    items: tuple[ManagedDatasetInfo, ...]
    limit: int
    offset: int
    total_count: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total_count

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[ManagedDatasetInfo]:
        return iter(self.items)

    @overload
    def __getitem__(self, index: int) -> ManagedDatasetInfo: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ManagedDatasetInfo, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ManagedDatasetInfo | tuple[ManagedDatasetInfo, ...]:
        return self.items[index]

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, requested_limit: int, requested_offset: int
    ) -> ManagedDatasetPage:
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not all(
            isinstance(item, Mapping) for item in raw_items
        ):
            raise ValueError("managed dataset list response must contain an items array")
        pagination = payload.get("pagination")
        pagination = pagination if isinstance(pagination, Mapping) else {}
        return cls(
            items=tuple(ManagedDatasetInfo.from_payload(item) for item in raw_items),
            limit=_nonnegative_int(pagination.get("limit", requested_limit), "pagination.limit"),
            offset=_nonnegative_int(
                pagination.get("offset", requested_offset), "pagination.offset"
            ),
            total_count=_nonnegative_int(
                pagination.get("total_count", len(raw_items)), "pagination.total_count"
            ),
        )


def _one_dimensional_values(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, TensorData):
        value = value.data
    elif isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{field_name} must be one-dimensional")
        value = value.detach().cpu().tolist()
    elif isinstance(value, Mapping):
        value = value.get("data")
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{field_name} must be a one-dimensional array")
    values = list(value)
    if any(
        isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray)
        for item in values
    ):
        raise ValueError(f"{field_name} must be one-dimensional")
    return values


def _managed_output_values(
    value: Any, field_name: str, *, token_identity: bool = False
) -> list[Any]:
    """Parse one audited inline managed-output vector shape."""

    allowed_dtypes = _MANAGED_TOKEN_OUTPUT_DTYPES if token_identity else _MANAGED_OUTPUT_DTYPES
    if isinstance(value, TensorData):
        if value.dtype not in allowed_dtypes:
            raise ValueError(f"{field_name} has an invalid managed output dtype")
    elif isinstance(value, Mapping):
        if set(value) != {"data", "dtype", "shape"}:
            raise ValueError(f"{field_name} has invalid managed tensor fields")
        dtype = value.get("dtype")
        if not isinstance(dtype, str) or dtype not in allowed_dtypes:
            raise ValueError(f"{field_name} has an invalid managed output dtype")
        data = value.get("data")
        shape = value.get("shape")
        if not isinstance(data, list) or not isinstance(shape, list) or len(shape) != 1:
            raise ValueError(f"{field_name} must be an exact one-dimensional tensor")
        dimension = shape[0]
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or int(dimension) != len(data)
        ):
            raise ValueError(f"{field_name} must be an exact one-dimensional tensor")
    return _one_dimensional_values(value, field_name)


def _redacted_token_values(value: Any, field_name: str, expected: int) -> tuple[int, ...]:
    values = _managed_output_values(value, field_name, token_identity=True)
    if len(values) != expected:
        raise ValueError(f"{field_name} length must equal input_token_count")
    if any(
        isinstance(token, bool)
        or not isinstance(token, Integral)
        or int(token) != WEAVER_REDACTED_TOKEN_ID
        for token in values
    ):
        raise ValueError(f"managed {field_name} may contain only the -8 sentinel")
    return tuple(int(token) for token in values)


def _finite_float(value: Any, field_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric values") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must contain only finite numeric values")
    return normalized


@dataclass(frozen=True, slots=True)
class SampleRefOutput:
    """Validated, position-aligned output for a managed sample."""

    datum_id: str
    sample_ref: SampleRef
    input_token_count: int
    target_tokens: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None
    elementwise_loss: tuple[float, ...] | None = None
    redacted_token_outputs: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    derived_outputs: Mapping[str, float | tuple[float, ...]] = field(default_factory=dict)

    @property
    def is_redacted(self) -> bool:
        return True

    def get_derived_output(self, name: str) -> float | tuple[float, ...] | None:
        normalized = _canonical_output_field(name)
        if normalized == "logprobs":
            return self.logprobs
        if normalized == "elementwise_loss":
            return self.elementwise_loss
        return self.derived_outputs.get(normalized)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SampleRefOutput:
        if payload.get("kind") != "sample_ref_output":
            raise ValueError("managed output kind must be 'sample_ref_output'")
        datum_id = _datum_id(payload.get("datum_id"))
        raw_ref = payload.get("sample_ref")
        if not isinstance(raw_ref, Mapping):
            raise ValueError("managed output must echo sample_ref")
        input_token_count = _positive_int(payload.get("input_token_count"), "input_token_count")

        target_tokens: tuple[int, ...] | None = None
        if "target_tokens" in payload:
            target_tokens = _redacted_token_values(
                payload["target_tokens"], "target_tokens", input_token_count
            )

        reserved = {
            "kind",
            "datum_id",
            "sample_ref",
            "input_token_count",
            "target_tokens",
        }
        aligned_outputs: dict[str, tuple[float, ...]] = {}
        redacted_token_outputs: dict[str, tuple[int, ...]] = {}
        derived_outputs: dict[str, float | tuple[float, ...]] = {}
        for field_name, raw_value in payload.items():
            if field_name in reserved:
                continue
            if not isinstance(field_name, str):
                raise ValueError("managed output field names must be strings")
            normalized = _canonical_output_field(field_name)
            if (
                _is_token_count_field(field_name)
                and isinstance(raw_value, Real)
                and not isinstance(raw_value, bool)
            ):
                if normalized in derived_outputs:
                    raise ValueError(f"duplicate managed output field {normalized}")
                derived_outputs[normalized] = _finite_float(raw_value, field_name)
                continue
            if _is_token_identity_field(field_name):
                redacted_token_outputs[field_name] = _redacted_token_values(
                    raw_value, field_name, input_token_count
                )
                continue
            if _is_forbidden_managed_output_field(field_name):
                raise ValueError(f"{field_name} is forbidden in a managed output")
            if normalized in _MANAGED_ALIGNED_OUTPUT_FIELDS:
                if normalized in aligned_outputs:
                    raise ValueError(f"duplicate managed output field {normalized}")
                values = _managed_output_values(raw_value, field_name)
                if len(values) != input_token_count:
                    raise ValueError(f"{field_name} length must equal input_token_count")
                aligned_outputs[normalized] = tuple(
                    _finite_float(value, field_name) for value in values
                )
                continue
            raise ValueError(f"unsupported managed output field {field_name}")

        for field_name, aligned_values in aligned_outputs.items():
            if field_name not in {"logprobs", "elementwise_loss"}:
                derived_outputs[field_name] = aligned_values

        return cls(
            datum_id=datum_id,
            sample_ref=SampleRef.from_payload(raw_ref),
            input_token_count=input_token_count,
            target_tokens=target_tokens,
            logprobs=aligned_outputs.get("logprobs"),
            elementwise_loss=aligned_outputs.get("elementwise_loss"),
            redacted_token_outputs=redacted_token_outputs,
            derived_outputs=derived_outputs,
        )
