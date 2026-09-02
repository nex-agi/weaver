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

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Iterator, Mapping, Sequence, overload

import torch

from .tensor import TensorData

WEAVER_REDACTED_TOKEN_ID = -8
MAX_DATUM_ID_LENGTH = 255


def _required_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _datum_id(value: Any) -> str:
    normalized = _required_name(value, "datum_id")
    if len(normalized) > MAX_DATUM_ID_LENGTH:
        raise ValueError(f"datum_id must be at most {MAX_DATUM_ID_LENGTH} characters")
    return normalized


_TOKEN_IDENTITY_FIELDS = frozenset(
    {
        "tokens",
        "token_ids",
        "input_tokens",
        "input_token_ids",
        "output_tokens",
        "output_token_ids",
        "target_tokens",
        "target_token_ids",
        "prompt_tokens",
        "prompt_token_ids",
        "completion_tokens",
        "completion_token_ids",
        "generated_tokens",
        "generated_token_ids",
        "sampled_tokens",
        "sampled_token_ids",
        "model_input",
        "messages",
        "raw_messages",
        "text",
        "raw_text",
        "logits",
        "full_logits",
        "top_k",
        "topk",
        "top_k_logits",
        "topk_logits",
        "top_k_tokens",
        "topk_tokens",
        "top_k_token_ids",
        "topk_token_ids",
    }
)


def _is_token_identity_field(field_name: str) -> bool:
    """Return whether a field can reveal token identities or vocabulary logits.

    Names such as ``per_token_kl`` and ``token_losses`` describe safe aligned
    numeric values and are intentionally not rejected merely for containing
    the word "token".
    """

    normalized = field_name.strip().lower().replace("-", "_")
    return normalized in _TOKEN_IDENTITY_FIELDS or normalized.endswith(
        ("_token_ids", "_tokens", "_labels")
    )


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
        object.__setattr__(self, "dataset", _required_name(self.dataset, "dataset"))
        object.__setattr__(self, "version", _required_name(self.version, "version"))
        object.__setattr__(self, "sample_idx", _nonnegative_int(self.sample_idx, "sample_idx"))

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "sample_idx": self.sample_idx,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SampleRef":
        return cls(
            dataset=_required_name(payload.get("dataset"), "dataset"),
            version=_required_name(payload.get("version"), "version"),
            sample_idx=_nonnegative_int(payload.get("sample_idx"), "sample_idx"),
        )


@dataclass(frozen=True, slots=True)
class SampleRefLength:
    """Model-bound effective input length for one :class:`SampleRef`."""

    sample_ref: SampleRef
    input_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_token_count",
            _positive_int(self.input_token_count, "input_token_count"),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SampleRefLength":
        return cls(
            sample_ref=SampleRef.from_payload(payload),
            input_token_count=_positive_int(payload.get("input_token_count"), "input_token_count"),
        )


def parse_sample_ref_lengths(requested: Sequence[SampleRef], payload: Any) -> list[SampleRefLength]:
    """Parse an order-preserving model-bound length response."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
        raise ValueError("sample length response must contain an items array")
    raw_items = payload["items"]
    if len(raw_items) != len(requested):
        raise ValueError(f"Expected {len(requested)} sample lengths, got {len(raw_items)}")

    resolved: list[SampleRefLength] = []
    known_counts: dict[SampleRef, int] = {}
    for index, (reference, item) in enumerate(zip(requested, raw_items)):
        if not isinstance(item, Mapping):
            raise ValueError(f"sample length item {index} must be an object")
        length = SampleRefLength.from_payload(item)
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
    def from_payload(cls, payload: Mapping[str, Any]) -> "ManagedDatasetInfo":
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
            name=_required_name(payload.get("name"), "name"),
            version=_required_name(payload.get("version"), "version"),
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
    ) -> "ManagedDatasetPage":
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
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a one-dimensional array")
    values = list(value)
    if any(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
        for item in values
    ):
        raise ValueError(f"{field_name} must be one-dimensional")
    return values


@dataclass(frozen=True, slots=True)
class SampleRefOutput:
    """Validated, position-aligned output for a managed sample."""

    datum_id: str
    sample_ref: SampleRef
    input_token_count: int
    target_tokens: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None
    elementwise_loss: tuple[float, ...] | None = None
    derived_outputs: Mapping[str, float | tuple[float, ...]] = field(default_factory=dict)

    @property
    def is_redacted(self) -> bool:
        return True

    def get_derived_output(self, name: str) -> float | tuple[float, ...] | None:
        if name == "logprobs":
            return self.logprobs
        if name == "elementwise_loss":
            return self.elementwise_loss
        return self.derived_outputs.get(name)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SampleRefOutput":
        if payload.get("kind") != "sample_ref_output":
            raise ValueError("managed output kind must be 'sample_ref_output'")
        datum_id = _datum_id(payload.get("datum_id"))
        raw_ref = payload.get("sample_ref")
        if not isinstance(raw_ref, Mapping):
            raise ValueError("managed output must echo sample_ref")
        input_token_count = _positive_int(payload.get("input_token_count"), "input_token_count")

        target_tokens: tuple[int, ...] | None = None
        if payload.get("target_tokens") is not None:
            raw_targets = _one_dimensional_values(payload["target_tokens"], "target_tokens")
            if len(raw_targets) != input_token_count:
                raise ValueError("target_tokens length must equal input_token_count")
            if any(
                isinstance(token, bool)
                or not isinstance(token, Integral)
                or int(token) != WEAVER_REDACTED_TOKEN_ID
                for token in raw_targets
            ):
                raise ValueError("managed target_tokens may contain only the -8 sentinel")
            target_tokens = tuple(int(token) for token in raw_targets)

        aligned: dict[str, tuple[float, ...] | None] = {}
        for field_name in ("logprobs", "elementwise_loss"):
            raw_value = payload.get(field_name)
            if raw_value is None:
                aligned[field_name] = None
                continue
            values = _one_dimensional_values(raw_value, field_name)
            if len(values) != input_token_count:
                raise ValueError(f"{field_name} length must equal input_token_count")
            try:
                aligned[field_name] = tuple(float(value) for value in values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must contain numeric values") from exc

        reserved = {
            "kind",
            "datum_id",
            "sample_ref",
            "input_token_count",
            "target_tokens",
            "logprobs",
            "elementwise_loss",
        }
        derived_outputs: dict[str, float | tuple[float, ...]] = {}
        for field_name, raw_value in payload.items():
            if field_name in reserved:
                continue
            if _is_token_identity_field(field_name):
                raise ValueError(
                    f"managed output contains forbidden token-bearing field {field_name!r}"
                )
            if isinstance(raw_value, Real) and not isinstance(raw_value, bool):
                derived_outputs[field_name] = float(raw_value)
                continue
            try:
                values = _one_dimensional_values(raw_value, field_name)
                numeric_values = tuple(float(value) for value in values)
            except (TypeError, ValueError):
                # Typed managed results expose only safe numeric derived values.
                # Unknown non-numeric control metadata remains in the legacy raw
                # result but is not promoted into this public typed view.
                continue
            if len(numeric_values) != input_token_count:
                raise ValueError(f"{field_name} length must equal input_token_count")
            derived_outputs[field_name] = numeric_values

        return cls(
            datum_id=datum_id,
            sample_ref=SampleRef.from_payload(raw_ref),
            input_token_count=input_token_count,
            target_tokens=target_tokens,
            logprobs=aligned["logprobs"],
            elementwise_loss=aligned["elementwise_loss"],
            derived_outputs=derived_outputs,
        )
