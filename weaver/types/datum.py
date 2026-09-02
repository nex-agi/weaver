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

"""Training datum representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence
from uuid import uuid4

import torch

from .managed_dataset import WEAVER_REDACTED_TOKEN_ID, SampleRef, _datum_id
from .model_input import ModelInput
from .tensor import TensorData, tensor_payload

_SAMPLE_REF_PROTECTED_INPUTS = frozenset({"target_tokens", "loss_mask", "weights", "sampling_mask"})


@dataclass(slots=True)
class Datum:
    model_input: ModelInput | None = None
    loss_fn_inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_ref: SampleRef | None = None
    datum_id: str | None = None

    def __post_init__(self) -> None:
        if (self.model_input is None) == (self.sample_ref is None):
            raise ValueError("Datum requires exactly one of model_input or sample_ref")
        if self.sample_ref is not None and self.datum_id is None:
            self.datum_id = f"d-{uuid4().hex}"
        if self.datum_id is not None:
            self.datum_id = _datum_id(self.datum_id)

        normalized: Dict[str, Any] = {}
        for key, value in self.loss_fn_inputs.items():
            # Handle TensorData objects (from tensor_payload)
            if isinstance(value, TensorData):
                normalized[key] = value.to_tensor()
            elif isinstance(value, torch.Tensor):
                normalized[key] = value
            elif _is_jagged_sequence(value):
                # Some loss inputs, such as sampling_mask, are jagged JSON payloads
                # rather than dense tensors and must be preserved as-is.
                normalized[key] = value
            else:
                normalized[key] = torch.as_tensor(value)
        self.loss_fn_inputs = normalized

        if self.sample_ref is not None:
            forbidden = set(normalized) & _SAMPLE_REF_PROTECTED_INPUTS
            if forbidden:
                rendered = ", ".join(sorted(forbidden))
                raise ValueError(
                    f"sample-ref Datum cannot provide server-owned loss inputs: {rendered}"
                )
        else:
            self._validate_token_targets()

    @property
    def kind(self) -> str:
        return "sample_ref" if self.sample_ref is not None else "token"

    @property
    def is_sample_ref(self) -> bool:
        return self.sample_ref is not None

    def to_payload(self) -> dict[str, object]:
        if self.sample_ref is not None:
            validate_sample_ref_loss_inputs([self], "")
        else:
            self._validate_token_targets()
        common: dict[str, object] = {
            "loss_fn_inputs": {
                name: (
                    values.to_dict()
                    if isinstance(values, TensorData)
                    else (
                        tensor_payload(values).to_dict()
                        if isinstance(values, torch.Tensor)
                        else values
                    )
                )
                for name, values in self.loss_fn_inputs.items()
            },
            **({"metadata": dict(self.metadata)} if self.metadata else {}),
        }
        if self.sample_ref is not None:
            if self.datum_id is None:  # guarded by from_sample_ref; direct construction is strict
                raise ValueError("sample-ref Datum requires datum_id")
            return {
                "kind": "sample_ref",
                "datum_id": self.datum_id,
                **self.sample_ref.to_payload(),
                **common,
            }

        assert self.model_input is not None
        payload: dict[str, object] = {"model_input": self.model_input.to_payload(), **common}
        if self.datum_id is not None:
            payload["datum_id"] = self.datum_id
        return payload

    @classmethod
    def from_raw(
        cls,
        *,
        model_input: ModelInput,
        loss_fn_inputs: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        datum_id: str | None = None,
    ) -> "Datum":
        return cls(
            model_input=model_input,
            loss_fn_inputs=dict(loss_fn_inputs),  # type: ignore[arg-type]
            metadata=dict(metadata or {}),
            datum_id=datum_id,
        )

    @classmethod
    def from_sample_ref(
        cls,
        *,
        dataset: str,
        version: str,
        sample_idx: int,
        loss_fn_inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        datum_id: str | None = None,
    ) -> "Datum":
        """Create one occurrence of an opaque managed-dataset sample."""

        return cls(
            sample_ref=SampleRef(dataset=dataset, version=version, sample_idx=sample_idx),
            datum_id=f"d-{uuid4().hex}" if datum_id is None else datum_id,
            loss_fn_inputs=dict(loss_fn_inputs or {}),
            metadata=dict(metadata or {}),
        )

    def with_loss_fn_inputs(
        self,
        updates: Mapping[str, Any] | None = None,
        **named_updates: Any,
    ) -> "Datum":
        """Return a same-kind copy with additional derived loss inputs."""

        merged = dict(self.loss_fn_inputs)
        merged.update(dict(updates or {}))
        merged.update(named_updates)
        return Datum(
            model_input=self.model_input,
            sample_ref=self.sample_ref,
            datum_id=self.datum_id,
            loss_fn_inputs=merged,
            metadata=dict(self.metadata),
        )

    def tensors(self) -> Dict[str, TensorData]:
        return {
            name: tensor_payload(values)
            for name, values in self.loss_fn_inputs.items()
            if isinstance(values, torch.Tensor)
        }

    def _validate_token_targets(self) -> None:
        raw_targets = self.loss_fn_inputs.get("target_tokens")
        if raw_targets is None:
            return
        targets = raw_targets.tolist() if hasattr(raw_targets, "tolist") else list(raw_targets)
        for token in targets:
            value = int(token)
            if value < 0 and value != -100:
                if value == WEAVER_REDACTED_TOKEN_ID:
                    raise ValueError("redacted -8 tokens are response-only and cannot be targets")
                raise ValueError("target_tokens may not contain negative token IDs other than -100")


def validate_sample_ref_loss_inputs(data: Sequence[Datum], loss_fn: str) -> None:
    """Reject attempts to override server-owned managed inputs.

    The server owns the evolving per-loss schema. The SDK intentionally does
    not close the protocol over a hard-coded loss-name or derived-field list.
    """

    del loss_fn
    for index, datum in enumerate(data):
        if not datum.is_sample_ref:
            continue
        protected = set(datum.loss_fn_inputs) & _SAMPLE_REF_PROTECTED_INPUTS
        if protected:
            rendered = ", ".join(sorted(protected))
            raise ValueError(
                f"datum {index}: sample-ref Datum cannot provide server-owned inputs: {rendered}"
            )


def _is_jagged_sequence(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    nested_lengths: list[int] = []
    saw_nested = False
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            saw_nested = True
            nested_lengths.append(len(item))
        elif saw_nested:
            return True
    return saw_nested and len(set(nested_lengths)) > 1
