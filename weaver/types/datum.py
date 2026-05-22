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

import torch

from .model_input import ModelInput
from .tensor import TensorData, tensor_payload


@dataclass(slots=True)
class Datum:
    model_input: ModelInput
    loss_fn_inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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

    def to_payload(self) -> dict[str, object]:
        return {
            "model_input": self.model_input.to_payload(),
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

    @classmethod
    def from_raw(
        cls,
        *,
        model_input: ModelInput,
        loss_fn_inputs: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> "Datum":
        return cls(
            model_input=model_input,
            loss_fn_inputs=dict(loss_fn_inputs),  # type: ignore[arg-type]
            metadata=dict(metadata or {}),
        )

    def tensors(self) -> Dict[str, TensorData]:
        return {
            name: tensor_payload(values)
            for name, values in self.loss_fn_inputs.items()
            if isinstance(values, torch.Tensor)
        }


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
