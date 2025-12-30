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
from typing import Dict, Mapping, Sequence

import torch

from .model_input import ModelInput
from .tensor import TensorData, tensor_payload


@dataclass(slots=True)
class Datum:
    model_input: ModelInput
    loss_fn_inputs: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: Dict[str, torch.Tensor] = {}
        for key, value in self.loss_fn_inputs.items():
            # Handle TensorData objects (from tensor_payload)
            if isinstance(value, TensorData):
                normalized[key] = value.to_tensor()
            else:
                normalized[key] = torch.as_tensor(value)
        self.loss_fn_inputs = normalized

    def to_payload(self) -> dict[str, object]:
        return {
            "model_input": self.model_input.to_payload(),
            "loss_fn_inputs": {
                name: tensor_payload(values).to_dict()
                for name, values in self.loss_fn_inputs.items()
            },
        }

    @classmethod
    def from_raw(
        cls,
        *,
        model_input: ModelInput,
        loss_fn_inputs: Mapping[str, torch.Tensor | Sequence[int] | Sequence[float]],
    ) -> "Datum":
        return cls(
            model_input=model_input,
            loss_fn_inputs=dict(loss_fn_inputs),  # type: ignore[arg-type]
        )

    def tensors(self) -> Dict[str, TensorData]:
        return {name: tensor_payload(values) for name, values in self.loss_fn_inputs.items()}
