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

"""Tensor utilities shared across type helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import torch
from typing_extensions import TypeAlias

ArrayLike: TypeAlias = Union[torch.Tensor, Sequence[float], Sequence[int]]

_TORCH_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.int64: "int64",
    torch.int32: "int32",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_TO_TORCH_DTYPE = {name: dtype for dtype, name in _TORCH_DTYPE_TO_NAME.items()}
_INT_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
}


def _infer_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype.is_floating_point:
        return torch.float32
    if tensor.dtype in _INT_DTYPES:
        return torch.int64
    raise TypeError(f"Unsupported dtype for tensor payload: {tensor.dtype}")


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return _NAME_TO_TORCH_DTYPE[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_NAME_TO_TORCH_DTYPE))
        raise TypeError(f"Unsupported dtype '{name}'. Supported: {supported}") from exc


def _coerce_tensor(values: ArrayLike, dtype: str | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    target_dtype = _resolve_dtype(dtype) if dtype is not None else _infer_dtype(tensor)
    return tensor.to(dtype=target_dtype)


@dataclass(slots=True)
class TensorData:
    """Serializable tensor representation understood by the Weaver server."""

    data: list[object]
    dtype: str

    def to_dict(self) -> dict[str, object]:
        return {"data": self.data, "dtype": self.dtype}

    @classmethod
    def from_array(cls, values: ArrayLike, dtype: str | None = None) -> "TensorData":
        tensor = _coerce_tensor(values, dtype)
        dtype_name = _TORCH_DTYPE_TO_NAME.get(tensor.dtype)
        if dtype_name is None:
            raise TypeError(f"Unsupported tensor dtype for serialization: {tensor.dtype}")
        return cls(data=tensor.cpu().tolist(), dtype=dtype_name)

    def to_tensor(self) -> torch.Tensor:
        torch_dtype = _resolve_dtype(self.dtype)
        return torch.as_tensor(self.data, dtype=torch_dtype)


def tensor_payload(values: ArrayLike, dtype: str | None = None) -> TensorData:
    return TensorData.from_array(values, dtype=dtype)
