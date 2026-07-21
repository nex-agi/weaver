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

"""Configuration helpers for the Weaver SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional, cast

_DEFAULT_BASE_URL = "https://weaver-console.nex-agi.cn"

TensorTransport = Literal["default", "http-binary"]
_TENSOR_TRANSPORTS = {"default", "http-binary"}
TensorCompression = Literal["raw", "zstd"]
_TENSOR_COMPRESSIONS = {"raw", "zstd"}


def _tensor_transport(value: str | None) -> TensorTransport:
    resolved = value or "default"
    if resolved not in _TENSOR_TRANSPORTS:
        supported = ", ".join(sorted(_TENSOR_TRANSPORTS))
        raise ValueError(f"Unsupported tensor transport {resolved!r}. Supported: {supported}")
    return cast(TensorTransport, resolved)


def _tensor_compression(value: str | None) -> TensorCompression:
    resolved = value or "raw"
    if resolved not in _TENSOR_COMPRESSIONS:
        supported = ", ".join(sorted(_TENSOR_COMPRESSIONS))
        raise ValueError(f"Unsupported tensor compression {resolved!r}. Supported: {supported}")
    return cast(TensorCompression, resolved)


@dataclass(slots=True)
class WeaverConfig:
    """Holds connection + auth settings for the Weaver server."""

    base_url: str = _DEFAULT_BASE_URL
    api_key: str | None = None
    tensor_transport: TensorTransport = "default"
    tensor_compression: TensorCompression = "raw"

    def __post_init__(self) -> None:
        self.tensor_transport = _tensor_transport(self.tensor_transport)
        self.tensor_compression = _tensor_compression(self.tensor_compression)
        if self.tensor_compression != "raw" and self.tensor_transport != "http-binary":
            raise ValueError("tensor_compression requires tensor_transport='http-binary'")

    @classmethod
    def from_env(
        cls,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tensor_transport: TensorTransport | None = None,
        tensor_compression: TensorCompression | None = None,
    ) -> "WeaverConfig":
        """Load configuration from kwargs with env fallbacks.

        The api_key should be the complete API key starting with 'sk-'
        obtained from the API Keys page.
        """

        return cls(
            base_url=base_url or os.getenv("WEAVER_BASE_URL") or _DEFAULT_BASE_URL,
            api_key=api_key or os.getenv("WEAVER_API_KEY"),
            tensor_transport=_tensor_transport(
                tensor_transport or os.getenv("WEAVER_TENSOR_TRANSPORT")
            ),
            tensor_compression=_tensor_compression(
                tensor_compression or os.getenv("WEAVER_TENSOR_COMPRESSION")
            ),
        )

    def require_auth(self) -> None:
        """Raise if auth credentials are missing."""

        if not self.api_key:
            raise RuntimeError(
                "Weaver credentials missing. Provide api_key or set "
                f"WEAVER_API_KEY environment variable. Get your API key "
                f"from the Weaver at {self.base_url}/api-keys"
            )
