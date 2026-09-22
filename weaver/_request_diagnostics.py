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

"""Diagnostics for streaming training submissions."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from .tensor_transport import MultipartLayout, TensorPack


@contextmanager
def multipart_request_diagnostics(
    logger: logging.Logger,
    headers: httpx.Headers,
    path: str,
    layout: MultipartLayout,
    pack: TensorPack,
) -> Iterator[None]:
    """Correlate a streaming submission without logging its contents or retrying it."""
    headers.setdefault("X-Trace-ID", str(uuid.uuid4()))
    fields: dict[str, Any] = {
        "request_id": headers["X-Trace-ID"],
        "method": "POST",
        "request_path": path,
        "content_length": layout.content_length,
        "manifest_bytes": layout.manifest_bytes,
        "tensor_pack_bytes": pack.size_bytes,
        "tensor_pack_decoded_bytes": pack.decoded_size_bytes,
        "tensor_codec": pack.codec,
    }
    started = time.monotonic()
    logger.info("Submitting multipart request: %s", fields, extra=fields)
    try:
        yield
    except Exception as exc:
        fields["elapsed_seconds"] = time.monotonic() - started
        fields["exception_type"] = type(exc).__name__
        logger.exception("Multipart request failed: %s", fields, extra=fields)
        raise
