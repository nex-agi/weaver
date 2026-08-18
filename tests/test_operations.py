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

from __future__ import annotations

import pytest

from weaver.operations import OperationHandle, WeaverOperationError


class _NoPollClient:
    def __init__(self) -> None:
        self.get_calls = 0

    def get(self, path: str):
        self.get_calls += 1
        raise AssertionError(f"completed operation must not be polled: {path}")


def test_precached_error_status_raises_without_polling() -> None:
    client = _NoPollClient()
    handle = OperationHandle(
        client=client,
        operation_id="op-1",
        _cached={"id": "op-1", "status": "error", "error": "operation_failed"},
    )

    with pytest.raises(WeaverOperationError) as exc_info:
        handle.result()

    assert exc_info.value.payload["error"] == "operation_failed"
    assert client.get_calls == 0


def test_operation_error_surfaces_structured_reason() -> None:
    payload = {
        "id": "op-1",
        "status": "error",
        "error": "operation_failed",
        "error_code": "context_length_exceeded",
        "error_message": (
            "request needs 34720 tokens (26528 input + 8192 completion), "
            "exceeding serving context length 32768"
        ),
        "error_details": {
            "source": "inference_engine",
            "upstream_status": 400,
            "max_context_length": 32768,
        },
    }
    handle = OperationHandle(client=_NoPollClient(), operation_id="op-1", _cached=payload)

    with pytest.raises(WeaverOperationError) as exc_info:
        handle.result()

    error = exc_info.value
    assert error.code == "context_length_exceeded"
    assert error.message == payload["error_message"]
    assert error.details == payload["error_details"]
    assert str(error) == f"Operation failed: context_length_exceeded: {payload['error_message']}"
