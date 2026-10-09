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

import asyncio

import pytest

from weaver.operations import AsyncOperationHandle, OperationHandle, WeaverOperationError


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
    assert str(error) == (
        f"Operation failed: context_length_exceeded: {payload['error_message']} (operation_id=op-1)"
    )


def test_operation_error_displays_diagnostic_and_correlation_context() -> None:
    error = WeaverOperationError(
        {
            "id": "op-checkpoint",
            "error": "operation_failed",
            "error_code": "export_checkpoint_source_missing",
            "error_message": "checkpoint files are no longer available to the converter",
            "retryable": False,
            "error_details": {
                "diagnostic_message": "ExportError: source_path does not exist: /gpfs/checkpoint",
                "stage": "export_checkpoint",
            },
        }
    )
    assert error.operation_id == "op-checkpoint"
    assert error.retryable is False
    assert "source_path does not exist: /gpfs/checkpoint" in str(error)
    assert "operation_id=op-checkpoint" in str(error)
    assert "stage=export_checkpoint" in str(error)


def test_operation_error_avoids_repeating_diagnostic() -> None:
    diagnostic = "cancelled: trainer stopped; no trainer will claim this operation"
    error = WeaverOperationError(
        {
            "id": "op-stopped",
            "error": "operation_failed",
            "error_message": "The forward/backward operation failed: " + diagnostic,
            "error_details": {"diagnostic_message": diagnostic},
        }
    )
    assert str(error).count(diagnostic) == 1


def test_operation_error_falls_back_when_old_server_has_no_details() -> None:
    handle = OperationHandle(
        client=_NoPollClient(),
        operation_id="op-legacy",
        _cached={"status": "error", "error": "operation_failed"},
    )
    with pytest.raises(WeaverOperationError) as caught:
        handle.result()
    assert "did not return a detailed failure reason" in str(caught.value)
    assert "operation_id=op-legacy" in str(caught.value)


@pytest.mark.parametrize(
    "details", [None, [], "invalid", {"diagnostic_message": [1, 2]}, {"diagnostic_message": "  \n"}]
)
def test_operation_error_handles_missing_or_malformed_details(details) -> None:
    error = WeaverOperationError({"error": "operation_failed", "error_details": details})
    assert "did not return a detailed failure reason" in str(error)


def test_async_operation_error_includes_handle_id_and_diagnostic() -> None:
    async def fail():
        handle = AsyncOperationHandle(
            client=None,
            operation_id="op-async",
            _cached={
                "status": "error",
                "error": "operation_failed",
                "error_details": {
                    "diagnostic_message": "CUDA out of memory: requested 12 GiB, available 4 GiB",
                },
            },
        )
        await handle.result()

    with pytest.raises(WeaverOperationError) as caught:
        asyncio.run(fail())
    assert "requested 12 GiB, available 4 GiB" in str(caught.value)
    assert "operation_id=op-async" in str(caught.value)
