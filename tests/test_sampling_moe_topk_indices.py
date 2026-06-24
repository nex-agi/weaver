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

"""Tests for return_moe_topk_indices support in SamplingClient."""

from unittest.mock import MagicMock

from weaver.sampling_client import SamplingClient
from weaver.types import ModelInput


def _make_client() -> tuple:
    """Create a SamplingClient with a mock service and return (client, mock_service)."""
    mock_service = MagicMock()
    client = SamplingClient(
        service=mock_service,
        sampling_session_id="sess-001",
        base_model="Qwen/Qwen3-8B",
    )
    return client, mock_service


def test_sample_request_includes_return_moe_topk_indices_flag():
    """When return_moe_topk_indices=True, the request body contains the flag at top level."""
    client, mock_service = _make_client()
    mock_handle = MagicMock()
    mock_handle.result.return_value = {
        "result": {"sequences": [{"tokens": [1], "text": "a", "stop_reason": "stop"}]}
    }
    mock_service.enqueue_operation.return_value = mock_handle

    prompt = ModelInput.from_ints([1, 2, 3])
    client.sample(prompt=prompt, return_moe_topk_indices=True)

    call_args = mock_service.enqueue_operation.call_args
    body = call_args[0][1]
    assert "return_moe_topk_indices" in body
    assert body["return_moe_topk_indices"] is True
    # Must NOT be inside sampling_params
    assert "return_moe_topk_indices" not in body["sampling_params"]


def test_sample_request_omits_flag_when_false():
    """When return_moe_topk_indices=False (default), the key is absent from the request body."""
    client, mock_service = _make_client()
    mock_handle = MagicMock()
    mock_handle.result.return_value = {
        "result": {"sequences": [{"tokens": [1], "text": "a", "stop_reason": "stop"}]}
    }
    mock_service.enqueue_operation.return_value = mock_handle

    prompt = ModelInput.from_ints([1, 2, 3])
    client.sample(prompt=prompt)

    call_args = mock_service.enqueue_operation.call_args
    body = call_args[0][1]
    assert "return_moe_topk_indices" not in body


def test_normalize_preserves_moe_topk_indices():
    """When the server response contains moe_topk_indices, the normalized result preserves it."""
    client, _ = _make_client()
    payload = {
        "result": {
            "sequences": [
                {
                    "tokens": [1, 2, 3],
                    "text": "hello",
                    "stop_reason": "stop",
                    "logprobs": [-0.5, -0.3, -0.1],
                    "moe_topk_indices": [[1, 7, 3, 5], [2, 4, 0, 6], [1, 3, 5, 7]],
                }
            ]
        }
    }

    result = client._normalize_sample_result(payload)

    assert "sequences" in result
    seq = result["sequences"][0]
    assert "moe_topk_indices" in seq
    assert seq["moe_topk_indices"] == [[1, 7, 3, 5], [2, 4, 0, 6], [1, 3, 5, 7]]


def test_normalize_omits_moe_topk_indices_when_absent():
    """When moe_topk_indices is not in the response, it's not in the normalized result."""
    client, _ = _make_client()
    payload = {
        "result": {
            "sequences": [
                {
                    "tokens": [1, 2, 3],
                    "text": "hello",
                    "stop_reason": "stop",
                    "logprobs": [-0.5, -0.3, -0.1],
                }
            ]
        }
    }

    result = client._normalize_sample_result(payload)

    assert "sequences" in result
    seq = result["sequences"][0]
    assert "moe_topk_indices" not in seq


def test_normalize_omits_moe_topk_indices_when_none():
    """When moe_topk_indices is None, it's not in the normalized result."""
    client, _ = _make_client()
    payload = {
        "result": {
            "sequences": [
                {
                    "tokens": [1, 2, 3],
                    "text": "hello",
                    "stop_reason": "stop",
                    "logprobs": [-0.5, -0.3, -0.1],
                    "moe_topk_indices": None,
                }
            ]
        }
    }

    result = client._normalize_sample_result(payload)

    assert "sequences" in result
    seq = result["sequences"][0]
    assert "moe_topk_indices" not in seq
