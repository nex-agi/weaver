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

"""Sampling-client support for public managed-sample references."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from weaver import _sampling_utils as _su
from weaver.async_sampling_client import AsyncSamplingClient
from weaver.sampling_client import SamplingClient
from weaver.types import LogprobsParams, ModelInput, SampleRef, SamplingParams


def _sample_response():
    return {
        "result": {"sequences": [{"tokens": [91, 92], "text": "answer", "stop_reason": "stop"}]}
    }


def _logprobs_response():
    return {"result": {"prompt_logprobs": [None, -0.4, -0.2]}}


def _sync_client() -> tuple[SamplingClient, MagicMock]:
    service = MagicMock()
    client = SamplingClient(
        service=service,
        sampling_session_id="sampling-1",
        base_model="Qwen/Qwen3-8B",
    )
    return client, service


def _async_client() -> tuple[AsyncSamplingClient, MagicMock]:
    service = MagicMock()
    service.datasets.get = AsyncMock()
    service.enqueue_operation = AsyncMock()
    client = AsyncSamplingClient(
        service=service,
        sampling_session_id="sampling-1",
        base_model="Qwen/Qwen3-8B",
    )
    return client, service


def test_sampling_payload_keeps_legacy_model_input_wire_unchanged():
    prompt = ModelInput.from_ints([1, 2, 3])
    body = _su.build_sample_body(
        prompt=prompt,
        sampling_params=SamplingParams(temperature=0.5),
        num_samples=2,
        include_prompt_logprobs=True,
        topk_prompt_logprobs=4,
        return_sampling_mask=False,
        return_old_logprob=False,
        return_moe_topk_indices=False,
    )

    assert body == {
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "sampling_params": {"temperature": 0.5, "top_p": 1.0, "top_k": -1},
        "num_samples": 2,
        "prompt_logprobs": True,
        "topk_prompt_logprobs": 4,
    }
    assert _su.build_logprobs_body(prompt, LogprobsParams()) == {
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]}
    }


def test_sampling_payload_serializes_only_the_public_sample_ref_identity():
    prompt = SampleRef(dataset="open-math", version="2026-09", sample_idx=17)
    expected = {
        "kind": "sample_ref",
        "dataset": "open-math",
        "version": "2026-09",
        "sample_idx": 17,
    }

    sample_body = _su.build_sample_body(
        prompt=prompt,
        sampling_params=None,
        num_samples=1,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
        return_sampling_mask=False,
        return_old_logprob=False,
        return_moe_topk_indices=False,
    )

    assert sample_body["prompt"] == expected
    assert _su.build_logprobs_body(prompt, None)["prompt"] == expected
    assert set(sample_body["prompt"]) == {"kind", "dataset", "version", "sample_idx"}


def test_sync_public_sample_ref_supports_sample_and_logprobs_with_cached_preflight():
    client, service = _sync_client()
    service.datasets.get.return_value = SimpleNamespace(content_visibility="public")
    sample_handle = MagicMock()
    sample_handle.result.return_value = _sample_response()
    logprobs_handle = MagicMock()
    logprobs_handle.result.return_value = _logprobs_response()
    service.enqueue_operation.side_effect = [sample_handle, logprobs_handle]
    prompt = SampleRef("open-math", "v1", 5)

    sample_result = client.sample(prompt=prompt)
    logprobs = client.compute_logprobs(prompt=prompt)

    assert sample_result["sequences"][0]["tokens"] == [91, 92]
    assert logprobs == [None, -0.4, -0.2]
    service.datasets.get.assert_called_once_with(name="open-math", version="v1")
    assert service.enqueue_operation.call_args_list[0] == call(
        "/api/v1/sampling-sessions/sampling-1/samples",
        {
            "prompt": {
                "kind": "sample_ref",
                "dataset": "open-math",
                "version": "v1",
                "sample_idx": 5,
            },
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    assert service.enqueue_operation.call_args_list[1] == call(
        "/api/v1/sampling-sessions/sampling-1/logprobs",
        {
            "prompt": {
                "kind": "sample_ref",
                "dataset": "open-math",
                "version": "v1",
                "sample_idx": 5,
            }
        },
    )


def test_sync_protected_sample_ref_is_rejected_before_enqueue_and_cached():
    client, service = _sync_client()
    service.datasets.get.return_value = SimpleNamespace(content_visibility="protected")
    prompt = SampleRef("secret", "v1", 0)

    with pytest.raises(ValueError, match="only support public SampleRef"):
        client.sample(prompt=prompt, wait=False)
    with pytest.raises(ValueError, match="only support public SampleRef"):
        client.compute_logprobs(prompt=prompt)

    service.datasets.get.assert_called_once_with(name="secret", version="v1")
    service.enqueue_operation.assert_not_called()


def test_model_input_sampling_never_fetches_the_dataset_catalog():
    client, service = _sync_client()
    handle = MagicMock()
    service.enqueue_operation.return_value = handle

    assert client.sample(prompt=ModelInput.from_ints([1]), wait=False) is handle

    service.datasets.get.assert_not_called()


def test_async_public_sample_ref_matches_sync_sampling_and_logprobs():
    async def run():
        client, service = _async_client()
        service.datasets.get.return_value = SimpleNamespace(content_visibility="public")
        sample_handle = MagicMock()
        sample_handle.result = AsyncMock(return_value=_sample_response())
        logprobs_handle = MagicMock()
        logprobs_handle.result = AsyncMock(return_value=_logprobs_response())
        service.enqueue_operation.side_effect = [sample_handle, logprobs_handle]
        prompt = SampleRef("open-math", "v1", 5)

        sample_result = await client.sample(prompt=prompt)
        logprobs = await client.compute_logprobs(prompt=prompt)
        return sample_result, logprobs, service

    sample_result, logprobs, service = asyncio.run(run())

    assert sample_result["sequences"][0]["tokens"] == [91, 92]
    assert logprobs == [None, -0.4, -0.2]
    service.datasets.get.assert_awaited_once_with(name="open-math", version="v1")
    assert service.enqueue_operation.await_count == 2
    assert service.enqueue_operation.call_args_list[0].args[1]["prompt"] == {
        "kind": "sample_ref",
        "dataset": "open-math",
        "version": "v1",
        "sample_idx": 5,
    }
    assert service.enqueue_operation.call_args_list[1].args[1]["prompt"] == {
        "kind": "sample_ref",
        "dataset": "open-math",
        "version": "v1",
        "sample_idx": 5,
    }


def test_async_protected_sample_ref_is_rejected_before_enqueue_and_cached():
    async def run():
        client, service = _async_client()
        service.datasets.get.return_value = SimpleNamespace(content_visibility="protected")
        prompt = SampleRef("secret", "v1", 0)

        with pytest.raises(ValueError, match="only support public SampleRef"):
            await client.sample(prompt=prompt, wait=False)
        with pytest.raises(ValueError, match="only support public SampleRef"):
            await client.compute_logprobs(prompt=prompt)
        return service

    service = asyncio.run(run())

    service.datasets.get.assert_awaited_once_with(name="secret", version="v1")
    service.enqueue_operation.assert_not_awaited()
