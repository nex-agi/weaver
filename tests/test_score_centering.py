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

"""Sampling protocol checks for rollout-time score-centering statistics."""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver.async_sampling_client import AsyncSamplingClient
from weaver.operations import AsyncOperationHandle, OperationHandle
from weaver.sampling_client import SamplingClient
from weaver.score_centering import SAMPLER_FIELDS, validate_sampler_result
from weaver.types import ModelInput, SamplingParams


def sequence():
    return {
        "tokens": [3, 4],
        "text": "answer",
        "stop_reason": "stop",
        "logprobs": [-1.0, -2.0],
        "sampler_logprobs": [-1.0, -2.0],
        "sampler_topk_ids": [[3, 5], [5, 6]],
        "sampler_topk_logprobs": [[-1.0, -2.0], [-1.0, -2.0]],
        "sampler_topk_mask": [[1, 1], [1, 1]],
        "sampler_distribution": {"schema": "behavior/unfiltered/v1", "temperature": 1.0},
    }


def test_sync_sampling_preserves_distribution_and_opt_in():
    service = MagicMock()
    service.enqueue_operation.return_value.result.return_value = {"sequences": [sequence()]}
    client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
    result = client.sample(
        prompt=ModelInput.from_ints([1, 2]),
        sampling_params=SamplingParams(score_centering={"head_size": 2}),
    )
    assert service.enqueue_operation.call_args.args[1]["score_centering"] == {
        "head_size": 2,
        "transport": "inline",
    }
    for key in SAMPLER_FIELDS:
        assert result["sequences"][0][key] == sequence()[key]
    client.sample(prompt=ModelInput.from_ints([1, 2]))
    assert "score_centering" not in service.enqueue_operation.call_args.args[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_params": SamplingParams(score_centering={"head_size": True})},
        {"sampling_params": SamplingParams(score_centering={"head_size": 129})},
        {"sampling_params": SamplingParams(temperature=0, score_centering={"head_size": 2})},
        {"sampling_params": SamplingParams(top_p=0.9, score_centering={"head_size": 2})},
        {"return_old_logprob": True},
        {"include_prompt_logprobs": True},
    ],
)
def test_invalid_request_fails_before_enqueue(kwargs):
    service = MagicMock()
    client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
    with pytest.raises(ValueError):
        client.sample(
            prompt=ModelInput.from_ints([1]),
            **({"sampling_params": SamplingParams(score_centering={"head_size": 2})} | kwargs),
        )
    service.enqueue_operation.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
def test_deferred_operation_cannot_silently_accept_old_server(async_mode):
    response = {"sequences": [{"tokens": [3], "text": "x"}]}
    state = {"status": "succeeded", "response": response}
    service = MagicMock()
    if async_mode:
        handle = AsyncOperationHandle(client=MagicMock(), operation_id="o", _cached=state)
        handle.wait = AsyncMock()
        service.enqueue_operation = AsyncMock(return_value=handle)
        client = AsyncSamplingClient(service=service, sampling_session_id="s", base_model="m")

        async def run():
            pending = await client.sample(
                prompt=ModelInput.from_ints([1]),
                sampling_params=SamplingParams(score_centering={"head_size": 2}),
                wait=False,
            )
            with pytest.raises(ValueError, match="missing sampler fields"):
                await pending.result()

        asyncio.run(run())
    else:
        handle = OperationHandle(client=MagicMock(), operation_id="o", _cached=state)
        handle.wait = MagicMock()
        service.enqueue_operation.return_value = handle
        client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
        pending = client.sample(
            prompt=ModelInput.from_ints([1]),
            sampling_params=SamplingParams(score_centering={"head_size": 2}),
            wait=False,
        )
        with pytest.raises(ValueError, match="missing sampler fields"):
            pending.result()


def test_async_normalization():
    service = MagicMock()
    handle = MagicMock()
    handle.result = AsyncMock(return_value={"result": {"sequences": [sequence()]}})
    service.enqueue_operation = AsyncMock(return_value=handle)
    client = AsyncSamplingClient(service=service, sampling_session_id="s", base_model="m")
    result = asyncio.run(
        client.sample(
            prompt=ModelInput.from_ints([1]),
            sampling_params=SamplingParams(score_centering={"head_size": 2}),
        )
    )
    assert result["sequences"][0]["sampler_topk_ids"] == sequence()["sampler_topk_ids"]


@pytest.mark.parametrize(
    "mutation", ["missing", "width", "duplicate", "nan", "mass", "sampled", "schema"]
)
def test_rejects_malformed_distribution(mutation):
    s = copy.deepcopy(sequence())
    if mutation == "missing":
        del s["sampler_topk_mask"]
    if mutation == "width":
        s["sampler_topk_ids"][0].pop()
    if mutation == "duplicate":
        s["sampler_topk_ids"][0] = [3, 3]
    if mutation == "nan":
        s["sampler_logprobs"][0] = float("nan")
    if mutation == "mass":
        s["sampler_topk_logprobs"][0] = [0.0, 0.0]
    if mutation == "sampled":
        s["sampler_logprobs"][0] = -4.0
    if mutation == "schema":
        s["sampler_distribution"]["schema"] = "unknown"
    with pytest.raises(ValueError):
        validate_sampler_result({"sequences": [s]}, 2)


def ref_sequence():
    import hashlib
    import struct

    s = sequence()
    for key in list(s):
        if key.startswith("sampler_topk_"):
            del s[key]
    s["sampler_distribution"]["weight_version"] = "v1"
    s["sampler_distribution_ref"] = {
        "schema": "weaver.sampler_distribution.v1",
        "distribution_schema": "behavior/unfiltered/v1",
        "top_k": 2,
        "token_count": 2,
        "weight_version": "v1",
        "tokens_sha256": hashlib.sha256(struct.pack("<ii", 3, 4)).hexdigest(),
    }
    return s


@pytest.mark.parametrize("async_mode", [False, True])
def test_ref_sampling_preserves_opaque_handle(async_mode):
    service = MagicMock()
    handle = MagicMock()
    payload = {"sequences": [ref_sequence()]}
    kwargs = dict(
        prompt=ModelInput.from_ints([1, 2]),
        sampling_params=SamplingParams(score_centering={"head_size": 2, "transport": "ref"}),
    )
    if async_mode:
        handle.result = AsyncMock(return_value=payload)
        service.enqueue_operation = AsyncMock(return_value=handle)
        client = AsyncSamplingClient(service=service, sampling_session_id="s", base_model="m")
        result = asyncio.run(client.sample(**kwargs))
    else:
        handle.result.return_value = payload
        service.enqueue_operation.return_value = handle
        client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
        result = client.sample(**kwargs)
    assert service.enqueue_operation.call_args.args[1]["score_centering"]["transport"] == "ref"
    assert (
        result["sequences"][0]["sampler_distribution_ref"]
        == ref_sequence()["sampler_distribution_ref"]
    )
    assert "sampler_topk_ids" not in result["sequences"][0]


@pytest.mark.parametrize("mutation", ["missing", "mixed", "version", "tokens"])
def test_ref_response_fails_closed(mutation):
    s = ref_sequence()
    if mutation == "missing":
        del s["sampler_distribution_ref"]
    if mutation == "mixed":
        s["sampler_topk_ids"] = [[3, 5], [5, 6]]
    if mutation == "version":
        s["sampler_distribution_ref"]["weight_version"] = "v2"
    if mutation == "tokens":
        s["tokens"] = [3, 5]
    with pytest.raises(ValueError):
        validate_sampler_result({"sequences": [s]}, 2, transport="ref")


def test_sc_options_are_independent_of_sampling_params():
    from weaver import _sampling_utils as su
    from weaver.types import SamplingParams, ScoreCenteringConfig

    config = ScoreCenteringConfig(head_size=128, transport="ref")
    params = SamplingParams(temperature=1, top_p=1, top_k=-1, score_centering=config)
    body = su.build_sample_body(
        prompt=ModelInput.from_ints([1]),
        sampling_params=params,
        num_samples=1,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
        return_sampling_mask=False,
        return_old_logprob=False,
        return_moe_topk_indices=False,
    )
    assert body["score_centering"] == config
    assert body["sampling_params"]["top_k"] == -1
    assert "topk_output_logprobs" not in body and "sampler_distribution_transport" not in body
    assert params.top_k == -1 and config == {"head_size": 128, "transport": "ref"}


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"head_size": 0},
        {"head_size": True},
        {"head_size": 129},
        {"head_size": 2, "top_k": 2},
        {"head_size": 2, "transport": "bad"},
    ],
)
def test_sc_options_validation(config):
    client = SamplingClient(service=MagicMock(), sampling_session_id="s")
    with pytest.raises(ValueError):
        client.sample(
            prompt=ModelInput.from_ints([1]), sampling_params=SamplingParams(score_centering=config)
        )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("transport", ["inline", "ref"])
def test_sampling_params_collects_sc_without_mutation(async_mode, transport):
    import asyncio
    from unittest.mock import AsyncMock

    from weaver.async_sampling_client import AsyncSamplingClient

    params = SamplingParams(score_centering={"head_size": 2, "transport": transport})
    response = {"sequences": [ref_sequence() if transport == "ref" else sequence()]}
    service = MagicMock()
    if async_mode:
        service.enqueue_operation = AsyncMock(return_value=MagicMock())
        client = AsyncSamplingClient(service=service, sampling_session_id="s", base_model="m")
        service.enqueue_operation.return_value.result = AsyncMock(return_value=response)
        result = asyncio.run(
            client.sample(prompt=ModelInput.from_ints([1, 2]), sampling_params=params)
        )
    else:
        service.enqueue_operation.return_value.result.return_value = response
        client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
        result = client.sample(prompt=ModelInput.from_ints([1, 2]), sampling_params=params)
    body = service.enqueue_operation.call_args.args[1]
    assert body["score_centering"] == params.score_centering
    assert "score_centering" not in body["sampling_params"]
    assert body["sampling_params"]["top_k"] == -1
    assert params.to_payload()["score_centering"] == {"head_size": 2, "transport": transport}
    assert result["sequences"][0]["tokens"] == response["sequences"][0]["tokens"]


@pytest.mark.parametrize(
    "legacy",
    [
        {"score_centering": {"head_size": 2}},
        {"topk_output_logprobs": 2},
        {"sampler_distribution_transport": "ref"},
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
def test_sample_rejects_removed_sc_keywords(legacy, async_mode):
    service = MagicMock()
    client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
    if async_mode:
        client = AsyncSamplingClient(service=service, sampling_session_id="s", base_model="m")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        client.sample(prompt=ModelInput.from_ints([1]), **legacy)
    service.enqueue_operation.assert_not_called()


def test_sampling_params_sc_does_not_override_sampling_top_k():
    service = MagicMock()
    client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
    params = SamplingParams(top_k=8, score_centering={"head_size": 128})
    with pytest.raises(ValueError, match="top_k=-1"):
        client.sample(prompt=ModelInput.from_ints([1]), sampling_params=params)
    assert params.top_k == 8
    service.enqueue_operation.assert_not_called()
