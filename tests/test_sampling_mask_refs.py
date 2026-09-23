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

import hashlib
import struct
from unittest.mock import Mock

import pytest

from weaver import types
from weaver._sampling_utils import build_sample_body, normalize_sample_result
from weaver.sampling_masks import validate_mask_result


def sequence():
    return {
        "tokens": [3, 4],
        "text": "x",
        "weight_version": "v1",
        "sampling_mask_ref": {
            "schema": "weaver.sampling_mask.v1",
            "token_count": 2,
            "weight_version": "v1",
            "empty_row": "full-vocabulary",
            "tokens_sha256": hashlib.sha256(struct.pack("<ii", 3, 4)).hexdigest(),
        },
    }


def test_ref_request_and_opaque_normalization():
    body = build_sample_body(
        prompt=types.ModelInput.from_ints([1, 2]),
        sampling_params=types.SamplingParams(top_k=8),
        num_samples=1,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
        return_sampling_mask=True,
        return_old_logprob=True,
        return_moe_topk_indices=False,
        sampling_mask_transport="ref",
    )
    assert body["sampling_mask_transport"] == "ref"
    payload = {"sequences": [sequence()]}
    validate_mask_result(payload)
    output = normalize_sample_result(
        payload, Mock(side_effect=AssertionError("no tokenizer needed"))
    )
    assert output["sequences"][0]["sampling_mask_ref"] == sequence()["sampling_mask_ref"]
    assert "sampling_masks" not in output["sequences"][0]


@pytest.mark.parametrize("bad", ["missing", "inline", "count", "tokens", "version"])
def test_reject_bad_ref_result(bad):
    seq = sequence()
    if bad == "missing":
        del seq["sampling_mask_ref"]
    elif bad == "inline":
        seq["sampling_masks"] = [[3], [4]]
    elif bad == "count":
        seq["sampling_mask_ref"]["token_count"] = 3
    elif bad == "tokens":
        seq["tokens"][0] = 9
    else:
        seq["weight_version"] = "v2"
    with pytest.raises(ValueError):
        validate_mask_result({"result": {"sequences": [seq]}})


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("wait", [False, True])
def test_sample_ref_validator_for_sync_async_and_deferred(async_mode, wait):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from weaver.async_sampling_client import AsyncSamplingClient
    from weaver.sampling_client import SamplingClient

    service, handle = MagicMock(), MagicMock()
    payload = {"sequences": [sequence()]}
    kwargs = dict(
        prompt=types.ModelInput.from_ints([1, 2]),
        return_sampling_mask=True,
        sampling_mask_transport="ref",
        wait=wait,
    )
    if async_mode:
        service.enqueue_operation = AsyncMock(return_value=handle)
        handle.result = AsyncMock(return_value=payload)
        client = AsyncSamplingClient(service=service, sampling_session_id="s", base_model="m")
        result = asyncio.run(client.sample(**kwargs))
    else:
        service.enqueue_operation.return_value = handle
        handle.result.return_value = payload
        client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
        result = client.sample(**kwargs)
    assert service.enqueue_operation.call_args.args[1]["sampling_mask_transport"] == "ref"
    handle._result_validator(payload)
    with pytest.raises(ValueError):
        handle._result_validator({"sequences": [{"tokens": [3, 4]}]})
    if wait:
        assert result["sequences"][0]["sampling_mask_ref"] == sequence()["sampling_mask_ref"]
    else:
        assert result is handle
