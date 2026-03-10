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

"""
E2E test for return_sampling_mask and return_old_logprob via weaver SDK.

Requires a running weaver-server and SGLang backend.

Usage:
    export WEAVER_BASE_URL=http://<weaver-server>:8080
    export WEAVER_API_KEY=sk-...
    python -m pytest tests/test_sampling_flags_e2e.py -v -s

    # Or with env overrides:
    WEAVER_BASE_URL=http://localhost:8080 WEAVER_API_KEY=sk-test \
    BASE_MODEL=Qwen/Qwen3-8B \
    python -m pytest tests/test_sampling_flags_e2e.py -v -s
"""

import os

import pytest

from weaver.service_client import ServiceClient
from weaver.types import ModelInput, SamplingParams

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")


@pytest.fixture(scope="module")
def sampling_client():
    svc = ServiceClient(
        base_url=os.environ.get("WEAVER_BASE_URL"),
        api_key=os.environ.get("WEAVER_API_KEY"),
    )
    svc.connect()
    client = svc.create_sampling_client(base_model=BASE_MODEL)
    yield client
    svc.close()


def test_sample_with_flags(sampling_client):
    """Verify old_logprobs and sampling_masks are returned when flags are set."""
    prompt_text = "The capital of France is"
    tokenizer = sampling_client._ensure_tokenizer()
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt = ModelInput.from_ints(tokens)

    params = SamplingParams(max_tokens=16, temperature=0.7)

    result = sampling_client.sample(
        prompt=prompt,
        sampling_params=params,
        return_sampling_mask=True,
        return_old_logprob=True,
    )

    assert "sequences" in result, f"missing sequences, keys: {list(result.keys())}"
    seq = result["sequences"][0]

    assert "tokens" in seq and len(seq["tokens"]) > 0
    assert "logprobs" in seq and len(seq["logprobs"]) > 0

    assert "old_logprobs" in seq, f"missing old_logprobs in sequence, keys: {list(seq.keys())}"
    assert len(seq["old_logprobs"]) == len(
        seq["tokens"]
    ), f"old_logprobs length {len(seq['old_logprobs'])} != tokens length {len(seq['tokens'])}"

    assert "sampling_masks" in seq, f"missing sampling_masks in sequence, keys: {list(seq.keys())}"
    assert len(seq["sampling_masks"]) > 0, "sampling_masks is empty"

    n = len(seq["tokens"])
    print(
        f"generated {n} tokens, got {len(seq['old_logprobs'])} old_logprobs, "
        f"{len(seq['sampling_masks'])} sampling_mask steps"
    )


def test_sample_without_flags(sampling_client):
    """Verify old_logprobs and sampling_masks are NOT returned when flags are off."""
    tokenizer = sampling_client._ensure_tokenizer()
    tokens = tokenizer.encode("Hello", add_special_tokens=False)
    prompt = ModelInput.from_ints(tokens)

    params = SamplingParams(max_tokens=8, temperature=0.7)

    result = sampling_client.sample(
        prompt=prompt,
        sampling_params=params,
    )

    seq = result["sequences"][0]
    assert "old_logprobs" not in seq, "old_logprobs should not be present when flag is off"
    assert "sampling_masks" not in seq, "sampling_masks should not be present when flag is off"
