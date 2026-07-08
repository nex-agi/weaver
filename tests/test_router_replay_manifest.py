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

"""Tests for server-side router-replay index-set creation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver._payloads import build_router_replay_index_set_body
from weaver.async_service_client import AsyncServiceClient
from weaver.service_client import ServiceClient

_VALUE_REFS = [
    {"storage": "gpfs", "format": "safetensors", "relative_path": "x/seq-0.safetensors"},
    {"storage": "gpfs", "format": "safetensors", "relative_path": "x/seq-1.safetensors"},
]
_SAMPLES = [{"value_ref": _VALUE_REFS[0]}, {"value_ref": _VALUE_REFS[1]}]
_ENDPOINT = "/api/v1/router-replay/index-sets"


def test_build_router_replay_index_set_body_normalizes_and_validates():
    body = build_router_replay_index_set_body("/model-a/", _VALUE_REFS)
    assert body == {"model_id": "model-a", "samples": _SAMPLES}


def test_build_router_replay_index_set_body_rejects_bad_input():
    with pytest.raises(ValueError):
        build_router_replay_index_set_body("", _VALUE_REFS)
    with pytest.raises(ValueError):
        build_router_replay_index_set_body("model-a", ["not-a-mapping"])


def test_service_client_posts_index_set_to_endpoint():
    client = ServiceClient.__new__(ServiceClient)
    client._http = MagicMock()
    client._http.post.return_value = {
        "replay_set_id": "r3-1-abcd",
        "index_set_uri": "weaver://model-a/router-replay/r3-1-abcd",
        "manifest_uri": "weaver://model-a/router-replay/r3-1-abcd/manifest.json",
    }

    out = client._create_router_replay_index_set("model-a", _VALUE_REFS)

    args, kwargs = client._http.post.call_args
    assert args[0] == _ENDPOINT
    assert kwargs["json"] == {"model_id": "model-a", "samples": _SAMPLES}
    assert out["manifest_uri"].endswith("manifest.json")


def test_async_service_client_posts_index_set_to_endpoint():
    client = AsyncServiceClient.__new__(AsyncServiceClient)
    http = AsyncMock()
    http.post.return_value = {
        "manifest_uri": "weaver://model-a/router-replay/r3-1-abcd/manifest.json"
    }
    client._http = http

    out = asyncio.run(client._create_router_replay_index_set("model-a", _VALUE_REFS))

    args, kwargs = http.post.call_args
    assert args[0] == _ENDPOINT
    assert kwargs["json"]["model_id"] == "model-a"
    assert out["manifest_uri"].endswith("manifest.json")
