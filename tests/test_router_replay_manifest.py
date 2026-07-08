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

"""Tests for server-side router-replay manifest persistence."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver._payloads import build_router_replay_manifest_body
from weaver.async_service_client import AsyncServiceClient
from weaver.service_client import ServiceClient

_MANIFEST = {"schema": "weaver.router_replay.index_set.v1", "shards": []}
_ENDPOINT = "/api/v1/router-replay/manifests"


def test_build_router_replay_manifest_body_normalizes_and_validates():
    body = build_router_replay_manifest_body("/model-a/", "set-1", _MANIFEST)
    assert body == {
        "model_id": "model-a",
        "replay_set_id": "set-1",
        "manifest": _MANIFEST,
    }


def test_build_router_replay_manifest_body_rejects_bad_input():
    with pytest.raises(ValueError):
        build_router_replay_manifest_body("", "set-1", _MANIFEST)
    with pytest.raises(ValueError):
        build_router_replay_manifest_body("model-a", "", _MANIFEST)
    with pytest.raises(ValueError):
        build_router_replay_manifest_body("model-a", "set-1", ["not", "a", "mapping"])


def test_service_client_posts_manifest_to_endpoint():
    client = ServiceClient.__new__(ServiceClient)
    client._http = MagicMock()
    client._http.post.return_value = {
        "index_set_uri": "weaver://model-a/router-replay/set-1",
        "manifest_uri": "weaver://model-a/router-replay/set-1/manifest.json",
    }

    out = client._write_router_replay_manifest("model-a", "set-1", _MANIFEST)

    args, kwargs = client._http.post.call_args
    assert args[0] == _ENDPOINT
    assert kwargs["json"] == {
        "model_id": "model-a",
        "replay_set_id": "set-1",
        "manifest": _MANIFEST,
    }
    assert out["manifest_uri"].endswith("manifest.json")


def test_async_service_client_posts_manifest_to_endpoint():
    client = AsyncServiceClient.__new__(AsyncServiceClient)
    http = AsyncMock()
    http.post.return_value = {"manifest_uri": "weaver://model-a/router-replay/set-1/manifest.json"}
    client._http = http

    out = asyncio.run(client._write_router_replay_manifest("model-a", "set-1", _MANIFEST))

    args, kwargs = http.post.call_args
    assert args[0] == _ENDPOINT
    assert kwargs["json"]["model_id"] == "model-a"
    assert out["manifest_uri"].endswith("manifest.json")
