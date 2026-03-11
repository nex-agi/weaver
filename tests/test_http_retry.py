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

"""Tests for HTTP retry behavior in APIClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from weaver._http import APIClient
from weaver.config import WeaverConfig


@pytest.fixture()
def config():
    return WeaverConfig(base_url="https://test.example.com", api_key="sk-test")


@pytest.fixture()
def client(config):
    c = APIClient(config, max_retries=3)
    yield c
    c.close()


class TestPostMaxRetriesOverride:
    """post() accepts a per-request max_retries override."""

    def test_post_no_retry_on_timeout(self, client):
        """POST with max_retries=1 should not retry on timeout."""
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = httpx.ReadTimeout("read timed out")

        with pytest.raises(httpx.ReadTimeout):
            client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert client._client.request.call_count == 1

    def test_post_default_retries(self, client):
        """POST without max_retries override retries up to the client default."""
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = httpx.ReadTimeout("read timed out")

        with pytest.raises(httpx.ReadTimeout):
            client.post("/api/v1/sessions", json={})

        # Client was created with max_retries=3
        assert client._client.request.call_count == 3

    def test_post_max_retries_success_on_second_attempt(self, client):
        """POST with default retries succeeds if second attempt works."""
        ok_response = MagicMock()
        ok_response.is_success = True
        ok_response.status_code = 200
        ok_response.content = b'{"id": "op-1"}'
        ok_response.json.return_value = {"id": "op-1"}

        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = [
            httpx.ReadTimeout("read timed out"),
            ok_response,
        ]

        result = client.post("/api/v1/sessions", json={})

        assert result == {"id": "op-1"}
        assert client._client.request.call_count == 2


class TestGetRetryUnchanged:
    """GET requests still use the default client-level retries."""

    def test_get_retries_on_timeout(self, client):
        """GET should retry up to client default on timeout."""
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = httpx.ReadTimeout("read timed out")

        with pytest.raises(httpx.ReadTimeout):
            client.get("/api/v1/models/m1")

        assert client._client.request.call_count == 3
