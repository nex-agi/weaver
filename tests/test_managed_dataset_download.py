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

"""Authenticated managed-dataset stream contract tests."""

from __future__ import annotations

import asyncio
import hashlib
import io

import httpx
import pytest

from weaver._async_http import AsyncAPIClient
from weaver._http import APIClient
from weaver.config import WeaverConfig


def _headers(content: bytes, **updates: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/x-ndjson",
        "Content-Length": str(len(content)),
        "X-Weaver-Content-SHA256": hashlib.sha256(content).hexdigest(),
        "X-Weaver-Content-Visibility": "public",
    }
    headers.update(updates)
    return headers


def test_sync_http_client_streams_authenticated_public_dataset():
    content = b'{"messages":[]}\n' * 3

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/managed-datasets/open/versions/v1/download"
        assert request.headers["x-weaver-api-key"] == "sk-test"
        assert request.headers["accept"] == "application/x-ndjson"
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers=_headers(content), stream=httpx.ByteStream(content))

    client = APIClient(WeaverConfig(base_url="https://example.test", api_key="sk-test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test",
        headers={"X-WEAVER-API-KEY": "sk-test"},
        transport=httpx.MockTransport(handler),
    )
    destination = io.BytesIO()
    try:
        metadata = client.download_managed_dataset(
            "/api/v1/managed-datasets/open/versions/v1/download", destination
        )
    finally:
        client.close()

    assert metadata == (len(content), hashlib.sha256(content).hexdigest())
    assert destination.getvalue() == content


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"Content-Type": "application/json"}, "application/x-ndjson"),
        ({"Content-Encoding": "gzip"}, "must not use content encoding"),
        ({"X-Weaver-Content-Visibility": "protected"}, "content_visibility=public"),
        ({"Content-Length": "invalid"}, "invalid Content-Length"),
        ({"X-Weaver-Content-SHA256": "A" * 64}, "lowercase SHA-256"),
    ],
)
def test_sync_http_client_rejects_invalid_public_dataset_headers(updates, match):
    content = b"{}\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=_headers(content, **updates),
            stream=httpx.ByteStream(content),
        )

    client = APIClient(WeaverConfig(base_url="https://example.test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ValueError, match=match):
            client.download_managed_dataset("/download", io.BytesIO())
    finally:
        client.close()


def test_sync_http_client_rejects_truncated_or_digest_mismatched_dataset():
    content = b"{}\n"

    def truncated(_request: httpx.Request) -> httpx.Response:
        headers = _headers(content)
        headers["Content-Length"] = str(len(content) + 1)
        return httpx.Response(200, headers=headers, stream=httpx.ByteStream(content))

    client = APIClient(WeaverConfig(base_url="https://example.test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(truncated)
    )
    try:
        with pytest.raises((ValueError, httpx.RemoteProtocolError)):
            client.download_managed_dataset("/download", io.BytesIO())
    finally:
        client.close()


def test_async_http_client_has_managed_dataset_download_parity():
    async def run() -> None:
        content = b'{"messages":[]}\n' * 3

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-weaver-api-key"] == "sk-test"
            return httpx.Response(200, headers=_headers(content), stream=httpx.ByteStream(content))

        client = AsyncAPIClient(WeaverConfig(base_url="https://example.test", api_key="sk-test"))
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://example.test",
            headers={"X-WEAVER-API-KEY": "sk-test"},
            transport=httpx.MockTransport(handler),
        )
        destination = io.BytesIO()
        try:
            metadata = await client.download_managed_dataset("/download", destination)
        finally:
            await client.aclose()

        assert metadata == (len(content), hashlib.sha256(content).hexdigest())
        assert destination.getvalue() == content

    asyncio.run(run())
