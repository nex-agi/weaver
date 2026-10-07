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

"""Negotiated direct result downloads and compatibility in both client stacks."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import logging
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import zstandard

from weaver._async_http import AsyncAPIClient
from weaver._http import APIClient
from weaver._tensor_read import (
    READ_MEDIA_TYPE,
    async_direct_result_stream,
    direct_result_stream,
    validate_read_plan,
)
from weaver.config import WeaverConfig


def fixture_plan(codec="raw"):
    raw = b"tensor-result-data" * 10000
    wire = zstandard.ZstdCompressor().compress(raw) if codec == "zstd" else raw
    pack = {
        "size_bytes": len(wire),
        "sha256": hashlib.sha256(wire).hexdigest(),
        "codec": codec,
        "decoded_size_bytes": len(raw),
    }
    op = str(uuid4())
    plan = {
        "version": 1,
        "operation_id": op,
        "tensor_pack": pack,
        "tos_host": "test.tos.example",
        "remote_read": {
            "lease_id": str(uuid4()),
            "ref": "artifact://generated/results/owned@version",
            "epoch": 1,
            "manifest_digest": "ab" * 32,
            "lease_remaining_seconds": 120,
            "blob": {
                "schema": "weaver-tensor-pack/v1/" + codec,
                "size_bytes": len(wire),
                "sha256": pack["sha256"],
            },
            "url": "https://test.tos.example/object?versionId=owned&X-Tos-Signature=secret-signature",
        },
    }
    return op, raw, wire, pack, plan


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("codec", ["raw", "zstd"])
@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "corrupt",
        "truncated",
        "redirect",
        "encoded",
        "transport",
        "foreign",
        "short_write",
        "release_outage",
    ],
)
def test_negotiated_direct_download(monkeypatch, caplog, async_mode, codec, fault):
    op, raw, wire, pack, plan = fixture_plan(codec)
    events, object_requests = [], []
    if fault == "foreign":
        plan["remote_read"]["url"] = "https://foreign.example/object?versionId=owned"

    def metadata(request):
        assert request.headers["X-Weaver-Api-Key"] == "existing-api-key"
        if request.method == "GET":
            assert request.headers.get_list("Accept") == [READ_MEDIA_TYPE]
            assert request.headers.get_list("Accept-Encoding") == ["identity"]
            assert request.headers.get("X-Weaver-Tensor-Read-ID")
            events.append("acquire")
            return httpx.Response(
                200,
                stream=httpx.ByteStream(json.dumps(plan).encode()),
                headers={"content-type": READ_MEDIA_TYPE},
            )
        assert request.url.path.endswith("/release")
        assert events[-1] == "object_closed" or fault in ("foreign", "transport")
        events.append("release")
        if fault == "release_outage":
            return httpx.Response(
                503, json={"error": {"code": "unavailable", "message": "outage", "retryable": True}}
            )
        return httpx.Response(204)

    body = wire
    if fault == "corrupt":
        body = bytes([wire[0] ^ 1]) + wire[1:]
    if fault == "truncated":
        body = wire[:-1]

    class SyncBytes(httpx.SyncByteStream):
        def __iter__(self):
            yield body

        def close(self):
            events.append("object_closed")

    class AsyncBytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body

        async def aclose(self):
            events.append("object_closed")

    def object_response(request, stream):
        object_requests.append(request)
        assert request.url.host == "test.tos.example"
        assert set(request.headers) == {"host", "accept-encoding"}
        assert request.headers["Accept-Encoding"] == "identity"
        if fault == "transport":
            raise httpx.ReadError("provider error with secret-signature", request=request)
        return httpx.Response(
            302 if fault == "redirect" else 200,
            headers={
                "content-encoding": "gzip" if fault == "encoded" else "identity",
                "location": "https://foreign.example/",
            },
            stream=stream,
        )

    class SyncTransport(httpx.BaseTransport):
        def __init__(self, **kwargs):
            assert kwargs == {"verify": True, "trust_env": False}

        def handle_request(self, request):
            return object_response(request, SyncBytes())

    class AsyncTransport(httpx.AsyncBaseTransport):
        def __init__(self, **kwargs):
            assert kwargs == {"verify": True, "trust_env": False}

        async def handle_async_request(self, request):
            return object_response(request, AsyncBytes())

    class ShortDestination(io.BytesIO):
        def write(self, data):
            return super().write(data[:-1])

    output = ShortDestination() if fault == "short_write" else io.BytesIO()
    # Production HTTP clients are constructed before patching the isolated byte transports.
    config = WeaverConfig(base_url="https://weaver.example", api_key="existing-api-key")
    caplog.set_level(logging.INFO)

    def verify_error(error):
        assert "secret-signature" not in str(error)
        assert "https://test.tos.example" not in str(error)

    if async_mode:

        async def exercise():
            client = AsyncAPIClient(config)
            await client._client.aclose()
            client._client = httpx.AsyncClient(
                base_url=config.base_url,
                headers=client._headers,
                transport=httpx.MockTransport(metadata),
            )
            monkeypatch.setattr(httpx, "AsyncHTTPTransport", AsyncTransport)
            try:
                if fault == "none":
                    await client.download_tensor_pack(op, output, **pack)
                else:
                    with pytest.raises(Exception) as error:
                        await client.download_tensor_pack(op, output, **pack)
                    verify_error(error.value)
            finally:
                await client.aclose()

        asyncio.run(exercise())
    else:
        client = APIClient(config)
        client._client.close()
        client._client = httpx.Client(
            base_url=config.base_url,
            headers=client._headers,
            transport=httpx.MockTransport(metadata),
        )
        monkeypatch.setattr(httpx, "HTTPTransport", SyncTransport)
        try:
            if fault == "none":
                client.download_tensor_pack(op, output, **pack)
            else:
                with pytest.raises(Exception) as error:
                    client.download_tensor_pack(op, output, **pack)
                verify_error(error.value)
        finally:
            client.close()
    assert events.count("acquire") == 1
    # A connection error has no opened byte response, but still releases the lease.
    assert events.count("release") == 1
    assert len(object_requests) == (0 if fault == "foreign" else 1)
    if fault == "none":
        assert output.getvalue() == raw
    assert "secret-signature" not in caplog.text


@pytest.mark.parametrize(
    "fault",
    [
        "expired",
        "late",
        "empty_version",
        "duplicate_version",
        "bool_ttl",
        "wrong_pack",
        "changed_lease",
    ],
)
def test_remote_read_descriptor_protection(fault):
    op, _, _, pack, plan = fixture_plan()
    began = time.monotonic()
    original = validate_read_plan(plan, op, pack, began)
    changed = copy.deepcopy(plan)
    read = changed["remote_read"]
    if fault == "expired":
        read["lease_remaining_seconds"] = 15
    elif fault == "late":
        began -= 115
    elif fault == "empty_version":
        read["url"] = "https://test.tos.example/object?versionId="
    elif fault == "duplicate_version":
        read["url"] += "&versionId="
    elif fault == "bool_ttl":
        read["lease_remaining_seconds"] = True
    elif fault == "wrong_pack":
        changed["tensor_pack"]["size_bytes"] += 1
    elif fault == "changed_lease":
        read["lease_id"] = str(uuid4())
    with pytest.raises(ValueError):
        validate_read_plan(changed, op, pack, began, original)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("renewal", ["success", "outage", "changed_identity"])
def test_stream_renews_and_stops_when_protection_is_lost(monkeypatch, async_mode, renewal):
    op, _, wire, pack, plan = fixture_plan()
    plan["remote_read"]["lease_remaining_seconds"] = 16
    events = []

    def metadata(path):
        if path.endswith("/renew"):
            events.append("renew")
            if renewal == "outage":
                raise ValueError("injected catalog outage")
            updated = copy.deepcopy(plan)
            updated["remote_read"]["lease_remaining_seconds"] = 120
            if renewal == "changed_identity":
                updated["remote_read"]["lease_id"] = str(uuid4())
            return updated
        assert path.endswith("/release")
        assert events[-1] == "closed"
        events.append("release")

    if async_mode:

        class SlowBytes(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.sleep(2.5)
                yield wire

            async def aclose(self):
                events.append("closed")

        class Transport(httpx.AsyncBaseTransport):
            def __init__(self, **kwargs):
                pass

            async def handle_async_request(self, request):
                return httpx.Response(200, stream=SlowBytes())

        monkeypatch.setattr(httpx, "AsyncHTTPTransport", Transport)

        async def request(method, path, **kwargs):
            return metadata(path)

        async def exercise():
            async with async_direct_result_stream(
                SimpleNamespace(_request=request), plan, op, pack, time.monotonic()
            ) as chunks:
                return b"".join([chunk async for chunk in chunks])

        def run():
            async def bounded():
                return await asyncio.wait_for(exercise(), timeout=8)

            return asyncio.run(bounded())
    else:

        class SlowBytes(httpx.SyncByteStream):
            def __iter__(self):
                time.sleep(2.5)
                yield wire

            def close(self):
                events.append("closed")

        class Transport(httpx.BaseTransport):
            def __init__(self, **kwargs):
                pass

            def handle_request(self, request):
                return httpx.Response(200, stream=SlowBytes())

        monkeypatch.setattr(httpx, "HTTPTransport", Transport)

        def run():
            client = SimpleNamespace(_request=lambda method, path, **kwargs: metadata(path))
            with direct_result_stream(client, plan, op, pack, time.monotonic()) as chunks:
                return b"".join(chunks)

    if renewal == "success":
        assert run() == wire
    else:
        with pytest.raises(ValueError, match="protection lost"):
            run()
    assert events == ["renew", "closed", "release"]


def test_async_cancellation_closes_response_releases_lease_and_stops_renewal(monkeypatch):
    op, _, _, pack, plan = fixture_plan()
    events = []

    async def exercise():
        entered = asyncio.Event()

        class WaitingBytes(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b"unreachable"

            async def aclose(self):
                events.append("closed")

        class Transport(httpx.AsyncBaseTransport):
            def __init__(self, **kwargs):
                pass

            async def handle_async_request(self, request):
                return httpx.Response(200, stream=WaitingBytes())

        monkeypatch.setattr(httpx, "AsyncHTTPTransport", Transport)

        async def request(method, path, **kwargs):
            assert path.endswith("/release")
            assert events == ["closed"]
            events.append("release")

        async def download():
            async with async_direct_result_stream(
                SimpleNamespace(_request=request), plan, op, pack, time.monotonic()
            ) as chunks:
                async for _ in chunks:
                    pytest.fail("download should remain pending")

        before = set(asyncio.all_tasks())
        task = asyncio.create_task(download())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert set(asyncio.all_tasks()) == before
        assert events == ["closed", "release"]

    asyncio.run(exercise())
