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

"""Tests for the asyncio client stack (AsyncAPIClient + AsyncOperationHandle)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from weaver import _async_http
from weaver._async_http import AsyncAPIClient
from weaver.config import WeaverConfig
from weaver.operations import AsyncOperationHandle, WeaverOperationError


@pytest.fixture()
def config():
    return WeaverConfig(base_url="https://test.example.com", api_key="sk-test")


def _ok_response(payload):
    resp = MagicMock()
    resp.is_success = True
    resp.status_code = 200
    resp.content = b'{"ok": true}'
    resp.json.return_value = payload
    return resp


# ---------------------------------------------------------------------------
# AsyncAPIClient retry behaviour (mirrors tests/test_http_retry.py)
# ---------------------------------------------------------------------------


class TestAsyncRetry:
    def test_post_no_retry_on_timeout(self, config, monkeypatch):
        monkeypatch.setattr(_async_http, "compute_retry_delay", lambda attempt: 0.0)

        async def run():
            client = AsyncAPIClient(config, max_retries=3)
            client._client = MagicMock()
            client._client.headers = {}
            client._client.request = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
            with pytest.raises(httpx.ReadTimeout):
                await client.post("/api/v1/models/m1/operations", json={}, max_retries=1)
            assert client._client.request.call_count == 1

        asyncio.run(run())

    def test_get_retries_until_default(self, config, monkeypatch):
        monkeypatch.setattr(_async_http, "compute_retry_delay", lambda attempt: 0.0)

        async def run():
            client = AsyncAPIClient(config, max_retries=3)
            client._client = MagicMock()
            client._client.headers = {}
            client._client.request = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
            with pytest.raises(httpx.ReadTimeout):
                await client.get("/api/v1/models/m1")
            assert client._client.request.call_count == 3

        asyncio.run(run())

    def test_connection_error_retried_with_max_retries_1(self, config, monkeypatch):
        monkeypatch.setattr(_async_http, "compute_retry_delay", lambda attempt: 0.0)

        async def run():
            client = AsyncAPIClient(config, max_retries=3)
            client._client = MagicMock()
            client._client.headers = {}
            client._client.request = AsyncMock(
                side_effect=[OSError(9, "Bad file descriptor"), _ok_response({"id": "op-1"})]
            )
            result = await client.post("/api/v1/models/m1/operations", json={}, max_retries=1)
            assert result == {"id": "op-1"}
            assert client._client.request.call_count == 2

        asyncio.run(run())

    def test_post_success_after_connection_retry_default(self, config, monkeypatch):
        monkeypatch.setattr(_async_http, "compute_retry_delay", lambda attempt: 0.0)

        async def run():
            client = AsyncAPIClient(config, max_retries=3)
            client._client = MagicMock()
            client._client.headers = {}
            client._client.request = AsyncMock(
                side_effect=[httpx.ConnectError("refused"), _ok_response({"ok": True})]
            )
            result = await client.post("/api/v1/sessions", json={})
            assert result == {"ok": True}
            assert client._client.request.call_count == 2

        asyncio.run(run())


# ---------------------------------------------------------------------------
# AsyncOperationHandle
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    """Async client stub that returns a scripted sequence of operation states."""

    def __init__(self, states):
        self._states = list(states)
        self.get_calls = 0

    async def get(self, path, *, params=None):
        self.get_calls += 1
        # Yield to the loop like a real network call would.
        await asyncio.sleep(0)
        return self._states[min(self.get_calls - 1, len(self._states) - 1)]


class TestAsyncOperationHandle:
    def test_result_polls_until_done(self, monkeypatch):
        monkeypatch.setenv("WEAVER_OPERATION_POLL_INTERVAL", "0.01")

        async def run():
            client = _FakeAsyncClient(
                [
                    {"id": "op-1", "status": "running"},
                    {"id": "op-1", "status": "done", "response": {"value": 42}},
                ]
            )
            handle = AsyncOperationHandle(
                client=client, operation_id="op-1", _cached={"id": "op-1", "status": "running"}
            )
            return await handle.result(), client.get_calls

        result, calls = asyncio.run(run())
        assert result == {"value": 42}
        assert calls >= 2

    def test_await_handle_directly(self, monkeypatch):
        monkeypatch.setenv("WEAVER_OPERATION_POLL_INTERVAL", "0.01")

        async def run():
            client = _FakeAsyncClient([{"id": "op-1", "status": "done", "response": {"value": 7}}])
            handle = AsyncOperationHandle(
                client=client, operation_id="op-1", _cached={"id": "op-1", "status": "running"}
            )
            return await handle  # __await__ shorthand for .result()

        assert asyncio.run(run()) == {"value": 7}

    def test_error_status_raises(self, monkeypatch):
        monkeypatch.setenv("WEAVER_OPERATION_POLL_INTERVAL", "0.01")

        async def run():
            client = _FakeAsyncClient([{"id": "op-1", "status": "error", "error": "boom"}])
            handle = AsyncOperationHandle(
                client=client, operation_id="op-1", _cached={"id": "op-1", "status": "running"}
            )
            with pytest.raises(WeaverOperationError):
                await handle.result()

        asyncio.run(run())

    def test_await_yields_event_loop_to_other_coroutines(self, monkeypatch):
        """The core requirement: awaiting a handle must not block the loop.

        While the handle is polling (asyncio.sleep between refreshes), an
        independent coroutine must get a chance to run.
        """
        monkeypatch.setenv("WEAVER_OPERATION_POLL_INTERVAL", "0.01")

        async def run():
            client = _FakeAsyncClient(
                [
                    {"id": "op-1", "status": "running"},
                    {"id": "op-1", "status": "running"},
                    {"id": "op-1", "status": "done", "response": "ok"},
                ]
            )
            handle = AsyncOperationHandle(
                client=client, operation_id="op-1", _cached={"id": "op-1", "status": "running"}
            )

            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.001)
                    ticks += 1

            task = asyncio.create_task(ticker())
            result = await handle.result()
            task.cancel()
            return result, ticks

        result, ticks = asyncio.run(run())
        assert result == "ok"
        assert ticks > 0  # the ticker ran concurrently while we awaited the handle

    def test_wait_all_runs_concurrently(self, monkeypatch):
        monkeypatch.setenv("WEAVER_OPERATION_POLL_INTERVAL", "0.01")

        async def run():
            handles = [
                AsyncOperationHandle(
                    client=_FakeAsyncClient([{"id": f"op-{i}", "status": "done", "response": i}]),
                    operation_id=f"op-{i}",
                    _cached={"id": f"op-{i}", "status": "running"},
                )
                for i in range(3)
            ]
            return await AsyncOperationHandle.wait_all(handles)

        assert asyncio.run(run()) == [0, 1, 2]
