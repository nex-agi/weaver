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
import atexit as _atexit
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from weaver import _async_http
from weaver._async_http import AsyncAPIClient
from weaver._http import WeaverAPIError
from weaver.async_service_client import AsyncServiceClient
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

    def test_explicit_max_retries_is_hard_bound_for_retryable_503(self, config, monkeypatch):
        monkeypatch.setattr(_async_http, "compute_retry_delay", lambda attempt: 0.0)

        async def run():
            draining = MagicMock()
            draining.is_success = False
            draining.status_code = 503
            draining.content = b'{"error":"server_draining"}'
            draining.text = "service temporarily unavailable"
            draining.headers = {}
            draining.json.return_value = {
                "error": "server_draining",
                "message": "service temporarily unavailable",
                "retryable": True,
            }
            ok = _ok_response({"id": "op-1"})
            client = AsyncAPIClient(config, max_retries=3)
            client._client = MagicMock()
            client._client.headers = {}
            client._client.request = AsyncMock(side_effect=[draining, ok])

            with pytest.raises(WeaverAPIError, match="server_draining"):
                await client.post("/api/v1/sampling-sessions/s1/sample", json={}, max_retries=1)

            assert client._client.request.call_count == 1

        asyncio.run(run())

    def test_post_default_retries_retryable_503(self, config, monkeypatch):
        monkeypatch.setattr(_async_http, "compute_retry_delay", lambda attempt: 0.0)

        async def run():
            draining = MagicMock()
            draining.is_success = False
            draining.status_code = 503
            draining.content = b'{"error":"server_draining"}'
            draining.text = "service temporarily unavailable"
            draining.headers = {}
            draining.json.return_value = {
                "error": "server_draining",
                "message": "service temporarily unavailable",
                "retryable": True,
            }
            ok = _ok_response({"id": "op-1"})
            client = AsyncAPIClient(config, max_retries=3)
            client._client = MagicMock()
            client._client.headers = {}
            client._client.request = AsyncMock(side_effect=[draining, ok])

            result = await client.post("/api/v1/sampling-sessions/s1/sample", json={})

            assert result == {"id": "op-1"}
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

    def test_error_status_surfaces_structured_reason(self, monkeypatch):
        monkeypatch.setenv("WEAVER_OPERATION_POLL_INTERVAL", "0.01")

        async def run():
            client = _FakeAsyncClient(
                [
                    {
                        "id": "op-1",
                        "status": "error",
                        "error": "operation_failed",
                        "error_code": "context_length_exceeded",
                        "error_message": "request exceeds serving context length",
                        "error_details": {"max_context_length": 32768},
                    }
                ]
            )
            handle = AsyncOperationHandle(
                client=client, operation_id="op-1", _cached={"id": "op-1", "status": "running"}
            )
            with pytest.raises(WeaverOperationError) as exc_info:
                await handle.result()
            return exc_info.value

        error = asyncio.run(run())
        assert error.code == "context_length_exceeded"
        assert error.message == "request exceeds serving context length"
        assert error.details == {"max_context_length": 32768}

    def test_precached_error_status_raises_without_polling(self):
        async def run():
            client = _FakeAsyncClient([])
            handle = AsyncOperationHandle(
                client=client,
                operation_id="op-1",
                _cached={"id": "op-1", "status": "error", "error": "boom"},
            )
            with pytest.raises(WeaverOperationError):
                await handle.result()
            return client.get_calls

        assert asyncio.run(run()) == 0

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


# ---------------------------------------------------------------------------
# AsyncServiceClient atexit safety net (parity with sync ServiceClient)
# ---------------------------------------------------------------------------


class TestAsyncAtexit:
    """The async client must terminate created models on exit, like the sync one."""

    def test_handler_terminates_created_models_on_fresh_client(self, monkeypatch):
        """The atexit handler spins a throwaway loop + fresh client and reuses
        ``terminate_model`` for every created model."""
        svc = AsyncServiceClient(base_url="https://test.example.com", api_key="sk-test")
        svc._created_models = ["m1", "m2"]

        fresh = MagicMock()
        fresh.post = AsyncMock(return_value={})
        fresh.aclose = AsyncMock()
        monkeypatch.setattr("weaver.async_service_client.AsyncAPIClient", lambda config: fresh)

        svc._atexit_terminate_created_models()

        posted_paths = [call.args[0] for call in fresh.post.call_args_list]
        assert posted_paths == [
            "/api/v1/models/m1/terminate",
            "/api/v1/models/m2/terminate",
        ]
        fresh.aclose.assert_awaited_once()
        assert svc._closed is True
        assert svc._http is None  # fresh client is dropped after use

    def test_handler_is_noop_after_aclose(self, monkeypatch):
        svc = AsyncServiceClient(base_url="https://test.example.com", api_key="sk-test")
        svc._created_models = ["m1"]
        svc._closed = True

        factory = MagicMock()
        monkeypatch.setattr("weaver.async_service_client.AsyncAPIClient", factory)

        svc._atexit_terminate_created_models()

        factory.assert_not_called()  # already closed -> nothing to do

    def test_handler_is_noop_without_created_models(self, monkeypatch):
        svc = AsyncServiceClient(base_url="https://test.example.com", api_key="sk-test")

        factory = MagicMock()
        monkeypatch.setattr("weaver.async_service_client.AsyncAPIClient", factory)

        svc._atexit_terminate_created_models()

        factory.assert_not_called()

    def test_connect_registers_and_aclose_unregisters(self, monkeypatch):
        svc = AsyncServiceClient(base_url="https://test.example.com", api_key="sk-test")

        registered: list = []
        unregistered: list = []
        monkeypatch.setattr(_atexit, "register", lambda fn: registered.append(fn))
        monkeypatch.setattr(_atexit, "unregister", lambda fn: unregistered.append(fn))

        svc._register_atexit()
        assert svc._atexit_registered is True
        assert registered == [svc._atexit_terminate_created_models]

        asyncio.run(svc.aclose())
        assert svc._atexit_registered is False
        assert unregistered == [svc._atexit_terminate_created_models]


def _build_async_atexit_script(marker_path: str, exit_code: str = "") -> str:
    """Build a subprocess script that verifies the atexit handler terminates
    created models when the process exits without ``aclose()``.

    Runs a real in-process HTTP server so the handler's throwaway-loop request
    path is exercised end-to-end; the server appends to ``marker_path`` whenever
    a ``/terminate`` is received.
    """
    return f"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from weaver.async_service_client import AsyncServiceClient

MARKER = {marker_path!r}


class _H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        path = self.path
        if path.endswith("/terminate"):
            with open(MARKER, "a") as fh:
                fh.write(path + chr(10))
            return self._send({{}})
        if path == "/api/v1/sessions":
            return self._send({{"id": "sess-1"}})
        if path.endswith("/models"):
            return self._send({{"id": "model-xyz", "base_model": "x"}})
        return self._send({{}})

    def do_GET(self):
        return self._send({{"items": []}})

    def log_message(self, *a, **k):
        return


httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
host, port = httpd.server_address
base_url = "http://%s:%d" % (host, port)


async def setup():
    client = AsyncServiceClient(base_url=base_url, api_key="sk-test", heartbeat_interval=1000)
    await client.connect()
    await client.create_model(base_model="x", training_mode="full_ft")
    # NOTE: deliberately no `await client.aclose()` -> exercise the atexit net.


asyncio.run(setup())
{exit_code}
"""


def test_atexit_terminates_created_models_on_normal_exit():
    """A process that creates a model via AsyncServiceClient and exits without
    aclose() must still terminate the model through the atexit handler."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / "terminated.marker"
        result = subprocess.run(
            [sys.executable, "-c", _build_async_atexit_script(str(marker))],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert marker.exists(), "no /terminate sent on normal exit"
        assert "/models/model-xyz/terminate" in marker.read_text()


def test_atexit_terminates_created_models_on_exception_exit():
    """The atexit safety net must also fire on an unhandled-exception exit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / "terminated.marker"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _build_async_atexit_script(str(marker), 'raise RuntimeError("boom")'),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert marker.exists(), "no /terminate sent on exception exit"
        assert "/models/model-xyz/terminate" in marker.read_text()
