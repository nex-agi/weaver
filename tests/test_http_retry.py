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

import http.server
import multiprocessing
import socketserver
import threading
from unittest.mock import MagicMock

import httpx
import pytest

from weaver import _http
from weaver._http import APIClient, WeaverAPIError, _is_connection_error, compute_retry_delay
from weaver.config import WeaverConfig


def _make_read_error_with_cause(
    errno: int = 9, msg: str = "Bad file descriptor"
) -> httpx.ReadError:
    """Build an httpx.ReadError whose __cause__ chain ends in an OSError.

    Mirrors how httpx wraps underlying socket errors via ``raise mapped_exc
    from original_oserror``.
    """
    try:
        raise OSError(errno, msg)
    except OSError as original:
        err = httpx.ReadError(f"[Errno {errno}] {msg}")
        err.__cause__ = original
        return err


def _error_response(
    code: str,
    *,
    status_code: int = 503,
    retryable: bool = True,
    retry_after: str | None = None,
) -> MagicMock:
    response = MagicMock()
    response.is_success = False
    response.status_code = status_code
    response.content = b'{"error":"temporary"}'
    response.text = "temporary"
    response.headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response.json.return_value = {
        "error": code,
        "message": "service temporarily unavailable",
        "retryable": retryable,
    }
    return response


def _ok_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.is_success = True
    response.status_code = 200
    response.content = b'{"ok":true}'
    response.json.return_value = payload
    return response


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


class TestServerDrainingRetry:
    """A pre-admission drain response is safe to retry even for POST."""

    def test_post_retries_server_draining_with_max_retries_1(self, client, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(_http.time, "sleep", sleeps.append)
        client._client = MagicMock()
        client._client.headers = {"Idempotency-Key": "sample-1"}
        client._client.request.side_effect = [
            _error_response("server_draining", retry_after="1"),
            _ok_response({"id": "op-1"}),
        ]

        result = client.post("/api/v1/sampling-sessions/s1/sample", json={}, max_retries=1)

        assert result == {"id": "op-1"}
        assert client._client.request.call_count == 2
        assert sleeps == [1.0]
        for request_call in client._client.request.call_args_list:
            assert request_call.kwargs["headers"]["Idempotency-Key"] == "sample-1"

    def test_other_retryable_post_error_remains_fatal(self, client, monkeypatch):
        sleep = MagicMock()
        monkeypatch.setattr(_http.time, "sleep", sleep)
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.return_value = _error_response("metering_unavailable")

        with pytest.raises(WeaverAPIError, match="metering_unavailable"):
            client.post("/api/v1/sampling-sessions/s1/sample", json={}, max_retries=1)

        assert client._client.request.call_count == 1
        sleep.assert_not_called()

    def test_server_draining_requires_retryable_contract(self, client, monkeypatch):
        sleep = MagicMock()
        monkeypatch.setattr(_http.time, "sleep", sleep)
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.return_value = _error_response("server_draining", retryable=False)

        with pytest.raises(WeaverAPIError, match="server_draining"):
            client.post("/api/v1/sampling-sessions/s1/sample", json={}, max_retries=1)

        assert client._client.request.call_count == 1
        sleep.assert_not_called()

    def test_server_draining_retry_budget_is_bounded(self, client, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(_http.time, "sleep", sleeps.append)
        monkeypatch.setattr(_http, "DEFAULT_SERVER_DRAINING_RETRIES", 2)
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.return_value = _error_response("server_draining", retry_after="1")

        with pytest.raises(WeaverAPIError, match="server_draining"):
            client.post("/api/v1/sampling-sessions/s1/sample", json={}, max_retries=1)

        assert client._client.request.call_count == 3
        assert sleeps == [1.0, 1.0]

    def test_retry_after_is_a_lower_bound(self):
        assert compute_retry_delay(1, "3") == 3.0
        assert compute_retry_delay(4, "1") == 4.0
        assert compute_retry_delay(2, "invalid") == 1.0


class TestConnectionErrorRetry:
    """Connection-level errors are retried regardless of max_retries."""

    def test_connection_error_retried_with_max_retries_1(self, client):
        """OSError (Bad file descriptor) retries even with max_retries=1."""
        ok_response = MagicMock()
        ok_response.is_success = True
        ok_response.status_code = 200
        ok_response.content = b'{"id": "op-1"}'
        ok_response.json.return_value = {"id": "op-1"}

        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = [
            OSError(9, "Bad file descriptor"),
            ok_response,
        ]

        result = client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert result == {"id": "op-1"}
        assert client._client.request.call_count == 2

    def test_connection_reset_retried_with_max_retries_1(self, client):
        """ConnectionResetError retries even with max_retries=1."""
        ok_response = MagicMock()
        ok_response.is_success = True
        ok_response.status_code = 200
        ok_response.content = b'{"ok": true}'
        ok_response.json.return_value = {"ok": True}

        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = [
            ConnectionResetError("Connection reset by peer"),
            ok_response,
        ]

        result = client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert result == {"ok": True}
        assert client._client.request.call_count == 2

    def test_connect_error_retried_with_max_retries_1(self, client):
        """httpx.ConnectError retries even with max_retries=1."""
        ok_response = MagicMock()
        ok_response.is_success = True
        ok_response.status_code = 200
        ok_response.content = b'{"ok": true}'
        ok_response.json.return_value = {"ok": True}

        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = [
            httpx.ConnectError("Connection refused"),
            ok_response,
        ]

        result = client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert result == {"ok": True}
        assert client._client.request.call_count == 2

    def test_connection_error_exhausts_after_default_retries(self, client):
        """Persistent connection errors raise after DEFAULT_CONNECTION_RETRIES."""
        from weaver._http import DEFAULT_CONNECTION_RETRIES

        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = OSError(9, "Bad file descriptor")

        with pytest.raises(WeaverAPIError, match="transport_unavailable") as exc_info:
            client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert exc_info.value.retryable is True
        assert isinstance(exc_info.value.__cause__, OSError)
        assert client._client.request.call_count == DEFAULT_CONNECTION_RETRIES

    def test_non_connection_error_not_retried_with_max_retries_1(self, client):
        """Non-connection errors (e.g., ReadTimeout) still respect max_retries=1."""
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = httpx.ReadTimeout("read timed out")

        with pytest.raises(httpx.ReadTimeout):
            client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert client._client.request.call_count == 1

    def test_remote_protocol_error_retried(self, client):
        """httpx.RemoteProtocolError retries even with max_retries=1."""
        ok_response = MagicMock()
        ok_response.is_success = True
        ok_response.status_code = 200
        ok_response.content = b'{"ok": true}'
        ok_response.json.return_value = {"ok": True}

        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = [
            httpx.RemoteProtocolError("Server disconnected"),
            ok_response,
        ]

        result = client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert result == {"ok": True}
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


class TestIsConnectionError:
    """_is_connection_error correctly classifies exceptions."""

    def test_direct_oserror(self):
        assert _is_connection_error(OSError(9, "Bad file descriptor"))

    def test_direct_connect_error(self):
        assert _is_connection_error(httpx.ConnectError("refused"))

    def test_direct_remote_protocol_error(self):
        assert _is_connection_error(httpx.RemoteProtocolError("reset"))

    def test_read_error_with_oserror_cause_is_connection_error(self):
        """Matches the actual traceback reported by NexRL."""
        assert _is_connection_error(_make_read_error_with_cause())

    def test_read_error_with_nested_cause_chain(self):
        """Walk multi-level __cause__ chains (httpx -> httpcore -> OSError)."""
        try:
            raise OSError(9, "Bad file descriptor")
        except OSError as original:
            mid = RuntimeError("httpcore layer")
            mid.__cause__ = original
            outer = httpx.ReadError("read error")
            outer.__cause__ = mid

        assert _is_connection_error(outer)

    def test_read_error_without_cause_is_not_connection_error(self):
        """A ReadError with no OSError in its chain is NOT retryable."""
        assert not _is_connection_error(httpx.ReadError("clean read error"))

    def test_read_timeout_is_not_connection_error(self):
        """Timeouts may happen after the request was partially delivered."""
        assert not _is_connection_error(httpx.ReadTimeout("timed out"))

    def test_generic_exception_is_not_connection_error(self):
        assert not _is_connection_error(ValueError("nope"))

    def test_context_chain_also_walked(self):
        """__context__ (implicit chain from raise-inside-except) is respected."""
        try:
            try:
                raise OSError(9, "Bad file descriptor")
            except OSError:
                raise httpx.ReadError("wrapped")  # sets __context__ implicitly
        except httpx.ReadError as e:
            assert _is_connection_error(e)

    def test_cycle_in_cause_chain_terminates(self):
        """A pathological __cause__ cycle must not infinite-loop."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert not _is_connection_error(a)


class TestReadErrorWithCauseRetried:
    """httpx.ReadError wrapping an OSError (the NexRL bug) is retried."""

    def test_read_error_ebadf_retried_with_max_retries_1(self, client):
        """Reproduces the exact NexRL traceback: ReadError[EBADF] must retry."""
        ok_response = MagicMock()
        ok_response.is_success = True
        ok_response.status_code = 200
        ok_response.content = b'{"id": "op-1"}'
        ok_response.json.return_value = {"id": "op-1"}

        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = [
            _make_read_error_with_cause(),
            ok_response,
        ]

        result = client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert result == {"id": "op-1"}
        assert client._client.request.call_count == 2

    def test_bare_read_error_respects_max_retries_1(self, client):
        """A ReadError with no OSError in its chain should NOT be force-retried."""
        client._client = MagicMock()
        client._client.headers = {}
        client._client.request.side_effect = httpx.ReadError("clean read error")

        with pytest.raises(httpx.ReadError):
            client.post("/api/v1/models/m1/operations", json={}, max_retries=1)

        assert client._client.request.call_count == 1


class TestForkSafety:
    """APIClient rebuilds httpx.Client when it observes a pid change."""

    def test_no_rebuild_when_pid_unchanged(self, client):
        """Same pid → _ensure_fresh_client is a no-op."""
        original = client._client
        client._ensure_fresh_client()
        assert client._client is original

    def test_rebuilds_when_pid_changes(self, client):
        """Simulate a fork by bumping the pid the client stored at init."""
        original = client._client
        # Pretend the client was created by some other (parent) process.
        client._pid = client._pid + 1

        client._ensure_fresh_client()

        assert client._client is not original
        import os as _os  # local import to avoid polluting module namespace

        assert client._pid == _os.getpid()

    def test_rebuild_does_not_close_inherited_client(self, client):
        """Closing the inherited client would touch the parent's FDs."""
        inherited = MagicMock()
        inherited.close = MagicMock()
        client._client = inherited
        client._pid = client._pid + 1  # pretend we were forked

        client._ensure_fresh_client()

        inherited.close.assert_not_called()
        assert client._client is not inherited

    def test_request_triggers_rebuild_on_pid_change(self, client):
        """The per-request hook rebuilds before the first call reaches httpx."""
        dead = MagicMock()
        dead.headers = {}
        dead.request.side_effect = AssertionError("inherited client must not be used after fork")
        client._client = dead
        client._pid = client._pid + 1  # simulate fork

        # With a freshly built client, request will fail with ConnectError
        # (no server listening) — that is fine; we only care the dead mock
        # never ran.
        with pytest.raises(Exception):  # ConnectError wrapped or raw
            client.post("/api/v1/sessions", json={}, max_retries=1)

        dead.request.assert_not_called()

    def test_close_is_noop_in_forked_child(self, client):
        """close() in a forked child must not touch the parent's FDs."""
        mock_client = MagicMock()
        client._client = mock_client
        client._pid = client._pid + 1  # simulate fork

        client.close()

        mock_client.close.assert_not_called()

    def test_close_runs_in_originating_process(self, client):
        """close() in the originating process closes the httpx client."""
        mock_client = MagicMock()
        client._client = mock_client

        client.close()

        mock_client.close.assert_called_once()


# Module-level HTTP handler: only top-level callables survive os.fork on
# platforms that use the 'spawn' start method (macOS default). Keeping it at
# module scope also makes pickling trivial.
class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        body = b'{"pid_ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs) -> None:  # silence test noise
        return


def _child_post(base_url: str, queue: multiprocessing.Queue, client: APIClient) -> None:
    """Entry point for the forked child in the e2e fork test."""
    try:
        # Re-create the WeaverConfig-side fields aren't needed; client state
        # travels via fork.
        result = client.post("/api/v1/echo", json={"hello": "child"}, max_retries=1)
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 - we surface the type name
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


@pytest.fixture()
def local_server():
    """Start a short-lived HTTP server on a random local port."""
    server = socketserver.TCPServer(("127.0.0.1", 0), _EchoHandler)
    server.allow_reuse_address = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


class TestForkE2E:
    """End-to-end: parent opens a connection, child inherits and must recover."""

    @pytest.mark.skipif(
        "fork" not in multiprocessing.get_all_start_methods(),
        reason="fork start method unavailable on this platform",
    )
    def test_forked_child_recovers_from_inherited_client(self, local_server):
        """Parent issues a request (opens a keep-alive socket), then forks.
        Child must rebuild its httpx.Client and successfully POST again.
        """
        config = WeaverConfig(base_url=local_server, api_key="sk-test")
        client = APIClient(config, max_retries=2)
        try:
            # Open a keep-alive socket in the parent so the child inherits
            # a real FD (the bug only reproduces when there's a live socket).
            first = client.post("/api/v1/echo", json={"hi": "parent"}, max_retries=1)
            assert first == {"pid_ok": True}

            ctx = multiprocessing.get_context("fork")
            queue: multiprocessing.Queue = ctx.Queue()
            proc = ctx.Process(target=_child_post, args=(local_server, queue, client))
            proc.start()
            proc.join(timeout=10.0)
            assert proc.exitcode == 0, f"child exited {proc.exitcode}"

            status, payload = queue.get(timeout=2.0)
            assert status == "ok", f"child failed: {payload}"
            assert payload == {"pid_ok": True}

            # Parent can still use its client after the child is gone.
            second = client.post("/api/v1/echo", json={"hi": "parent2"}, max_retries=1)
            assert second == {"pid_ok": True}
        finally:
            client.close()
