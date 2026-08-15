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

"""End-to-end deadlock/liveness tests for the async client over real sockets.

Unlike the mocked tests, these drive ``AsyncServiceClient`` and
``AsyncOperationHandle`` against a real threaded HTTP server with high
concurrency, a live heartbeat task, and clean shutdown. Every test carries a
hard ``pytest.mark.timeout`` plus an inner ``asyncio.wait_for`` so that a
deadlock or an accidental blocking call FAILS the test instead of hanging.
"""

import asyncio
import json
import os
import re
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from weaver.async_service_client import AsyncServiceClient
from weaver.operations import AsyncOperationHandle

# Operations report "running" for this many polls, then "done".
_POLLS_TILL_DONE = 2


class _Handler(BaseHTTPRequestHandler):
    """Minimal stand-in for the Weaver operations API."""

    # HTTP/1.1 enables keep-alive so httpx reuses connections instead of
    # opening a fresh socket (and server thread) per request.
    protocol_version = "HTTP/1.1"

    polls: dict = defaultdict(int)
    op_counter = 0
    lock = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        cls.polls = defaultdict(int)
        cls.op_counter = 0

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        path = self.path

        if path == "/api/v1/sessions":
            return self._send(200, {"id": "sess-1"})
        if path.endswith("/heartbeat"):
            return self._send(200, {})
        if re.match(r"/api/v1/sessions/[^/]+/models$", path):
            return self._send(200, {"id": "model-1", "base_model": "x"})
        if path.endswith("/terminate"):
            return self._send(200, {})

        # Anything else is an operation submission.
        with _Handler.lock:
            _Handler.op_counter += 1
            op_id = f"op-{_Handler.op_counter}"
        return self._send(200, {"id": op_id, "status": "running"})

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        match = re.match(r"/api/v1/operations/(.+)", self.path)
        if match:
            op_id = match.group(1)
            with _Handler.lock:
                _Handler.polls[op_id] += 1
                count = _Handler.polls[op_id]
            if count >= _POLLS_TILL_DONE:
                return self._send(200, {"id": op_id, "status": "done", "response": {"op": op_id}})
            return self._send(200, {"id": op_id, "status": "running"})
        if self.path.partition("?")[0] == "/api/v1/supported-models":
            return self._send(200, {"items": [], "pagination": {"total_count": 0}})
        return self._send(404, {})

    def log_message(self, *_args, **_kwargs) -> None:  # silence test noise
        return


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("WEAVER_OPERATION_POLL_INTERVAL", "0.02")
    # The real-socket fixture must never traverse a developer/CI HTTP proxy.
    # Some environments define NO_PROXY but omit loopback addresses, which
    # turns this liveness test into a proxy load test and produces spurious
    # 502 responses under its 200-request burst.
    no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    loopback_no_proxy = ",".join(part for part in (no_proxy, "127.0.0.1", "localhost") if part)
    monkeypatch.setenv("NO_PROXY", loopback_no_proxy)
    monkeypatch.setenv("no_proxy", loopback_no_proxy)
    _Handler.reset()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


def _svc(base_url: str, **kwargs) -> AsyncServiceClient:
    return AsyncServiceClient(base_url=base_url, api_key="sk-test", **kwargs)


@pytest.mark.timeout(30)
def test_high_concurrency_no_deadlock(server):
    """200 operations submitted and awaited concurrently must all complete."""

    async def run():
        async with _svc(server, heartbeat_interval=0.05) as svc:
            handles = await asyncio.wait_for(
                asyncio.gather(
                    *[svc.enqueue_operation("/op/submit", {"i": i}) for i in range(200)]
                ),
                timeout=20,
            )
            results = await asyncio.wait_for(
                asyncio.gather(*[h.result() for h in handles]), timeout=20
            )
            return results

    results = asyncio.run(run())
    assert len(results) == 200
    assert all(r["op"].startswith("op-") for r in results)
    assert len({r["op"] for r in results}) == 200  # no cross-talk between ops


@pytest.mark.timeout(30)
def test_event_loop_stays_responsive_under_load(server):
    """A high-frequency ticker keeps advancing while 50 ops are in flight.

    If any blocking call sneaks into the async path, the loop would stall and
    the ticker would barely advance.
    """

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.001)
                ticks += 1

        async with _svc(server, heartbeat_interval=0.05) as svc:
            task = asyncio.create_task(ticker())
            handles = await asyncio.gather(
                *[svc.enqueue_operation("/op/submit", {}) for _ in range(50)]
            )
            await asyncio.wait_for(asyncio.gather(*[h.result() for h in handles]), timeout=20)
            task.cancel()
            return ticks

    ticks = asyncio.run(run())
    assert ticks > 20  # the loop ran the ticker many times during the awaits


@pytest.mark.timeout(30)
def test_same_handle_awaited_concurrently(server):
    """Awaiting the same handle from two coroutines must not deadlock."""

    async def run():
        async with _svc(server) as svc:
            handle = await svc.enqueue_operation("/op/submit", {})
            return await asyncio.wait_for(
                asyncio.gather(handle.result(), handle.result()), timeout=20
            )

    a, b = asyncio.run(run())
    assert a == b


@pytest.mark.timeout(30)
def test_wait_all_concurrent(server):
    async def run():
        async with _svc(server) as svc:
            handles = await asyncio.gather(
                *[svc.enqueue_operation("/op/submit", {}) for _ in range(20)]
            )
            return await asyncio.wait_for(AsyncOperationHandle.wait_all(handles), timeout=20)

    results = asyncio.run(run())
    assert len(results) == 20


@pytest.mark.timeout(30)
def test_lifecycle_create_model_and_shutdown(server):
    """connect -> create_model -> aclose (terminate + heartbeat cancel) must not hang."""

    async def run():
        async with _svc(server, heartbeat_interval=0.05) as svc:
            tc = await svc.create_model(base_model="x")
            assert tc.model_id == "model-1"
            # let the heartbeat fire a few times concurrently
            await asyncio.sleep(0.15)
        return True

    assert asyncio.run(run()) is True


@pytest.mark.timeout(30)
def test_double_aclose_is_idempotent(server):
    async def run():
        svc = _svc(server, heartbeat_interval=0.05)
        await svc.connect()
        await svc.aclose()
        await svc.aclose()  # second close must be a no-op, not a hang
        return True

    assert asyncio.run(run()) is True
