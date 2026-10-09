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

"""Pre-model preparation parity and bounded readiness waiting."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from weaver._http import WeaverAPIError
from weaver.managed_dataset_client import AsyncManagedDatasetsClient, ManagedDatasetsClient
from weaver.types import SampleRef


def response(refs):
    return {"items": [{**ref.to_payload(), "input_token_count": 3} for ref in refs]}


def pending():
    return WeaverAPIError(409, "managed_dataset_preparing", "requested samples are preparing", True)


def test_sync_preparation_waits_before_any_model_or_training_operation():
    refs = [SampleRef("synthetic", "v1", 0)]
    post = Mock(side_effect=[pending(), response(refs)])
    service = SimpleNamespace(
        http=SimpleNamespace(post=post), session_id="synthetic-session", ensure_session=Mock()
    )
    lengths = ManagedDatasetsClient(service).prepare_sample_ref_lengths(
        refs, base_model="synthetic-model", training_max_sequence_length=32, poll_interval=0.001
    )
    assert lengths[0].input_token_count == 3
    assert post.call_count == 2
    assert all(
        call.args[0] == "/api/v1/sessions/synthetic-session/managed-dataset-sample-lengths"
        for call in post.call_args_list
    )
    assert all(
        call.kwargs["json"]["training_max_sequence_length"] == 32
        and call.kwargs["max_retries"] == 1
        for call in post.call_args_list
    )


def test_async_preparation_waiting_keeps_caller_event_loop_responsive():
    async def run():
        refs = [SampleRef("synthetic", "v1", 0)]
        post = AsyncMock(side_effect=[pending(), response(refs)])
        service = SimpleNamespace(
            http=SimpleNamespace(post=post),
            session_id="synthetic-session",
            ensure_session=AsyncMock(),
        )
        ticks = []

        async def tick():
            for _ in range(3):
                ticks.append(1)
                await asyncio.sleep(0.001)

        lengths, _ = await asyncio.wait_for(
            asyncio.gather(
                AsyncManagedDatasetsClient(service).prepare_sample_ref_lengths(
                    refs,
                    base_model="synthetic-model",
                    training_max_sequence_length=32,
                    poll_interval=0.01,
                ),
                tick(),
            ),
            timeout=1,
        )
        assert lengths[0].input_token_count == 3 and len(ticks) == 3
        assert post.call_count == 2 and service.ensure_session.await_count == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "code,retryable",
    [
        ("managed_dataset_not_found", False),
        ("managed_dataset_protocol_error", False),
        ("managed_dataset_preparation_failed", False),
        ("other_failure", True),
    ],
)
def test_authorization_and_protocol_errors_are_never_readiness_retries(code, retryable):
    error = WeaverAPIError(409, code, "synthetic failure", retryable)
    post = Mock(side_effect=error)
    service = SimpleNamespace(
        http=SimpleNamespace(post=post), session_id="synthetic-session", ensure_session=Mock()
    )
    with pytest.raises(WeaverAPIError):
        ManagedDatasetsClient(service).prepare_sample_ref_lengths(
            [SampleRef("d", "v", 0)],
            base_model="synthetic-model",
            training_max_sequence_length=32,
            poll_interval=0.001,
        )
    assert post.call_count == 1


def test_timeout_and_wait_false_do_not_submit_training_work():
    service = SimpleNamespace(
        http=SimpleNamespace(post=Mock(side_effect=pending())),
        session_id="synthetic-session",
        ensure_session=Mock(),
    )
    client = ManagedDatasetsClient(service)
    with pytest.raises(TimeoutError):
        client.prepare_sample_ref_lengths(
            [SampleRef("d", "v", 0)],
            base_model="synthetic-model",
            training_max_sequence_length=32,
            timeout=0.003,
            poll_interval=0.001,
        )
    with pytest.raises(WeaverAPIError):
        client.prepare_sample_ref_lengths(
            [SampleRef("d", "v", 0)],
            base_model="synthetic-model",
            training_max_sequence_length=32,
            wait=False,
        )


def test_preparation_chunks_are_bounded_and_response_order_is_checked():
    refs = [SampleRef("d", "v", i) for i in range(257)]
    post = Mock(
        side_effect=lambda _path, **kwargs: response(
            [SampleRef(**ref) for ref in kwargs["json"]["items"]]
        )
    )
    service = SimpleNamespace(
        http=SimpleNamespace(post=post), session_id="synthetic-session", ensure_session=Mock()
    )
    result = ManagedDatasetsClient(service).prepare_sample_ref_lengths(
        refs, base_model="synthetic-model", training_max_sequence_length=32
    )
    assert [len(call.kwargs["json"]["items"]) for call in post.call_args_list] == [256, 1]
    assert [item.sample_ref for item in result] == refs
    post.side_effect = None
    post.return_value = response([SampleRef("other", "v", 0)])
    with pytest.raises(ValueError):
        ManagedDatasetsClient(service).prepare_sample_ref_lengths(
            refs[:1], base_model="synthetic-model", training_max_sequence_length=32
        )


@pytest.mark.parametrize("maximum", [0, 1, True, 3.5])
def test_invalid_budget_does_not_create_a_session(maximum):
    service = SimpleNamespace(ensure_session=Mock())
    with pytest.raises(ValueError):
        ManagedDatasetsClient(service).prepare_sample_ref_lengths(
            [SampleRef("d", "v", 0)],
            base_model="synthetic-model",
            training_max_sequence_length=maximum,
        )
    service.ensure_session.assert_not_called()


@pytest.fixture
def preparation_server():
    import json
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        counts = {}
        lock = threading.Lock()
        delay = 0.01

        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            identity = body["items"][0]["sample_idx"]
            with self.lock:
                count = self.counts.get(identity, 0) + 1
                self.counts[identity] = count
            time.sleep(self.delay)
            status = 409 if count == 1 else 200
            payload = (
                {"error": "managed_dataset_preparing", "message": "preparing", "retryable": True}
                if status == 409
                else {"items": [{**ref, "input_token_count": 3} for ref in body["items"]]}
            )
            encoded = json.dumps(payload).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.timeout(10)
def test_real_socket_async_preparation_concurrency_and_cancellation(preparation_server):
    import time

    from weaver._async_http import AsyncAPIClient
    from weaver.config import WeaverConfig

    url, handler = preparation_server

    async def run():
        http = AsyncAPIClient(WeaverConfig(base_url=url, api_key="synthetic-key"))
        service = SimpleNamespace(
            http=http, session_id="synthetic-session", ensure_session=AsyncMock()
        )
        client = AsyncManagedDatasetsClient(service)
        ticks = []
        stop = asyncio.Event()

        async def ticker():
            while not stop.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.002)

        probe = asyncio.create_task(ticker())
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        client.prepare_sample_ref_lengths(
                            [SampleRef("d", "v", i)],
                            base_model="synthetic",
                            training_max_sequence_length=32,
                            timeout=2,
                            poll_interval=0.01,
                        )
                        for i in range(12)
                    ]
                ),
                timeout=3,
            )
            assert all(result[0].input_token_count == 3 for result in results)
            assert len(ticks) >= 8
            assert all(count == 2 for count in handler.counts.values())
            handler.delay = 0.25
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    client.prepare_sample_ref_lengths(
                        [SampleRef("d", "v", 99)],
                        base_model="synthetic",
                        training_max_sequence_length=32,
                        timeout=0.05,
                    ),
                    timeout=1,
                )
            assert time.monotonic() - started < 0.2
            assert handler.counts[99] == 1
        finally:
            stop.set()
            await probe
            await http.aclose()

    asyncio.run(run())


@pytest.mark.timeout(10)
def test_real_socket_sync_slow_response_uses_remaining_io_budget(preparation_server):
    import time

    from weaver._http import APIClient
    from weaver.config import WeaverConfig

    url, handler = preparation_server
    handler.delay = 0.25
    with APIClient(WeaverConfig(base_url=url, api_key="synthetic-key")) as http:
        service = SimpleNamespace(http=http, session_id="synthetic-session", ensure_session=Mock())
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            ManagedDatasetsClient(service).prepare_sample_ref_lengths(
                [SampleRef("d", "v", 99)],
                base_model="synthetic",
                training_max_sequence_length=32,
                timeout=0.05,
            )
        assert time.monotonic() - started < 0.2
        assert handler.counts[99] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_slow_success_cannot_start_next_chunk_after_deadline(asynchronous):
    import time

    refs = [SampleRef("d", "v", i) for i in range(257)]
    calls = []

    def post(_path, **kwargs):
        calls.append(kwargs)
        time.sleep(0.02)
        return response(refs[:256])

    async def apost(_path, **kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0.02)
        return response(refs[:256])

    if asynchronous:
        service = SimpleNamespace(
            http=SimpleNamespace(post=apost), session_id="s", ensure_session=AsyncMock()
        )
        with pytest.raises(TimeoutError):
            asyncio.run(
                AsyncManagedDatasetsClient(service).prepare_sample_ref_lengths(
                    refs, base_model="m", training_max_sequence_length=32, timeout=0.01
                )
            )
    else:
        service = SimpleNamespace(
            http=SimpleNamespace(post=post), session_id="s", ensure_session=Mock()
        )
        with pytest.raises(TimeoutError):
            ManagedDatasetsClient(service).prepare_sample_ref_lengths(
                refs, base_model="m", training_max_sequence_length=32, timeout=0.01
            )
    assert len(calls) == 1
    assert "deadline" in calls[0]


@pytest.mark.parametrize("expire_on", [1, 2])
def test_async_expiry_does_not_construct_unawaited_coroutine(monkeypatch, recwarn, expire_on):
    import gc

    from weaver import managed_dataset_client as module

    checks = []

    def remaining(_deadline, _now):
        checks.append(1)
        if len(checks) == expire_on:
            raise TimeoutError("expired")
        return 1.0

    monkeypatch.setattr(module, "preparation_remaining", remaining)
    service = SimpleNamespace(
        http=SimpleNamespace(post=AsyncMock()), session_id="s", ensure_session=AsyncMock()
    )
    with pytest.raises(TimeoutError):
        asyncio.run(
            AsyncManagedDatasetsClient(service).prepare_sample_ref_lengths(
                [SampleRef("d", "v", 0)], base_model="m", training_max_sequence_length=32
            )
        )
    gc.collect()
    assert not [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
    service.http.post.assert_not_called()
    assert service.ensure_session.call_count == expire_on - 1
