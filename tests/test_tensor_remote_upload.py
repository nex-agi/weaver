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

"""Actual SDK transport dispatch, safe fallback and retained-source recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from uuid import uuid4

import httpx
import pytest

from weaver import _tensor_upload as transport
from weaver._async_http import AsyncAPIClient
from weaver._http import APIClient
from weaver.config import WeaverConfig
from weaver.tensor_transport import TensorPack
from weaver.tensor_upload import TensorUploadInterrupted


@pytest.mark.parametrize("phase", ["record", "part"])
def test_event_loop_shutdown_joins_disk_worker_before_source_cleanup(tmp_path, monkeypatch, phase):
    fixture = UploadFixture(tmp_path)
    entered, release, main_returned, cleanup = (threading.Event() for _ in range(4))
    errors = []
    if phase == "record":
        original = transport._record

        def blocked(*args):
            entered.set()
            assert release.wait(10)
            return original(*args)

        monkeypatch.setattr(transport, "_record", blocked)
    else:
        original = transport._part_chunks

        def blocked(*args):
            entered.set()
            assert release.wait(10)
            yield from original(*args)

        monkeypatch.setattr(transport, "_part_chunks", blocked)
    monkeypatch.setattr(
        transport, "_async_bulk_client", lambda timeout: httpx.MockTransport(fixture.bulk)
    )

    async def main():
        client = AsyncAPIClient(
            WeaverConfig(base_url="https://weaver.example", api_key="existing-sdk-test-key")
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://weaver.example",
            headers=client._headers,
            transport=httpx.MockTransport(fixture.control),
            trust_env=False,
        )

        async def upload():
            try:
                await client.post_tensor_multipart(
                    fixture.path, request=fixture.request, tensor_pack=fixture.pack
                )
            finally:
                cleanup.set()
                fixture.pack.close()
                await client.aclose()

        asyncio.create_task(upload())
        deadline = time.monotonic() + 5
        while not entered.is_set():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        main_returned.set()
        # asyncio.run now cancels ALL tasks, including an internal to_thread
        # Task. An executor Future must remain owned until the actual IO exits.

    def runner():
        try:
            asyncio.run(main())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=runner)
    thread.start()
    try:
        assert main_returned.wait(5)
        assert not cleanup.wait(
            0.2
        ), "request cleanup escaped while its disk thread was still running"
        assert fixture.pack.path.is_file()
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors
    assert cleanup.is_set() and fixture.pack.path.is_file()
    assert fixture.pack._remote_recovery is not None
    record, retained = transport._load_journal(
        fixture.pack._remote_recovery, "https://weaver.example"
    )
    assert retained.path == fixture.pack.path and record["request"] == fixture.request


class UploadFixture:
    def __init__(self, tmp_path, fault=""):
        self.path = "/api/v1/models/" + str(uuid4()) + "/forward-passes"
        self.wire = b"source-tensor-data" * (300000 if fault == "cookie" else 10)
        tmp_path.mkdir(parents=True, exist_ok=True)
        source = tmp_path / "pack.bin"
        source.write_bytes(self.wire)
        self.pack = TensorPack(source, len(self.wire), hashlib.sha256(self.wire).hexdigest())
        self.request = {
            "payload": {"tensor_transport": "http-binary", "tensor_compression": "raw", "seq_id": 1}
        }
        self.operation = str(uuid4())
        self.session = str(uuid4())
        self.publisher = str(uuid4())
        self.ref = "artifact://generated/inputs/" + str(uuid4()) + "@" + str(uuid4())
        self.fault = fault
        self.prepares = 0
        self.submissions = 0
        self.multipart = 0
        self.parts = []
        self.upload_id = None
        self.first_session = None

    def view(self, ready=False):
        return {
            "version": 1,
            "upload_id": self.upload_id,
            "operation_id": self.operation,
            "state": "REMOTE_PUBLISHED" if ready else "REMOTE_PREPARED",
            "ready": ready,
            "tos_hosts": ["test.tos.example"],
            "session": {
                "upload_session_id": self.session,
                "operation_id": self.publisher,
                "ref": self.ref,
                "state": "DURABLE" if ready else "UPLOADING",
                "blob": {
                    "schema": "weaver-tensor-input-pack/v1/raw",
                    "size_bytes": len(self.wire),
                    "sha256": self.pack.sha256,
                },
                "part_size": 4 << 20,
                "part_count": (len(self.wire) + (4 << 20) - 1) // (4 << 20),
            },
        }

    def control(self, request):
        assert request.headers.get("X-Weaver-API-Key") == "existing-sdk-test-key"
        path = request.url.path
        if request.headers.get("Content-Type", "").startswith("multipart/"):
            self.multipart += 1
            return httpx.Response(202, json={"id": self.operation})
        body = json.loads(request.content) if request.content else {}
        if path.endswith("/tensor-uploads"):
            self.prepares += 1
            if self.upload_id is None:
                self.upload_id = body["upload_id"]
            assert body["upload_id"] == self.upload_id
            if self.fault == "missing":
                return httpx.Response(404, text="404 page not found")
            if self.fault == "disabled" or (
                self.fault == "ambiguous-disabled" and self.prepares > 1
            ):
                return httpx.Response(
                    409, json={"error": "tensor_remote_upload_disabled", "retryable": False}
                )
            if (
                self.fault == "ambiguous-disabled"
                or self.fault == "lost-prepare"
                and self.prepares == 1
            ):
                raise httpx.ReadError("lost prepare ACK", request=request)
            if self.fault == "foreign":
                return httpx.Response(404, json={"error": "model_not_found"})
            return httpx.Response(200, json=self.view(ready=self.submissions > 0))
        if path.endswith("/parts"):
            number = body["part_number"]
            offset = (number - 1) * (4 << 20)
            size = min(4 << 20, len(self.wire) - offset)
            result = {
                "upload_session_id": self.session,
                "part_number": number,
                "size_bytes": size,
                "headers": {"Content-Length": str(size), "Content-MD5": body["content_md5"]},
                "remaining_seconds": 130,
                "url": "https://test.tos.example/object?uploadId=owned&partNumber="
                + str(number)
                + "&X-Tos-Expires=120&X-Tos-Date="
                + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            }
            if self.fault == "host":
                result["url"] = result["url"].replace("test.tos.example", "foreign.example")
            if self.fault == "auth":
                result["headers"]["Authorization"] = "must-not-forward"
            if self.fault == "part":
                result["part_number"] += 1
            return httpx.Response(200, json=result)
        if path.endswith("/complete"):
            return httpx.Response(200, json=self.view(ready=True))
        if request.method == "GET":
            return httpx.Response(200, json=self.view(ready=True))
        if path.endswith("/forward-passes"):
            self.submissions += 1
            assert body == {**self.request, "tensor_input_upload_id": self.upload_id}
            assert request.headers["Idempotency-Key"] == self.upload_id
            if self.fault == "lost-admission":
                raise httpx.ReadError("lost admission ACK", request=request)
            return httpx.Response(202, json={"id": self.operation})
        raise AssertionError("unexpected control path")

    def bulk(self, request):
        assert request.url.host == "test.tos.example"
        assert not {"authorization", "cookie", "x-weaver-api-key"} & set(request.headers)
        self.parts.append(request.content)
        if self.fault == "bulk-failure":
            raise httpx.WriteError("secret signed URL must not escape", request=request)
        if self.fault == "redirect":
            return httpx.Response(307, headers={"Location": "https://foreign.example"})
        return httpx.Response(200, headers={"Set-Cookie": "cloud-cookie=never-forward; Path=/"})


def run_fixture(fixture, async_mode, monkeypatch):
    if async_mode:

        async def run():
            client = AsyncAPIClient(
                WeaverConfig(base_url="https://weaver.example", api_key="existing-sdk-test-key")
            )
            await client._client.aclose()
            client._client = httpx.AsyncClient(
                base_url="https://weaver.example",
                headers=client._headers,
                transport=httpx.MockTransport(fixture.control),
            )
            monkeypatch.setattr(
                transport,
                "_async_bulk_client",
                lambda timeout: httpx.MockTransport(fixture.bulk),
            )
            try:
                return await client.post_tensor_multipart(
                    fixture.path, request=fixture.request, tensor_pack=fixture.pack
                )
            finally:
                await client.aclose()

        return asyncio.run(run())
    client = APIClient(
        WeaverConfig(base_url="https://weaver.example", api_key="existing-sdk-test-key")
    )
    client._client.close()
    client._client = httpx.Client(
        base_url="https://weaver.example",
        headers=client._headers,
        transport=httpx.MockTransport(fixture.control),
    )
    monkeypatch.setattr(
        transport,
        "_bulk_client",
        lambda timeout: httpx.MockTransport(fixture.bulk),
    )
    try:
        return client.post_tensor_multipart(
            fixture.path, request=fixture.request, tensor_pack=fixture.pack
        )
    finally:
        client.close()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("fault", ["", "lost-prepare", "cookie"])
def test_upload_actual_sdk_dispatch(tmp_path, monkeypatch, async_mode, fault):
    fixture = UploadFixture(tmp_path, fault)
    result = run_fixture(fixture, async_mode, monkeypatch)
    assert result["id"] == fixture.operation
    assert b"".join(fixture.parts) == fixture.wire
    assert fixture.multipart == 0 and fixture.submissions == 1
    assert fixture.pack._remote_recovery is None
    fixture.pack.close()
    assert not fixture.pack.path.exists()
    assert not list(tmp_path.glob(".weaver-tensor-upload-*.json"))


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("fault", ["disabled", "missing"])
def test_definitive_initial_rejection_preserves_legacy_multipart(
    tmp_path, monkeypatch, async_mode, fault
):
    fixture = UploadFixture(tmp_path, fault)
    assert run_fixture(fixture, async_mode, monkeypatch)["id"] == fixture.operation
    assert fixture.multipart == 1 and not fixture.parts
    fixture.pack.close()
    assert not fixture.pack.path.exists()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "foreign",
        "host",
        "auth",
        "part",
        "redirect",
        "bulk-failure",
        "ambiguous-disabled",
        "lost-admission",
    ],
)
def test_no_unsafe_fallback_and_unique_source_retained(tmp_path, monkeypatch, async_mode, fault):
    fixture = UploadFixture(tmp_path, fault)
    with pytest.raises(TensorUploadInterrupted) as caught:
        run_fixture(fixture, async_mode, monkeypatch)
    failure = caught.value
    fixture.pack.close()
    assert fixture.pack.path.read_bytes() == fixture.wire
    assert failure.recovery_path.is_file() and failure.upload_id == fixture.upload_id
    assert failure.recovery_path.stat().st_mode & 0o777 == 0o600
    assert "existing-sdk-test-key" not in failure.recovery_path.read_text()
    assert "signed URL" not in str(failure) and fixture.multipart == 0
    if fault in {"foreign", "host", "auth", "part", "ambiguous-disabled"}:
        assert not fixture.parts
    fixture.pack._retain_source = False
    fixture.pack.close()
    failure.recovery_path.unlink()


@pytest.mark.parametrize("async_mode", [False, True])
def test_bulk_does_not_log_signed_url(tmp_path, monkeypatch, caplog, async_mode):
    import logging

    fixture = UploadFixture(tmp_path)
    caplog.set_level(logging.INFO, logger="httpx")
    run_fixture(fixture, async_mode, monkeypatch)
    assert "/object?" not in caplog.text
    fixture.pack.close()


@pytest.mark.timeout(15)
def test_async_cancel_after_admission_keeps_valid_recovery(tmp_path, monkeypatch):
    import threading

    fixture = UploadFixture(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = transport._finish_journal

    def finish(pack):
        entered.set()
        assert release.wait(10)
        original(pack)

    monkeypatch.setattr(transport, "_finish_journal", finish)
    monkeypatch.setattr(
        transport, "_async_bulk_client", lambda timeout: httpx.MockTransport(fixture.bulk)
    )

    async def exercise():
        client = AsyncAPIClient(
            WeaverConfig(base_url="https://weaver.example", api_key="existing-sdk-test-key")
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://weaver.example",
            headers=client._headers,
            transport=httpx.MockTransport(fixture.control),
        )
        task = asyncio.create_task(
            client.post_tensor_multipart(
                fixture.path, request=fixture.request, tensor_pack=fixture.pack
            )
        )
        try:
            deadline = time.monotonic() + 5
            while not entered.is_set():
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
            assert fixture.submissions == 1
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            fixture.pack.close()
            recovery = getattr(caught.value, "recovery_path", None)
            assert (
                recovery is not None and recovery.is_file()
            ), "cancelled delivery needs a real resume journal"
            assert fixture.pack.path.is_file()
            assert json.loads(recovery.read_text())["upload_id"] == fixture.upload_id
            recovered = await client.resume_tensor_upload(str(recovery))
            assert recovered["id"] == fixture.operation
            assert not recovery.exists() and not fixture.pack.path.exists()
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await client.aclose()

    asyncio.run(asyncio.wait_for(exercise(), timeout=12))


@pytest.mark.timeout(20)
def test_async_public_upload_socket_concurrency_and_ticker(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    fixtures = [UploadFixture(tmp_path / str(number)) for number in range(4)]
    by_model = {fixture.path.split("/")[4]: fixture for fixture in fixtures}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            fixture = by_model[self.path.split("/")[4]]
            reply = fixture.control(
                httpx.Request(
                    "POST", "http://localhost" + self.path, headers=dict(self.headers), content=body
                )
            )
            content = reply.content
            self.send_response(reply.status_code)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def cloud(request):
        await asyncio.sleep(0.1)
        assert not {"authorization", "cookie", "x-weaver-api-key"} & set(request.headers)
        return httpx.Response(200)

    monkeypatch.setattr(transport, "_async_bulk_client", lambda timeout: httpx.MockTransport(cloud))

    async def exercise():
        ticks = 0
        running = True

        async def tick():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(tick())
        client = AsyncAPIClient(
            WeaverConfig(
                base_url=f"http://127.0.0.1:{server.server_port}", api_key="existing-sdk-test-key"
            )
        )
        try:
            results = await asyncio.gather(
                *(
                    client.post_tensor_multipart(
                        fixture.path, request=fixture.request, tensor_pack=fixture.pack
                    )
                    for fixture in fixtures
                )
            )
            assert [item["id"] for item in results] == [fixture.operation for fixture in fixtures]
            assert ticks >= 10
        finally:
            running = False
            ticker.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            await client.aclose()

    async def bounded_main():
        await asyncio.wait_for(exercise(), 15)
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]

    try:
        asyncio.run(bounded_main())
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
        for fixture in fixtures:
            fixture.pack.close()
