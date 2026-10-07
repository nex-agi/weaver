from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

from weaver import _artifact_read
from weaver._artifacts import ArtifactFile, descriptor_files
from weaver.async_service_client import AsyncServiceClient
from weaver.service_client import ServiceClient

DATA = b"owned remote HF model bytes"
ARTIFACT = str(uuid4())
LEASE = str(uuid4())
ENTRY = ArtifactFile("nested/weights.safetensors", "", len(DATA), hashlib.sha256(DATA).hexdigest())


def plan():
    return {
        "version": 1,
        "artifact_id": ARTIFACT,
        "name": ENTRY.name,
        "tos_host": "test.tos.example",
        "remote_read": {
            "lease_id": LEASE,
            "ref": "artifact://exports/test/id@generation",
            "epoch": 1,
            "manifest_digest": "a" * 64,
            "lease_remaining_seconds": 60,
            "file": {"path": "hf/" + ENTRY.name, "size_bytes": ENTRY.size, "sha256": ENTRY.sha256},
            "url": "https://test.tos.example/object?versionId=owned-version",
        },
    }


@pytest.mark.parametrize(
    "mutation", ["host", "version", "file", "digest", "epoch", "deadline", "renew"]
)
def test_read_metadata_rejects_changed_content(mutation):
    raw = plan()
    previous = _artifact_read._plan(raw, ARTIFACT, ENTRY, time.monotonic())[2]
    if mutation == "host":
        raw["tos_host"] = "foreign.example"
    if mutation == "version":
        raw["remote_read"]["url"] += "&versionId=foreign"
    if mutation == "file":
        raw["remote_read"]["file"]["size_bytes"] += 1
    if mutation == "digest":
        raw["remote_read"]["manifest_digest"] = "unknown"
    if mutation == "epoch":
        raw["remote_read"]["epoch"] = True
    if mutation == "deadline":
        raw["remote_read"]["lease_remaining_seconds"] = 10
    if mutation == "renew":
        raw["remote_read"]["url"] = "https://test.tos.example/object?versionId=changed"
    with pytest.raises(ValueError):
        _artifact_read._plan(raw, ARTIFACT, ENTRY, time.monotonic(), previous)


class SyncStream(httpx.SyncByteStream):
    def __init__(self, data, closed, *, slow=False):
        self.data, self.closed, self.slow = data, closed, slow

    def __iter__(self):
        if self.slow:
            time.sleep(0.06)
        yield self.data

    def close(self):
        self.closed.append(True)


class AsyncStream(httpx.AsyncByteStream):
    def __init__(self, data, closed, *, slow=False):
        self.data, self.closed, self.slow = data, closed, slow

    async def __aiter__(self):
        if self.slow:
            await asyncio.sleep(0.06)
        yield self.data

    async def aclose(self):
        self.closed.append(True)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("scenario", ["success", "bad-sha", "too-long", "renew-fails", "empty"])
def test_direct_download_atomic_release_and_integrity(tmp_path, monkeypatch, async_mode, scenario):
    data = DATA
    entry = ENTRY
    if scenario == "bad-sha":
        data = b"x" * len(DATA)
    if scenario == "too-long":
        data += b"extra"
    if scenario == "empty":
        data = b""
        entry = ArtifactFile(ENTRY.name, "", 0, hashlib.sha256(b"").hexdigest())
    raw = plan()
    raw["remote_read"]["file"] = {
        "path": "hf/" + entry.name,
        "size_bytes": entry.size,
        "sha256": entry.sha256,
    }
    actions, closed, requests = [], [], []
    monkeypatch.setattr(_artifact_read, "min", lambda *_: 0.01, raising=False)

    def post(path, *, json, **_):
        assert path == f"/api/v1/artifacts/{ARTIFACT}/download-read"
        actions.append(json["action"])
        if json["action"] == "release":
            assert scenario == "empty" or closed
            return None
        if json["action"] == "renew" and scenario == "renew-fails":
            raise RuntimeError("owned simulated lease loss")
        return copy.deepcopy(raw)

    def response(request):
        requests.append(request)
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        return httpx.Response(
            200, stream=(AsyncStream if async_mode else SyncStream)(data, closed, slow=True)
        )

    if async_mode:

        class Transport:
            def __init__(self, **kwargs):
                assert kwargs == {"verify": True, "trust_env": False}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def handle_async_request(self, request):
                return response(request)

        monkeypatch.setattr(httpx, "AsyncHTTPTransport", Transport)
        client = MagicMock()
        client.post = AsyncMock(side_effect=post)

        def invoke():
            return asyncio.run(
                _artifact_read.async_download_managed_file(client, ARTIFACT, entry, tmp_path)
            )

    else:

        class Transport:
            def __init__(self, **kwargs):
                assert kwargs == {"verify": True, "trust_env": False}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def handle_request(self, request):
                return response(request)

        monkeypatch.setattr(httpx, "HTTPTransport", Transport)
        client = MagicMock()
        client.post.side_effect = post

        def invoke():
            return _artifact_read.download_managed_file(client, ARTIFACT, entry, tmp_path)

    if scenario in ("success", "empty"):
        invoke()
        assert (tmp_path / entry.name).read_bytes() == data
        count = len(actions)
        invoke()
        assert len(actions) == count, "complete files need no lease"
    else:
        with pytest.raises((ValueError, RuntimeError)):
            invoke()
        assert not (tmp_path / entry.name).exists()
    assert actions[0] == "acquire" and actions[-1] == "release"
    if scenario != "empty":
        assert "renew" in actions


@pytest.mark.parametrize("async_mode", [False, True])
def test_public_download_routes_to_managed_reader(tmp_path, monkeypatch, async_mode):
    descriptor = {
        "artifact_id": ARTIFACT,
        "managed_read": {"version": 1},
        "files": [{"name": ENTRY.name, "size": ENTRY.size, "sha256": ENTRY.sha256}],
    }
    files = descriptor_files(descriptor)
    assert files == [ENTRY]
    seen = []
    if async_mode:

        async def download(client, artifact, entry, dest):
            seen.append((artifact, entry, dest))

        monkeypatch.setattr(_artifact_read, "async_download_managed_file", download)
        client = AsyncServiceClient.__new__(AsyncServiceClient)
        client._http = MagicMock(get=AsyncMock(return_value=descriptor))
        client._resolve_weights_artifact_id = AsyncMock(return_value=ARTIFACT)
        result = asyncio.run(client.download_weights(ARTIFACT, tmp_path, verify=False))
    else:

        def download(client, artifact, entry, dest):
            seen.append((artifact, entry, dest))

        monkeypatch.setattr(_artifact_read, "download_managed_file", download)
        client = ServiceClient.__new__(ServiceClient)
        client._http = MagicMock()
        client.http.get.return_value = descriptor
        client._resolve_weights_artifact_id = MagicMock(return_value=ARTIFACT)
        result = client.download_weights(ARTIFACT, tmp_path, verify=False)
    assert result == tmp_path and seen == [(ARTIFACT, ENTRY, tmp_path)]


def test_async_hash_cancellation_waits_before_closing_file(tmp_path, monkeypatch):
    began, finished = threading.Event(), threading.Event()

    def blocking(fd):
        began.set()
        time.sleep(0.05)
        os_fstat = os.fstat(fd)
        assert os_fstat.st_ino > 0
        finished.set()

    async def run():
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            task = asyncio.create_task(_artifact_read._async_io(blocking, fd))
            while not began.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        finally:
            os.close(fd)

    asyncio.run(run())


def test_public_async_download_double_cancel_drains_hash_before_release(tmp_path, monkeypatch):
    entered, leave, finished = threading.Event(), threading.Event(), threading.Event()
    closed, actions = [], []
    original = _artifact_read.check_downloaded_file_at

    def hash_with_barrier(fd, name, entry, **kwargs):
        entered.set()
        assert leave.wait(2), "test owns a bounded IO barrier"
        original(fd, name, entry, **kwargs)
        assert os.fstat(fd).st_ino > 0
        finished.set()

    class Transport:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def handle_async_request(self, request):
            return httpx.Response(200, stream=AsyncStream(DATA, closed))

    def post(path, *, json, **_):
        actions.append(json["action"])
        if json["action"] == "release":
            assert finished.is_set() and closed
            return None
        return plan()

    monkeypatch.setattr(_artifact_read, "check_downloaded_file_at", hash_with_barrier)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", Transport)
    descriptor = {
        "artifact_id": ARTIFACT,
        "managed_read": {"version": 1},
        "files": [{"name": ENTRY.name, "size": ENTRY.size, "sha256": ENTRY.sha256}],
    }
    client = AsyncServiceClient.__new__(AsyncServiceClient)
    client._http = MagicMock(
        get=AsyncMock(return_value=descriptor), post=AsyncMock(side_effect=post)
    )
    client._resolve_weights_artifact_id = AsyncMock(return_value=ARTIFACT)

    async def run():
        download = asyncio.create_task(client.download_weights(ARTIFACT, tmp_path))

        async def cancel_twice():
            while not entered.is_set():
                await asyncio.sleep(0.001)
            download.cancel()
            await asyncio.sleep(0.01)
            download.cancel()
            await asyncio.sleep(0.01)
            assert not download.done() and "release" not in actions
            leave.set()
            with pytest.raises(asyncio.CancelledError):
                await download

        try:
            await asyncio.wait_for(cancel_twice(), timeout=2)
        finally:
            leave.set()
            if not download.done():
                download.cancel()
            await asyncio.gather(download, return_exceptions=True)

    asyncio.run(run())
    assert actions[-1] == "release" and finished.is_set()
    assert not (tmp_path / ENTRY.name).exists()


@pytest.mark.parametrize("async_mode", [False, True])
def test_managed_retry_resumes_exact_file_with_range(tmp_path, monkeypatch, async_mode):
    requests, actions = [], []
    data = bytes(range(256)) * 2048
    offset = 256 << 10
    entry = ArtifactFile(ENTRY.name, "", len(data), hashlib.sha256(data).hexdigest())

    def post(path, *, json, **_):
        actions.append(json["action"])
        if json["action"] == "release":
            return None
        raw = plan()
        raw["remote_read"]["file"].update(size_bytes=entry.size, sha256=entry.sha256)
        return raw

    class BrokenSync(httpx.SyncByteStream):
        def __iter__(self):
            yield data[:offset]
            raise httpx.ReadError("owned fixture connection drop")

    class BrokenAsync(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield data[:offset]
            raise httpx.ReadError("owned fixture connection drop")

    def response(request):
        requests.append(request)
        if len(requests) == 1:
            assert "range" not in request.headers
            return httpx.Response(200, stream=BrokenAsync() if async_mode else BrokenSync())
        assert request.headers["range"] == f"bytes={offset}-"
        assert request.url.params["versionId"] == "owned-version"
        return httpx.Response(
            206,
            headers={"content-range": f"bytes {offset}-{len(data) - 1}/{len(data)}"},
            stream=(AsyncStream if async_mode else SyncStream)(data[offset:], []),
        )

    if async_mode:

        class Transport:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def handle_async_request(self, request):
                return response(request)

        monkeypatch.setattr(httpx, "AsyncHTTPTransport", Transport)
        client = MagicMock(post=AsyncMock(side_effect=post))
        asyncio.run(_artifact_read.async_download_managed_file(client, ARTIFACT, entry, tmp_path))
    else:

        class Transport:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def handle_request(self, request):
                return response(request)

        monkeypatch.setattr(httpx, "HTTPTransport", Transport)
        client = MagicMock()
        client.post.side_effect = post
        _artifact_read.download_managed_file(client, ARTIFACT, entry, tmp_path)
    assert (tmp_path / entry.name).read_bytes() == data
    assert len(requests) == 2 and actions == ["acquire", "release"]
