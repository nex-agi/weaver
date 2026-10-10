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

"""Leased direct HF downloads. API traffic contains metadata, never file bytes."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import re
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import httpx

from ._artifacts import (
    ArtifactFile,
    check_downloaded_file_at,
    is_file_already_complete_at,
    resume_offset_at,
)
from ._safeio import open_for_write, open_parent_fd, rename_within, supports_dir_fd


def _plan(
    raw: Any, artifact_id: str, entry: ArtifactFile, began: float, previous: tuple | None = None
) -> tuple[str, float, tuple]:
    try:
        read = raw["remote_read"]
        parts = urlsplit(read["url"])
        versions = parse_qs(parts.query, keep_blank_values=True).get("versionId", [])
        identity = (
            read["lease_id"],
            read["ref"],
            read["epoch"],
            read["manifest_digest"],
            tuple(versions),
        )
        valid = (
            type(raw["version"]) is int
            and raw["version"] == 1
            and raw["artifact_id"] == artifact_id
            and raw["name"] == entry.name
            and str(UUID(read["lease_id"])) == read["lease_id"]
            and UUID(read["lease_id"]).int != 0
            and type(read["epoch"]) is int
            and read["epoch"] == 1
            and isinstance(read["ref"], str)
            and read["ref"].startswith("artifact://")
            and re.fullmatch(r"[0-9a-f]{64}", read["manifest_digest"]) is not None
            and read["file"]
            == {"path": "hf/" + entry.name, "size_bytes": entry.size, "sha256": entry.sha256}
            and type(read["file"]["size_bytes"]) is int
            and type(read["lease_remaining_seconds"]) is int
            and 16 <= read["lease_remaining_seconds"] <= 3600
            and len(read["url"]) <= 16384
            and parts.scheme == "https"
            and parts.hostname == raw["tos_host"]
            and bool(parts.hostname)
            and parts.port in (None, 443)
            and not parts.username
            and not parts.password
            and not parts.fragment
            and len(versions) == 1
            and versions[0] not in ("", "null")
            and (previous is None or previous == identity)
        )
        deadline = began + read["lease_remaining_seconds"] - 10
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("invalid HF read protection") from None
    if not valid or deadline - time.monotonic() <= 5:
        raise ValueError("invalid or expired HF read protection")
    return read["url"], deadline, identity


def _body(entry: ArtifactFile, action: str, lease: str = "") -> dict:
    return {"name": entry.name, "action": action, "lease_id": lease}


def _target(dest: Path, entry: ArtifactFile):
    if not supports_dir_fd():
        raise RuntimeError("managed HF downloads require filesystem descriptor support")
    relative = PurePosixPath(entry.name)
    return open_parent_fd(dest, relative, create=True), relative.name, relative.name + ".part"


async def _async_io(function, *args, **kwargs):
    # Hashing multi-GB files must leave the event loop free to renew leases.
    # A cancelled caller waits for its IO worker before closing the inode fd.
    task = asyncio.get_running_loop().run_in_executor(
        None, functools.partial(function, *args, **kwargs)
    )
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


def _response_ok(response: httpx.Response, offset: int, size: int) -> None:
    expected = 206 if offset else 200
    if (
        response.status_code != expected
        or response.headers.get("content-encoding", "identity") != "identity"
    ):
        raise ValueError("HF object request rejected")
    if offset and response.headers.get("content-range") != f"bytes {offset}-{size - 1}/{size}":
        raise ValueError("HF object range changed")
    length = response.headers.get("content-length")
    if length is not None and length != str(size - offset):
        raise ValueError("HF object length changed")


def _request(url: str, offset: int) -> httpx.Request:
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return httpx.Request(
        "GET",
        url,
        headers=headers,
        extensions={"timeout": {"connect": 5, "read": 20, "write": 20, "pool": 5}},
    )


def download_managed_file(client: Any, artifact_id: str, entry: ArtifactFile, dest: Path) -> None:
    fd, final, part = _target(dest, entry)
    path = f"/api/v1/artifacts/{artifact_id}/download-read"
    lease = ""
    stop = threading.Event()
    thread = None
    failures: list[Exception] = []
    try:
        if is_file_already_complete_at(fd, final, entry, verify=True):
            return
        began = time.monotonic()
        request = _body(entry, "acquire")
        request["request_id"] = str(uuid4())
        raw = client.post(path, json=request)
        # Save a canonical returned lease for cleanup even if validation fails.
        candidate = raw.get("remote_read", {}).get("lease_id") if isinstance(raw, dict) else None
        if isinstance(candidate, str) and str(UUID(candidate)) == candidate:
            lease = candidate
        url, deadline, identity = _plan(raw, artifact_id, entry, began)
        protection = {"url": url, "deadline": deadline}

        def check():
            if failures or time.monotonic() >= protection["deadline"]:
                raise ValueError("HF read protection lost")

        def renew():
            while not stop.wait(min(20, (deadline - began) / 3)):
                try:
                    started = time.monotonic()
                    updated = client.post(path, json=_body(entry, "renew", lease), max_retries=1)
                    url, until, _ = _plan(updated, artifact_id, entry, started, identity)
                    protection.update(url=url, deadline=until)
                except Exception as error:
                    failures.append(error)
                    return

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        # Low-level transport avoids API credentials, redirects, proxies and
        # client INFO logs containing signed object URLs.
        with httpx.HTTPTransport(verify=True, trust_env=False) as transport:
            for attempt in range(4):
                check()
                offset = resume_offset_at(fd, part, entry)
                if offset == entry.size:
                    if entry.size == 0:
                        with open_for_write(fd, part, append=False):
                            pass
                    break
                try:
                    response = transport.handle_request(_request(protection["url"], offset))
                    with (
                        contextlib.closing(response),
                        open_for_write(fd, part, append=offset > 0) as sink,
                    ):
                        _response_ok(response, offset, entry.size)
                        received = offset
                        for chunk in response.iter_raw(256 << 10):
                            check()
                            received += len(chunk)
                            if received > entry.size:
                                raise ValueError("HF object exceeds manifest length")
                            sink.write(chunk)
                        check()
                        if received != entry.size:
                            raise ValueError("HF object truncated")
                    break
                except httpx.TransportError:
                    if attempt == 3:
                        raise ValueError("HF object transport failed") from None
            check_downloaded_file_at(fd, part, entry, verify=True)
            check()
            rename_within(fd, part, final)
    finally:
        stop.set()
        if thread is not None:
            thread.join()
        try:
            if lease:
                client.post(path, json=_body(entry, "release", lease), max_retries=1)
        finally:
            os.close(fd)


async def async_download_managed_file(
    client: Any, artifact_id: str, entry: ArtifactFile, dest: Path
) -> None:
    fd, final, part = _target(dest, entry)
    path = f"/api/v1/artifacts/{artifact_id}/download-read"
    lease = ""
    task = None
    failures: list[Exception] = []
    try:
        if await _async_io(is_file_already_complete_at, fd, final, entry, verify=True):
            return
        began = time.monotonic()
        request = _body(entry, "acquire")
        request["request_id"] = str(uuid4())
        raw = await client.post(path, json=request)
        candidate = raw.get("remote_read", {}).get("lease_id") if isinstance(raw, dict) else None
        if isinstance(candidate, str) and str(UUID(candidate)) == candidate:
            lease = candidate
        url, deadline, identity = _plan(raw, artifact_id, entry, began)
        protection = {"url": url, "deadline": deadline}

        def check():
            if failures or time.monotonic() >= protection["deadline"]:
                raise ValueError("HF read protection lost")

        async def renew():
            while True:
                await asyncio.sleep(min(20, (deadline - began) / 3))
                try:
                    started = time.monotonic()
                    updated = await client.post(
                        path, json=_body(entry, "renew", lease), max_retries=1
                    )
                    url, until, _ = _plan(updated, artifact_id, entry, started, identity)
                    protection.update(url=url, deadline=until)
                except Exception as error:
                    failures.append(error)
                    return

        task = asyncio.create_task(renew())
        async with httpx.AsyncHTTPTransport(verify=True, trust_env=False) as transport:
            for attempt in range(4):
                check()
                offset = resume_offset_at(fd, part, entry)
                if offset == entry.size:
                    if entry.size == 0:
                        with open_for_write(fd, part, append=False):
                            pass
                    break
                try:
                    response = await transport.handle_async_request(
                        _request(protection["url"], offset)
                    )
                    try:
                        _response_ok(response, offset, entry.size)
                        with open_for_write(fd, part, append=offset > 0) as sink:
                            received = offset
                            async for chunk in response.aiter_raw(256 << 10):
                                check()
                                received += len(chunk)
                                if received > entry.size:
                                    raise ValueError("HF object exceeds manifest length")
                                sink.write(chunk)
                            check()
                            if received != entry.size:
                                raise ValueError("HF object truncated")
                    finally:
                        await response.aclose()
                    break
                except httpx.TransportError:
                    if attempt == 3:
                        raise ValueError("HF object transport failed") from None
            await _async_io(check_downloaded_file_at, fd, part, entry, verify=True)
            check()
            rename_within(fd, part, final)
    finally:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            if lease:
                await client.post(path, json=_body(entry, "release", lease), max_retries=1)
        finally:
            os.close(fd)
