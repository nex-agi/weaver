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

"""Shared metadata validation and leased direct result streams."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx

READ_MEDIA_TYPE = "application/vnd.weaver.tensor-read+json"


def decode_read_metadata(raw: bytes) -> dict[str, Any]:
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError("invalid result read metadata") from None
    if not isinstance(result, dict):
        raise ValueError("invalid result read metadata")
    return result


def lease_path(plan: dict[str, Any], operation_id: str) -> str:
    try:
        lease = plan["remote_read"]["lease_id"]
        if str(UUID(lease)) != lease or str(UUID(operation_id)) != operation_id:
            raise ValueError
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ValueError("invalid result read lease identity") from None
    return f"/api/v1/operations/{operation_id}/tensor-pack/leases/{lease}"


@dataclass(frozen=True)
class ReadPlan:
    url: str
    deadline: float
    identity: tuple[str, str, int, str]


def validate_read_plan(
    plan: dict[str, Any],
    operation_id: str,
    pack: dict[str, Any],
    started_at: float,
    previous: ReadPlan | None = None,
) -> ReadPlan:
    try:
        read = plan["remote_read"]
        actual = plan["tensor_pack"]
        url = read["url"]
        parts = urlsplit(url)
        version = parse_qs(parts.query, keep_blank_values=True).get("versionId", [])
        remaining = read["lease_remaining_seconds"]
        identity = (read["lease_id"], read["ref"], read["epoch"], read["manifest_digest"])
        valid = (
            type(plan["version"]) is int
            and plan["version"] == 1
            and plan["operation_id"] == operation_id
            and all(actual[k] == pack[k] for k in ("size_bytes", "codec", "decoded_size_bytes"))
            and actual["sha256"].lower() == pack["sha256"]
            and read["blob"]
            == {
                "schema": "weaver-tensor-pack/v1/" + pack["codec"],
                "size_bytes": pack["size_bytes"],
                "sha256": pack["sha256"],
            }
            and type(read["epoch"]) is int
            and read["epoch"] == 1
            and isinstance(read["ref"], str)
            and read["ref"].startswith("artifact://")
            and isinstance(read["manifest_digest"], str)
            and re.fullmatch("[0-9a-f]{64}", read["manifest_digest"]) is not None
            and type(remaining) is int
            and 1 <= remaining <= 3600
            and len(url) <= 16384
            and parts.scheme == "https"
            and parts.hostname == plan["tos_host"]
            and bool(parts.hostname)
            and parts.port in (None, 443)
            and not parts.username
            and not parts.password
            and not parts.fragment
            and len(version) == 1
            and version[0] not in ("", "null")
            and (previous is None or previous.identity == identity)
        )
        deadline = started_at + remaining - 10
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("invalid result read descriptor") from None
    if not valid or deadline - time.monotonic() <= 5:
        raise ValueError("invalid or expired result read protection")
    return ReadPlan(url, deadline, identity)


@contextlib.contextmanager
def direct_result_stream(
    client: Any,
    plan: dict[str, Any],
    operation_id: str,
    pack: dict[str, Any],
    started_at: float,
) -> Iterator[Iterator[bytes]]:
    path = lease_path(plan, operation_id)
    stop = threading.Event()
    failures: list[Exception] = []
    thread = None
    try:
        initial = validate_read_plan(plan, operation_id, pack, started_at)
        protection = {"deadline": initial.deadline}
        interval = min(20, (initial.deadline - time.monotonic()) / 3)

        def renew():
            while not stop.wait(interval):
                try:
                    began = time.monotonic()
                    updated = client._request("POST", path + "/renew", max_retries=1)
                    protection["deadline"] = validate_read_plan(
                        updated, operation_id, pack, began, initial
                    ).deadline
                except Exception as error:
                    failures.append(error)
                    return

        def check():
            if failures or time.monotonic() >= protection["deadline"]:
                raise ValueError("result read protection lost")

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        # The transport has no auth/cookies/proxy/redirect logic. Using it
        # directly also avoids httpx.Client INFO logging of the signed URL.
        with httpx.HTTPTransport(verify=True, trust_env=False) as transport:
            try:
                check()
                request = httpx.Request(
                    "GET",
                    initial.url,
                    headers={"Accept-Encoding": "identity"},
                    extensions={"timeout": {"connect": 5, "read": 20, "write": 20, "pool": 5}},
                )
                with contextlib.closing(transport.handle_request(request)) as response:
                    if (
                        response.status_code != 200
                        or response.headers.get("content-encoding", "identity") != "identity"
                    ):
                        raise ValueError("direct result object request failed")

                    def chunks():
                        iterator = response.iter_raw(256 << 10)
                        while True:
                            check()
                            try:
                                chunk = next(iterator)
                            except StopIteration:
                                break
                            check()
                            yield chunk

                    yield chunks()
            except httpx.HTTPError:
                raise ValueError("direct result object transport failed") from None
    finally:
        active_error = sys.exc_info()[1]
        stop.set()
        if thread is not None:
            thread.join()
        # Release happens after the byte response is closed, even on cancellation.
        try:
            client._request("POST", path + "/release", max_retries=1)
        except Exception:
            if active_error is None:
                raise


@contextlib.asynccontextmanager
async def async_direct_result_stream(
    client: Any,
    plan: dict[str, Any],
    operation_id: str,
    pack: dict[str, Any],
    started_at: float,
) -> AsyncIterator[AsyncIterator[bytes]]:
    path = lease_path(plan, operation_id)
    task = None
    failures: list[Exception] = []
    try:
        initial = validate_read_plan(plan, operation_id, pack, started_at)
        protection = {"deadline": initial.deadline}
        interval = min(20, (initial.deadline - time.monotonic()) / 3)

        async def renew():
            while True:
                await asyncio.sleep(interval)
                try:
                    began = time.monotonic()
                    updated = await client._request("POST", path + "/renew", max_retries=1)
                    protection["deadline"] = validate_read_plan(
                        updated, operation_id, pack, began, initial
                    ).deadline
                except Exception as error:
                    failures.append(error)
                    return

        def check():
            if failures or time.monotonic() >= protection["deadline"]:
                raise ValueError("result read protection lost")

        task = asyncio.create_task(renew())
        async with httpx.AsyncHTTPTransport(verify=True, trust_env=False) as transport:
            try:
                check()
                request = httpx.Request(
                    "GET",
                    initial.url,
                    headers={"Accept-Encoding": "identity"},
                    extensions={"timeout": {"connect": 5, "read": 20, "write": 20, "pool": 5}},
                )
                response = await transport.handle_async_request(request)
                try:
                    if (
                        response.status_code != 200
                        or response.headers.get("content-encoding", "identity") != "identity"
                    ):
                        raise ValueError("direct result object request failed")

                    async def chunks():
                        iterator = response.aiter_raw(256 << 10)
                        while True:
                            check()
                            try:
                                chunk = await iterator.__anext__()
                            except StopAsyncIteration:
                                break
                            check()
                            yield chunk

                    yield chunks()
                finally:
                    await response.aclose()
            except httpx.HTTPError:
                raise ValueError("direct result object transport failed") from None
    finally:
        active_error = sys.exc_info()[1]
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            await client._request("POST", path + "/release", max_retries=1)
        except Exception:
            if active_error is None:
                raise
