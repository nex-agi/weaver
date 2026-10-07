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

"""Bounded direct input IO; sync and async share protocol and journal semantics."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import time
from pathlib import Path
from typing import Any, cast

import httpx
from opentelemetry.propagate import inject

from ._payloads import (
    remote_tensor_fallback,
    remote_tensor_prepare_body,
    remote_tensor_upload_route,
    validate_remote_tensor_part,
    validate_remote_tensor_view,
)
from ._tensor_upload_journal import (
    _create_journal,
    _finish_journal,
    _load_journal,
    _part_chunks,
    _part_md5,
    _persist_record,
)
from .tensor_transport import TensorPack
from .tensor_upload import TensorUploadInterrupted

FALLBACK = object()
RETRIES = 3
UPLOAD_SECONDS = 600
MAX_RESPONSE_BYTES = 1 << 20


def _check_response(response: httpx.Response, expected: int) -> dict[str, Any]:
    from ._http import raise_for_response

    if response.status_code != expected:
        raise_for_response(response)
        raise ValueError("unexpected remote input response status")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("remote input metadata response exceeds its bound")
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("invalid remote input metadata response")
    return result


def _rpc(
    client: Any, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    client._ensure_fresh_client()
    last: BaseException | None = None
    for attempt in range(RETRIES):
        owned_headers = dict(headers or {})
        inject(owned_headers)
        try:
            with client._client.stream(method, path, json=body, headers=owned_headers) as reply:
                content = bytearray()
                for chunk in reply.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_RESPONSE_BYTES:
                        raise ValueError("remote input metadata response exceeds its bound")
                response = httpx.Response(
                    reply.status_code,
                    headers=reply.headers,
                    content=bytes(content),
                    request=reply.request,
                )
                response.extensions["weaver_no_prior_upload_ambiguity"] = attempt == 0
            if response.status_code not in {408, 429, 500, 502, 503, 504}:
                return response
        except httpx.HTTPError as error:
            last = error
        if attempt + 1 < RETRIES:
            time.sleep(0.1 * 2**attempt)
    if last is not None:
        raise RuntimeError("remote input control request failed") from None
    return response


async def _async_rpc(
    client: Any, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    client._ensure_fresh_client()
    last: BaseException | None = None
    for attempt in range(RETRIES):
        owned_headers = dict(headers or {})
        inject(owned_headers)
        try:
            async with client._client.stream(
                method, path, json=body, headers=owned_headers
            ) as reply:
                content = bytearray()
                async for chunk in reply.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_RESPONSE_BYTES:
                        raise ValueError("remote input metadata response exceeds its bound")
                response = httpx.Response(
                    reply.status_code,
                    headers=reply.headers,
                    content=bytes(content),
                    request=reply.request,
                )
                response.extensions["weaver_no_prior_upload_ambiguity"] = attempt == 0
            if response.status_code not in {408, 429, 500, 502, 503, 504}:
                return response
        except httpx.HTTPError as error:
            last = error
        if attempt + 1 < RETRIES:
            await asyncio.sleep(0.1 * 2**attempt)
    if last is not None:
        raise RuntimeError("remote input control request failed") from None
    return response


async def _disk(function: Any, *args: Any) -> Any:
    # File hashing/fsync/part IO can block on GPFS. Cancellation joins the worker
    # before request-finally cleanup can delete its source or abandon a journal.
    task = asyncio.get_running_loop().run_in_executor(
        None, functools.partial(function, *args)
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(task)
        with contextlib.suppress(BaseException):
            task.result()
        raise


def _record(client: Any, path: str, request: Any, pack: TensorPack) -> tuple[dict[str, Any], bool]:
    body = json.loads(json.dumps(request, allow_nan=False))
    if pack._remote_recovery is None:
        _, record = _create_journal(
            client._base_url, path, body, pack, client._client.headers.get("Idempotency-Key")
        )
        return record, True
    record, loaded = _load_journal(pack._remote_recovery, client._base_url)
    if (
        record["path"] != path
        or record["request"] != body
        or loaded.path != pack.path.absolute()
        or (loaded.size_bytes, loaded.sha256, loaded.codec, loaded.decoded_size_bytes)
        != (pack.size_bytes, pack.sha256.lower(), pack.codec, pack.decoded_size_bytes)
    ):
        raise ValueError("original tensor upload request changed")
    return record, False


def _interrupted(pack: TensorPack, record: dict[str, Any] | None) -> TensorUploadInterrupted:
    assert pack._remote_recovery is not None
    return TensorUploadInterrupted(
        pack._remote_recovery, pack.path, record["upload_id"] if record else "unknown"
    )


def _stream_part(
    bulk: httpx.BaseTransport,
    part: dict[str, Any],
    pack: TensorPack,
    offset: int,
    size: int,
    timeout: Any,
) -> None:
    chunks = _part_chunks(pack, offset, size)
    try:
        request = httpx.Request(
            "PUT",
            part["url"],
            headers=part["headers"],
            content=chunks,
            extensions={"timeout": httpx.Timeout(timeout).as_dict()},
        )
        with contextlib.closing(bulk.handle_request(request)) as response:
            if response.status_code != 200:
                raise ValueError("remote tensor part was not accepted")
    except httpx.HTTPError:
        raise RuntimeError("remote tensor part transfer failed") from None
    finally:
        chunks.close()


async def _async_stream_part(
    bulk: httpx.AsyncBaseTransport,
    part: dict[str, Any],
    pack: TensorPack,
    offset: int,
    size: int,
    timeout: Any,
) -> None:
    chunks = _part_chunks(pack, offset, size)

    async def content():
        try:
            while True:
                value = await _disk(next, chunks, None)
                if value is None:
                    break
                yield value
        finally:
            await _disk(chunks.close)

    content_generator = content()
    try:
        request = httpx.Request(
            "PUT",
            part["url"],
            headers=part["headers"],
            content=content_generator,
            extensions={"timeout": httpx.Timeout(timeout).as_dict()},
        )
        response = await bulk.handle_async_request(request)
        try:
            if response.status_code != 200:
                raise ValueError("remote tensor part was not accepted")
        finally:
            await response.aclose()
    except httpx.HTTPError:
        raise RuntimeError("remote tensor part transfer failed") from None
    finally:
        await content_generator.aclose()


def _bulk_client(timeout: Any) -> httpx.BaseTransport:
    # The low-level transport has no Client cookie jar, redirect policy or
    # httpx INFO request logger that could disclose a signed query.
    return httpx.HTTPTransport(verify=True, trust_env=False)


def _async_bulk_client(timeout: Any) -> httpx.AsyncBaseTransport:
    return httpx.AsyncHTTPTransport(verify=True, trust_env=False)


def submit_remote_tensor(client: Any, path: str, request: Any, pack: TensorPack) -> Any:
    route = remote_tensor_upload_route(path)
    if route is None:
        if pack._remote_recovery is not None:
            raise _interrupted(pack, None)
        return FALLBACK
    if not pack._remote_lock.acquire(blocking=False):
        raise ValueError("tensor pack already has an active upload")
    record = None
    try:
        pack._retain_source = True
        record, first = _record(client, path, request, pack)
        root, _ = route
        response = _rpc(client, "POST", root, remote_tensor_prepare_body(record))
        if remote_tensor_fallback(
            response,
            first_attempt=first
            and response.extensions.get("weaver_no_prior_upload_ambiguity") is True,
        ):
            _finish_journal(pack)
            return FALLBACK
        view = validate_remote_tensor_view(_check_response(response, 200), record)
        original = view
        owned = root + "/" + record["upload_id"]
        deadline = time.monotonic() + UPLOAD_SECONDS
        if not view["ready"] and view["session"]["state"] == "UPLOADING":
            cloud_id = None
            # A fresh bulk client has no Weaver auth, cookies, proxy or redirects.
            with _bulk_client(client._timeout) as bulk:
                for number in range(1, view["session"]["part_count"] + 1):
                    offset = (number - 1) * view["session"]["part_size"]
                    size = min(view["session"]["part_size"], pack.size_bytes - offset)
                    md5 = _part_md5(pack, offset, size)
                    for attempt in range(RETRIES):
                        signed = _check_response(
                            _rpc(
                                client,
                                "POST",
                                owned + "/parts",
                                {"part_number": number, "content_md5": md5},
                            ),
                            200,
                        )
                        cloud_id = validate_remote_tensor_part(
                            signed, view, number, size, md5, cloud_id
                        )
                        try:
                            _stream_part(bulk, signed, pack, offset, size, client._timeout)
                            break
                        except RuntimeError:
                            if attempt + 1 == RETRIES:
                                raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError("remote tensor upload timed out")
        while not view["ready"]:
            response = _rpc(client, "POST", owned + "/complete", {})
            view = validate_remote_tensor_view(_check_response(response, 200), record, original)
            if not view["ready"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote tensor verification timed out")
                time.sleep(0.1)
                view = validate_remote_tensor_view(
                    _check_response(_rpc(client, "GET", owned), 200), record, original
                )
        final = {**record["request"], "tensor_input_upload_id": record["upload_id"]}
        operation = _check_response(
            _rpc(client, "POST", path, final, {"Idempotency-Key": record["idempotency_key"]}), 202
        )
        if operation.get("id") != view["operation_id"]:
            raise ValueError("remote input submission changed its original operation")
        _finish_journal(pack)
        return operation
    except Exception:
        if pack._remote_recovery is not None:
            raise _interrupted(pack, record) from None
        pack._retain_source = False
        raise
    finally:
        pack._remote_lock.release()


async def async_submit_remote_tensor(client: Any, path: str, request: Any, pack: TensorPack) -> Any:
    route = remote_tensor_upload_route(path)
    if route is None:
        if pack._remote_recovery is not None:
            raise _interrupted(pack, None)
        return FALLBACK
    if not pack._remote_lock.acquire(blocking=False):
        raise ValueError("tensor pack already has an active upload")
    record = None
    delivered_journal = None
    try:
        pack._retain_source = True
        record, first = await _disk(_record, client, path, request, pack)
        root, _ = route
        response = await _async_rpc(client, "POST", root, remote_tensor_prepare_body(record))
        if remote_tensor_fallback(
            response,
            first_attempt=first
            and response.extensions.get("weaver_no_prior_upload_ambiguity") is True,
        ):
            await _disk(_finish_journal, pack)
            return FALLBACK
        view = validate_remote_tensor_view(_check_response(response, 200), record)
        original = view
        owned = root + "/" + record["upload_id"]
        deadline = time.monotonic() + UPLOAD_SECONDS
        if not view["ready"] and view["session"]["state"] == "UPLOADING":
            cloud_id = None
            async with _async_bulk_client(client._timeout) as bulk:
                for number in range(1, view["session"]["part_count"] + 1):
                    offset = (number - 1) * view["session"]["part_size"]
                    size = min(view["session"]["part_size"], pack.size_bytes - offset)
                    md5 = await _disk(_part_md5, pack, offset, size)
                    for attempt in range(RETRIES):
                        signed = _check_response(
                            await _async_rpc(
                                client,
                                "POST",
                                owned + "/parts",
                                {"part_number": number, "content_md5": md5},
                            ),
                            200,
                        )
                        cloud_id = validate_remote_tensor_part(
                            signed, view, number, size, md5, cloud_id
                        )
                        try:
                            await _async_stream_part(
                                bulk, signed, pack, offset, size, client._timeout
                            )
                            break
                        except RuntimeError:
                            if attempt + 1 == RETRIES:
                                raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError("remote tensor upload timed out")
        while not view["ready"]:
            response = await _async_rpc(client, "POST", owned + "/complete", {})
            view = validate_remote_tensor_view(_check_response(response, 200), record, original)
            if not view["ready"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote tensor verification timed out")
                await asyncio.sleep(0.1)
                view = validate_remote_tensor_view(
                    _check_response(await _async_rpc(client, "GET", owned), 200), record, original
                )
        final = {**record["request"], "tensor_input_upload_id": record["upload_id"]}
        operation = _check_response(
            await _async_rpc(
                client, "POST", path, final, {"Idempotency-Key": record["idempotency_key"]}
            ),
            202,
        )
        if operation.get("id") != view["operation_id"]:
            raise ValueError("remote input submission changed its original operation")
        delivered_journal = pack._remote_recovery
        await _disk(_finish_journal, pack)
        return operation
    except asyncio.CancelledError as error:
        if delivered_journal is not None and record is not None and pack._remote_recovery is None:
            # The operation was accepted but cancellation prevented delivery.
            # Cleanup IO was joined; restore the identical nonce/key before
            # request-finally may release its sole local source.
            pack._retain_source = True
            with contextlib.suppress(asyncio.CancelledError):
                await _disk(_persist_record, delivered_journal, pack, record)
        if pack._remote_recovery is not None:
            cast(Any, error).recovery_path = pack._remote_recovery
        else:
            pack._retain_source = False
        raise
    except Exception:
        if pack._remote_recovery is not None:
            raise _interrupted(pack, record) from None
        pack._retain_source = False
        raise
    finally:
        pack._remote_lock.release()


def resume_remote_tensor(client: Any, recovery_path: str | Path) -> Any:
    record, pack = _load_journal(Path(recovery_path), client._base_url)
    try:
        return submit_remote_tensor(client, record["path"], record["request"], pack)
    finally:
        pack.close()


async def async_resume_remote_tensor(client: Any, recovery_path: str | Path) -> Any:
    record, pack = await _disk(_load_journal, Path(recovery_path), client._base_url)
    try:
        return await async_submit_remote_tensor(client, record["path"], record["request"], pack)
    finally:
        await _disk(pack.close)
