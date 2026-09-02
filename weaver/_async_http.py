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

"""Async HTTP client for interacting with the Weaver server.

This is the asyncio twin of :class:`weaver._http.APIClient`. It mirrors the
retry, connection-error, trace-propagation and fork-safety behaviour of the
synchronous client, but every blocking point (the network round trip and the
backoff sleeps) is awaited so the event loop stays free for other coroutines.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from typing import Any, BinaryIO, Mapping, MutableMapping

import httpx
from opentelemetry import baggage, context, trace
from opentelemetry.propagate import inject
from opentelemetry.trace import Status, StatusCode

from ._http import (
    DEFAULT_CONNECTION_LIMITS,
    DEFAULT_CONNECTION_RETRIES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_TIMEOUT,
    TENSOR_PACK_CHUNK_BYTES,
    USER_AGENT,
    DownloadURLExpiredError,
    WeaverAPIError,
    _is_connection_error,
    _validate_tensor_pack_download,
    _validate_tensor_pack_response_length,
    _validate_tensor_pack_response_metadata,
    apply_request_span_attributes,
    compute_retry_delay,
    extract_model_id_from_path,
    raise_for_response,
)
from ._telemetry import get_tracer
from .config import TensorCompression, WeaverConfig
from .tensor_transport import MultipartLayout, TensorPack, decompress_zstd_tensor_pack

logger = logging.getLogger(__name__)


def build_async_download_client(timeout: httpx.Timeout | float | None = None) -> httpx.AsyncClient:
    """Asyncio twin of :func:`weaver._http.build_download_client`.

    Presigned URLs are self-authorizing plain object-storage URLs; the Weaver
    API key must never be sent to an external host, so this client carries no
    auth headers and no Weaver base URL — only a User-Agent.
    """
    return httpx.AsyncClient(
        timeout=timeout or DOWNLOAD_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True,
    )


async def async_stream_download_to_file(
    url: str,
    sink: BinaryIO,
    *,
    client: httpx.AsyncClient,
    resume_from: int = 0,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> int:
    """Asyncio twin of :func:`weaver._http.stream_download_to_file`.

    Network reads are awaited so the event loop stays free; the per-chunk
    ``sink.write`` is a buffered local-disk write that is fast relative to the
    awaited network reads, which keeps the loop responsive without a thread
    hop per chunk.

    The caller owns *sink* — see the sync twin for why opening it here would
    defeat the descriptor anchoring in :mod:`weaver._safeio`.

    Returns:
        Total bytes now present in *sink*.

    Raises:
        DownloadURLExpiredError: The URL was rejected with 401/403 — refresh
            the download descriptor and retry with a fresh URL.
        WeaverAPIError: Any other non-success response.
    """
    headers: dict[str, str] = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
    async with client.stream("GET", url, headers=headers) as response:
        if response.status_code in (401, 403):
            await response.aread()
            raise DownloadURLExpiredError(
                f"presigned URL rejected with HTTP {response.status_code}; "
                "the download descriptor must be re-fetched for a fresh URL"
            )
        if response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
            # The requested offset is at/past EOF: the bytes on disk already
            # cover the file. The caller verifies size/sha before trusting it.
            await response.aread()
            return resume_from
        if not response.is_success:
            await response.aread()
            raise_for_response(response)
        partial = response.status_code == httpx.codes.PARTIAL_CONTENT
        if partial:
            written = resume_from
        else:
            # The server ignored the Range header and is sending the whole
            # body. The sink may be open for append and already hold a
            # partial, so drop those bytes rather than doubling the file.
            sink.seek(0)
            sink.truncate()
            written = 0
        async for chunk in response.aiter_bytes(chunk_size):
            sink.write(chunk)
            written += len(chunk)
    return written


async def _await_blocking_io(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Finish one in-flight file operation before propagating cancellation."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException:
            pass
        raise


async def _open_temporary_file() -> BinaryIO:
    """Open a temporary file off-loop without leaking it on cancellation."""

    task = asyncio.create_task(asyncio.to_thread(tempfile.TemporaryFile, mode="w+b"))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_close_completed_file)
        raise


def _close_completed_file(task: "asyncio.Task[BinaryIO]") -> None:
    try:
        task.result().close()
    except BaseException:
        pass


class AsyncAPIClient:
    """Thin wrapper around ``httpx.AsyncClient`` with Weaver-specific behaviour."""

    def __init__(
        self,
        config: WeaverConfig,
        *,
        timeout: httpx.Timeout | float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self._base_url = config.base_url.rstrip("/")
        headers: MutableMapping[str, str] = {"User-Agent": USER_AGENT}
        if config.api_key:
            headers["X-WEAVER-API-KEY"] = config.api_key
        self._headers = headers
        self._timeout = timeout or DEFAULT_TIMEOUT

        self._client = self._build_client()
        self._pid = os.getpid()
        self._max_retries = max_retries

        self._tracer = get_tracer()

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._headers,
            limits=DEFAULT_CONNECTION_LIMITS,
        )

    def _ensure_fresh_client(self) -> None:
        """Rebuild the underlying client if the process changed (post-fork).

        See :meth:`weaver._http.APIClient._ensure_fresh_client` for the
        rationale. The inherited client is dropped without ``aclose()`` because
        its socket FDs belong to the parent process.
        """
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        self._client = self._build_client()
        self._pid = current_pid

    async def __aenter__(self) -> "AsyncAPIClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        # Only close the client in the process that created it; closing a
        # fork-inherited client would write shutdown bytes over the parent's FDs.
        if os.getpid() == self._pid:
            await self._client.aclose()

    async def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post(
        self,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> Any:
        return await self._request("POST", path, params=params, json=json, max_retries=max_retries)

    async def post_tensor_multipart(
        self,
        path: str,
        *,
        request: Mapping[str, Any],
        tensor_pack: TensorPack,
    ) -> Any:
        """Submit one non-retryable operation with a binary tensor attachment."""

        layout = MultipartLayout(request, tensor_pack)
        model_id = extract_model_id_from_path(path)
        with self._tracer.start_as_current_span("weaver.post", kind=trace.SpanKind.CLIENT) as span:
            apply_request_span_attributes(span, "POST", path, model_id)
            self._ensure_fresh_client()
            headers = dict(self._client.headers or {})
            headers["Content-Type"] = layout.content_type
            headers["Content-Length"] = str(layout.content_length)
            inject(headers)
            try:
                response = await self._client.request(
                    "POST",
                    path,
                    content=layout.async_stream(),
                    headers=headers,
                )
                span.set_attribute("http.status_code", response.status_code)
                if not response.is_success:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    raise_for_response(response)
                span.set_status(Status(StatusCode.OK))
                if response.status_code == httpx.codes.NO_CONTENT or not response.content:
                    return None
                return response.json()
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    async def download_tensor_pack(
        self,
        operation_id: str,
        destination: BinaryIO,
        *,
        size_bytes: int,
        sha256: str,
        codec: TensorCompression = "raw",
        decoded_size_bytes: int | None = None,
    ) -> None:
        """Download one bounded, verified operation result tensor pack."""

        path = f"/api/v1/operations/{operation_id}/tensor-pack"
        expected_digest, expected_decoded_size = _validate_tensor_pack_download(
            size_bytes, sha256, codec, decoded_size_bytes
        )
        digest = hashlib.sha256()
        received = 0
        compressed = await _open_temporary_file() if codec == "zstd" else None
        wire_destination = compressed if compressed is not None else destination
        with self._tracer.start_as_current_span("weaver.get", kind=trace.SpanKind.CLIENT) as span:
            apply_request_span_attributes(span, "GET", path, None)
            self._ensure_fresh_client()
            headers = dict(self._client.headers or {})
            headers["Accept-Encoding"] = "identity"
            inject(headers)
            try:
                async with self._client.stream("GET", path, headers=headers) as response:
                    span.set_attribute("http.status_code", response.status_code)
                    if not response.is_success:
                        await response.aread()
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                        raise_for_response(response)
                    _validate_tensor_pack_response_length(response, size_bytes)
                    _validate_tensor_pack_response_metadata(
                        response,
                        codec=codec,
                        decoded_size_bytes=expected_decoded_size,
                    )
                    async for chunk in response.aiter_raw(chunk_size=TENSOR_PACK_CHUNK_BYTES):
                        if received + len(chunk) > size_bytes:
                            raise ValueError(
                                f"downloaded tensor pack exceeds expected {size_bytes} bytes"
                            )
                        await _await_blocking_io(wire_destination.write, chunk)
                        digest.update(chunk)
                        received += len(chunk)
                if received != size_bytes:
                    raise ValueError(
                        f"downloaded tensor pack has {received} bytes, expected {size_bytes}"
                    )
                if digest.hexdigest() != expected_digest:
                    raise ValueError(
                        "downloaded tensor pack SHA-256 does not match operation metadata"
                    )
                await _await_blocking_io(wire_destination.flush)
                await _await_blocking_io(wire_destination.seek, 0)
                if compressed is not None:
                    await _await_blocking_io(
                        decompress_zstd_tensor_pack,
                        compressed,
                        destination,
                        expected_decoded_size,
                    )
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                if compressed is not None:
                    await _await_blocking_io(compressed.close)

    async def patch(self, path: str, *, json: Any) -> Any:
        return await self._request("PATCH", path, json=json)

    async def delete(self, path: str) -> Any:
        return await self._request("DELETE", path)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        max_retries: int | None = None,
    ) -> Any:
        model_id = extract_model_id_from_path(path)

        ctx = context.get_current()
        if model_id:
            ctx = baggage.set_baggage("model_id", model_id, context=ctx)

        with self._tracer.start_as_current_span(
            f"weaver.{method.lower()}",
            context=ctx,
            kind=trace.SpanKind.CLIENT,
        ) as span:
            apply_request_span_attributes(span, method, path, model_id)
            return await self._request_with_retries(
                span, method, path, params=params, json=json, max_retries=max_retries
            )

    async def _request_with_retries(
        self,
        span: trace.Span,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        max_retries: int | None = None,
    ) -> Any:
        """Async mirror of :meth:`APIClient._request_with_retries`.

        Connection-level errors are retried up to ``DEFAULT_CONNECTION_RETRIES``
        regardless of *max_retries* (the request never reached the server, so it
        is safe even for non-idempotent methods). Server-declared retryable
        errors are retried for idempotent methods and 503 responses.
        """
        effective_max_retries = max_retries if max_retries is not None else self._max_retries
        last_exception: Exception | None = None
        request_attempt = 0
        connection_error_count = 0

        while request_attempt < effective_max_retries:
            try:
                self._ensure_fresh_client()

                headers = dict(self._client.headers or {})
                inject(headers)  # Adds 'traceparent' header with trace context

                response = await self._client.request(
                    method, path, params=params, json=json, headers=headers
                )

                span.set_attribute("http.status_code", response.status_code)

                if response.is_success:
                    span.set_status(Status(StatusCode.OK))
                    if response.status_code == httpx.codes.NO_CONTENT:
                        return None
                    if not response.content:
                        return None
                    return response.json()

                span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                raise_for_response(response)

            except WeaverAPIError as e:
                last_exception = e
                span.record_exception(e)
                request_attempt += 1
                retryable_503 = e.status_code == httpx.codes.SERVICE_UNAVAILABLE
                if retryable_503:
                    effective_max_retries = max(effective_max_retries, self._max_retries)
                is_last_attempt = request_attempt >= effective_max_retries
                idempotent_method = method.upper() in {"GET", "HEAD", "OPTIONS"}

                if (
                    (not e.retryable)
                    or (not idempotent_method and not retryable_503)
                    or is_last_attempt
                ):
                    span.set_status(Status(StatusCode.ERROR, "API error"))
                    raise

                logger.debug(
                    "Retryable API error (attempt %d/%d): %s %s - [%d] %s: %s",
                    request_attempt,
                    effective_max_retries,
                    method,
                    path,
                    e.status_code,
                    e.code,
                    e.message,
                )
                await asyncio.sleep(compute_retry_delay(request_attempt))

            except Exception as e:  # pylint: disable=broad-except
                last_exception = e
                span.record_exception(e)

                if _is_connection_error(e):
                    connection_error_count += 1

                    logger.debug(
                        "Connection error (attempt %d/%d): %s %s - %s: %s",
                        connection_error_count,
                        DEFAULT_CONNECTION_RETRIES,
                        method,
                        path,
                        type(e).__name__,
                        str(e),
                    )

                    if connection_error_count >= DEFAULT_CONNECTION_RETRIES:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                f"Connection failed after {connection_error_count} attempts",
                            )
                        )
                        logger.error(
                            "Connection error after %d attempts: %s %s - %s",
                            connection_error_count,
                            method,
                            path,
                            str(e),
                        )
                        raise

                    delay = compute_retry_delay(connection_error_count)
                    logger.debug("Retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
                    continue

                request_attempt += 1
                is_last_attempt = request_attempt >= effective_max_retries

                if is_last_attempt:
                    span.set_status(
                        Status(StatusCode.ERROR, f"Failed after {effective_max_retries} retries")
                    )

                logger.debug(
                    "HTTP request failed (attempt %d/%d): %s %s - %s: %s",
                    request_attempt,
                    effective_max_retries,
                    method,
                    path,
                    type(e).__name__,
                    str(e),
                )

                if is_last_attempt:
                    logger.error(
                        "HTTP request failed after %d retries: %s %s - %s",
                        effective_max_retries,
                        method,
                        path,
                        str(e),
                    )
                    raise

                delay = compute_retry_delay(request_attempt)
                logger.debug("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")
