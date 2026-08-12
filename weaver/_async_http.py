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
import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

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
    USER_AGENT,
    DownloadURLExpiredError,
    WeaverAPIError,
    _is_connection_error,
    apply_request_span_attributes,
    compute_retry_delay,
    extract_model_id_from_path,
    raise_for_response,
)
from ._telemetry import get_tracer
from .config import WeaverConfig

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
    dest: Path,
    *,
    client: httpx.AsyncClient,
    resume_from: int = 0,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> int:
    """Asyncio twin of :func:`weaver._http.stream_download_to_file`.

    Network reads are awaited so the event loop stays free; the per-chunk
    ``fh.write`` is a buffered local-disk write that is fast relative to the
    awaited network reads, which keeps the loop responsive without a thread
    hop per chunk.

    Returns:
        Total bytes now present in *dest*.

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
        mode = "ab" if partial else "wb"
        written = resume_from if partial else 0
        with open(dest, mode) as fh:
            async for chunk in response.aiter_bytes(chunk_size):
                fh.write(chunk)
                written += len(chunk)
    return written


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
        errors are retried only for idempotent methods.
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
                is_last_attempt = request_attempt >= effective_max_retries
                idempotent_method = method.upper() in {"GET", "HEAD", "OPTIONS"}

                if (not e.retryable) or (not idempotent_method) or is_last_attempt:
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
