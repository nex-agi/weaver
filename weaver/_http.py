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

"""HTTP client utilities for interacting with the Weaver server."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Mapping, MutableMapping

import httpx
from opentelemetry import baggage, context, trace
from opentelemetry.propagate import inject
from opentelemetry.trace import Status, StatusCode

from . import __version__
from ._telemetry import get_tracer
from .config import WeaverConfig

USER_AGENT: str = f"weaver-sdk/{__version__}"  # type: ignore[has-type]

# default timeout is 1 minute
DEFAULT_TIMEOUT = httpx.Timeout(timeout=60, connect=5.0)
DEFAULT_MAX_RETRIES = 10
DEFAULT_CONNECTION_RETRIES = 3
DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=20)

INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 10.0

# Transport-layer errors that indicate the request never reached the server.
# Safe to retry regardless of idempotency because no server-side state was created.
CONNECTION_ERRORS = (OSError, httpx.ConnectError, httpx.RemoteProtocolError)

logger = logging.getLogger(__name__)


def _is_connection_error(exc: BaseException) -> bool:
    """Return True if *exc* represents a transport-level failure.

    Either the exception itself is in :data:`CONNECTION_ERRORS`, or its
    ``__cause__``/``__context__`` chain contains an :class:`OSError`. httpx
    wraps low-level OS errors into :class:`httpx.ReadError` / ``WriteError``
    via ``raise mapped_exc(...) from original_oserror``, so walking the chain
    recovers that signal. Treating those as connection errors is safe: an
    OSError on the client socket (e.g. ``EBADF`` after fork, ``EPIPE`` on a
    dead keep-alive) means the request bytes never left the process, so
    retrying cannot duplicate non-idempotent server-side effects.
    """
    if isinstance(exc, CONNECTION_ERRORS):
        return True
    seen: set[int] = set()
    cur: BaseException | None = exc.__cause__ or exc.__context__
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, OSError):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


class WeaverAPIError(RuntimeError):
    """Structured error returned by the Weaver API.

    Optional capacity fields intentionally use integer nano-USD values and
    decimal USD strings. This avoids losing very small charges to binary
    floating-point rounding while still giving CLI users readable amounts.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        retryable: bool,
        *,
        request_id: str | None = None,
        retry_after: str | None = None,
        required_nanos: int | None = None,
        available_nanos: int | None = None,
        required_usd: str | None = None,
        available_usd: str | None = None,
        details: Mapping[str, Any] | None = None,
    ):
        super().__init__(f"[{status_code}] {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.request_id = request_id
        self.retry_after = retry_after
        self.required_nanos = required_nanos
        self.available_nanos = available_nanos
        self.required_usd = required_usd
        self.available_usd = available_usd
        self.details = dict(details or {})


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _usd_from_nanos(value: int | None) -> str | None:
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    whole, fractional = divmod(absolute, 1_000_000_000)
    if fractional == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{fractional:09d}".rstrip("0")


def extract_model_id_from_path(path: str) -> str | None:
    """Extract ``model_id`` from an API path for trace/baggage propagation.

    Patterns:
    - ``/api/v1/models/{model_id}/...``
    - ``/api/v1/models/{model_id}``

    Returns:
        model_id if found, None otherwise.
    """
    match = re.match(r"/api/v\d+/models/([^/]+)", path)
    if match:
        return match.group(1)
    return None


def raise_for_response(response: httpx.Response) -> None:
    """Convert a non-success httpx response into a :class:`WeaverAPIError`."""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    raw_details = payload.get("details")
    details = raw_details if isinstance(raw_details, dict) else {}

    def detail(name: str) -> Any:
        # Accept the canonical nested contract and the early top-level shape so
        # SDK upgrades do not have to be synchronized exactly with a server deploy.
        return details.get(name, payload.get(name))

    required_nanos = _optional_int(detail("required_nanos"))
    available_nanos = _optional_int(detail("available_nanos"))
    raise WeaverAPIError(
        response.status_code,
        code=payload.get("error", "unknown_error"),
        message=payload.get("message", response.text),
        retryable=bool(payload.get("retryable", False)),
        request_id=_optional_string(payload.get("request_id"))
        or _optional_string(response.headers.get("X-Request-ID")),
        retry_after=_optional_string(payload.get("retry_after"))
        or _optional_string(response.headers.get("Retry-After")),
        required_nanos=required_nanos,
        available_nanos=available_nanos,
        required_usd=_optional_string(detail("required_usd")) or _usd_from_nanos(required_nanos),
        available_usd=_optional_string(detail("available_usd")) or _usd_from_nanos(available_nanos),
        details=details,
    )


def apply_request_span_attributes(
    span: trace.Span, method: str, path: str, model_id: str | None
) -> None:
    """Set the standard Weaver request attributes on *span* (shared sync/async)."""
    span.set_attribute("http.method", method)
    span.set_attribute("http.url", path)
    span.set_attribute("http.user_agent", USER_AGENT)
    if model_id:
        span.set_attribute("model_id", model_id)
        span.set_attribute("weaver.model_id", model_id)  # Alternative key

    span_context = span.get_span_context()
    if span_context.is_valid:
        trace_id = format(span_context.trace_id, "032x")
        if model_id:
            logger.debug("API request trace_id: %s, model_id: %s", trace_id, model_id)
        else:
            logger.debug("API request trace_id: %s", trace_id)


def compute_retry_delay(attempt: int) -> float:
    """Exponential backoff delay for retry *attempt* (1-based)."""
    return min(INITIAL_RETRY_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)


class APIClient:
    """Thin wrapper around httpx.Client with Weaver-specific behavior."""

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

        # Initialize tracer for distributed tracing
        self._tracer = get_tracer()

    def _build_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._headers,
            limits=DEFAULT_CONNECTION_LIMITS,
        )

    def _ensure_fresh_client(self) -> None:
        """Rebuild ``self._client`` if the current process differs from the one
        that created it.

        httpx.Client holds OS-level socket file descriptors inside its
        connection pool. After ``os.fork`` (the default on Linux for
        ``multiprocessing``), those FDs are inherited by the child but point
        at sockets the child never opened; any read/write against them fails
        with ``OSError: [Errno 9] Bad file descriptor``. Detecting the pid
        change and constructing a fresh client (without touching the
        inherited one — closing it would attempt to write ``Connection:
        close`` over the same dead FDs) gives every child its own pool.
        """
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        # Drop the reference to the inherited client without calling close():
        # its FDs belong to the parent's sockets and are unsafe to touch here.
        self._client = self._build_client()
        self._pid = current_pid

    def __enter__(self) -> "APIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        # Only close the client in the process that created it; closing a
        # fork-inherited client would write shutdown bytes over FDs that
        # belong to the parent.
        if os.getpid() == self._pid:
            self._client.close()

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> Any:
        return self._request("POST", path, params=params, json=json, max_retries=max_retries)

    def patch(self, path: str, *, json: Any) -> Any:
        return self._request("PATCH", path, json=json)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        max_retries: int | None = None,
    ) -> Any:
        # Extract model_id from path if present (e.g., /api/v1/models/{model_id}/...)
        model_id = self._extract_model_id_from_path(path)

        # Set model_id in baggage if found (propagates automatically)
        ctx = context.get_current()
        if model_id:
            ctx = baggage.set_baggage("model_id", model_id, context=ctx)

        # Create a span for this API call with context
        with self._tracer.start_as_current_span(
            f"weaver.{method.lower()}",
            context=ctx,
            kind=trace.SpanKind.CLIENT,
        ) as span:
            # Add span attributes
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", path)
            span.set_attribute("http.user_agent", USER_AGENT)

            # Add model_id as span attribute for easy filtering in APMPlus
            if model_id:
                span.set_attribute("model_id", model_id)
                span.set_attribute("weaver.model_id", model_id)  # Alternative key

            # Log trace_id and model_id for debugging
            span_context = span.get_span_context()
            if span_context.is_valid:
                trace_id = format(span_context.trace_id, "032x")
                if model_id:
                    logger.debug("API request trace_id: %s, model_id: %s", trace_id, model_id)
                else:
                    logger.debug("API request trace_id: %s", trace_id)

            # Execute the request with retries
            return self._request_with_retries(
                span, method, path, params=params, json=json, max_retries=max_retries
            )

    def _extract_model_id_from_path(self, path: str) -> str | None:
        return extract_model_id_from_path(path)

    def _request_with_retries(
        self,
        span: trace.Span,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        max_retries: int | None = None,
    ) -> Any:
        """Execute HTTP request with retry logic and trace context injection.

        Connection-level errors (stale sockets, refused connections) are retried
        up to ``DEFAULT_CONNECTION_RETRIES`` times regardless of *max_retries*
        because the request never reached the server and is therefore safe to
        retry even for non-idempotent methods.

        Args:
            span: OpenTelemetry span for tracing.
            method: HTTP method (GET, POST, etc.).
            path: API path.
            params: Query parameters.
            json: JSON body.
            max_retries: Per-request override for maximum number of attempts.
                When provided, overrides the client-level ``self._max_retries``.
        """
        effective_max_retries = max_retries if max_retries is not None else self._max_retries
        last_exception: Exception | None = None
        request_attempt = 0
        connection_error_count = 0

        while request_attempt < effective_max_retries:
            try:
                # Rebuild the httpx.Client if we are in a forked child — the
                # inherited socket pool would otherwise surface as EBADF.
                self._ensure_fresh_client()

                # Inject trace context into HTTP headers
                headers = dict(self._client.headers or {})
                inject(headers)  # Adds 'traceparent' header with trace context

                # Make the HTTP request
                response = self._client.request(
                    method, path, params=params, json=json, headers=headers
                )

                # Record response status
                span.set_attribute("http.status_code", response.status_code)

                if response.is_success:
                    span.set_status(Status(StatusCode.OK))
                    if response.status_code == httpx.codes.NO_CONTENT:
                        return None
                    if not response.content:
                        return None
                    return response.json()

                # Non-success status
                span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                self._raise_error(response)

            except WeaverAPIError as e:
                # Retry server-declared retryable errors for idempotent methods
                # only (e.g. GET operation polling surviving a transient read);
                # POST stays fatal to avoid duplicating non-idempotent work.
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
                delay = min(INITIAL_RETRY_DELAY * (2 ** (request_attempt - 1)), MAX_RETRY_DELAY)
                time.sleep(delay)

            except Exception as e:  # pylint: disable=broad-except
                last_exception = e
                span.record_exception(e)

                if _is_connection_error(e):
                    # Connection-level error — the request never reached the
                    # server, so it is safe to retry regardless of max_retries.
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

                    delay = min(
                        INITIAL_RETRY_DELAY * (2 ** (connection_error_count - 1)),
                        MAX_RETRY_DELAY,
                    )
                    logger.debug("Retrying in %.1fs...", delay)
                    time.sleep(delay)
                    continue

                request_attempt += 1
                is_last_attempt = request_attempt >= effective_max_retries

                if is_last_attempt:
                    span.set_status(
                        Status(StatusCode.ERROR, f"Failed after {effective_max_retries} retries")
                    )

                # Log the error
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

                # Wait before retrying with exponential backoff
                delay = min(INITIAL_RETRY_DELAY * (2 ** (request_attempt - 1)), MAX_RETRY_DELAY)
                logger.debug("Retrying in %.1fs...", delay)
                time.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    def _raise_error(self, response: httpx.Response) -> None:
        raise_for_response(response)


def backoff_delays(
    initial: float = INITIAL_RETRY_DELAY,
    factor: float = 2.0,
    maximum: float = MAX_RETRY_DELAY,
):
    """Generate exponential backoff delays for retries."""
    delay = initial
    while True:
        yield delay
        delay = min(delay * factor, maximum)
