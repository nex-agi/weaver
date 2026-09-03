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

import hashlib
import logging
import os
import re
import tempfile
import time
from typing import Any, BinaryIO, Mapping, MutableMapping

import httpx
from opentelemetry import baggage, context, trace
from opentelemetry.propagate import inject
from opentelemetry.trace import Status, StatusCode

from . import __version__
from ._telemetry import get_tracer
from .config import TensorCompression, WeaverConfig
from .tensor_transport import (
    MultipartLayout,
    TensorPack,
    _bounded_tensor_pack_size,
    decompress_zstd_tensor_pack,
)

USER_AGENT: str = f"weaver-sdk/{__version__}"  # type: ignore[has-type]

# default timeout is 1 minute
DEFAULT_TIMEOUT = httpx.Timeout(timeout=60, connect=5.0)
DEFAULT_MAX_RETRIES = 10
DEFAULT_CONNECTION_RETRIES = 3
DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=20)
TENSOR_PACK_CHUNK_BYTES = 8 * 1024 * 1024

INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 10.0

# Transport-layer errors that indicate the request never reached the server.
# Safe to retry regardless of idempotency because no server-side state was created.
CONNECTION_ERRORS = (OSError, httpx.ConnectError, httpx.RemoteProtocolError)

# Artifact file downloads stream multi-GB safetensors shards; chunks are
# written to disk as they arrive so memory stays flat.
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB
# Per-read timeout, not whole-transfer: httpx applies ``read`` between socket
# reads, so a long download stays alive as long as bytes keep flowing.
DOWNLOAD_TIMEOUT = httpx.Timeout(timeout=60.0, connect=10.0)
MANAGED_DATASET_CONTENT_TYPE = "application/x-ndjson"
MANAGED_DATASET_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

logger = logging.getLogger(__name__)


def _managed_dataset_download_metadata(response: httpx.Response) -> tuple[int, str]:
    """Validate the authenticated public-dataset stream contract."""

    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != MANAGED_DATASET_CONTENT_TYPE:
        raise ValueError(
            f"managed dataset download must use {MANAGED_DATASET_CONTENT_TYPE}, "
            f"got {content_type or 'no Content-Type'}"
        )
    content_encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
    if content_encoding != "identity":
        raise ValueError("managed dataset download must not use content encoding")
    if response.headers.get("X-Weaver-Content-Visibility") != "public":
        raise ValueError("managed dataset download must be marked content_visibility=public")
    raw_size = response.headers.get("Content-Length")
    try:
        size_bytes = int(raw_size) if raw_size is not None else -1
    except ValueError as exc:
        raise ValueError("managed dataset download has an invalid Content-Length") from exc
    if size_bytes < 0:
        raise ValueError("managed dataset download requires a non-negative Content-Length")
    sha256 = response.headers.get("X-Weaver-Content-SHA256", "")
    if not _SHA256_HEX_RE.fullmatch(sha256):
        raise ValueError("managed dataset download requires a lowercase SHA-256 response header")
    return size_bytes, sha256


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


class DownloadURLExpiredError(RuntimeError):
    """A presigned artifact URL was rejected (HTTP 401/403).

    Presigned URLs are short-lived (~15 minutes); callers recover by
    re-fetching the download descriptor for fresh URLs and retrying.
    """


def build_download_client(timeout: httpx.Timeout | float | None = None) -> httpx.Client:
    """Build a bare client for downloading presigned artifact URLs.

    Presigned URLs are self-authorizing plain object-storage URLs; the Weaver
    API key must never be sent to an external host, so this client carries no
    auth headers and no Weaver base URL — only a User-Agent.
    """
    return httpx.Client(
        timeout=timeout or DOWNLOAD_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True,
    )


def stream_download_to_file(
    url: str,
    sink: BinaryIO,
    *,
    client: httpx.Client,
    resume_from: int = 0,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> int:
    """Stream a plain (non-API) URL into the open file *sink*.

    Opening the destination is the caller's job, deliberately: the download
    path anchors its writes to a directory descriptor (:mod:`weaver._safeio`)
    so an untrusted descriptor file name cannot be re-pointed at another
    directory between validation and write. Resolving a path here would put
    that decision back into this function, where the anchor is not available.

    Args:
        url: Presigned download URL. No Weaver auth headers are attached;
            *client* must be a bare client (see :func:`build_download_client`).
        sink: Open binary file the body is written into. Must be opened for
            append when *resume_from* is non-zero, for truncating write
            otherwise.
        client: Bare ``httpx.Client`` used for the request.
        resume_from: Byte offset already present in *sink*; sends a ``Range``
            header so an interrupted download resumes instead of restarting.
        chunk_size: Streaming chunk size in bytes.

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
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code in (401, 403):
            response.read()
            raise DownloadURLExpiredError(
                f"presigned URL rejected with HTTP {response.status_code}; "
                "the download descriptor must be re-fetched for a fresh URL"
            )
        if response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
            # The requested offset is at/past EOF: the bytes on disk already
            # cover the file. The caller verifies size/sha before trusting it.
            response.read()
            return resume_from
        if not response.is_success:
            response.read()
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
        for chunk in response.iter_bytes(chunk_size):
            sink.write(chunk)
            written += len(chunk)
    return written


def _validate_tensor_pack_expectation(size_bytes: int, sha256: str) -> str:
    _bounded_tensor_pack_size(size_bytes, "tensor_pack.size_bytes")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("tensor pack sha256 must be a SHA-256 hex digest")
    try:
        bytes.fromhex(sha256)
    except ValueError as exc:
        raise ValueError("tensor pack sha256 must be a SHA-256 hex digest") from exc
    return sha256.lower()


def _validate_tensor_pack_download(
    size_bytes: int,
    sha256: str,
    codec: TensorCompression,
    decoded_size_bytes: int | None,
) -> tuple[str, int]:
    digest = _validate_tensor_pack_expectation(size_bytes, sha256)
    if codec not in {"raw", "zstd"}:
        raise ValueError("tensor pack codec must be 'raw' or 'zstd'")
    decoded_size = _bounded_tensor_pack_size(
        size_bytes if decoded_size_bytes is None else decoded_size_bytes,
        "tensor_pack.decoded_size_bytes",
    )
    if codec == "raw" and decoded_size != size_bytes:
        raise ValueError("raw tensor pack decoded_size_bytes must equal size_bytes")
    return digest, decoded_size


def _validate_tensor_pack_response_length(response: httpx.Response, expected: int) -> None:
    raw_length = response.headers.get("Content-Length")
    if raw_length is None:
        raise ValueError("tensor pack response is missing Content-Length")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise ValueError("tensor pack response has invalid Content-Length") from exc
    if content_length != expected:
        raise ValueError(
            f"tensor pack response Content-Length is {content_length}, expected {expected}"
        )


def _validate_tensor_pack_response_metadata(
    response: httpx.Response,
    *,
    codec: TensorCompression,
    decoded_size_bytes: int,
) -> None:
    response_codec = response.headers.get("X-Weaver-Tensor-Codec", "raw")
    if response_codec not in {"raw", "zstd"}:
        raise ValueError(f"tensor pack response has unsupported codec {response_codec!r}")
    raw_decoded_size = response.headers.get("X-Weaver-Tensor-Decoded-Size")
    if raw_decoded_size is None:
        if response_codec != "raw":
            raise ValueError("zstd tensor pack response is missing decoded size")
        response_decoded_size = int(response.headers["Content-Length"])
    else:
        try:
            response_decoded_size = int(raw_decoded_size)
        except ValueError as exc:
            raise ValueError("tensor pack response has invalid decoded size") from exc
        if response_decoded_size < 0:
            raise ValueError("tensor pack response has invalid decoded size")
    if response_codec != codec:
        raise ValueError(f"tensor pack response codec is {response_codec!r}, expected {codec!r}")
    if response_decoded_size != decoded_size_bytes:
        raise ValueError(
            "tensor pack response decoded size is "
            f"{response_decoded_size}, expected {decoded_size_bytes}"
        )


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

    def post_tensor_multipart(
        self,
        path: str,
        *,
        request: Mapping[str, Any],
        tensor_pack: TensorPack,
    ) -> Any:
        """Submit one non-retryable operation with a binary tensor attachment."""

        layout = MultipartLayout(request, tensor_pack)
        model_id = self._extract_model_id_from_path(path)
        with self._tracer.start_as_current_span("weaver.post", kind=trace.SpanKind.CLIENT) as span:
            apply_request_span_attributes(span, "POST", path, model_id)
            self._ensure_fresh_client()
            headers = dict(self._client.headers or {})
            headers["Content-Type"] = layout.content_type
            headers["Content-Length"] = str(layout.content_length)
            inject(headers)
            try:
                response = self._client.request(
                    "POST",
                    path,
                    content=layout.sync_stream(),
                    headers=headers,
                )
                span.set_attribute("http.status_code", response.status_code)
                if not response.is_success:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    self._raise_error(response)
                span.set_status(Status(StatusCode.OK))
                if response.status_code == httpx.codes.NO_CONTENT or not response.content:
                    return None
                return response.json()
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def download_tensor_pack(
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
        compressed = tempfile.TemporaryFile(mode="w+b") if codec == "zstd" else None
        wire_destination = compressed if compressed is not None else destination
        with self._tracer.start_as_current_span("weaver.get", kind=trace.SpanKind.CLIENT) as span:
            apply_request_span_attributes(span, "GET", path, None)
            self._ensure_fresh_client()
            headers = dict(self._client.headers or {})
            headers["Accept-Encoding"] = "identity"
            inject(headers)
            try:
                with self._client.stream("GET", path, headers=headers) as response:
                    span.set_attribute("http.status_code", response.status_code)
                    if not response.is_success:
                        response.read()
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                        self._raise_error(response)
                    _validate_tensor_pack_response_length(response, size_bytes)
                    _validate_tensor_pack_response_metadata(
                        response,
                        codec=codec,
                        decoded_size_bytes=expected_decoded_size,
                    )
                    for chunk in response.iter_raw(chunk_size=TENSOR_PACK_CHUNK_BYTES):
                        if received + len(chunk) > size_bytes:
                            raise ValueError(
                                f"downloaded tensor pack exceeds expected {size_bytes} bytes"
                            )
                        wire_destination.write(chunk)
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
                wire_destination.flush()
                wire_destination.seek(0)
                if compressed is not None:
                    decompress_zstd_tensor_pack(
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
                    compressed.close()

    def download_managed_dataset(self, path: str, destination: BinaryIO) -> tuple[int, str]:
        """Stream one authenticated public JSONL dataset with exact integrity checks."""

        digest = hashlib.sha256()
        received = 0
        with self._tracer.start_as_current_span("weaver.get", kind=trace.SpanKind.CLIENT) as span:
            apply_request_span_attributes(span, "GET", path, None)
            self._ensure_fresh_client()
            headers = {
                key: value
                for key, value in (self._client.headers or {}).items()
                if key.lower() not in {"accept", "accept-encoding"}
            }
            headers["Accept"] = MANAGED_DATASET_CONTENT_TYPE
            headers["Accept-Encoding"] = "identity"
            inject(headers)
            try:
                with self._client.stream("GET", path, headers=headers) as response:
                    span.set_attribute("http.status_code", response.status_code)
                    if not response.is_success:
                        response.read()
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                        self._raise_error(response)
                    expected_size, expected_sha256 = _managed_dataset_download_metadata(response)
                    for chunk in response.iter_raw(chunk_size=MANAGED_DATASET_DOWNLOAD_CHUNK_BYTES):
                        if received + len(chunk) > expected_size:
                            raise ValueError("managed dataset download exceeds its Content-Length")
                        destination.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                if received != expected_size:
                    raise ValueError(
                        f"managed dataset download has {received} bytes, expected "
                        f"{expected_size}"
                    )
                if digest.hexdigest() != expected_sha256:
                    raise ValueError("managed dataset download SHA-256 mismatch")
                span.set_status(Status(StatusCode.OK))
                return received, expected_sha256
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

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
                # and for 503 responses. A retryable 503 means the server did
                # not accept the request, so operation POSTs are safe to repeat.
                last_exception = e
                span.record_exception(e)
                request_attempt += 1
                retryable_503 = e.status_code == httpx.codes.SERVICE_UNAVAILABLE
                if retryable_503 and max_retries is None:
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
                        transport_error = WeaverAPIError(
                            503,
                            "transport_unavailable",
                            f"{method} {path} failed after {connection_error_count} connection attempts",
                            True,
                        )
                        raise transport_error from e

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
