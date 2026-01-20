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
import time
from typing import Any, Mapping, MutableMapping

import re

import httpx
from opentelemetry import baggage, trace
from opentelemetry.propagate import inject
from opentelemetry.trace import Status, StatusCode

from . import __version__
from ._telemetry import get_tracer
from .config import WeaverConfig

USER_AGENT: str = f"weaver-sdk/{__version__}"  # type: ignore[has-type]

# default timeout is 1 minute
DEFAULT_TIMEOUT = httpx.Timeout(timeout=60, connect=5.0)
DEFAULT_MAX_RETRIES = 10
DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=20)

INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 10.0

logger = logging.getLogger(__name__)


class WeaverAPIError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str, retryable: bool):
        super().__init__(f"[{status_code}] {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable


class APIClient:
    """Thin wrapper around httpx.Client with Weaver-specific behavior."""

    def __init__(
        self,
        config: WeaverConfig,
        *,
        timeout: httpx.Timeout | float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        base_url = config.base_url.rstrip("/")
        headers: MutableMapping[str, str] = {"User-Agent": USER_AGENT}
        if config.api_key:
            headers["X-WEAVER-API-KEY"] = config.api_key

        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout or DEFAULT_TIMEOUT,
            headers=headers,
            limits=DEFAULT_CONNECTION_LIMITS,
        )
        self._max_retries = max_retries
        
        # Initialize tracer for distributed tracing
        self._tracer = get_tracer()

    def __enter__(self) -> "APIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, *, json: Any = None) -> Any:
        return self._request("POST", path, json=json)

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
    ) -> Any:
        # Extract model_id from path if present (e.g., /api/v1/models/{model_id}/...)
        model_id = self._extract_model_id_from_path(path)
        
        # Set model_id in baggage if found (propagates automatically)
        ctx = trace.get_current()
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
            return self._request_with_retries(span, method, path, params=params, json=json)
    
    def _extract_model_id_from_path(self, path: str) -> str | None:
        """
        Extract model_id from API path.
        
        Patterns:
        - /api/v1/models/{model_id}/...
        - /api/v1/models/{model_id}
        
        Returns:
            model_id if found, None otherwise
        """
        # Match /api/v1/models/{model_id}/... or /api/v1/models/{model_id}
        match = re.match(r"/api/v\d+/models/([^/]+)", path)
        if match:
            return match.group(1)
        return None

    def _request_with_retries(
        self,
        span: trace.Span,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        """Execute HTTP request with retry logic and trace context injection."""
        last_exception = None

        for attempt in range(self._max_retries):
            try:
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

            except WeaverAPIError:
                # Don't retry WeaverAPIError (4xx errors are not retryable)
                span.set_status(Status(StatusCode.ERROR, "API error"))
                raise
                
            except Exception as e:
                last_exception = e
                is_last_attempt = attempt == self._max_retries - 1

                # Record error in span
                span.record_exception(e)
                if is_last_attempt:
                    span.set_status(Status(StatusCode.ERROR, f"Failed after {self._max_retries} retries"))

                # Log the error
                logger.debug(
                    "HTTP request failed (attempt %d/%d): %s %s - %s: %s",
                    attempt + 1,
                    self._max_retries,
                    method,
                    path,
                    type(e).__name__,
                    str(e),
                )

                if is_last_attempt:
                    logger.error(
                        "HTTP request failed after %d retries: %s %s - %s",
                        self._max_retries,
                        method,
                        path,
                        str(e),
                    )
                    raise

                # Wait before retrying with exponential backoff
                delay = min(INITIAL_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
                logger.debug("Retrying in %.1fs...", delay)
                time.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    def _raise_error(self, response: httpx.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        raise WeaverAPIError(
            response.status_code,
            code=payload.get("error", "unknown_error"),
            message=payload.get("message", response.text),
            retryable=bool(payload.get("retryable", False)),
        )


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
