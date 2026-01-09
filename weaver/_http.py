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

import httpx

from . import __version__
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
        last_exception = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.request(method, path, params=params, json=json)
                if response.is_success:
                    if response.status_code == httpx.codes.NO_CONTENT:
                        return None
                    if not response.content:
                        return None
                    return response.json()
                self._raise_error(response)

            except Exception as e:
                last_exception = e
                is_last_attempt = attempt == self._max_retries - 1

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
