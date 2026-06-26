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

"""Helpers for Weaver long-running operations."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from ._http import APIClient, WeaverAPIError, backoff_delays
from ._utils import extract_id, lookup_case_insensitive

if TYPE_CHECKING:
    from ._async_http import AsyncAPIClient

logger = logging.getLogger(__name__)

_TRANSIENT_OPERATION_POLL_ERROR_FRAGMENTS = (
    "unexpected eof",
    "connection reset",
    "broken pipe",
    "bad connection",
    "server closed the connection",
)


def _positive_float_env(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {raw!r}")
    return value


def _operation_poll_transient_retries() -> int:
    """Number of transient operation-refresh failures to tolerate per wait."""

    return _nonnegative_int_env("WEAVER_OPERATION_POLL_TRANSIENT_RETRIES", 20)


def _is_transient_operation_poll_error(exc: WeaverAPIError) -> bool:
    if exc.retryable:
        return True
    if exc.status_code >= 500:
        return True

    # Legacy weaver-server mis-maps repo/read errors to 404 not_found; tolerate
    # those transient I/O messages while old servers are live (a real missing
    # operation, without a transient fragment, still stays fatal).
    message = f"{exc.code} {exc.message}".lower()
    return (
        exc.status_code == 404
        and exc.code == "not_found"
        and any(fragment in message for fragment in _TRANSIENT_OPERATION_POLL_ERROR_FRAGMENTS)
    )


def _operation_poll_delays() -> Iterator[float]:
    """Yield operation polling delays.

    ``WEAVER_OPERATION_POLL_INTERVAL`` sets a fixed interval for latency-
    sensitive loops (e.g. RL training waiting step-by-step on one operation);
    otherwise fall back to the default exponential backoff.
    """
    fixed_interval = _positive_float_env("WEAVER_OPERATION_POLL_INTERVAL")
    if fixed_interval is not None:
        while True:
            yield fixed_interval
    yield from backoff_delays()


class WeaverOperationError(RuntimeError):
    def __init__(self, payload: Dict[str, Any]):
        message = lookup_case_insensitive(payload, "error") or "operation_failed"
        super().__init__(f"Operation failed: {message}")
        self.payload = payload


class _OperationHandleMixin:
    """Pure (IO-free) status accessors shared by sync and async handles."""

    operation_id: str
    _cached: Dict[str, Any]

    @property
    def status(self) -> Optional[str]:
        status = lookup_case_insensitive(self._cached, "status")
        return str(status).lower() if status is not None else None

    @property
    def response(self) -> Any:
        return lookup_case_insensitive(self._cached, "response")

    @property
    def error(self) -> Optional[str]:
        err = lookup_case_insensitive(self._cached, "error")
        return str(err) if err else None

    def done(self) -> bool:
        status = self.status
        return status in {"done", "error"}

    def _raise_if_failed(self) -> None:
        if self.status == "error":
            raise WeaverOperationError(self._cached)


@dataclass
class OperationHandle(_OperationHandleMixin):
    client: APIClient
    operation_id: str
    _cached: Dict[str, Any]

    @classmethod
    def from_payload(cls, client: APIClient, payload: Dict[str, Any]) -> "OperationHandle":
        op_id = extract_id(payload)
        return cls(client=client, operation_id=str(op_id), _cached=payload)

    def refresh(self) -> Dict[str, Any]:
        path = f"/api/v1/operations/{self.operation_id}"
        self._cached = self.client.get(path)
        return self._cached

    def wait(self) -> Dict[str, Any]:
        transient_error_count = 0
        max_transient_errors = _operation_poll_transient_retries()
        if self.done():
            return self._cached
        for delay in _operation_poll_delays():
            time.sleep(delay)
            try:
                self.refresh()
            except WeaverAPIError as exc:
                if (
                    transient_error_count < max_transient_errors
                    and _is_transient_operation_poll_error(exc)
                ):
                    transient_error_count += 1
                    logger.warning(
                        "Transient error while polling operation %s; retrying (%d/%d): %s",
                        self.operation_id,
                        transient_error_count,
                        max_transient_errors,
                        exc,
                    )
                    continue
                raise
            if self.done():
                break
        if not self.done():
            raise WeaverAPIError(504, "timeout", "Operation polling timed out", True)
        self._raise_if_failed()
        return self._cached

    def result(self) -> Any:
        payload = self.wait()
        return lookup_case_insensitive(payload, "response")

    @classmethod
    def wait_all(cls, handles: List["OperationHandle"]) -> List[Any]:
        """Wait for all handles to complete and return their results.

        Args:
            handles: List of OperationHandle instances to wait on.

        Returns:
            List of results in the same order as the input handles.
        """
        return [h.result() for h in handles]


@dataclass
class AsyncOperationHandle(_OperationHandleMixin):
    """Asyncio twin of :class:`OperationHandle`.

    Polling uses ``asyncio.sleep`` so awaiting a handle yields the event loop
    to other coroutines while the server works. The handle is also directly
    awaitable: ``result = await handle`` is shorthand for ``await
    handle.result()``.
    """

    client: "AsyncAPIClient"
    operation_id: str
    _cached: Dict[str, Any]

    @classmethod
    def from_payload(
        cls, client: "AsyncAPIClient", payload: Dict[str, Any]
    ) -> "AsyncOperationHandle":
        op_id = extract_id(payload)
        return cls(client=client, operation_id=str(op_id), _cached=payload)

    async def refresh(self) -> Dict[str, Any]:
        path = f"/api/v1/operations/{self.operation_id}"
        self._cached = await self.client.get(path)
        return self._cached

    async def wait(self) -> Dict[str, Any]:
        transient_error_count = 0
        max_transient_errors = _operation_poll_transient_retries()
        if self.done():
            return self._cached
        for delay in _operation_poll_delays():
            await asyncio.sleep(delay)
            try:
                await self.refresh()
            except WeaverAPIError as exc:
                if (
                    transient_error_count < max_transient_errors
                    and _is_transient_operation_poll_error(exc)
                ):
                    transient_error_count += 1
                    logger.warning(
                        "Transient error while polling operation %s; retrying (%d/%d): %s",
                        self.operation_id,
                        transient_error_count,
                        max_transient_errors,
                        exc,
                    )
                    continue
                raise
            if self.done():
                break
        if not self.done():
            raise WeaverAPIError(504, "timeout", "Operation polling timed out", True)
        self._raise_if_failed()
        return self._cached

    async def result(self) -> Any:
        payload = await self.wait()
        return lookup_case_insensitive(payload, "response")

    def __await__(self):
        return self.result().__await__()

    @classmethod
    async def wait_all(cls, handles: List["AsyncOperationHandle"]) -> List[Any]:
        """Concurrently wait for all handles and return results in input order."""
        return await asyncio.gather(*(handle.result() for handle in handles))


def build_operation_handle(client: APIClient, payload: Dict[str, Any]) -> OperationHandle:
    return OperationHandle.from_payload(client, payload)


def build_async_operation_handle(
    client: "AsyncAPIClient", payload: Dict[str, Any]
) -> AsyncOperationHandle:
    return AsyncOperationHandle.from_payload(client, payload)
