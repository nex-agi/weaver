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

"""Sync/async managed-dataset catalog clients."""

from __future__ import annotations

import asyncio
import builtins
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

import httpx

from ._http import WeaverAPIError
from ._managed_preparation import (
    PREPARATION_CHUNK_ITEMS,
    preparation_remaining,
    preparation_request,
    preparation_retry_delay,
    validate_preparation_wait,
)
from .types.managed_dataset import (
    ManagedDatasetInfo,
    ManagedDatasetPage,
    SampleRef,
    SampleRefLength,
    _dataset_name,
    _dataset_version,
    parse_sample_ref_lengths,
)


def _page_params(
    *,
    limit: int,
    offset: int,
    name: str | None,
    status: str | None,
    compatible_model: str | None,
) -> dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    for key, value in (
        ("name", name),
        ("status", status),
        ("compatible_model", compatible_model),
    ):
        if value is not None:
            normalized = _dataset_name(value, "name") if key == "name" else value.strip()
            if not normalized:
                raise ValueError(f"{key} must not be blank")
            params[key] = normalized
    return params


def _catalog_path(name: str, version: str) -> str:
    normalized_name = _dataset_name(name, "name")
    normalized_version = _dataset_version(version)
    return (
        f"/api/v1/managed-datasets/{quote(normalized_name, safe='')}"
        f"/versions/{quote(normalized_version, safe='')}"
    )


class ManagedDatasetsClient:
    """Authorized managed-dataset catalog bound to a synchronous service."""

    def __init__(self, service: Any) -> None:
        self._service = service

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        name: str | None = None,
        status: str | None = None,
        compatible_model: str | None = None,
    ) -> ManagedDatasetPage:
        params = _page_params(
            limit=limit,
            offset=offset,
            name=name,
            status=status,
            compatible_model=compatible_model,
        )
        payload = self._service.http.get("/api/v1/managed-datasets", params=params)
        if not isinstance(payload, Mapping):
            raise ValueError("managed dataset list response must be an object")
        return ManagedDatasetPage.from_payload(
            payload, requested_limit=limit, requested_offset=offset
        )

    def get(self, *, name: str, version: str) -> ManagedDatasetInfo:
        payload = self._service.http.get(_catalog_path(name, version))
        if not isinstance(payload, Mapping):
            raise ValueError("managed dataset response must be an object")
        info = ManagedDatasetInfo.from_payload(payload)
        expected = (_dataset_name(name, "name"), _dataset_version(version))
        if (info.name, info.version) != expected:
            raise ValueError("managed dataset response does not match the requested version")
        return info

    def prepare_sample_ref_lengths(
        self,
        refs: Sequence[SampleRef],
        *,
        base_model: str,
        training_max_sequence_length: int,
        wait: bool = True,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> builtins.list[SampleRefLength]:
        """Prepare requested samples before creating a GPU-backed training model.

        Args:
            refs: Current batch or bounded lookahead references.
            base_model: Exact model registry name.
            training_max_sequence_length: Exact subsequent training budget.
            wait: Wait for explicitly retryable preparation states.
            timeout: Preparation deadline in seconds, including session setup.
                Async cancels in-flight calls; sync caps each socket phase to
                remaining time and rejects late replies.
            poll_interval: Minimum retry delay, in seconds.

        Returns:
            Authoritative lengths in reference order; no token arrays or private IDs.
        """
        requested = list(refs)
        validate_preparation_wait(wait=wait, timeout=timeout, poll_interval=poll_interval)
        if not requested:
            return []
        # Validate before even initializing a session or submitting CPU work.
        preparation_request(
            requested[:PREPARATION_CHUNK_ITEMS],
            base_model=base_model,
            training_max_sequence_length=training_max_sequence_length,
        )
        if not all(isinstance(ref, SampleRef) for ref in requested):
            raise TypeError("refs must contain only SampleRef values")
        deadline = time.monotonic() + timeout
        self._service.ensure_session()
        preparation_remaining(deadline, time.monotonic())
        path = f"/api/v1/sessions/{self._service.session_id}/managed-dataset-sample-lengths"
        results: builtins.list[SampleRefLength] = []
        known: dict[SampleRef, int] = {}
        for start in range(0, len(requested), PREPARATION_CHUNK_ITEMS):
            chunk = requested[start : start + PREPARATION_CHUNK_ITEMS]
            body = preparation_request(
                chunk,
                base_model=base_model,
                training_max_sequence_length=training_max_sequence_length,
            )
            while True:
                try:
                    preparation_remaining(deadline, time.monotonic())
                    payload = self._service.http.post(
                        path, json=body, max_retries=1, deadline=deadline
                    )
                    preparation_remaining(deadline, time.monotonic())
                    break
                except httpx.TimeoutException as error:
                    raise TimeoutError("Timed out preparing managed samples") from error
                except WeaverAPIError as error:
                    delay = preparation_retry_delay(
                        error,
                        wait=wait,
                        remaining=deadline - time.monotonic(),
                        poll_interval=poll_interval,
                    )
                    time.sleep(delay)
            parsed = parse_sample_ref_lengths(chunk, payload)
            for item in parsed:
                if (
                    item.input_token_count >= training_max_sequence_length
                    or known.setdefault(item.sample_ref, item.input_token_count)
                    != item.input_token_count
                ):
                    raise ValueError("prepared sample lengths violate their exact contract")
            results.extend(parsed)
        return results


class AsyncManagedDatasetsClient:
    """Authorized managed-dataset catalog bound to an asynchronous service."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        name: str | None = None,
        status: str | None = None,
        compatible_model: str | None = None,
    ) -> ManagedDatasetPage:
        params = _page_params(
            limit=limit,
            offset=offset,
            name=name,
            status=status,
            compatible_model=compatible_model,
        )
        payload = await self._service.http.get("/api/v1/managed-datasets", params=params)
        if not isinstance(payload, Mapping):
            raise ValueError("managed dataset list response must be an object")
        return ManagedDatasetPage.from_payload(
            payload, requested_limit=limit, requested_offset=offset
        )

    async def get(self, *, name: str, version: str) -> ManagedDatasetInfo:
        payload = await self._service.http.get(_catalog_path(name, version))
        if not isinstance(payload, Mapping):
            raise ValueError("managed dataset response must be an object")
        info = ManagedDatasetInfo.from_payload(payload)
        expected = (_dataset_name(name, "name"), _dataset_version(version))
        if (info.name, info.version) != expected:
            raise ValueError("managed dataset response does not match the requested version")
        return info

    async def prepare_sample_ref_lengths(
        self,
        refs: Sequence[SampleRef],
        *,
        base_model: str,
        training_max_sequence_length: int,
        wait: bool = True,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> builtins.list[SampleRefLength]:
        """Prepare a bounded sample window without allocating a training model.

        Args:
            refs: Current batch or bounded lookahead references.
            base_model: Exact model registry name.
            training_max_sequence_length: Exact subsequent training budget.
            wait: Wait for explicitly retryable preparation states.
            timeout: Preparation deadline in seconds, including session setup.
                Async cancels in-flight calls; sync caps each socket phase to
                remaining time and rejects late replies.
            poll_interval: Minimum retry delay, in seconds.

        Returns:
            Authoritative lengths in reference order.
        """
        requested = list(refs)
        validate_preparation_wait(wait=wait, timeout=timeout, poll_interval=poll_interval)
        if not requested:
            return []
        preparation_request(
            requested[:PREPARATION_CHUNK_ITEMS],
            base_model=base_model,
            training_max_sequence_length=training_max_sequence_length,
        )
        if not all(isinstance(ref, SampleRef) for ref in requested):
            raise TypeError("refs must contain only SampleRef values")
        deadline = time.monotonic() + timeout
        remaining = preparation_remaining(deadline, time.monotonic())
        await asyncio.wait_for(self._service.ensure_session(), timeout=remaining)
        path = f"/api/v1/sessions/{self._service.session_id}/managed-dataset-sample-lengths"
        results: builtins.list[SampleRefLength] = []
        known: dict[SampleRef, int] = {}
        for start in range(0, len(requested), PREPARATION_CHUNK_ITEMS):
            chunk = requested[start : start + PREPARATION_CHUNK_ITEMS]
            body = preparation_request(
                chunk,
                base_model=base_model,
                training_max_sequence_length=training_max_sequence_length,
            )
            while True:
                try:
                    remaining = preparation_remaining(deadline, time.monotonic())
                    payload = await asyncio.wait_for(
                        self._service.http.post(path, json=body, max_retries=1, deadline=deadline),
                        timeout=remaining,
                    )
                    preparation_remaining(deadline, time.monotonic())
                    break
                except httpx.TimeoutException as error:
                    raise TimeoutError("Timed out preparing managed samples") from error
                except WeaverAPIError as error:
                    delay = preparation_retry_delay(
                        error,
                        wait=wait,
                        remaining=deadline - time.monotonic(),
                        poll_interval=poll_interval,
                    )
                    await asyncio.sleep(delay)
            parsed = parse_sample_ref_lengths(chunk, payload)
            for item in parsed:
                if (
                    item.input_token_count >= training_max_sequence_length
                    or known.setdefault(item.sample_ref, item.input_token_count)
                    != item.input_token_count
                ):
                    raise ValueError("prepared sample lengths violate their exact contract")
            results.extend(parsed)
        return results
