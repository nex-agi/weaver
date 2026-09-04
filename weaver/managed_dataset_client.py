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

from typing import Any, Mapping
from urllib.parse import quote

from .types.managed_dataset import (
    ManagedDatasetInfo,
    ManagedDatasetPage,
    _dataset_name,
    _dataset_version,
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
