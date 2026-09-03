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

"""Sync/async managed-dataset catalog and verified public downloads."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote
from uuid import uuid4

from ._safeio import (
    legacy_open_for_write,
    open_for_write,
    open_parent_fd,
    publish_within,
    supports_dir_fd,
    unlink_within,
)
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


def _download_path(name: str, version: str) -> str:
    return f"{_catalog_path(name, version)}/download"


def _destination_path(destination: str | Path, *, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    path = Path(destination).expanduser()
    if path.exists() and path.is_dir():
        raise ValueError("managed dataset download destination must be a file path")
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(f"managed dataset download destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _flush_and_sync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


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

    def download(
        self,
        *,
        name: str,
        version: str,
        destination: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Download one authorized public dataset as canonical JSONL.

        The authenticated response is streamed to a same-directory ``.part``
        file, checked against its exact size and SHA-256 headers, then
        atomically published. Protected datasets are never downloadable. An
        existing destination is preserved unless ``overwrite=True``.
        """

        info = self.get(name=name, version=version)
        if info.content_visibility != "public":
            raise ValueError("protected managed datasets cannot be downloaded")
        destination_path = _destination_path(destination, overwrite=overwrite)
        part_name = f".{destination_path.name}.{uuid4().hex}.part"
        if supports_dir_fd():
            relative = PurePosixPath(destination_path.name)
            parent_fd = open_parent_fd(destination_path.parent, relative, create=False)
            try:
                try:
                    with open_for_write(parent_fd, part_name, append=False) as handle:
                        self._service.http.download_managed_dataset(
                            _download_path(name, version), handle
                        )
                        _flush_and_sync(handle)
                    publish_within(
                        parent_fd,
                        part_name,
                        destination_path.name,
                        overwrite=overwrite,
                    )
                    os.fsync(parent_fd)
                except BaseException:
                    unlink_within(parent_fd, part_name)
                    raise
            finally:
                os.close(parent_fd)
            return destination_path

        part_path = destination_path.with_name(part_name)
        try:
            with legacy_open_for_write(part_path, append=False) as handle:
                self._service.http.download_managed_dataset(_download_path(name, version), handle)
                _flush_and_sync(handle)
            if overwrite:
                part_path.replace(destination_path)
            else:
                os.link(part_path, destination_path, follow_symlinks=False)
                part_path.unlink()
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        return destination_path


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

    async def download(
        self,
        *,
        name: str,
        version: str,
        destination: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Async counterpart of :meth:`ManagedDatasetsClient.download`."""

        info = await self.get(name=name, version=version)
        if info.content_visibility != "public":
            raise ValueError("protected managed datasets cannot be downloaded")
        destination_path = _destination_path(destination, overwrite=overwrite)
        part_name = f".{destination_path.name}.{uuid4().hex}.part"
        if supports_dir_fd():
            relative = PurePosixPath(destination_path.name)
            parent_fd = open_parent_fd(destination_path.parent, relative, create=False)
            try:
                try:
                    with open_for_write(parent_fd, part_name, append=False) as handle:
                        await self._service.http.download_managed_dataset(
                            _download_path(name, version), handle
                        )
                        await asyncio.to_thread(_flush_and_sync, handle)
                    publish_within(
                        parent_fd,
                        part_name,
                        destination_path.name,
                        overwrite=overwrite,
                    )
                    os.fsync(parent_fd)
                except BaseException:
                    unlink_within(parent_fd, part_name)
                    raise
            finally:
                os.close(parent_fd)
            return destination_path

        part_path = destination_path.with_name(part_name)
        try:
            with legacy_open_for_write(part_path, append=False) as handle:
                await self._service.http.download_managed_dataset(
                    _download_path(name, version), handle
                )
                await asyncio.to_thread(_flush_and_sync, handle)
            if overwrite:
                part_path.replace(destination_path)
            else:
                os.link(part_path, destination_path, follow_symlinks=False)
                part_path.unlink()
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        return destination_path
