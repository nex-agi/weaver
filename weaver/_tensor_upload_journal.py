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

"""Private, credential-free recovery records; no network or event-loop ownership."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Generator
from pathlib import Path
from typing import Any
from uuid import uuid4

from .tensor_transport import TensorPack

CHUNK_BYTES = 1 << 20
MAX_JOURNAL_BYTES = 17 << 20


def _source_fd(pack: TensorPack) -> int:
    fd = os.open(pack.path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != pack.size_bytes:
            raise ValueError("tensor source is not the expected regular file")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _verify_source(pack: TensorPack) -> None:
    fd = _source_fd(pack)
    try:
        digest = hashlib.sha256()
        while chunk := os.read(fd, CHUNK_BYTES):
            digest.update(chunk)
        if digest.hexdigest() != pack.sha256.lower():
            raise ValueError("tensor source digest changed")
    finally:
        os.close(fd)


def _sync_directory(parent: Path) -> None:
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _create_journal(
    base_url: str, path: str, body: dict[str, Any], pack: TensorPack, idempotency_key: str | None
) -> tuple[Path, dict[str, Any]]:
    _verify_source(pack)
    upload_id = str(uuid4())
    journal = pack.path.absolute().with_name(".weaver-tensor-upload-" + upload_id + ".json")
    record = {
        "version": 1,
        "protocol": "weaver-tensor-input-recovery/v1",
        "base_url": base_url,
        "path": path,
        "request": body,
        "source_path": str(pack.path.absolute()),
        "upload_id": upload_id,
        "idempotency_key": idempotency_key or upload_id,
        "tensor_pack": {
            "size_bytes": pack.size_bytes,
            "sha256": pack.sha256.lower(),
            "codec": pack.codec,
            "decoded_size_bytes": pack.decoded_size_bytes,
        },
    }
    _persist_record(journal, pack, record)
    return journal, record


def _persist_record(journal: Path, pack: TensorPack, record: dict[str, Any]) -> None:
    raw = json.dumps(record, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > MAX_JOURNAL_BYTES:
        raise ValueError("tensor upload recovery metadata exceeds its bound")
    fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        _sync_directory(journal.parent)
    except BaseException:
        journal.unlink(missing_ok=True)
        raise
    pack._remote_recovery = journal
    pack._retain_source = True


def _load_journal(journal: Path, base_url: str) -> tuple[dict[str, Any], TensorPack]:
    fd = os.open(journal, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > MAX_JOURNAL_BYTES
        ):
            raise ValueError("tensor recovery requires a private owned regular journal")
        record = json.loads(source.read(MAX_JOURNAL_BYTES + 1))
    if (
        record.get("version") != 1
        or record.get("protocol") != "weaver-tensor-input-recovery/v1"
        or record.get("base_url") != base_url
    ):
        raise ValueError("tensor recovery endpoint or protocol changed")
    pack = TensorPack(Path(record["source_path"]), **record["tensor_pack"])
    _verify_source(pack)
    pack._remote_recovery, pack._retain_source = journal, True
    return record, pack


def _finish_journal(pack: TensorPack) -> None:
    if pack._remote_recovery is not None:
        journal = pack._remote_recovery
        journal.unlink(missing_ok=True)
        _sync_directory(journal.parent)
    pack._remote_recovery, pack._retain_source = None, False


def _part_md5(pack: TensorPack, offset: int, size: int) -> str:
    import base64

    fd = _source_fd(pack)
    try:
        digest, remaining = hashlib.md5(usedforsecurity=False), size
        while remaining:
            chunk = os.pread(fd, min(CHUNK_BYTES, remaining), offset)
            if not chunk:
                raise ValueError("tensor source ended before its signed part")
            digest.update(chunk)
            offset, remaining = offset + len(chunk), remaining - len(chunk)
        return base64.b64encode(digest.digest()).decode()
    finally:
        os.close(fd)


def _part_chunks(pack: TensorPack, offset: int, size: int) -> Generator[bytes, None, None]:
    fd = _source_fd(pack)
    try:
        remaining = size
        while remaining:
            chunk = os.pread(fd, min(CHUNK_BYTES, remaining), offset)
            if not chunk:
                raise ValueError("tensor source ended before its signed part")
            yield chunk
            offset, remaining = offset + len(chunk), remaining - len(chunk)
    finally:
        os.close(fd)
