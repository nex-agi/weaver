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

"""Filesystem access anchored to an open directory descriptor.

Artifact descriptors name their own destination files, so ``download_weights``
turns untrusted text into local paths. Name validation and a resolved-parent
containment check both reason about *names*, and a name is only true at the
instant it is resolved: between the check and the write, anyone able to create
entries in the destination tree can replace an intermediate directory with a
symlink and the bytes follow it out of the tree. ``O_NOFOLLOW`` does not close
that window — it refuses a symlink only at the final component.

This module removes the window rather than narrowing it. The destination is
walked one component at a time with ``openat`` semantics (``O_NOFOLLOW`` on
every component below the root), and stat, open, rename and unlink are then
issued relative to the descriptor that walk returns. A descriptor pins an
*inode*, not a name: once the walk succeeds, renaming that directory or
replacing it with a symlink cannot move the write, because nothing afterwards
re-traverses the name.

Both client stacks share these helpers. They are ordinary blocking ``os``
calls, which is what the async stack wants: each one is a single metadata
syscall, no more expensive than the per-chunk ``fh.write`` the async streamer
already performs inline, while the unbounded work (walking a directory tree,
hashing a multi-GB shard) is handed to a worker thread by the caller via
``asyncio.to_thread``. The event loop is never held for an unbounded time.

``dir_fd`` is a POSIX facility; on Windows ``os.supports_dir_fd`` is empty and
there is no ``O_NOFOLLOW``. Windows is not a supported execution environment
for this SDK, so :func:`supports_dir_fd` gates the anchored implementation and
callers take a single documented fallback branch there
(:func:`legacy_open_for_write`) that keeps the previous path-based behaviour.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path, PurePosixPath
from typing import BinaryIO

# Absent on Windows. Falling back to 0 keeps the flag arithmetic total; the
# anchored code paths that rely on them are gated by supports_dir_fd().
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

# How a refused component surfaces: ELOOP when O_NOFOLLOW meets a symlink,
# ENOTDIR when the component is (or points at) a non-directory.
_UNSAFE_COMPONENT_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR})


def supports_dir_fd() -> bool:
    """Whether this platform can anchor filesystem calls to a directory fd.

    Evaluated per call rather than cached at import so tests can exercise the
    fallback branch by emptying ``os.supports_dir_fd``. It is a handful of set
    lookups against a fixed-size set, once per downloaded file.
    """
    return (
        os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
    )


def open_parent_fd(root: Path, rel: PurePosixPath, *, create: bool) -> int:
    """Open the directory that will hold *rel*'s final component.

    *root* is opened by path: it is the caller's own destination, so resolving
    a symlink there is intended. Every component below it is opened with
    ``O_NOFOLLOW``, so a symlink standing anywhere along the way fails the walk
    instead of redirecting it — including one planted between two iterations,
    since each step is resolved relative to the descriptor of the step above.

    Args:
        root: Destination directory; must already exist.
        rel: Validated relative path. Only ``rel.parts[:-1]`` is walked; the
            final component is the caller's business.
        create: Create missing intermediate directories while descending.

    Returns:
        A directory descriptor owned by the caller, who MUST close it.

    Raises:
        ValueError: A component is a symlink or is not a directory.
        OSError: The walk failed for an ordinary IO reason (missing component
            with ``create=False``, permissions, ...).
    """
    fd = os.open(os.fspath(root), os.O_RDONLY | _O_DIRECTORY)
    try:
        for part in rel.parts[:-1]:
            child_fd = _open_directory(fd, part, create=create)
            os.close(fd)
            fd = child_fd
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_directory(parent_fd: int, name: str, *, create: bool) -> int:
    """Open (optionally creating) the subdirectory *name* under *parent_fd*."""
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
    except OSError as exc:
        _reject_unsafe_component(name, exc)
        raise
    try:
        os.mkdir(name, dir_fd=parent_fd)
    except FileExistsError:
        # Another writer created it between our open and our mkdir. Re-open
        # below and let O_NOFOLLOW decide whether what appeared is a directory
        # or a symlink someone planted in the same window.
        pass
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        _reject_unsafe_component(name, exc)
        raise


def _reject_unsafe_component(name: str, exc: OSError) -> None:
    """Translate a refused directory component into :class:`ValueError`.

    Returns normally for unrelated ``OSError``s so the caller re-raises them
    as the IO errors they are.
    """
    if exc.errno in _UNSAFE_COMPONENT_ERRNOS:
        raise ValueError(
            f"unsafe path component in the download destination: {name!r} "
            "is a symlink or not a directory"
        ) from exc


def _reject_symlink_leaf(name: str, exc: OSError, *, verb: str) -> None:
    """Translate ``ELOOP`` on a final component into :class:`ValueError`."""
    if exc.errno == errno.ELOOP:
        raise ValueError(f"refusing to {verb} through a symlink: {name}") from exc


def open_for_write(parent_fd: int, name: str, *, append: bool) -> BinaryIO:
    """Open *name* under *parent_fd* for writing, never following a symlink.

    Args:
        parent_fd: Directory descriptor from :func:`open_parent_fd`.
        name: Single path component; never contains a separator.
        append: Keep the existing bytes and append to them (a resumed
            download); otherwise truncate what is there.

    Raises:
        ValueError: *name* is a symlink.
    """
    flags = os.O_WRONLY | os.O_CREAT | _O_NOFOLLOW
    flags |= os.O_APPEND if append else os.O_TRUNC
    try:
        fd = os.open(name, flags, 0o666, dir_fd=parent_fd)
    except OSError as exc:
        _reject_symlink_leaf(name, exc, verb="write")
        raise
    try:
        if append:
            return os.fdopen(fd, "ab")
        return os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise


def open_for_read(parent_fd: int, name: str) -> BinaryIO | None:
    """Open *name* under *parent_fd* for reading, or None when it is missing.

    Raises:
        ValueError: *name* is a symlink.
    """
    try:
        fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        _reject_symlink_leaf(name, exc, verb="read")
        raise
    try:
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def stat_no_follow(parent_fd: int, name: str) -> os.stat_result | None:
    """``lstat`` *name* under *parent_fd*, or None when it does not exist.

    Not following the link is what lets callers *detect* a planted symlink
    (and reject it) instead of silently measuring its target.
    """
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def rename_within(parent_fd: int, src_name: str, dst_name: str) -> None:
    """Rename *src_name* to *dst_name*, both resolved against *parent_fd*.

    ``rename`` never follows a symlink at either end, so publishing a download
    replaces whatever entry holds the final name rather than writing through
    it, and the anchor keeps both ends inside the directory the walk verified.
    """
    os.rename(src_name, dst_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)


def unlink_within(parent_fd: int, name: str) -> None:
    """Unlink *name* under *parent_fd*; a name that is already gone is fine."""
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _no_follow_opener(path: str, flags: int) -> int:
    """Open refusing to follow a symlink at the final component."""
    return os.open(path, flags | _O_NOFOLLOW)


def legacy_open_for_write(path: Path, *, append: bool) -> BinaryIO:
    """Path-based fallback for platforms without ``dir_fd`` support.

    Only reachable where :func:`supports_dir_fd` is false — Windows, which is
    not a supported execution environment for this SDK. It preserves the
    behaviour that predates the anchored walk: refuse a symlink standing at
    the destination, then open with ``O_NOFOLLOW`` where the platform has it.
    That leaves the check-to-use window the anchored path removes, which is
    the accepted cost of the unsupported platform.

    Raises:
        ValueError: *path* is a symlink.
    """
    if path.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {path}")
    if append:
        return open(path, "ab", opener=_no_follow_opener)
    return open(path, "wb", opener=_no_follow_opener)
