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

"""Offload large tensor payloads off the HTTP path via a shared store.

When ``WEAVER_BLOB_OFFLOAD`` is enabled, large dense tensor fields are written
to a shared store rooted at ``WEAVER_BLOB_ROOT`` and the wire payload carries a
small reference (``{"$blob": {...}}``) instead of the inline numbers. Both the
producer (this SDK) and the consumer (the trainer, which imports this module)
resolve the reference's relative ``key`` against their own ``WEAVER_BLOB_ROOT``
— only the key crosses the wire, never an absolute path.

Disabled by default: with the flag off, callers keep emitting inline payloads
and this module is never touched, so the wire format is byte-identical.

Config is read under either name: the SDK side reads ``WEAVER_BLOB_OFFLOAD`` /
``WEAVER_BLOB_ROOT``; the auto-provisioned trainer receives the same settings
through the orchestrator's ``WEAVER_TRAINER_*`` passthrough (which strips the
prefix), so it reads ``TRAINER_BLOB_OFFLOAD`` / ``TRAINER_BLOB_ROOT``.
"""

from __future__ import annotations

import functools
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .types.payload_ref import PayloadRef

logger = logging.getLogger(__name__)

# Marker key identifying an offloaded-blob reference on the wire. Its value is a
# ``PayloadRef`` (the SDK-wide large-payload envelope shared with router-replay):
# ``{storage, format, relative_path, dtype, size_bytes}`` plus the tensor-specific
# ``shape``/``codec``/``offset``/``splits`` carried in ``PayloadRef.metadata``.
BLOB_KEY = "$blob"
_CODEC = "raw"  # raw little-endian C-contiguous buffer
_STORAGE = "filesystem"  # PayloadRef.storage (a shared POSIX mount)
_FORMAT = "raw-tensor"  # PayloadRef.format (raw buffer, read via this store)

# dtype name (shared with types.tensor) <-> numpy dtype.
_NAME_TO_NUMPY: dict[str, np.dtype] = {
    "int64": np.dtype("<i8"),
    "int32": np.dtype("<i4"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}
_TORCH_TO_NAME: dict[torch.dtype, str] = {
    torch.int64: "int64",
    torch.int32: "int32",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_TO_TORCH: dict[str, torch.dtype] = {n: d for d, n in _TORCH_TO_NAME.items()}


def is_blob_ref(obj: Any) -> bool:
    """True if ``obj`` is an offloaded-blob reference on the wire."""
    return isinstance(obj, dict) and BLOB_KEY in obj


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _blob_env(suffix: str) -> Optional[str]:
    """Read a blob config value under either accepted name.

    The SDK process reads ``WEAVER_BLOB_<suffix>`` from its own environment.
    The trainer receives the same config through the orchestrator's
    ``WEAVER_TRAINER_*`` passthrough, which strips the prefix to
    ``TRAINER_BLOB_<suffix>``. Both names resolve to the same setting so a
    single value can be declared once on each side without renaming.
    """
    return os.getenv(f"WEAVER_BLOB_{suffix}") or os.getenv(f"TRAINER_BLOB_{suffix}")


class BlobStore:
    """Filesystem-backed store for offloaded tensors (POSIX/shared mount).

    The concrete backend is a directory tree under ``WEAVER_BLOB_ROOT``; the
    code never names it beyond that env var. Keys are relative paths of the
    form ``<run>/<model_id>/seq-<seq_id>/<field>.bin``.
    """

    def __init__(self) -> None:
        self._enabled = _truthy(_blob_env("OFFLOAD"))
        root = _blob_env("ROOT")
        self._root: Optional[Path] = Path(root) if root else None
        # Keep the last K seq directories per model; older ones are GC'd.
        # K must exceed the producer's max in-flight depth: ops are submitted
        # asynchronously and the consumer parses them one at a time, so each
        # blob must survive until its op is claimed. A too-small K would delete
        # blobs that are still queued. 64 covers the current pipelined submission
        # depth (one step's chunks, drained at a per-step barrier) with margin;
        # override via WEAVER_BLOB_KEEP_LAST_K / TRAINER_BLOB_KEEP_LAST_K.
        try:
            self._keep_last_k = max(1, int(_blob_env("KEEP_LAST_K") or "64"))
        except ValueError:
            self._keep_last_k = 64
        # unique per process/run; only the writer needs this (it ships the key).
        self._run = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        self._lock = threading.Lock()
        self._pruned_seq: dict[str, int] = {}  # model_id -> last seq pruned up to
        self._write_seq: dict[str, int] = {}  # model_id -> next auto write seq
        # Reader-side cache of whole packed files. A packed chunk is read once
        # and its many per-field references are sliced out of memory, instead of
        # re-reading (or re-opening) the same file per reference.
        self._read_cache: "OrderedDict[str, bytes]" = OrderedDict()
        try:
            self._read_cache_max = max(1, int(_blob_env("READ_CACHE") or "4"))
        except ValueError:
            self._read_cache_max = 4

        if self._enabled:
            if self._root is None:
                logger.warning(
                    "Blob offload requested but the store root is empty "
                    "(set WEAVER_BLOB_ROOT, or TRAINER_BLOB_ROOT on the trainer); "
                    "falling back to inline HTTP payloads."
                )
                self._enabled = False
            else:
                try:
                    self._root.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    logger.warning(
                        "WEAVER_BLOB_ROOT %s not writable (%s); "
                        "falling back to inline HTTP payloads.",
                        self._root,
                        exc,
                    )
                    self._enabled = False

        if self._enabled:
            logger.info(
                "Blob offload ENABLED: root=%s run=%s keep_last_k=%d",
                self._root,
                self._run,
                self._keep_last_k,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run(self) -> str:
        return self._run

    # ------------------------------------------------------------------ write
    def put_tensor(
        self,
        tensor: torch.Tensor,
        *,
        model_id: str,
        seq_id: int,
        field: str,
    ) -> dict[str, Any]:
        """Write ``tensor`` to a blob and return its ``$blob`` reference.

        The tensor is stored as a raw little-endian C-contiguous buffer; dtype
        and shape travel in the reference so the reader can reconstruct it
        without re-parsing JSON.
        """
        assert self._root is not None  # guarded by ``enabled``
        dtype_name, shape, buf = _tensor_to_buf(tensor)

        key = f"{self._run}/{model_id}/seq-{seq_id}/{_safe_field(field)}.bin"
        self._write_atomic(self._root / key, [buf])
        self._maybe_gc(model_id, seq_id)

        return {
            BLOB_KEY: PayloadRef(
                storage=_STORAGE,
                format=_FORMAT,
                relative_path=key,
                dtype=dtype_name,
                size_bytes=len(buf),
                metadata={"shape": shape, "codec": _CODEC},
            ).to_payload()
        }

    def _write_atomic(self, dest: Path, parts: Sequence[bytes]) -> None:
        """Write ``parts`` to ``dest`` atomically (temp file + fsync + rename)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.tmp")
        with open(tmp, "wb") as fh:
            for part in parts:
                fh.write(part)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)

    def open_pack(self, *, model_id: str, seq_id: int) -> "PackWriter":
        """Open a writer that packs one chunk's tensors into a single file.

        Use this instead of many :meth:`put_tensor` calls when serializing a
        batch: every per-datum field becomes one slice of one file rather than
        its own tiny file (one create + fsync + rename each).
        """
        assert self._root is not None  # guarded by ``enabled``
        return PackWriter(self, model_id=model_id, seq_id=seq_id)

    # ------------------------------------------------------------------- read
    def _safe_key_path(self, relative_path: str) -> Path:
        """``self._root / relative_path``, refusing anything that could escape the
        root.

        ``relative_path`` arrives on the wire inside an (untrusted) training
        payload, so a crafted ref must not let the reader resolve a file outside
        ``WEAVER_BLOB_ROOT`` — reject absolute paths (which would replace the
        root), ``..``/``.`` components, and backslash separators.
        """
        assert self._root is not None
        parts = relative_path.split("/")
        if (
            not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or ".." in parts
            or "." in parts
        ):
            raise ValueError(f"unsafe blob relative_path: {relative_path!r}")
        return self._root / relative_path

    def get_array(self, ref: dict[str, Any]) -> np.ndarray:
        """Resolve a ``$blob`` reference to a numpy array (reader side)."""
        if self._root is None:
            raise RuntimeError(
                "Got a $blob reference but the store root is not configured "
                "(set WEAVER_BLOB_ROOT, or TRAINER_BLOB_ROOT on the trainer)."
            )
        pr = PayloadRef.from_payload(ref[BLOB_KEY])
        codec = pr.metadata.get("codec", _CODEC)
        if codec != _CODEC:
            raise ValueError(f"Unsupported blob codec: {codec}")
        np_dtype = _NAME_TO_NUMPY.get(pr.dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported blob dtype: {pr.dtype}")
        if pr.relative_path is None:
            raise ValueError("$blob reference is missing relative_path")
        path = self._safe_key_path(pr.relative_path)
        nbytes = pr.size_bytes
        offset = pr.metadata.get("offset")
        if offset is not None:
            # Packed file: this reference is one slice of a shared file. Read the
            # file once (cached) and slice, so a chunk's many references cost one
            # read rather than one per reference.
            if nbytes is None:
                raise ValueError(f"Blob {pr.relative_path} has offset but no size_bytes")
            whole = self._read_file_cached(path)
            end = offset + nbytes
            if end > len(whole):
                raise ValueError(
                    f"Blob {pr.relative_path} slice [{offset}:{end}] out of range "
                    f"(file length {len(whole)}; partial/corrupt read)."
                )
            buf = whole[offset:end]
        else:
            buf = path.read_bytes()
            if nbytes is not None and len(buf) != nbytes:
                raise ValueError(
                    f"Blob {pr.relative_path} length {len(buf)} != expected {nbytes} "
                    "(partial/corrupt read)."
                )
        # .copy() so the array owns writable memory (np.frombuffer is read-only).
        return np.frombuffer(buf, dtype=np_dtype).reshape(pr.metadata["shape"]).copy()

    def _read_file_cached(self, path: Path) -> bytes:
        """Return the full bytes of ``path``, caching whole packed files.

        The I/O happens outside the lock; a concurrent double-read is harmless
        (the content is identical), and the cache keeps the last K files so a
        chunk's references all hit the same in-memory buffer.
        """
        key = str(path)
        with self._lock:
            cached = self._read_cache.get(key)
            if cached is not None:
                self._read_cache.move_to_end(key)
                return cached
        data = path.read_bytes()
        with self._lock:
            self._read_cache[key] = data
            self._read_cache.move_to_end(key)
            while len(self._read_cache) > self._read_cache_max:
                self._read_cache.popitem(last=False)
        return data

    def get_tensor(self, ref: dict[str, Any]) -> torch.Tensor:
        """Resolve a ``$blob`` reference to a torch tensor (reader side)."""
        return torch.from_numpy(self.get_array(ref))

    # ---------------------------------------------------------------- results
    def next_write_seq(self, model_id: str) -> int:
        """Monotonic seq for store-side (e.g. trainer result) writes.

        Unlike forward/forward_backward inputs (whose seq is the operation seq
        from the SDK), result blobs are written by the consumer side, which has
        no operation seq — so it auto-assigns one here. The keep-last-K GC then
        prunes old result blobs the same way.
        """
        with self._lock:
            seq = self._write_seq.get(model_id, 0)
            self._write_seq[model_id] = seq + 1
        return seq

    def put_packed(
        self,
        arrays: Sequence[Any],
        *,
        model_id: str,
        field: str,
    ) -> dict[str, Any]:
        """Write a list of 1-D float sequences as one concatenated blob.

        The returned ``$blob`` reference carries per-sequence lengths under
        ``"splits"`` so :meth:`get_packed` can reconstruct the original list in
        order. One blob per call (not per element) keeps the file/fsync count
        low. Order is preserved exactly — the reader splits back position-for-
        position.
        """
        seqs = [torch.as_tensor(a, dtype=torch.float32).reshape(-1) for a in arrays]
        splits = [int(s.numel()) for s in seqs]
        flat = torch.cat(seqs) if seqs else torch.zeros(0, dtype=torch.float32)
        ref = self.put_tensor(
            flat,
            model_id=model_id,
            seq_id=self.next_write_seq(model_id),
            field=field,
        )
        ref[BLOB_KEY]["splits"] = splits
        return ref

    def get_packed(self, ref: dict[str, Any]) -> list[np.ndarray]:
        """Resolve a packed ``$blob`` reference back to its list of arrays."""
        flat = self.get_array(ref)
        splits = ref[BLOB_KEY].get("splits")
        if splits is None:
            return [flat]
        total = int(sum(splits))
        if total != flat.size:
            raise ValueError(
                f"Blob {ref[BLOB_KEY].get('relative_path')} splits sum to {total} "
                f"but the array has {flat.size} elements (corrupt/mismatched descriptor)."
            )
        out: list[np.ndarray] = []
        offset = 0
        for length in splits:
            out.append(flat[offset : offset + length])
            offset += length
        return out

    # --------------------------------------------------------------------- gc
    def _maybe_gc(self, model_id: str, seq_id: int) -> None:
        """Prune seq directories older than the last K for this model."""
        with self._lock:
            if self._pruned_seq.get(model_id, -1) >= seq_id:
                return
            self._pruned_seq[model_id] = seq_id
        assert self._root is not None
        model_dir = self._root / self._run / model_id
        cutoff = seq_id - self._keep_last_k
        if cutoff < 0:
            return
        try:
            for child in model_dir.glob("seq-*"):
                try:
                    n = int(child.name.split("seq-", 1)[1])
                except (IndexError, ValueError):
                    continue
                if n <= cutoff:
                    _rmtree_quiet(child)
        except OSError:
            pass  # GC is best-effort; never fail a training step on it.


def _tensor_to_buf(tensor: torch.Tensor) -> tuple[str, list[int], bytes]:
    """Materialize ``tensor`` to (dtype name, shape, raw C-contiguous bytes)."""
    dtype_name = _TORCH_TO_NAME.get(tensor.dtype)
    if dtype_name is None:
        raise TypeError(f"Unsupported dtype for blob offload: {tensor.dtype}")
    arr = tensor.detach().cpu().contiguous().numpy()
    return dtype_name, list(arr.shape), arr.tobytes(order="C")


class PackWriter:
    """Accumulate one chunk's tensors into a single packed blob.

    A training chunk's per-datum input fields would otherwise become hundreds
    of tiny files — one create + fsync + rename each, which dominates wall time
    on a shared store. ``PackWriter`` buffers them and writes one file on
    :meth:`commit`; every reference it returns carries a byte ``offset`` into
    that file so the reader slices its tensor back out. Not thread-safe: one
    writer serves one chunk's serialization.
    """

    def __init__(self, store: "BlobStore", *, model_id: str, seq_id: int) -> None:
        self._store = store
        self._model_id = model_id
        self._seq_id = seq_id
        self._key = f"{store.run}/{model_id}/seq-{seq_id}/pack.bin"
        self._parts: list[bytes] = []
        self._offset = 0

    def put_tensor(self, tensor: torch.Tensor, *, field: str) -> dict[str, Any]:
        """Append ``tensor`` to the pack and return its offset reference.

        ``field`` is accepted for call-site symmetry with
        :meth:`BlobStore.put_tensor`; the slice is identified by ``offset``.
        """
        del field  # offset (not a file name) identifies the slice
        dtype_name, shape, buf = _tensor_to_buf(tensor)
        offset = self._offset
        self._parts.append(buf)
        self._offset += len(buf)
        return {
            BLOB_KEY: PayloadRef(
                storage=_STORAGE,
                format=_FORMAT,
                relative_path=self._key,
                dtype=dtype_name,
                size_bytes=len(buf),
                metadata={"shape": shape, "codec": _CODEC, "offset": offset},
            ).to_payload()
        }

    def commit(self) -> None:
        """Flush the accumulated tensors to one file (one fsync + rename)."""
        if not self._parts:
            return
        # PackWriter is a friend of BlobStore (same module); the writes below go
        # through BlobStore's internal helpers intentionally.
        store = self._store
        # pylint: disable=protected-access
        root = store._root
        assert root is not None
        store._write_atomic(root / self._key, self._parts)
        self._parts = []
        store._maybe_gc(self._model_id, self._seq_id)


def _safe_field(field: str) -> str:
    """Sanitize a field name into a single safe path segment."""
    keep = [c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in field]
    out = "".join(keep).strip("._") or "field"
    return out


def _rmtree_quiet(path: Path) -> None:
    try:
        for child in path.iterdir():
            if child.is_dir():
                _rmtree_quiet(child)
            else:
                child.unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        pass


@functools.lru_cache(maxsize=1)
def get_blob_store() -> BlobStore:
    """Return the process-wide blob store (created lazily on first use)."""
    return BlobStore()
