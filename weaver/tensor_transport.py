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

"""Binary transport for dense cross-entropy training tensors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import secrets
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence, cast

import httpx
import torch
import zstandard

from .config import TensorCompression, TensorTransport
from .types import Datum
from .types.tensor import TensorData, tensor_payload

TENSOR_KEY = "$tensor"
_RAW_CODEC = "raw"
_FORMAT = "raw-tensor"
_STREAM_CHUNK_BYTES = 8 * 1024 * 1024
_ZSTD_LEVEL = 3
_ZSTD_THREADS = 4
_TORCH_TO_NAME: dict[torch.dtype, str] = {
    torch.int64: "int64",
    torch.int32: "int32",
    torch.float32: "float32",
    torch.float64: "float64",
}


@dataclass(slots=True)
class TensorPack:
    """One temporary HTTP tensor pack owned by a prepared request."""

    path: Path
    size_bytes: int
    sha256: str
    codec: TensorCompression = "raw"
    decoded_size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.codec not in ("raw", "zstd"):
            raise ValueError("tensor pack codec must be 'raw' or 'zstd'")
        if self.decoded_size_bytes is None:
            if self.codec != "raw":
                raise ValueError("zstd tensor packs require decoded_size_bytes")
            self.decoded_size_bytes = self.size_bytes
        if self.codec == "raw" and self.decoded_size_bytes != self.size_bytes:
            raise ValueError("raw tensor pack decoded_size_bytes must equal size_bytes")

    def close(self) -> None:
        """Remove the client-local temporary pack."""

        self.path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class TensorPackMetadata:
    """Trusted bounds and digest for one operation result pack."""

    size_bytes: int
    sha256: str
    codec: TensorCompression
    decoded_size_bytes: int


@dataclass(slots=True)
class SerializedTrainingData:
    """Serialized datums plus an optional HTTP attachment."""

    data: list[dict[str, Any]]
    tensor_pack: TensorPack | None = None


@dataclass(slots=True)
class PreparedOperationBody:
    """An operation JSON body and its optional HTTP tensor pack."""

    body: dict[str, Any]
    tensor_pack: TensorPack | None = None

    def close(self) -> None:
        """Release request-local resources after submission."""

        if self.tensor_pack is not None:
            self.tensor_pack.close()


class _DigestingWriter:
    """Hash compressed bytes while forwarding them to a file."""

    def __init__(self, handle: BinaryIO, digest: Any) -> None:
        self._handle = handle
        self._digest = digest

    def write(self, data: bytes | bytearray | memoryview) -> int:
        self._digest.update(data)
        return self._handle.write(data)

    def flush(self) -> None:
        self._handle.flush()


class _PackWriter:
    """Write request tensors incrementally and produce offset descriptors."""

    def __init__(
        self,
        *,
        compression: TensorCompression,
    ) -> None:
        if compression not in ("raw", "zstd"):
            raise ValueError(f"unsupported tensor compression: {compression!r}")

        self._codec: TensorCompression = compression
        self._offset: int = 0
        self._sha256: Any = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            prefix="weaver-tensors-", suffix=".bin", delete=False
        ) as handle:
            self._path = Path(handle.name)
        self._resources: ExitStack = ExitStack()
        self._file: BinaryIO = self._resources.enter_context(self._path.open("wb"))
        if self._codec == "zstd":
            sink = _DigestingWriter(self._file, self._sha256)
            self._handle: BinaryIO = cast(
                BinaryIO,
                zstandard.ZstdCompressor(level=_ZSTD_LEVEL, threads=_ZSTD_THREADS).stream_writer(
                    cast(BinaryIO, sink), closefd=False
                ),
            )
        else:
            self._handle = self._file

    def put_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Append one tensor and return its transport-specific descriptor."""

        dtype_name = _TORCH_TO_NAME.get(tensor.dtype)
        if dtype_name is None:
            raise TypeError(f"unsupported dtype for tensor transport: {tensor.dtype}")
        array = tensor.detach().cpu().contiguous().numpy()
        raw = memoryview(array).cast("B")  # type: ignore[arg-type]

        offset = self._offset
        size_bytes = raw.nbytes
        self._handle.write(raw)
        if self._codec == "raw":
            self._sha256.update(raw)
        self._offset += size_bytes

        descriptor = {
            "format": _FORMAT,
            "codec": _RAW_CODEC,
            "dtype": dtype_name,
            "shape": list(array.shape),
            "offset": offset,
            "size_bytes": size_bytes,
        }
        return {TENSOR_KEY: descriptor}

    def commit(self) -> TensorPack:
        """Finish and return the HTTP attachment metadata."""

        if self._codec == "zstd":
            self._handle.close()
            self._file.flush()
        else:
            self._handle.flush()
        self._resources.close()
        return TensorPack(
            self._path,
            self._path.stat().st_size,
            self._sha256.hexdigest(),
            codec=self._codec,
            decoded_size_bytes=self._offset,
        )

    def abort(self) -> None:
        """Close and remove an unpublished pack."""

        try:
            if not self._handle.closed:
                self._handle.close()
        finally:
            self._resources.close()
        self._path.unlink(missing_ok=True)


def serialize_training_data(
    data: Sequence[Datum],
    *,
    loss_fn: str,
    transport: TensorTransport,
    compression: TensorCompression = "zstd",
) -> SerializedTrainingData:
    """Serialize datums, optimizing dense cross-entropy tensors only."""

    if loss_fn != "cross_entropy" or transport == "default":
        return SerializedTrainingData([datum.to_payload() for datum in data])

    writer = _PackWriter(compression=compression)
    try:
        payload: list[dict[str, Any]] = []
        for datum in data:
            chunks: list[dict[str, Any]] = []
            for chunk in datum.model_input.chunks:
                tokens = torch.as_tensor(chunk.tokens, dtype=torch.int64)
                chunks.append({"type": chunk.type, "tokens": writer.put_tensor(tokens)})

            loss_inputs: dict[str, Any] = {}
            for name, value in datum.loss_fn_inputs.items():
                if name in {"target_tokens", "weights"} and isinstance(
                    value, (torch.Tensor, TensorData)
                ):
                    tensor = value if isinstance(value, torch.Tensor) else value.to_tensor()
                    wire_dtype = torch.int64 if name == "target_tokens" else torch.float32
                    loss_inputs[name] = writer.put_tensor(tensor.to(dtype=wire_dtype))
                elif isinstance(value, TensorData):
                    loss_inputs[name] = value.to_dict()
                elif isinstance(value, torch.Tensor):
                    loss_inputs[name] = tensor_payload(value).to_dict()
                else:
                    loss_inputs[name] = value
            datum_payload: dict[str, Any] = {
                "model_input": {"chunks": chunks},
                "loss_fn_inputs": loss_inputs,
            }
            if datum.metadata:
                datum_payload["metadata"] = dict(datum.metadata)
            payload.append(datum_payload)
        tensor_pack = writer.commit()
    except BaseException:
        writer.abort()
        raise

    return SerializedTrainingData(payload, tensor_pack)


class MultipartLayout:
    """Byte-identical multipart layout shared by sync and async clients."""

    def __init__(self, request: Mapping[str, Any], tensor_pack: TensorPack) -> None:
        boundary = f"weaver-{secrets.token_hex(16)}"
        manifest = json.dumps(
            {
                "version": 1,
                "request": request,
                "tensor_pack": {
                    "size_bytes": tensor_pack.size_bytes,
                    "sha256": tensor_pack.sha256,
                    "codec": tensor_pack.codec,
                    "decoded_size_bytes": tensor_pack.decoded_size_bytes,
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        marker = boundary.encode("ascii")
        self.prefix = (
            b"--"
            + marker
            + b'\r\nContent-Disposition: form-data; name="manifest"\r\n'
            + b"Content-Type: application/json\r\n\r\n"
            + manifest
            + b"\r\n--"
            + marker
            + b'\r\nContent-Disposition: form-data; name="tensor_pack"; filename="pack.bin"\r\n'
            + b"Content-Type: application/octet-stream\r\n\r\n"
        )
        self.suffix = b"\r\n--" + marker + b"--\r\n"
        self.content_type = f"multipart/form-data; boundary={boundary}"
        self.content_length = len(self.prefix) + tensor_pack.size_bytes + len(self.suffix)
        self.pack_path = tensor_pack.path

    def sync_stream(self) -> "SyncMultipartStream":
        return SyncMultipartStream(self)

    def async_stream(self) -> "AsyncMultipartStream":
        return AsyncMultipartStream(self)


class SyncMultipartStream(httpx.SyncByteStream):
    """Replayable synchronous stream for one multipart tensor request."""

    def __init__(self, layout: MultipartLayout) -> None:
        self._layout = layout

    def __iter__(self) -> Iterator[bytes]:
        yield self._layout.prefix
        with self._layout.pack_path.open("rb") as handle:
            while chunk := handle.read(_STREAM_CHUNK_BYTES):
                yield chunk
        yield self._layout.suffix


class AsyncMultipartStream(httpx.AsyncByteStream):
    """Asynchronous stream that keeps pack file I/O off the event loop."""

    def __init__(self, layout: MultipartLayout) -> None:
        self._layout = layout

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        yield self._layout.prefix
        handle = await _open_binary_file(self._layout.pack_path)
        try:
            while chunk := await _await_blocking_io(handle.read, _STREAM_CHUNK_BYTES):
                yield chunk
        finally:
            await _await_blocking_io(handle.close)
        yield self._layout.suffix


def result_uses_http_tensor_pack(payload: Mapping[str, Any]) -> bool:
    """Return whether detailed logprobs reference an HTTP result pack."""

    result = payload.get("result")
    outputs = result.get("loss_fn_outputs") if isinstance(result, Mapping) else None
    if not isinstance(outputs, list):
        return False
    for output in outputs:
        if not isinstance(output, Mapping):
            continue
        logprobs = output.get("logprobs")
        if logprobs is None:
            logprobs = output.get("Logprobs")
        if isinstance(logprobs, Mapping) and TENSOR_KEY in logprobs:
            return True
    return False


def result_tensor_pack_metadata(payload: Mapping[str, Any]) -> TensorPackMetadata:
    """Parse the required size and digest for an HTTP result pack."""

    metadata = payload.get("tensor_pack")
    if not isinstance(metadata, Mapping):
        raise ValueError("HTTP tensor result is missing tensor_pack metadata")
    expected_size = _nonnegative_int(metadata.get("size_bytes"), "tensor_pack.size_bytes")
    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("tensor_pack.sha256 must be a SHA-256 hex digest")
    try:
        bytes.fromhex(expected_sha256)
    except ValueError as exc:
        raise ValueError("tensor_pack.sha256 must be a SHA-256 hex digest") from exc
    codec = metadata.get("codec", "raw")
    if codec not in ("raw", "zstd"):
        raise ValueError("tensor_pack.codec must be 'raw' or 'zstd'")
    raw_decoded_size = metadata.get("decoded_size_bytes")
    if raw_decoded_size is None:
        if codec != "raw":
            raise ValueError("zstd tensor packs require tensor_pack.decoded_size_bytes")
        decoded_size = expected_size
    else:
        decoded_size = _nonnegative_int(raw_decoded_size, "tensor_pack.decoded_size_bytes")
    if codec == "raw" and decoded_size != expected_size:
        raise ValueError("raw tensor pack decoded_size_bytes must equal size_bytes")
    return TensorPackMetadata(
        size_bytes=expected_size,
        sha256=expected_sha256.lower(),
        codec=cast(TensorCompression, codec),
        decoded_size_bytes=decoded_size,
    )


def decompress_zstd_tensor_pack(
    source: BinaryIO,
    destination: BinaryIO,
    decoded_size_bytes: int,
) -> None:
    """Decode exactly one bounded Zstandard frame into ``destination``."""

    expected_size = _nonnegative_int(decoded_size_bytes, "tensor_pack.decoded_size_bytes")
    source.seek(0)
    destination.seek(0)
    destination.truncate(0)
    decoded = 0
    reader = zstandard.ZstdDecompressor().stream_reader(
        source,
        read_across_frames=True,
        closefd=False,
    )
    try:
        while True:
            # Reading one byte past the declared size detects expansion and
            # concatenated frames without allowing an unbounded allocation.
            read_size = min(_STREAM_CHUNK_BYTES, expected_size - decoded + 1)
            try:
                output = reader.read(read_size)
            except zstandard.ZstdError as exc:
                raise ValueError(
                    "zstd tensor pack is invalid, truncated, or has trailing compressed data"
                ) from exc
            if not output:
                break
            if decoded == expected_size:
                raise ValueError("zstd tensor pack has trailing compressed data")
            if decoded + len(output) > expected_size:
                raise ValueError(f"decoded tensor pack exceeds expected {expected_size} bytes")
            destination.write(output)
            decoded += len(output)
    finally:
        reader.close()

    if decoded != expected_size:
        raise ValueError(
            "zstd tensor pack is truncated or size-mismatched: "
            f"decoded tensor pack has {decoded} bytes, expected {expected_size}"
        )
    destination.flush()
    destination.seek(0)


def materialize_http_tensor(reference: Mapping[str, Any], pack: BinaryIO) -> torch.Tensor:
    descriptor = reference.get(TENSOR_KEY)
    if not isinstance(descriptor, Mapping):
        raise ValueError("$tensor must contain an object descriptor")
    return _materialize_raw_tensor_slice(descriptor, pack, name="$tensor")


def _materialize_raw_tensor_slice(
    descriptor: Mapping[str, Any], source: BinaryIO, *, name: str
) -> torch.Tensor:
    dtypes = {
        "int64": torch.int64,
        "int32": torch.int32,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype_name = descriptor.get("dtype")
    dtype = dtypes.get(dtype_name) if isinstance(dtype_name, str) else None
    if descriptor.get("format") != _FORMAT or descriptor.get("codec", _RAW_CODEC) != _RAW_CODEC:
        raise ValueError(f"{name} must use raw-tensor/raw")
    if dtype is None:
        raise TypeError(f"unsupported {name} dtype: {dtype_name!r}")
    raw_shape = descriptor.get("shape")
    if not isinstance(raw_shape, list):
        raise ValueError(f"{name} shape must be a list")
    shape = [_nonnegative_int(value, f"{name}.shape") for value in raw_shape]
    size_bytes = _nonnegative_int(descriptor.get("size_bytes"), f"{name}.size_bytes")
    offset = _nonnegative_int(descriptor.get("offset", 0), f"{name}.offset")
    if math.prod(shape) * torch.empty((), dtype=dtype).element_size() != size_bytes:
        raise ValueError(f"{name} size_bytes does not match shape and dtype")
    if size_bytes == 0:
        return torch.empty(shape, dtype=dtype)

    data = bytearray(size_bytes)
    view = memoryview(data)
    source.seek(offset)
    read = 0
    while read < size_bytes:
        count = source.readinto(view[read:])  # type: ignore[attr-defined]
        if not count:
            break
        read += count
    if read != size_bytes:
        raise ValueError(f"{name} slice exceeds the tensor pack")
    return torch.frombuffer(data, dtype=dtype).reshape(shape)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


async def _await_blocking_io(function: Any, *args: Any) -> Any:
    """Finish one in-flight file operation before propagating cancellation."""

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException:
            pass
        raise


async def _open_binary_file(path: Path) -> BinaryIO:
    """Open a pack off-loop without leaking the handle on cancellation."""

    task = asyncio.create_task(asyncio.to_thread(path.open, "rb"))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_close_completed_file)
        raise


def _close_completed_file(task: "asyncio.Task[BinaryIO]") -> None:
    try:
        task.result().close()
    except BaseException:
        pass
