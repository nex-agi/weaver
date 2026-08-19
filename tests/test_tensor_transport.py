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

"""Tests for training tensor transport."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
from email.parser import BytesParser
from email.policy import default as email_policy
from types import MethodType, SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
import zstandard

from weaver._async_http import AsyncAPIClient
from weaver._http import APIClient
from weaver._payloads import (
    parse_logprob_tensors,
    prepare_forward_backward_operation,
)
from weaver.async_training_client import AsyncTrainingClient, _build_training_payload
from weaver.config import WeaverConfig
from weaver.operations import AsyncOperationHandle, OperationHandle
from weaver.tensor_transport import (
    TENSOR_KEY,
    MultipartLayout,
    PreparedOperationBody,
    TensorPack,
    decompress_zstd_tensor_pack,
    materialize_http_tensor,
    result_tensor_pack_metadata,
    result_uses_http_tensor_pack,
)
from weaver.training_client import TrainingClient
from weaver.types import Datum, ModelInput


def _datum() -> Datum:
    return Datum(
        model_input=ModelInput.from_ints([11, 12, 13]),
        loss_fn_inputs={
            "target_tokens": torch.tensor([12, 13, 14], dtype=torch.int64),
            "weights": torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
        },
        metadata={"source": "test"},
    )


def _payload(transport="default", loss_fn="cross_entropy"):
    prepared = _prepare(transport=transport, loss_fn=loss_fn)
    assert prepared.tensor_pack is None
    return prepared.body["payload"]


def _prepare(
    *, transport="default", compression="raw", loss_fn="cross_entropy", seq_id=7, data=None
):
    return prepare_forward_backward_operation(
        model_id="model-1",
        seq_id=seq_id,
        data=[_datum()] if data is None else data,
        loss_fn=loss_fn,
        loss_fn_config=None,
        request_metadata=None,
        tensor_transport=transport,
        tensor_compression=compression,
    )


def _multipart_parts(layout: MultipartLayout) -> dict[str, bytes]:
    body = b"".join(layout.sync_stream())
    assert len(body) == layout.content_length
    message = BytesParser(policy=email_policy).parsebytes(
        f"Content-Type: {layout.content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def test_default_transport_remains_inline():
    payload = _payload()
    datum = payload["forward_backward_input"]["data"][0]

    assert payload["tensor_transport"] == "default"
    assert "tensor_compression" not in payload
    assert datum["model_input"]["chunks"][0]["tokens"] == [11, 12, 13]
    assert datum["loss_fn_inputs"]["target_tokens"]["data"] == [12, 13, 14]
    assert datum["loss_fn_inputs"]["weights"]["data"] == [0.0, 1.0, 1.0]


def test_http_binary_defaults_to_zstd():
    prepared = prepare_forward_backward_operation(
        model_id="model-1",
        seq_id=7,
        data=[_datum()],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        request_metadata=None,
        tensor_transport="http-binary",
    )
    try:
        assert prepared.body["payload"]["tensor_compression"] == "zstd"
        assert prepared.tensor_pack is not None
        assert prepared.tensor_pack.codec == "zstd"
    finally:
        prepared.close()


def test_http_binary_normalizes_dense_input_dtypes():
    datum = Datum(
        model_input=ModelInput.from_ints([11, 12]),
        loss_fn_inputs={
            "target_tokens": torch.tensor([12, 13], dtype=torch.int32),
            "weights": torch.tensor([0.0, 1.0], dtype=torch.float16),
        },
    )

    prepared = _prepare(transport="http-binary", data=[datum])
    try:
        wire = prepared.body["payload"]["forward_backward_input"]["data"][0]
        assert wire["loss_fn_inputs"]["target_tokens"][TENSOR_KEY]["dtype"] == "int64"
        assert wire["loss_fn_inputs"]["weights"][TENSOR_KEY]["dtype"] == "float32"
    finally:
        prepared.close()


def test_http_binary_builds_one_path_free_pack():
    prepared = _prepare(transport="http-binary")
    assert prepared.tensor_pack is not None
    pack_path = prepared.tensor_pack.path
    try:
        payload = prepared.body["payload"]
        datum = payload["forward_backward_input"]["data"][0]
        refs = [
            datum["model_input"]["chunks"][0]["tokens"],
            datum["loss_fn_inputs"]["target_tokens"],
            datum["loss_fn_inputs"]["weights"],
        ]
        assert all(TENSOR_KEY in ref for ref in refs)
        assert all(
            "path" not in descriptor and "uri" not in descriptor and "storage" not in descriptor
            for ref in refs
            for descriptor in [ref[TENSOR_KEY]]
        )
        assert prepared.tensor_pack.size_bytes == 60
        assert prepared.tensor_pack.sha256 == hashlib.sha256(pack_path.read_bytes()).hexdigest()

        layout = MultipartLayout(prepared.body, prepared.tensor_pack)
        parts = _multipart_parts(layout)
        manifest = json.loads(parts["manifest"])
        assert set(parts) == {"manifest", "tensor_pack"}
        assert manifest == {
            "version": 1,
            "request": prepared.body,
            "tensor_pack": {
                "size_bytes": 60,
                "sha256": prepared.tensor_pack.sha256,
                "codec": "raw",
                "decoded_size_bytes": 60,
            },
        }
        assert parts["tensor_pack"] == pack_path.read_bytes()
    finally:
        prepared.close()
    assert not pack_path.exists()


def test_http_binary_zstd_streams_one_compressed_pack():
    prepared = _prepare(transport="http-binary", compression="zstd")
    assert prepared.tensor_pack is not None
    pack = prepared.tensor_pack
    try:
        wire = pack.path.read_bytes()
        decoded = zstandard.ZstdDecompressor().decompress(
            wire, max_output_size=pack.decoded_size_bytes
        )
        assert pack.codec == "zstd"
        assert pack.decoded_size_bytes == 60
        assert pack.size_bytes == len(wire)
        assert pack.sha256 == hashlib.sha256(wire).hexdigest()
        assert decoded == b"".join(
            [
                np.asarray([11, 12, 13], dtype="<i8").tobytes(),
                np.asarray([12, 13, 14], dtype="<i8").tobytes(),
                np.asarray([0.0, 1.0, 1.0], dtype="<f4").tobytes(),
            ]
        )

        parts = _multipart_parts(MultipartLayout(prepared.body, pack))
        manifest = json.loads(parts["manifest"])
        assert manifest["request"]["payload"]["tensor_compression"] == "zstd"
        assert manifest["tensor_pack"] == {
            "size_bytes": len(wire),
            "sha256": pack.sha256,
            "codec": "zstd",
            "decoded_size_bytes": 60,
        }
        assert parts["tensor_pack"] == wire
    finally:
        prepared.close()


def test_http_binary_is_cross_entropy_only():
    prepared = _prepare(transport="http-binary", compression="zstd", loss_fn="forward_logprob")

    assert prepared.tensor_pack is None
    payload = prepared.body["payload"]
    assert payload["tensor_transport"] == "http-binary"
    assert payload["tensor_compression"] == "zstd"
    assert payload["forward_backward_input"]["data"][0]["model_input"]["chunks"][0]["tokens"] == [
        11,
        12,
        13,
    ]


def test_http_binary_rejects_protocol_limit_before_upload(monkeypatch):
    monkeypatch.setattr("weaver.tensor_transport._MAX_TENSOR_PACK_BYTES", 1)

    with pytest.raises(ValueError, match="decoded_size_bytes.*exceeds maximum"):
        _prepare(transport="http-binary")


def test_sync_and_async_multipart_streams_are_identical():
    prepared = _prepare(transport="http-binary")
    assert prepared.tensor_pack is not None
    try:
        layout = MultipartLayout(prepared.body, prepared.tensor_pack)
        sync_body = b"".join(layout.sync_stream())

        async def collect() -> bytes:
            chunks = [chunk async for chunk in layout.async_stream()]
            return b"".join(chunks)

        assert asyncio.run(collect()) == sync_body
    finally:
        prepared.close()


def test_async_multipart_file_reads_yield_the_event_loop():
    prepared = _prepare(transport="http-binary")
    assert prepared.tensor_pack is not None

    async def collect() -> int:
        layout = MultipartLayout(prepared.body, prepared.tensor_pack)
        ticks = 0
        running = True

        async def ticker() -> None:
            nonlocal ticks
            while running:
                await asyncio.sleep(0)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            _ = b"".join([chunk async for chunk in layout.async_stream()])
        finally:
            running = False
            await task
        return ticks

    try:
        assert asyncio.run(collect()) > 0
    finally:
        prepared.close()


def test_parse_logprobs_materializes_http_tensor_pack():
    values = np.asarray([-0.2, -0.4], dtype="<f4")
    result = {
        "result": {
            "loss_fn_outputs": [
                {
                    "logprobs": {
                        TENSOR_KEY: {
                            "format": "raw-tensor",
                            "codec": "raw",
                            "dtype": "float32",
                            "shape": [2],
                            "offset": 0,
                            "size_bytes": values.nbytes,
                        }
                    }
                }
            ]
        },
        "tensor_pack": {
            "size_bytes": values.nbytes,
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        },
    }

    pack = io.BytesIO(values.tobytes())
    tensors = parse_logprob_tensors(result, [_datum()], tensor_pack=pack)

    assert tensors[0].tolist() == pytest.approx([-0.2, -0.4])
    assert tensors[0].requires_grad


def test_materialize_http_tensor_checks_pack_bounds_before_allocating():
    ref = {
        TENSOR_KEY: {
            "format": "raw-tensor",
            "codec": "raw",
            "dtype": "float32",
            "shape": [1_000_000_000],
            "offset": 0,
            "size_bytes": 4_000_000_000,
        }
    }

    with pytest.raises(ValueError, match="slice exceeds the tensor pack"):
        materialize_http_tensor(ref, io.BytesIO(b"tiny"))


def test_downloaded_tensor_pack_metadata_is_strict():
    payload = {"tensor_pack": {"size_bytes": 4, "sha256": "0" * 64}}

    metadata = result_tensor_pack_metadata(payload)
    assert metadata.codec == "raw"
    assert metadata.decoded_size_bytes == 4


def test_downloaded_tensor_pack_metadata_reports_protocol_limit():
    payload = {
        "tensor_pack": {
            "size_bytes": (8 << 30) + 1,
            "sha256": "0" * 64,
        }
    }

    with pytest.raises(ValueError, match="size_bytes.*exceeds maximum"):
        result_tensor_pack_metadata(payload)


def test_zstd_result_metadata_and_exact_decompression():
    decoded = b"tensor-pack" * 100
    wire = zstandard.ZstdCompressor().compress(decoded)
    payload = {
        "tensor_pack": {
            "size_bytes": len(wire),
            "sha256": hashlib.sha256(wire).hexdigest(),
            "codec": "zstd",
            "decoded_size_bytes": len(decoded),
        }
    }

    metadata = result_tensor_pack_metadata(payload)
    destination = io.BytesIO()
    decompress_zstd_tensor_pack(io.BytesIO(wire), destination, metadata.decoded_size_bytes)

    assert metadata.codec == "zstd"
    assert destination.read() == decoded


def test_zstd_decompression_accepts_one_streaming_frame():
    decoded = b"streaming tensor pack" * 100
    wire_file = io.BytesIO()
    compressor = zstandard.ZstdCompressor(write_checksum=True).stream_writer(
        wire_file, closefd=False
    )
    compressor.write(decoded)
    compressor.close()
    destination = io.BytesIO()

    decompress_zstd_tensor_pack(wire_file, destination, len(decoded))

    assert destination.read() == decoded


def test_zstd_decompression_rejects_concatenated_frames():
    compressor = zstandard.ZstdCompressor()
    wire = compressor.compress(b"abc") + compressor.compress(b"def")

    with pytest.raises(ValueError, match="exactly one frame"):
        decompress_zstd_tensor_pack(io.BytesIO(wire), io.BytesIO(), 6)


@pytest.mark.parametrize("codec", ["raw", "zstd"])
def test_operation_results_transparently_materialize_and_cache_binary_pack(codec):
    values = np.asarray([-0.2, -0.4], dtype="<f4")
    binary = _http_logprob_result(values, codec=codec)
    inline = _inline_logprob_result(values)
    client = _SyncTensorPackClient(values.tobytes(), codec)

    binary_handle = OperationHandle(
        client=client,
        operation_id="op-1",
        _cached={"status": "done", "response": binary},
    )
    inline_handle = OperationHandle(
        client=SimpleNamespace(),
        operation_id="op-2",
        _cached={"status": "done", "response": inline},
    )

    waited = binary_handle.wait()
    assert waited["response"] == inline
    assert binary_handle.response == inline
    assert binary_handle.result() == inline
    assert "tensor_pack" not in binary_handle.result()
    assert client.calls == 1
    assert inline_handle.result() is inline


def test_result_tensor_pack_detection_does_not_scan_inline_outputs():
    class NoTraversal(dict):
        def values(self):
            raise AssertionError("inline output was traversed")

    assert not result_uses_http_tensor_pack(NoTraversal(result={"values": [1, 2, 3]}))
    assert result_uses_http_tensor_pack({"tensor_pack": {"size_bytes": 1}})


@pytest.mark.parametrize("codec", ["raw", "zstd"])
def test_async_operation_result_transparently_materializes_and_caches_pack(codec):
    async def exercise() -> None:
        values = np.asarray([-0.2, -0.4], dtype="<f4")
        result = _http_logprob_result(values, codec=codec)
        client = _AsyncTensorPackClient(values.tobytes(), codec)
        handle = AsyncOperationHandle(
            client=client,
            operation_id="op-1",
            _cached={"status": "done", "response": result},
        )

        waited = await handle.wait()
        expected = _inline_logprob_result(values)
        assert waited["response"] == expected
        assert handle.response == expected
        assert await handle.result() == expected
        assert "tensor_pack" not in await handle.result()
        assert client.calls == 1

    asyncio.run(exercise())


def test_operation_wait_all_materializes_binary_results():
    values = np.asarray([-0.2, -0.4], dtype="<f4")
    clients = [_SyncTensorPackClient(values.tobytes(), "raw") for _ in range(2)]
    handles = [
        OperationHandle(
            client=client,
            operation_id="op-1",
            _cached={"status": "done", "response": _http_logprob_result(values)},
        )
        for client in clients
    ]

    assert OperationHandle.wait_all(handles) == [_inline_logprob_result(values)] * 2
    assert [client.calls for client in clients] == [1, 1]


def test_async_operation_wait_all_materializes_binary_results():
    async def exercise() -> None:
        values = np.asarray([-0.2, -0.4], dtype="<f4")
        clients = [_AsyncTensorPackClient(values.tobytes(), "raw") for _ in range(2)]
        handles = [
            AsyncOperationHandle(
                client=client,
                operation_id="op-1",
                _cached={"status": "done", "response": _http_logprob_result(values)},
            )
            for client in clients
        ]

        assert await AsyncOperationHandle.wait_all(handles) == [_inline_logprob_result(values)] * 2
        assert [client.calls for client in clients] == [1, 1]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("wire", "decoded_size", "error"),
    [
        (zstandard.ZstdCompressor().compress(b"0123456789"), 9, "exceeds expected"),
        (zstandard.ZstdCompressor().compress(b"0123456789"), 11, "has 10 bytes"),
        (zstandard.ZstdCompressor().compress(b"0123456789")[:-1], 10, "truncated"),
        (
            zstandard.ZstdCompressor().compress(b"0123456789") + b"trailing",
            10,
            "exactly one frame",
        ),
    ],
)
def test_zstd_decompression_rejects_invalid_bounds(wire, decoded_size, error):
    with pytest.raises(ValueError, match=error):
        decompress_zstd_tensor_pack(io.BytesIO(wire), io.BytesIO(), decoded_size)


def test_sync_http_client_submits_multipart_without_retry():
    prepared = _prepare(transport="http-binary")
    assert prepared.tensor_pack is not None
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = request.read()
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        assert int(request.headers["content-length"]) == len(body)
        raise httpx.WriteError("ambiguous failure", request=request)

    client = APIClient(WeaverConfig(base_url="https://example.test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(httpx.WriteError, match="ambiguous failure"):
            client.post_tensor_multipart(
                "/api/v1/models/model-1/forward-backward-passes",
                request=prepared.body,
                tensor_pack=prepared.tensor_pack,
            )
        assert calls == 1
    finally:
        client.close()
        prepared.close()


def test_sync_http_client_downloads_result_pack_once():
    payload = b"tensor-pack"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/api/v1/operations/op-1/tensor-pack"
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(payload))},
            stream=httpx.ByteStream(payload),
        )

    client = APIClient(WeaverConfig(base_url="https://example.test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )
    destination = io.BytesIO()
    try:
        client.download_tensor_pack(
            "op-1",
            destination,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    finally:
        client.close()

    assert calls == 1
    assert destination.read() == payload


def test_sync_http_client_rejects_protocol_limit_before_request():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = APIClient(WeaverConfig(base_url="https://example.test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ValueError, match="size_bytes.*exceeds maximum"):
            client.download_tensor_pack(
                "op-1",
                io.BytesIO(),
                size_bytes=(8 << 30) + 1,
                sha256="0" * 64,
            )
    finally:
        client.close()

    assert calls == 0


def test_sync_http_client_verifies_then_decompresses_zstd_result_pack():
    decoded = b"tensor-pack" * 100
    wire = zstandard.ZstdCompressor().compress(decoded)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(wire)),
                "X-Weaver-Tensor-Codec": "zstd",
                "X-Weaver-Tensor-Decoded-Size": str(len(decoded)),
            },
            stream=httpx.ByteStream(wire),
        )

    client = APIClient(WeaverConfig(base_url="https://example.test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )
    destination = io.BytesIO()
    try:
        client.download_tensor_pack(
            "op-1",
            destination,
            size_bytes=len(wire),
            sha256=hashlib.sha256(wire).hexdigest(),
            codec="zstd",
            decoded_size_bytes=len(decoded),
        )
    finally:
        client.close()

    assert destination.read() == decoded


@pytest.mark.parametrize(
    ("headers", "payload", "error"),
    [
        ({}, b"tensor-pack", "missing Content-Length"),
        ({"Content-Length": "1"}, b"tensor-pack", "Content-Length is 1"),
        ({"Content-Length": "10"}, b"tensor-pack", "exceeds expected 10 bytes"),
    ],
)
def test_sync_http_client_bounds_result_pack(headers, payload, error):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=httpx.ByteStream(payload))

    client = APIClient(WeaverConfig(base_url="https://example.test"))
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )
    destination = io.BytesIO()
    try:
        with pytest.raises(ValueError, match=error):
            client.download_tensor_pack(
                "op-1", destination, size_bytes=10, sha256=hashlib.sha256(payload).hexdigest()
            )
    finally:
        client.close()


def test_async_http_client_streams_multipart_and_result_pack():
    async def exercise() -> None:
        prepared = _prepare(transport="http-binary")
        assert prepared.tensor_pack is not None
        uploads = 0
        downloads = 0
        result_pack = b"result-pack" * 100
        result_wire = zstandard.ZstdCompressor().compress(result_pack)

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal uploads, downloads
            if request.method == "POST":
                uploads += 1
                body = await request.aread()
                assert int(request.headers["content-length"]) == len(body)
                return httpx.Response(202, json={"id": "op-1", "status": "pending"})
            downloads += 1
            assert request.url.path == "/api/v1/operations/op-1/tensor-pack"
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(len(result_wire)),
                    "X-Weaver-Tensor-Codec": "zstd",
                    "X-Weaver-Tensor-Decoded-Size": str(len(result_pack)),
                },
                stream=httpx.ByteStream(result_wire),
            )

        client = AsyncAPIClient(WeaverConfig(base_url="https://example.test"))
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        destination = io.BytesIO()
        try:
            response = await client.post_tensor_multipart(
                "/api/v1/models/model-1/forward-backward-passes",
                request=prepared.body,
                tensor_pack=prepared.tensor_pack,
            )
            await client.download_tensor_pack(
                "op-1",
                destination,
                size_bytes=len(result_wire),
                sha256=hashlib.sha256(result_wire).hexdigest(),
                codec="zstd",
                decoded_size_bytes=len(result_pack),
            )
        finally:
            await client.aclose()
            prepared.close()

        assert response["id"] == "op-1"
        assert uploads == 1
        assert downloads == 1
        assert destination.read() == result_pack

    asyncio.run(exercise())


def _inline_logprob_result(values: np.ndarray) -> dict:
    return {
        "result": {
            "loss_fn_outputs": [
                {
                    "logprobs": {
                        "data": values.tolist(),
                        "dtype": "float32",
                        "shape": list(values.shape),
                    }
                }
            ]
        }
    }


def _http_logprob_result(values: np.ndarray, *, codec: str = "raw") -> dict:
    decoded = values.tobytes()
    wire = zstandard.ZstdCompressor().compress(decoded) if codec == "zstd" else decoded
    return {
        "result": {
            "loss_fn_outputs": [
                {
                    "logprobs": {
                        TENSOR_KEY: {
                            "format": "raw-tensor",
                            "codec": "raw",
                            "dtype": "float32",
                            "shape": list(values.shape),
                            "offset": 0,
                            "size_bytes": values.nbytes,
                        }
                    }
                }
            ]
        },
        "tensor_pack": {
            "size_bytes": len(wire),
            "sha256": hashlib.sha256(wire).hexdigest(),
            "codec": codec,
            "decoded_size_bytes": len(decoded),
        },
    }


class _SyncTensorPackClient:
    def __init__(self, decoded: bytes, codec: str) -> None:
        self.decoded = decoded
        self.codec = codec
        self.wire = zstandard.ZstdCompressor().compress(decoded) if codec == "zstd" else decoded
        self.calls = 0

    def download_tensor_pack(self, operation_id, destination, **metadata):
        assert operation_id == "op-1"
        assert metadata == {
            "size_bytes": len(self.wire),
            "sha256": hashlib.sha256(self.wire).hexdigest(),
            "codec": self.codec,
            "decoded_size_bytes": len(self.decoded),
        }
        self.calls += 1
        destination.write(self.decoded)
        destination.seek(0)


class _AsyncTensorPackClient(_SyncTensorPackClient):
    async def download_tensor_pack(self, operation_id, destination, **metadata):
        super().download_tensor_pack(operation_id, destination, **metadata)


def _nested_http_tensor_result() -> tuple[dict, bytes]:
    values = np.asarray([0.25, 0.5], dtype="<f4")
    result = _http_logprob_result(values)
    reference = result["result"]["loss_fn_outputs"][0].pop("logprobs")
    result["result"]["loss_fn_outputs"][0]["custom"] = {"nested": [reference, reference]}
    return result, values.tobytes()


def test_sync_operation_materializes_every_nested_result_tensor():
    result, pack = _nested_http_tensor_result()
    download_client = _SyncTensorPackClient(pack, "raw")
    handle = OperationHandle(
        client=download_client,
        operation_id="op-1",
        _cached={"status": "done", "response": result},
    )

    materialized = handle.result()

    output = materialized["result"]["loss_fn_outputs"][0]
    expected = {"data": [0.25, 0.5], "dtype": "float32", "shape": [2]}
    assert output["custom"]["nested"] == [expected, expected]
    assert "tensor_pack" not in materialized
    assert TENSOR_KEY in result["result"]["loss_fn_outputs"][0]["custom"]["nested"][0]


def test_async_operation_materializes_every_nested_result_tensor():
    async def exercise() -> None:
        result, pack = _nested_http_tensor_result()
        handle = AsyncOperationHandle(
            client=_AsyncTensorPackClient(pack, "raw"),
            operation_id="op-1",
            _cached={"status": "done", "response": result},
        )

        materialized = await handle.result()

        output = materialized["result"]["loss_fn_outputs"][0]
        expected = {"data": [0.25, 0.5], "dtype": "float32", "shape": [2]}
        assert output["custom"]["nested"] == [expected, expected]
        assert "tensor_pack" not in materialized

    asyncio.run(exercise())


def test_sync_custom_loss_downloads_one_operation_pack():
    values = np.asarray([-0.2, -0.4, -0.6], dtype="<f4")
    result = _http_logprob_result(values)

    class DownloadClient:
        calls = 0

        def download_tensor_pack(
            self,
            operation_id,
            destination,
            *,
            size_bytes,
            sha256,
            codec,
            decoded_size_bytes,
        ):
            assert operation_id == "op-1"
            assert size_bytes == values.nbytes
            assert sha256 == hashlib.sha256(values.tobytes()).hexdigest()
            assert codec == "raw"
            assert decoded_size_bytes == values.nbytes
            self.calls += 1
            destination.write(values.tobytes())
            destination.seek(0)

    download_client = DownloadClient()
    handle = OperationHandle(
        client=download_client,
        operation_id="op-1",
        _cached={"status": "done", "response": result},
    )
    client = TrainingClient(
        service=SimpleNamespace(http=download_client),
        model_id="model-1",
        base_model="base",
        session_id="session-1",
    )

    def fake_forward(self, *args, **kwargs):
        assert kwargs["wait"] is False
        return handle

    def fake_forward_backward(self, *args, **kwargs):
        return {}

    client.forward = MethodType(fake_forward, client)
    client.forward_backward = MethodType(fake_forward_backward, client)

    output = client.forward_backward_custom(
        [_datum()], lambda _data, logprobs: (logprobs[0].sum(), {"ok": True})
    )

    assert download_client.calls == 1
    assert output["metrics"] == {"ok": True}


def test_async_custom_loss_downloads_one_operation_pack():
    async def exercise() -> None:
        values = np.asarray([-0.2, -0.4, -0.6], dtype="<f4")
        result = _http_logprob_result(values)

        class DownloadClient:
            calls = 0

            async def download_tensor_pack(
                self,
                operation_id,
                destination,
                *,
                size_bytes,
                sha256,
                codec,
                decoded_size_bytes,
            ):
                assert operation_id == "op-1"
                assert size_bytes == values.nbytes
                assert sha256 == hashlib.sha256(values.tobytes()).hexdigest()
                assert codec == "raw"
                assert decoded_size_bytes == values.nbytes
                self.calls += 1
                destination.write(values.tobytes())
                destination.seek(0)

        download_client = DownloadClient()

        client = AsyncTrainingClient(
            service=SimpleNamespace(http=download_client),
            model_id="model-1",
            base_model="base",
            session_id="session-1",
        )
        handle = AsyncOperationHandle(
            client=download_client,
            operation_id="op-1",
            _cached={"status": "done", "response": result},
        )

        async def fake_forward(self, *args, **kwargs):
            assert kwargs["wait"] is False
            return handle

        async def fake_forward_backward(self, *args, **kwargs):
            return {}

        client.forward = MethodType(fake_forward, client)
        client.forward_backward = MethodType(fake_forward_backward, client)
        output = await client.forward_backward_custom(
            [_datum()], lambda _data, logprobs: (logprobs[0].sum(), {"ok": True})
        )

        assert download_client.calls == 1
        assert output["metrics"] == {"ok": True}

    asyncio.run(exercise())


def test_cancelled_async_payload_build_removes_eventual_temp_pack(tmp_path):
    started = threading.Event()
    release = threading.Event()
    pack_path = tmp_path / "cancelled-pack.bin"

    def builder(**_kwargs):
        pack_path.write_bytes(b"pack")
        started.set()
        release.wait(timeout=5)
        return PreparedOperationBody(
            {"payload": {}},
            TensorPack(pack_path, 4, hashlib.sha256(b"pack").hexdigest()),
        )

    async def exercise() -> None:
        task = asyncio.create_task(
            _build_training_payload(
                builder,
                loss_fn="cross_entropy",
                tensor_transport="http-binary",
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):
            if not pack_path.exists():
                break
            await asyncio.sleep(0.01)

    asyncio.run(exercise())
    assert not pack_path.exists()
