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

"""Tests for HF weights download: descriptor plumbing and download_weights."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import PurePosixPath
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from weaver import _async_http, _http, _safeio, async_service_client, service_client
from weaver._artifacts import (
    descriptor_files,
    parse_download_target,
    select_artifact_payload,
)
from weaver._http import DownloadURLExpiredError, stream_download_to_file
from weaver.async_service_client import AsyncServiceClient
from weaver.cli import cli
from weaver.service_client import ServiceClient
from weaver.types.weights_artifact import WeightsArtifact

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FILES: Dict[str, bytes] = {
    "adapter_config.json": b'{"r": 32}',
    "adapter_model.safetensors": b"\x00" * 4096 + b"tensor-bytes",
    "sub/tokenizer.json": b'{"vocab": {}}',
}

# Single-file artifact: lets resume/retry tests assert on one request stream
# without interleaving from the other shards.
SINGLE_FILE: Dict[str, bytes] = {"shard.bin": bytes(range(256)) * 8}

CHECKPOINT_URI = "weaver://mdl-123/checkpoints/step-5"
ARTIFACT_UUID_2 = "bbbb2222-cc33-4d44-8e55-ffff6666aaaa"
CHECKPOINT_UUID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
ARTIFACT_UUID = "aaaa1111-bb22-4c33-8d44-eeee5555ffff"
ADAPTER_ARTIFACT = {
    "id": ARTIFACT_UUID,
    "checkpoint_id": CHECKPOINT_UUID,
    "model_id": "mdl-123",
    "kind": "hf_adapter",
    "status": "completed",
    "uri": f"{CHECKPOINT_URI}/artifacts/hf_adapter",
}
MODEL_ARTIFACT = {
    "id": ARTIFACT_UUID_2,
    "checkpoint_id": CHECKPOINT_UUID,
    "model_id": "mdl-123",
    "kind": "hf_model",
    "status": "completed",
    "uri": f"{CHECKPOINT_URI}/artifacts/hf_model",
}


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _descriptor(
    files: Dict[str, bytes],
    *,
    sig: str = "ok",
    sha_overrides: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    overrides = sha_overrides or {}
    return {
        "artifact_id": ARTIFACT_UUID,
        "kind": "hf_adapter",
        "total_bytes": sum(len(content) for content in files.values()),
        "files": [
            {
                "name": name,
                "size": len(content),
                "sha256": overrides.get(name, _sha(content)),
                "url": f"https://tos.example.com/files/{name}?sig={sig}",
                "url_expires_at": "2026-08-12T00:15:00Z",
            }
            for name, content in files.items()
        ],
    }


def _presigned_handler(files: Dict[str, bytes], calls: list | None = None):
    """MockTransport handler emulating presigned object-storage URLs."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        assert "X-WEAVER-API-KEY" not in request.headers  # bare client only
        if request.url.params.get("sig") == "expired":
            return httpx.Response(403, text="<Error>expired</Error>")
        name = request.url.path[len("/files/") :]
        content = files[name]
        range_header = request.headers.get("Range")
        if range_header:
            start = int(range_header.split("=", 1)[1].split("-", 1)[0])
            if start >= len(content):
                return httpx.Response(416)
            return httpx.Response(206, content=content[start:])
        return httpx.Response(200, content=content)

    return handler


def _make_sync_client(api_get_side_effect) -> ServiceClient:
    client = ServiceClient(base_url="https://test.example.com", api_key="sk-test")
    client._http = MagicMock()
    client._http.get.side_effect = api_get_side_effect
    return client


def _make_async_client(api_get_side_effect) -> AsyncServiceClient:
    client = AsyncServiceClient(base_url="https://test.example.com", api_key="sk-test")
    client._http = MagicMock()
    client._http.get = AsyncMock(side_effect=api_get_side_effect)
    return client


def _api_routes(descriptor: Dict[str, Any], artifacts: list | None = None):
    """Standard API GET routing for resolution + descriptor fetch."""

    routes = {
        "/api/v1/models/mdl-123/checkpoints": {
            "items": [{"id": CHECKPOINT_UUID, "path": CHECKPOINT_URI, "type": "weight"}]
        },
        f"/api/v1/checkpoints/{CHECKPOINT_UUID}/artifacts": {
            "items": artifacts if artifacts is not None else [dict(ADAPTER_ARTIFACT)]
        },
        f"/api/v1/artifacts/{ARTIFACT_UUID}/download": descriptor,
        f"/api/v1/artifacts/{ARTIFACT_UUID_2}/download": descriptor,
    }

    def side_effect(path, **_kwargs):
        return routes[path]

    return side_effect


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestParseDownloadTarget:
    def test_artifact_uri(self):
        parsed = parse_download_target(f"{CHECKPOINT_URI}/artifacts/hf_adapter")
        assert parsed.model_id == "mdl-123"
        assert parsed.checkpoint_path == CHECKPOINT_URI
        assert parsed.kind == "hf_adapter"
        assert parsed.artifact_id is None

    def test_checkpoint_uri(self):
        parsed = parse_download_target(CHECKPOINT_URI)
        assert parsed.checkpoint_path == CHECKPOINT_URI
        assert parsed.kind is None

    def test_raw_artifact_id(self):
        parsed = parse_download_target(ARTIFACT_UUID)
        assert parsed.artifact_id == ARTIFACT_UUID

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown artifact kind"):
            parse_download_target(f"{CHECKPOINT_URI}/artifacts/onnx")

    def test_malformed_weaver_uri_rejected(self):
        with pytest.raises(ValueError, match="Unrecognized weaver URI"):
            parse_download_target("weaver://mdl-123/weights/step-5")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_download_target("  ")


class TestSelectArtifactPayload:
    def test_single_completed(self):
        selected = select_artifact_payload([dict(ADAPTER_ARTIFACT)], None, context="t")
        assert selected["id"] == ARTIFACT_UUID

    def test_kind_filter(self):
        items = [dict(ADAPTER_ARTIFACT), dict(MODEL_ARTIFACT)]
        selected = select_artifact_payload(items, "hf_model", context="t")
        assert selected["id"] == ARTIFACT_UUID_2

    def test_no_completed_tells_user_to_export(self):
        items = [{**ADAPTER_ARTIFACT, "status": "pending"}]
        with pytest.raises(RuntimeError, match="export_weights"):
            select_artifact_payload(items, None, context="t")

    def test_ambiguous_kinds_require_kind(self):
        items = [dict(ADAPTER_ARTIFACT), dict(MODEL_ARTIFACT)]
        with pytest.raises(ValueError, match="disambiguate"):
            select_artifact_payload(items, None, context="t")


class TestDescriptorFiles:
    def test_normalizes_entries(self):
        files = descriptor_files(_descriptor(FILES))
        assert [entry.name for entry in files] == list(FILES)
        assert files[0].size == len(FILES["adapter_config.json"])
        assert files[0].sha256 == _sha(FILES["adapter_config.json"])
        assert files[0].url.startswith("https://tos.example.com/")

    def test_rejects_traversal_names(self):
        bad = _descriptor({"../evil.bin": b"x"})
        with pytest.raises(ValueError, match="unsafe file name"):
            descriptor_files(bad)

    def test_rejects_absolute_names(self):
        bad = _descriptor({"/etc/passwd": b"x"})
        with pytest.raises(ValueError, match="unsafe file name"):
            descriptor_files(bad)

    @pytest.mark.parametrize("name", [".", "./"])
    def test_rejects_names_that_normalize_to_nothing(self, name):
        # Regression: PurePosixPath(".").parts is empty, so "." used to pass
        # validation; the download path then aliased dest_dir itself and the
        # .part landed OUTSIDE the requested directory as its sibling.
        bad = _descriptor({name: b"x"})
        with pytest.raises(ValueError, match="unsafe file name"):
            descriptor_files(bad)

    @pytest.mark.parametrize("name", ["a//b", "a/./b", "foo/", "./foo"])
    def test_rejects_non_canonical_names(self, name):
        bad = _descriptor({name: b"x"})
        with pytest.raises(ValueError, match="non-canonical file name"):
            descriptor_files(bad)

    @pytest.mark.parametrize(
        "name",
        [
            "..\\owned.bin",
            "a\\..\\owned.bin",
            "C:\\Users\\victim\\owned.bin",
            "\\Windows\\owned.bin",
            "C:foo",
        ],
    )
    def test_rejects_windows_traversal_names(self, name):
        # PurePosixPath treats these as single ordinary names, but the final
        # dest_dir / name write uses HOST semantics — on Windows they escape
        # or replace dest_dir.
        with pytest.raises(ValueError, match="unsafe file name"):
            descriptor_files(_descriptor({name: b"x"}))

    @pytest.mark.parametrize(
        "name",
        ["CON", "NUL", "con.txt", "a/PRN/b", "COM1", "foo.", "foo ", "a/b:stream", "d/trail. /x"],
    )
    def test_rejects_windows_special_components(self, name):
        # Device names (any extension), trailing dot/space aliases and ADS
        # syntax target something other than the manifest path on Windows.
        with pytest.raises(ValueError, match="unsafe file name"):
            descriptor_files(_descriptor({name: b"x"}))

    def test_rejects_duplicate_names(self):
        descriptor = _descriptor(FILES)
        descriptor["files"].append(dict(descriptor["files"][0]))
        with pytest.raises(ValueError, match="duplicate file name"):
            descriptor_files(descriptor)

    def test_rejects_empty_descriptor(self):
        with pytest.raises(ValueError, match="no files"):
            descriptor_files({"files": []})


# ---------------------------------------------------------------------------
# stream_download_to_file (sync + async low-level helper)
# ---------------------------------------------------------------------------


class TestStreamDownload:
    """The streamer writes into a sink its caller opened (and anchored)."""

    def test_full_download(self, tmp_path):
        content = FILES["adapter_model.safetensors"]
        transport = httpx.MockTransport(_presigned_handler(FILES))
        dest = tmp_path / "shard.bin"
        with httpx.Client(transport=transport) as client:
            with open(dest, "wb") as sink:
                written = stream_download_to_file(
                    "https://tos.example.com/files/adapter_model.safetensors?sig=ok",
                    sink,
                    client=client,
                )
        assert written == len(content)
        assert dest.read_bytes() == content

    def test_resume_appends_via_range(self, tmp_path):
        content = FILES["adapter_model.safetensors"]
        transport = httpx.MockTransport(_presigned_handler(FILES))
        dest = tmp_path / "shard.bin"
        dest.write_bytes(content[:100])
        with httpx.Client(transport=transport) as client:
            with open(dest, "ab") as sink:
                written = stream_download_to_file(
                    "https://tos.example.com/files/adapter_model.safetensors?sig=ok",
                    sink,
                    client=client,
                    resume_from=100,
                )
        assert written == len(content)
        assert dest.read_bytes() == content

    def test_restart_when_server_ignores_range(self, tmp_path):
        content = b"full-content"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=content)  # no Range support

        dest = tmp_path / "f.bin"
        dest.write_bytes(b"stale-partial")
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            # Opened for append because the caller intended to resume; the 200
            # forces the streamer to drop the stale bytes rather than append.
            with open(dest, "ab") as sink:
                written = stream_download_to_file(
                    "https://tos.example.com/f", sink, client=client, resume_from=13
                )
        assert written == len(content)
        assert dest.read_bytes() == content

    def test_expired_url_raises_dedicated_error(self, tmp_path):
        transport = httpx.MockTransport(_presigned_handler(FILES))
        with httpx.Client(transport=transport) as client:
            with open(tmp_path / "f.bin", "wb") as sink:
                with pytest.raises(DownloadURLExpiredError):
                    stream_download_to_file(
                        "https://tos.example.com/files/adapter_config.json?sig=expired",
                        sink,
                        client=client,
                    )

    def test_async_twin_full_download(self, tmp_path):
        content = FILES["adapter_model.safetensors"]
        transport = httpx.MockTransport(_presigned_handler(FILES))
        dest = tmp_path / "shard.bin"

        async def run():
            async with httpx.AsyncClient(transport=transport) as client:
                with open(dest, "wb") as sink:
                    return await _async_http.async_stream_download_to_file(
                        "https://tos.example.com/files/adapter_model.safetensors?sig=ok",
                        sink,
                        client=client,
                    )

        written = asyncio.run(run())
        assert written == len(content)
        assert dest.read_bytes() == content

    def test_async_twin_restart_when_server_ignores_range(self, tmp_path):
        content = b"full-content"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=content)  # no Range support

        dest = tmp_path / "f.bin"
        dest.write_bytes(b"stale-partial")

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with open(dest, "ab") as sink:
                    return await _async_http.async_stream_download_to_file(
                        "https://tos.example.com/f", sink, client=client, resume_from=13
                    )

        written = asyncio.run(run())
        assert written == len(content)
        assert dest.read_bytes() == content


# ---------------------------------------------------------------------------
# ServiceClient.download_weights (sync)
# ---------------------------------------------------------------------------


class TestDownloadWeights:
    def _patch_transport(self, monkeypatch, handler):
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def test_parallel_download_by_artifact_id(self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_transport(monkeypatch, _presigned_handler(FILES, calls))
        client = _make_sync_client(_api_routes(_descriptor(FILES)))

        dest = client.download_weights(ARTIFACT_UUID, tmp_path / "out")

        assert dest == tmp_path / "out"
        for name, content in FILES.items():
            assert (dest / name).read_bytes() == content
        assert not list(dest.rglob("*.part"))
        assert len(calls) == len(FILES)
        # Only the descriptor endpoint is hit for a raw artifact id.
        client.http.get.assert_called_once_with(f"/api/v1/artifacts/{ARTIFACT_UUID}/download")

    def test_checkpoint_uri_resolution_flow(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        client = _make_sync_client(_api_routes(_descriptor(FILES)))

        client.download_weights(CHECKPOINT_URI, tmp_path / "out")

        paths = [call.args[0] for call in client.http.get.call_args_list]
        assert paths == [
            "/api/v1/models/mdl-123/checkpoints",
            f"/api/v1/checkpoints/{CHECKPOINT_UUID}/artifacts",
            f"/api/v1/artifacts/{ARTIFACT_UUID}/download",
        ]

    def test_artifact_uri_selects_kind(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        artifacts = [dict(ADAPTER_ARTIFACT), dict(MODEL_ARTIFACT)]
        client = _make_sync_client(_api_routes(_descriptor(FILES), artifacts=artifacts))

        client.download_weights(f"{CHECKPOINT_URI}/artifacts/hf_model", tmp_path / "out")

        paths = [call.args[0] for call in client.http.get.call_args_list]
        assert paths[-1] == f"/api/v1/artifacts/{ARTIFACT_UUID_2}/download"

    def test_weights_artifact_object_target(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        client = _make_sync_client(_api_routes(_descriptor(FILES)))

        artifact = WeightsArtifact.from_payload(ADAPTER_ARTIFACT)
        client.download_weights(artifact, tmp_path / "out")

        client.http.get.assert_called_once_with(f"/api/v1/artifacts/{ARTIFACT_UUID}/download")

    def test_no_completed_artifact_never_exports_implicitly(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        pending = [{**ADAPTER_ARTIFACT, "status": "pending"}]
        client = _make_sync_client(_api_routes(_descriptor(FILES), artifacts=pending))

        with pytest.raises(RuntimeError, match="export_weights"):
            client.download_weights(CHECKPOINT_URI, tmp_path / "out")
        client.http.post.assert_not_called()

    def test_ambiguous_artifacts_require_kind(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        artifacts = [dict(ADAPTER_ARTIFACT), dict(MODEL_ARTIFACT)]
        client = _make_sync_client(_api_routes(_descriptor(FILES), artifacts=artifacts))

        with pytest.raises(ValueError, match="disambiguate"):
            client.download_weights(CHECKPOINT_URI, tmp_path / "out")

    def test_kind_conflicting_with_artifact_uri_rejected(self, tmp_path, monkeypatch):
        client = _make_sync_client(_api_routes(_descriptor(FILES)))
        with pytest.raises(ValueError, match="conflicts"):
            client.download_weights(
                f"{CHECKPOINT_URI}/artifacts/hf_adapter", tmp_path / "out", kind="hf_model"
            )

    def test_invalid_kind_rejected(self, tmp_path):
        client = _make_sync_client(_api_routes(_descriptor(FILES)))
        with pytest.raises(ValueError, match="kind must be one of"):
            client.download_weights(ARTIFACT_UUID, tmp_path / "out", kind="onnx")

    def test_sha256_mismatch_raises_and_removes_file(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        descriptor = _descriptor(FILES, sha_overrides={"adapter_config.json": "0" * 64})
        client = _make_sync_client(_api_routes(descriptor))

        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            client.download_weights(ARTIFACT_UUID, tmp_path / "out")
        assert not (tmp_path / "out" / "adapter_config.json").exists()
        assert not (tmp_path / "out" / "adapter_config.json.part").exists()

    def test_sha256_mismatch_ignored_when_verify_false(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        descriptor = _descriptor(FILES, sha_overrides={"adapter_config.json": "0" * 64})
        client = _make_sync_client(_api_routes(descriptor))

        dest = client.download_weights(ARTIFACT_UUID, tmp_path / "out", verify=False)

        assert (dest / "adapter_config.json").read_bytes() == FILES["adapter_config.json"]

    def test_expired_url_refreshes_descriptor(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        descriptor_calls = {"count": 0}

        def api_get(path, **_kwargs):
            if path == f"/api/v1/artifacts/{ARTIFACT_UUID}/download":
                descriptor_calls["count"] += 1
                # First descriptor hands out already-expired URLs; the re-fetch
                # (triggered by the 403) returns working ones.
                sig = "expired" if descriptor_calls["count"] == 1 else "ok"
                return _descriptor(FILES, sig=sig)
            raise AssertionError(f"unexpected API GET {path}")

        client = _make_sync_client(api_get)
        dest = client.download_weights(ARTIFACT_UUID, tmp_path / "out")

        assert descriptor_calls["count"] >= 2
        for name, content in FILES.items():
            assert (dest / name).read_bytes() == content

    def test_completed_files_are_skipped_on_rerun(self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_transport(monkeypatch, _presigned_handler(FILES, calls))
        client = _make_sync_client(_api_routes(_descriptor(FILES)))
        dest = tmp_path / "out"
        (dest / "sub").mkdir(parents=True)
        for name, content in FILES.items():
            (dest / name).write_bytes(content)

        client.download_weights(ARTIFACT_UUID, dest)

        assert not calls  # everything already on disk and verified

    def test_unsafe_descriptor_name_rejected(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        client = _make_sync_client(_api_routes(_descriptor({"../evil.bin": b"x"})))

        with pytest.raises(ValueError, match="unsafe file name"):
            client.download_weights(ARTIFACT_UUID, tmp_path / "out")

    def test_resumes_from_existing_part_file(self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_transport(monkeypatch, _presigned_handler(SINGLE_FILE, calls))
        client = _make_sync_client(_api_routes(_descriptor(SINGLE_FILE)))
        dest = tmp_path / "out"
        dest.mkdir()
        # A previous interrupted run left the first 100 bytes on disk.
        (dest / "shard.bin.part").write_bytes(SINGLE_FILE["shard.bin"][:100])

        client.download_weights(ARTIFACT_UUID, dest, max_concurrency=1)

        assert (dest / "shard.bin").read_bytes() == SINGLE_FILE["shard.bin"]
        assert calls[0].headers["Range"] == "bytes=100-"

    def test_oversized_part_is_discarded_and_refetched(self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_transport(monkeypatch, _presigned_handler(SINGLE_FILE, calls))
        client = _make_sync_client(_api_routes(_descriptor(SINGLE_FILE)))
        dest = tmp_path / "out"
        dest.mkdir()
        # Longer than the manifest size: not resumable, must restart at 0.
        (dest / "shard.bin.part").write_bytes(b"x" * (len(SINGLE_FILE["shard.bin"]) + 10))

        client.download_weights(ARTIFACT_UUID, dest, max_concurrency=1)

        assert (dest / "shard.bin").read_bytes() == SINGLE_FILE["shard.bin"]
        assert "Range" not in calls[0].headers

    def test_transport_error_retries_with_range_resume(self, tmp_path, monkeypatch):
        content = SINGLE_FILE["shard.bin"]
        seen: list = []

        def flaky(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) == 1:
                # First attempt dies after the partial write already landed.
                raise httpx.ReadError("connection reset")
            range_header = request.headers.get("Range")
            start = int(range_header.split("=", 1)[1].split("-", 1)[0]) if range_header else 0
            return httpx.Response(206 if range_header else 200, content=content[start:])

        self._patch_transport(monkeypatch, flaky)
        client = _make_sync_client(_api_routes(_descriptor(SINGLE_FILE)))
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "shard.bin.part").write_bytes(content[:100])

        client.download_weights(ARTIFACT_UUID, dest, max_concurrency=1)

        assert (dest / "shard.bin").read_bytes() == content
        assert len(seen) == 2
        assert seen[1].headers["Range"] == "bytes=100-"

    def test_max_concurrency_must_be_positive(self, tmp_path):
        client = _make_sync_client(_api_routes(_descriptor(FILES)))
        with pytest.raises(ValueError, match="max_concurrency"):
            client.download_weights(ARTIFACT_UUID, tmp_path / "out", max_concurrency=0)


# ---------------------------------------------------------------------------
# AsyncServiceClient.download_weights (async twin)
# ---------------------------------------------------------------------------


class TestAsyncDownloadWeights:
    def _patch_transport(self, monkeypatch, handler):
        monkeypatch.setattr(
            async_service_client,
            "build_async_download_client",
            lambda timeout=None: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    def test_parallel_download_by_artifact_id(self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_transport(monkeypatch, _presigned_handler(FILES, calls))
        client = _make_async_client(_api_routes(_descriptor(FILES)))

        dest = asyncio.run(client.download_weights(ARTIFACT_UUID, tmp_path / "out"))

        for name, content in FILES.items():
            assert (dest / name).read_bytes() == content
        assert not list(dest.rglob("*.part"))
        assert len(calls) == len(FILES)
        client.http.get.assert_awaited_once_with(f"/api/v1/artifacts/{ARTIFACT_UUID}/download")

    def test_checkpoint_uri_resolution_flow(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        client = _make_async_client(_api_routes(_descriptor(FILES)))

        asyncio.run(client.download_weights(CHECKPOINT_URI, tmp_path / "out"))

        paths = [call.args[0] for call in client.http.get.call_args_list]
        assert paths == [
            "/api/v1/models/mdl-123/checkpoints",
            f"/api/v1/checkpoints/{CHECKPOINT_UUID}/artifacts",
            f"/api/v1/artifacts/{ARTIFACT_UUID}/download",
        ]

    def test_no_completed_artifact_never_exports_implicitly(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        pending = [{**ADAPTER_ARTIFACT, "status": "pending"}]
        client = _make_async_client(_api_routes(_descriptor(FILES), artifacts=pending))

        with pytest.raises(RuntimeError, match="export_weights"):
            asyncio.run(client.download_weights(CHECKPOINT_URI, tmp_path / "out"))

    def test_sha256_mismatch_raises(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        descriptor = _descriptor(FILES, sha_overrides={"adapter_config.json": "0" * 64})
        client = _make_async_client(_api_routes(descriptor))

        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            asyncio.run(client.download_weights(ARTIFACT_UUID, tmp_path / "out"))

    def test_expired_url_refreshes_descriptor(self, tmp_path, monkeypatch):
        self._patch_transport(monkeypatch, _presigned_handler(FILES))
        descriptor_calls = {"count": 0}

        async def api_get(path, **_kwargs):
            if path == f"/api/v1/artifacts/{ARTIFACT_UUID}/download":
                descriptor_calls["count"] += 1
                sig = "expired" if descriptor_calls["count"] == 1 else "ok"
                return _descriptor(FILES, sig=sig)
            raise AssertionError(f"unexpected API GET {path}")

        client = AsyncServiceClient(base_url="https://test.example.com", api_key="sk-test")
        client._http = MagicMock()
        client._http.get = AsyncMock(side_effect=api_get)

        dest = asyncio.run(client.download_weights(ARTIFACT_UUID, tmp_path / "out"))

        assert descriptor_calls["count"] >= 2
        for name, content in FILES.items():
            assert (dest / name).read_bytes() == content

    def test_resumes_from_existing_part_file(self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_transport(monkeypatch, _presigned_handler(SINGLE_FILE, calls))
        client = _make_async_client(_api_routes(_descriptor(SINGLE_FILE)))
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "shard.bin.part").write_bytes(SINGLE_FILE["shard.bin"][:100])

        asyncio.run(client.download_weights(ARTIFACT_UUID, dest, max_concurrency=1))

        assert (dest / "shard.bin").read_bytes() == SINGLE_FILE["shard.bin"]
        assert calls[0].headers["Range"] == "bytes=100-"

    def test_max_concurrency_must_be_positive(self, tmp_path):
        client = _make_async_client(_api_routes(_descriptor(FILES)))
        with pytest.raises(ValueError, match="max_concurrency"):
            asyncio.run(client.download_weights(ARTIFACT_UUID, tmp_path / "out", max_concurrency=0))


# ---------------------------------------------------------------------------
# CLI: weaver checkpoint download
# ---------------------------------------------------------------------------


class TestCheckpointDownloadCLI:
    def test_download_invokes_service_client(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
        monkeypatch.delenv("WEAVER_API_KEY", raising=False)
        client = MagicMock()
        client.download_weights.return_value = tmp_path / "out"
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli,
                [
                    "checkpoint",
                    "download",
                    f"{CHECKPOINT_URI}/artifacts/hf_adapter",
                    "-o",
                    str(tmp_path / "out"),
                    "--kind",
                    "hf_adapter",
                ],
            )

        assert result.exit_code == 0
        client.connect.assert_called_once_with(ensure_session=False)
        client.download_weights.assert_called_once_with(
            f"{CHECKPOINT_URI}/artifacts/hf_adapter",
            str(tmp_path / "out"),
            kind="hf_adapter",
        )
        client.close.assert_called_once_with()

    def test_download_error_exits_nonzero(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
        monkeypatch.delenv("WEAVER_API_KEY", raising=False)
        client = MagicMock()
        client.download_weights.side_effect = RuntimeError("No completed HF weights artifact")
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli,
                ["checkpoint", "download", ARTIFACT_UUID, "-o", str(tmp_path / "out")],
            )

        assert result.exit_code == 1
        assert "No completed HF weights artifact" in result.output


# ---------------------------------------------------------------------------
# Bare download client never carries Weaver credentials
# ---------------------------------------------------------------------------


class TestBareDownloadClient:
    def test_sync_client_has_no_api_key_header(self):
        with _http.build_download_client() as client:
            assert "X-WEAVER-API-KEY" not in client.headers
            assert client.headers.get("User-Agent", "").startswith("weaver-sdk/")

    def test_async_client_has_no_api_key_header(self):
        async def run():
            async with _async_http.build_async_download_client() as client:
                assert "X-WEAVER-API-KEY" not in client.headers
                assert client.headers.get("User-Agent", "").startswith("weaver-sdk/")

        asyncio.run(run())


class TestSymlinkContainment:
    def test_symlink_inside_dest_cannot_escape(self, tmp_path, monkeypatch):
        calls: list = []
        outside = tmp_path / "outside"
        outside.mkdir()
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "link").symlink_to(outside)
        files = {"link/owned.bin": b"x"}
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(_presigned_handler(files, calls))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(files)))
        with pytest.raises(ValueError, match="outside the destination"):
            client.download_weights(ARTIFACT_UUID, dest)
        assert not (outside / "owned.bin").exists()
        assert not (outside / "owned.bin.part").exists()
        assert not calls  # rejected before a single byte was fetched

    def test_async_symlink_inside_dest_cannot_escape(self, tmp_path, monkeypatch):
        calls: list = []
        outside = tmp_path / "outside"
        outside.mkdir()
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "link").symlink_to(outside)
        files = {"link/owned.bin": b"x"}
        monkeypatch.setattr(
            async_service_client,
            "build_async_download_client",
            lambda timeout=None: httpx.AsyncClient(
                transport=httpx.MockTransport(_presigned_handler(files, calls))
            ),
        )
        client = _make_async_client(_api_routes(_descriptor(files)))
        with pytest.raises(ValueError, match="outside the destination"):
            asyncio.run(client.download_weights(ARTIFACT_UUID, dest))
        assert not (outside / "owned.bin").exists()
        assert not calls


# ---------------------------------------------------------------------------
# Descriptor-anchored destination walk (weaver._safeio)
# ---------------------------------------------------------------------------

requires_dir_fd = pytest.mark.skipif(
    not _safeio.supports_dir_fd(), reason="platform has no dir_fd support"
)


@requires_dir_fd
class TestAnchoredWalk:
    def test_creates_and_returns_the_leaf_directory(self, tmp_path):
        dest = tmp_path / "out"
        dest.mkdir()

        parent_fd = _safeio.open_parent_fd(dest, PurePosixPath("a/b/file.bin"), create=True)
        try:
            assert (dest / "a" / "b").is_dir()
            with _safeio.open_for_write(parent_fd, "file.bin", append=False) as sink:
                sink.write(b"payload")
        finally:
            os.close(parent_fd)

        assert (dest / "a" / "b" / "file.bin").read_bytes() == b"payload"

    def test_symlinked_intermediate_is_rejected(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "sub").symlink_to(outside)

        with pytest.raises(ValueError, match="unsafe path component"):
            _safeio.open_parent_fd(dest, PurePosixPath("sub/owned.bin"), create=True)

    def test_walk_creates_nothing_outside_before_rejecting(self, tmp_path):
        # Regression for the reported side effect: mkdir(parents=True) used to
        # run ahead of the containment check and materialized a directory in
        # the link target before anything raised.
        outside = tmp_path / "outside"
        outside.mkdir()
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "link").symlink_to(outside)

        with pytest.raises(ValueError, match="unsafe path component"):
            _safeio.open_parent_fd(dest, PurePosixPath("link/newdir/file.bin"), create=True)

        assert list(outside.iterdir()) == []

    def test_symlink_to_a_directory_inside_dest_is_also_rejected(self, tmp_path):
        # The resolved-parent check would accept this one. The walk follows NO
        # component, which is what makes the anchor unconditional.
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "real").mkdir()
        (dest / "link").symlink_to(dest / "real")

        with pytest.raises(ValueError, match="unsafe path component"):
            _safeio.open_parent_fd(dest, PurePosixPath("link/owned.bin"), create=True)

    def test_file_where_a_directory_is_expected_is_rejected(self, tmp_path):
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "sub").write_bytes(b"not a directory")

        with pytest.raises(ValueError, match="unsafe path component"):
            _safeio.open_parent_fd(dest, PurePosixPath("sub/owned.bin"), create=True)

    def test_anchor_survives_a_directory_swap(self, tmp_path):
        """The TOCTOU kill in isolation: swap the name AFTER the walk.

        A descriptor names an inode. Once the walk has one, re-pointing the
        path it came from is inert — no later call re-traverses the name.
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "sub").mkdir()

        parent_fd = _safeio.open_parent_fd(dest, PurePosixPath("sub/owned.bin"), create=True)
        try:
            # Everything an attacker winning the race could do to the name.
            os.rename(dest / "sub", dest / "sub-moved")
            (dest / "sub").symlink_to(outside)

            with _safeio.open_for_write(parent_fd, "owned.bin.part", append=False) as sink:
                sink.write(b"anchored")
            _safeio.rename_within(parent_fd, "owned.bin.part", "owned.bin")
            assert os.listdir(parent_fd) == ["owned.bin"]
        finally:
            os.close(parent_fd)

        assert (dest / "sub-moved" / "owned.bin").read_bytes() == b"anchored"
        assert list(outside.iterdir()) == []


@requires_dir_fd
class TestDownloadAnchoring:
    """End-to-end: the bytes land where the walk pointed, not where the name does."""

    @staticmethod
    def _swapping_handler(files, dest, outside, swapped):
        """Presigned handler that swaps ``dest/sub`` out mid-download.

        By the time a request reaches the transport, ``_download_weights_file``
        has already walked and pinned the parent descriptor, so this is the
        exact interleaving a racing attacker needs — made deterministic. The
        first attempt then dies with a transport error, which puts the *whole*
        remainder of the download after the swap: the resume stat, the reopen,
        the write, the hash and the publishing rename.
        """
        base = _presigned_handler(files)

        def handler(request: httpx.Request) -> httpx.Response:
            if not swapped:
                os.rename(dest / "sub", dest / "sub-moved")
                (dest / "sub").symlink_to(outside)
                swapped.append(True)
                raise httpx.ReadError("connection reset")
            return base(request)

        return handler

    def _prepare(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "sub").mkdir()
        return dest, outside, {"sub/owned.bin": b"anchored-bytes"}

    def _assert_landed_inside(self, dest, outside, swapped):
        assert swapped, "the swap must actually have happened mid-download"
        assert (dest / "sub-moved" / "owned.bin").read_bytes() == b"anchored-bytes"
        assert not (outside / "owned.bin").exists()
        assert not (outside / "owned.bin.part").exists()
        assert list(outside.iterdir()) == []

    def test_sync_write_and_publish_follow_the_descriptor(self, tmp_path, monkeypatch):
        dest, outside, files = self._prepare(tmp_path)
        swapped: list = []
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(self._swapping_handler(files, dest, outside, swapped))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(files)))

        client.download_weights(ARTIFACT_UUID, dest)

        self._assert_landed_inside(dest, outside, swapped)

    def test_async_write_and_publish_follow_the_descriptor(self, tmp_path, monkeypatch):
        dest, outside, files = self._prepare(tmp_path)
        swapped: list = []
        monkeypatch.setattr(
            async_service_client,
            "build_async_download_client",
            lambda timeout=None: httpx.AsyncClient(
                transport=httpx.MockTransport(self._swapping_handler(files, dest, outside, swapped))
            ),
        )
        client = _make_async_client(_api_routes(_descriptor(files)))

        asyncio.run(client.download_weights(ARTIFACT_UUID, dest))

        self._assert_landed_inside(dest, outside, swapped)


class TestUnanchoredFallback:
    """Platforms without ``dir_fd`` (Windows) keep the previous path-based flow."""

    @pytest.fixture
    def no_dir_fd(self, monkeypatch):
        monkeypatch.setattr(os, "supports_dir_fd", frozenset())
        assert not _safeio.supports_dir_fd()

    def test_sync_download_uses_the_legacy_path(self, tmp_path, monkeypatch, no_dir_fd):
        def unreachable(*_args, **_kwargs):
            raise AssertionError("the anchored walk must not run without dir_fd support")

        monkeypatch.setattr(service_client, "open_parent_fd", unreachable)
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(_presigned_handler(FILES))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(FILES)))

        dest = client.download_weights(ARTIFACT_UUID, tmp_path / "out")

        for name, content in FILES.items():
            assert (dest / name).read_bytes() == content
        assert not list(dest.rglob("*.part"))

    def test_async_download_uses_the_legacy_path(self, tmp_path, monkeypatch, no_dir_fd):
        def unreachable(*_args, **_kwargs):
            raise AssertionError("the anchored walk must not run without dir_fd support")

        monkeypatch.setattr(async_service_client, "open_parent_fd", unreachable)
        monkeypatch.setattr(
            async_service_client,
            "build_async_download_client",
            lambda timeout=None: httpx.AsyncClient(
                transport=httpx.MockTransport(_presigned_handler(FILES))
            ),
        )
        client = _make_async_client(_api_routes(_descriptor(FILES)))

        dest = asyncio.run(client.download_weights(ARTIFACT_UUID, tmp_path / "out"))

        for name, content in FILES.items():
            assert (dest / name).read_bytes() == content
        assert not list(dest.rglob("*.part"))

    def test_legacy_path_still_resumes(self, tmp_path, monkeypatch, no_dir_fd):
        calls: list = []
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(_presigned_handler(SINGLE_FILE, calls))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(SINGLE_FILE)))
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "shard.bin.part").write_bytes(SINGLE_FILE["shard.bin"][:100])

        client.download_weights(ARTIFACT_UUID, dest, max_concurrency=1)

        assert (dest / "shard.bin").read_bytes() == SINGLE_FILE["shard.bin"]
        assert calls[0].headers["Range"] == "bytes=100-"

    def test_legacy_path_still_rejects_a_preplanted_part_symlink(
        self, tmp_path, monkeypatch, no_dir_fd
    ):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"")
        dest = tmp_path / "out"
        dest.mkdir()
        files = {"owned.bin": b"x"}
        (dest / "owned.bin.part").symlink_to(outside)
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(_presigned_handler(files))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(files)))

        with pytest.raises(ValueError, match="refusing to write through a symlink"):
            client.download_weights(ARTIFACT_UUID, dest)
        assert outside.read_bytes() == b""


class TestUriModelIdGuard:
    @pytest.mark.parametrize(
        "uri",
        [
            "weaver://../checkpoints/x",
            "weaver://%2e%2e/checkpoints/x",
            "weaver://a\\b/checkpoints/x",
            "weaver://.hidden/checkpoints/x",
        ],
    )
    def test_parse_rejects_unsafe_model_ids(self, uri):
        with pytest.raises(ValueError, match="unsafe model id|Unrecognized weaver URI"):
            parse_download_target(uri)

    def test_download_rejects_unsafe_model_id_before_any_request(self, tmp_path):
        client = _make_sync_client(None)
        with pytest.raises(ValueError, match="unsafe model id"):
            client.download_weights("weaver://../checkpoints/x", tmp_path / "out")
        client.http.get.assert_not_called()

    def test_async_download_rejects_unsafe_model_id(self, tmp_path):
        client = _make_async_client(None)
        with pytest.raises(ValueError, match="unsafe model id"):
            asyncio.run(client.download_weights("weaver://../checkpoints/x", tmp_path / "out"))
        client.http.get.assert_not_called()


class TestPartSymlinkGuard:
    """A ``.part`` name held by a symlink is refused, not written through.

    The anchored path catches this at the resume ``lstat``, before anything is
    opened: the link is seen as a link instead of being measured through.
    """

    def test_preplanted_part_symlink_is_rejected(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"")
        dest = tmp_path / "out"
        dest.mkdir()
        files = {"owned.bin": b"x"}
        (dest / "owned.bin.part").symlink_to(outside)
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(_presigned_handler(files))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(files)))
        with pytest.raises(ValueError, match="refusing to write through a symlink"):
            client.download_weights(ARTIFACT_UUID, dest)
        assert outside.read_bytes() == b""

    def test_async_preplanted_part_symlink_is_rejected(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"")
        dest = tmp_path / "out"
        dest.mkdir()
        files = {"owned.bin": b"x"}
        (dest / "owned.bin.part").symlink_to(outside)
        monkeypatch.setattr(
            async_service_client,
            "build_async_download_client",
            lambda timeout=None: httpx.AsyncClient(
                transport=httpx.MockTransport(_presigned_handler(files))
            ),
        )
        client = _make_async_client(_api_routes(_descriptor(files)))
        with pytest.raises(ValueError, match="refusing to write through a symlink"):
            asyncio.run(client.download_weights(ARTIFACT_UUID, dest))
        assert outside.read_bytes() == b""

    @requires_dir_fd
    def test_symlink_at_the_final_name_is_replaced_not_written_through(self, tmp_path, monkeypatch):
        # A link planted at the *published* name must not be treated as
        # finished work, and the publishing rename must replace the link
        # itself rather than following it.
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"untouched")
        dest = tmp_path / "out"
        dest.mkdir()
        files = {"owned.bin": b"real-content"}
        (dest / "owned.bin").symlink_to(outside)
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(_presigned_handler(files))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(files)))

        client.download_weights(ARTIFACT_UUID, dest)

        assert not (dest / "owned.bin").is_symlink()
        assert (dest / "owned.bin").read_bytes() == b"real-content"
        assert outside.read_bytes() == b"untouched"


@requires_dir_fd
class TestHardLinkGuard:
    """A ``.part`` name that resolves to a multiply-linked inode is refused.

    ``O_NOFOLLOW`` guards symlinks only; a pre-planted hard link
    (``os.link(outside, dest/x.part)``) points a stable name at an inode that
    can live outside the tree. The fstat-on-the-open-descriptor check
    (:func:`weaver._safeio._reject_hard_link`) rejects it, and the fresh
    ``O_CREAT | O_EXCL`` open never truncates a pre-existing name.
    """

    HARDLINK_FILES = {"owned.bin": b"real-downloaded-content"}

    def _patch_sync(self, monkeypatch):
        monkeypatch.setattr(
            service_client,
            "build_download_client",
            lambda timeout=None: httpx.Client(
                transport=httpx.MockTransport(_presigned_handler(self.HARDLINK_FILES))
            ),
        )

    def _patch_async(self, monkeypatch):
        monkeypatch.setattr(
            async_service_client,
            "build_async_download_client",
            lambda timeout=None: httpx.AsyncClient(
                transport=httpx.MockTransport(_presigned_handler(self.HARDLINK_FILES))
            ),
        )

    def test_open_for_write_refuses_a_preplanted_hard_link(self, tmp_path):
        # The primitive, exercised directly in both modes. Append opens the
        # inode and rejects it on the fstat; a fresh O_EXCL open never opens
        # the occupied name at all and refuses the race — both safe, and
        # neither writes a byte, so the outside inode is untouched.
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"SECRET")
        dest = tmp_path / "out"
        dest.mkdir()
        os.link(outside, dest / "x.part")
        parent_fd = _safeio.open_parent_fd(dest, PurePosixPath("x"), create=False)
        try:
            with pytest.raises(ValueError, match="hard link"):
                _safeio.open_for_write(parent_fd, "x.part", append=True)
            with pytest.raises(ValueError, match="between validation and creation"):
                _safeio.open_for_write(parent_fd, "x.part", append=False)
        finally:
            os.close(parent_fd)
        assert outside.read_bytes() == b"SECRET"
        assert os.stat(outside).st_nlink == 2

    def test_sync_preplanted_hard_linked_part_is_refused(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"AAA")  # shorter than the manifest -> a "resumable" partial
        dest = tmp_path / "out"
        dest.mkdir()
        os.link(outside, dest / "owned.bin.part")
        self._patch_sync(monkeypatch)
        client = _make_sync_client(_api_routes(_descriptor(self.HARDLINK_FILES)))

        with pytest.raises(ValueError, match="hard link"):
            client.download_weights(ARTIFACT_UUID, dest)

        assert outside.read_bytes() == b"AAA"  # byte-for-byte unchanged
        assert os.stat(outside).st_nlink == 2

    def test_async_preplanted_hard_linked_part_is_refused(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"AAA")
        dest = tmp_path / "out"
        dest.mkdir()
        os.link(outside, dest / "owned.bin.part")
        self._patch_async(monkeypatch)
        client = _make_async_client(_api_routes(_descriptor(self.HARDLINK_FILES)))

        with pytest.raises(ValueError, match="hard link"):
            asyncio.run(client.download_weights(ARTIFACT_UUID, dest))

        assert outside.read_bytes() == b"AAA"
        assert os.stat(outside).st_nlink == 2

    @staticmethod
    def _racing_resume(module, dest, outside):
        """Wrap ``resume_offset_at`` to swap the ``.part`` for a hard link.

        Fires AFTER validation returns and BEFORE the write is opened — the
        exact window an attacker races — so only the fstat on the opened
        descriptor can catch it. Deterministic: no real race needed.
        """
        real = module.resume_offset_at

        def racing(parent_fd, part_name, entry):
            offset = real(parent_fd, part_name, entry)
            # Repoint the validated name at an outside inode.
            os.unlink(dest / part_name)
            os.link(outside, dest / part_name)
            return offset

        return racing

    def test_sync_hard_link_added_after_validation_is_caught_on_fstat(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"SECRET")
        dest = tmp_path / "out"
        dest.mkdir()
        # A legitimate single-linked partial exists at validation time.
        (dest / "owned.bin.part").write_bytes(b"AA")
        self._patch_sync(monkeypatch)
        monkeypatch.setattr(
            service_client, "resume_offset_at", self._racing_resume(service_client, dest, outside)
        )
        client = _make_sync_client(_api_routes(_descriptor(self.HARDLINK_FILES)))

        with pytest.raises(ValueError, match="hard link"):
            client.download_weights(ARTIFACT_UUID, dest)

        assert outside.read_bytes() == b"SECRET"
        assert os.stat(outside).st_nlink == 2

    def test_async_hard_link_added_after_validation_is_caught_on_fstat(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"SECRET")
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "owned.bin.part").write_bytes(b"AA")
        self._patch_async(monkeypatch)
        monkeypatch.setattr(
            async_service_client,
            "resume_offset_at",
            self._racing_resume(async_service_client, dest, outside),
        )
        client = _make_async_client(_api_routes(_descriptor(self.HARDLINK_FILES)))

        with pytest.raises(ValueError, match="hard link"):
            asyncio.run(client.download_weights(ARTIFACT_UUID, dest))

        assert outside.read_bytes() == b"SECRET"
        assert os.stat(outside).st_nlink == 2

    def test_normal_download_still_succeeds_over_a_single_linked_partial(
        self, tmp_path, monkeypatch
    ):
        # Regression guard: the fstat check must not reject an ordinary resume.
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "owned.bin.part").write_bytes(self.HARDLINK_FILES["owned.bin"][:5])
        self._patch_sync(monkeypatch)
        client = _make_sync_client(_api_routes(_descriptor(self.HARDLINK_FILES)))

        dest_ret = client.download_weights(ARTIFACT_UUID, dest, max_concurrency=1)

        assert (dest_ret / "owned.bin").read_bytes() == self.HARDLINK_FILES["owned.bin"]
        assert not list(dest_ret.rglob("*.part"))
