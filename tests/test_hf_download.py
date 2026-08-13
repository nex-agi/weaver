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
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from weaver import _async_http, _http, async_service_client, service_client
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
    def test_full_download(self, tmp_path):
        content = FILES["adapter_model.safetensors"]
        transport = httpx.MockTransport(_presigned_handler(FILES))
        dest = tmp_path / "shard.bin"
        with httpx.Client(transport=transport) as client:
            written = stream_download_to_file(
                "https://tos.example.com/files/adapter_model.safetensors?sig=ok",
                dest,
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
            written = stream_download_to_file(
                "https://tos.example.com/files/adapter_model.safetensors?sig=ok",
                dest,
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
            written = stream_download_to_file(
                "https://tos.example.com/f", dest, client=client, resume_from=13
            )
        assert written == len(content)
        assert dest.read_bytes() == content

    def test_expired_url_raises_dedicated_error(self, tmp_path):
        transport = httpx.MockTransport(_presigned_handler(FILES))
        with httpx.Client(transport=transport) as client:
            with pytest.raises(DownloadURLExpiredError):
                stream_download_to_file(
                    "https://tos.example.com/files/adapter_config.json?sig=expired",
                    tmp_path / "f.bin",
                    client=client,
                )

    def test_async_twin_full_download(self, tmp_path):
        content = FILES["adapter_model.safetensors"]
        transport = httpx.MockTransport(_presigned_handler(FILES))
        dest = tmp_path / "shard.bin"

        async def run():
            async with httpx.AsyncClient(transport=transport) as client:
                return await _async_http.async_stream_download_to_file(
                    "https://tos.example.com/files/adapter_model.safetensors?sig=ok",
                    dest,
                    client=client,
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
                transport=httpx.MockTransport(_presigned_handler(files))
            ),
        )
        client = _make_sync_client(_api_routes(_descriptor(files)))
        with pytest.raises(ValueError, match="outside the destination"):
            client.download_weights(ARTIFACT_UUID, dest)
        assert not (outside / "owned.bin").exists()
        assert not (outside / "owned.bin.part").exists()

    def test_async_symlink_inside_dest_cannot_escape(self, tmp_path, monkeypatch):
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
                transport=httpx.MockTransport(_presigned_handler(files))
            ),
        )
        client = _make_async_client(_api_routes(_descriptor(files)))
        with pytest.raises(ValueError, match="outside the destination"):
            asyncio.run(client.download_weights(ARTIFACT_UUID, dest))
        assert not (outside / "owned.bin").exists()
