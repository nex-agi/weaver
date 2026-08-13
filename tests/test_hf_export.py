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

"""Tests for HF weights export: WeightsArtifact type and export_weights."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from weaver.async_training_client import AsyncTrainingClient
from weaver.cli import cli
from weaver.operations import AsyncOperationHandle, OperationHandle
from weaver.training_client import TrainingClient
from weaver.types.checkpoint import Checkpoint
from weaver.types.weights_artifact import WeightsArtifact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECKPOINT_UUID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

ARTIFACT_PAYLOAD: Dict[str, Any] = {
    "id": "aaaa1111-bb22-4c33-8d44-eeee5555ffff",
    "checkpoint_id": CHECKPOINT_UUID,
    "model_id": "mdl-123",
    "kind": "hf_adapter",
    "status": "completed",
    "uri": "weaver://mdl-123/checkpoints/step-5/artifacts/hf_adapter",
    "size_bytes": 1024,
    "manifest": {"format_version": 1, "files": [{"name": "adapter_model.safetensors"}]},
    "ttl_seconds": 604800,
    "expires_at": "2026-08-19T00:00:00Z",
    "created_at": "2026-08-12T00:00:00Z",
    "updated_at": "2026-08-12T00:10:00Z",
}


def _make_training_client() -> TrainingClient:
    """Create a TrainingClient with a mocked ServiceClient."""
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    return TrainingClient(
        service=service,
        model_id="mdl-123",
        base_model="Qwen/Qwen3-8B",
        session_id="sess-abc",
    )


def _make_async_training_client() -> AsyncTrainingClient:
    """Create an AsyncTrainingClient with a mocked AsyncServiceClient."""
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    service.http.post = AsyncMock()
    service.http.get = AsyncMock()
    return AsyncTrainingClient(
        service=service,
        model_id="mdl-123",
        base_model="Qwen/Qwen3-8B",
        session_id="sess-abc",
    )


def _done_operation(response: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": "op-1", "status": "done", "response": response}


# ---------------------------------------------------------------------------
# WeightsArtifact type
# ---------------------------------------------------------------------------


class TestWeightsArtifactType:
    def test_from_payload(self):
        artifact = WeightsArtifact.from_payload(ARTIFACT_PAYLOAD)
        assert artifact.id == "aaaa1111-bb22-4c33-8d44-eeee5555ffff"
        assert artifact.checkpoint_id == CHECKPOINT_UUID
        assert artifact.model_id == "mdl-123"
        assert artifact.kind == "hf_adapter"
        assert artifact.status == "completed"
        assert artifact.uri == "weaver://mdl-123/checkpoints/step-5/artifacts/hf_adapter"
        assert artifact.size_bytes == 1024
        assert artifact.manifest == ARTIFACT_PAYLOAD["manifest"]
        assert artifact.error is None
        assert artifact.ttl_seconds == 604800
        assert artifact.expires_at == "2026-08-19T00:00:00Z"
        assert artifact.created_at == "2026-08-12T00:00:00Z"
        assert artifact.updated_at == "2026-08-12T00:10:00Z"

    def test_from_payload_minimal(self):
        artifact = WeightsArtifact.from_payload({"id": "art-2"})
        assert artifact.id == "art-2"
        assert artifact.kind == "hf_model"
        assert artifact.status is None
        assert artifact.manifest is None
        assert artifact.size_bytes is None

    def test_from_payload_error_state(self):
        artifact = WeightsArtifact.from_payload(
            {"id": "art-3", "kind": "hf_model", "status": "error", "error": "conversion failed"}
        )
        assert artifact.status == "error"
        assert artifact.error == "conversion failed"

    def test_is_frozen(self):
        artifact = WeightsArtifact(id="aaaa1111-bb22-4c33-8d44-eeee5555ffff")
        with pytest.raises(AttributeError):
            artifact.id = "art-2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrainingClient.export_weights (sync)
# ---------------------------------------------------------------------------


class TestExportWeights:
    def test_one_step_export_without_checkpoint(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)

        artifact = tc.export_weights()

        assert isinstance(artifact, WeightsArtifact)
        assert artifact.id == "aaaa1111-bb22-4c33-8d44-eeee5555ffff"
        args = tc._service.http.post.call_args
        assert args[0][0] == "/api/v1/models/mdl-123/export-hf"
        assert args[1]["json"] == {
            "format": "huggingface",
            "merge_adapter": False,
            "ttl_seconds": 604800,
        }
        assert args[1]["max_retries"] == 1

    def test_checkpoint_export_with_checkpoint_object(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)

        ckpt = Checkpoint(id=CHECKPOINT_UUID, path="weaver://mdl-123/checkpoints/step-5")
        artifact = tc.export_weights(checkpoint=ckpt, merge_adapter=True, force=True)

        assert isinstance(artifact, WeightsArtifact)
        args = tc._service.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/export"
        assert args[1]["json"] == {
            "format": "huggingface",
            "merge_adapter": True,
            "ttl_seconds": 604800,
            "force": True,
        }

    def test_checkpoint_export_with_raw_id(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)

        tc.export_weights(checkpoint=CHECKPOINT_UUID, ttl_seconds=None)

        args = tc._service.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/export"
        assert args[1]["json"]["ttl_seconds"] is None

    def test_checkpoint_export_resolves_weaver_path(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = {
            "items": [
                {"id": "ckpt-0", "path": "weaver://mdl-123/checkpoints/step-1"},
                {"id": CHECKPOINT_UUID, "path": "weaver://mdl-123/checkpoints/step-5"},
            ]
        }
        tc._service.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)

        tc.export_weights(checkpoint="weaver://mdl-123/checkpoints/step-5")

        tc._service.http.get.assert_called_once_with("/api/v1/models/mdl-123/checkpoints")
        args = tc._service.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/export"

    def test_checkpoint_export_unknown_path_raises(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = {"items": []}

        with pytest.raises(ValueError, match="No checkpoint with path"):
            tc.export_weights(checkpoint="weaver://mdl-123/checkpoints/missing")
        tc._service.http.post.assert_not_called()

    def test_checkpoint_export_foreign_model_path_raises(self):
        tc = _make_training_client()

        with pytest.raises(ValueError, match="belongs to model other"):
            tc.export_weights(checkpoint="weaver://other/checkpoints/step-5")
        tc._service.http.post.assert_not_called()

    def test_idempotent_completed_hit_returns_artifact_even_without_wait(self):
        # HTTP 200 idempotent hit: the response body is the artifact itself,
        # so there is no operation to hand back even when wait=False.
        tc = _make_training_client()
        tc._service.http.post.return_value = dict(ARTIFACT_PAYLOAD)

        artifact = tc.export_weights(checkpoint=CHECKPOINT_UUID, wait=False)

        assert isinstance(artifact, WeightsArtifact)
        assert artifact.status == "completed"

    def test_no_wait_returns_operation_handle(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = {"id": "op-9", "status": "pending"}

        handle = tc.export_weights(checkpoint=CHECKPOINT_UUID, wait=False)

        assert isinstance(handle, OperationHandle)
        assert handle.operation_id == "op-9"

    def test_wait_parses_operation_response_into_artifact(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)

        artifact = tc.export_weights(checkpoint=CHECKPOINT_UUID)

        assert isinstance(artifact, WeightsArtifact)
        assert artifact.kind == "hf_adapter"
        assert artifact.checkpoint_id == CHECKPOINT_UUID

    def test_empty_checkpoint_reference_raises(self):
        tc = _make_training_client()
        with pytest.raises(ValueError, match="must not be empty"):
            tc.export_weights(checkpoint="   ")

    def test_checkpoint_object_without_id_raises(self):
        tc = _make_training_client()
        ckpt = Checkpoint(id="", path="weaver://mdl-123/checkpoints/step-5")
        with pytest.raises(ValueError, match="no id"):
            tc.export_weights(checkpoint=ckpt)


# ---------------------------------------------------------------------------
# AsyncTrainingClient.export_weights (async twin)
# ---------------------------------------------------------------------------


class TestAsyncExportWeights:
    def test_one_step_export_without_checkpoint(self):
        tc = _make_async_training_client()
        tc._service.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)

        artifact = asyncio.run(tc.export_weights())

        assert isinstance(artifact, WeightsArtifact)
        assert artifact.id == "aaaa1111-bb22-4c33-8d44-eeee5555ffff"
        args = tc._service.http.post.call_args
        assert args[0][0] == "/api/v1/models/mdl-123/export-hf"
        assert args[1]["json"] == {
            "format": "huggingface",
            "merge_adapter": False,
            "ttl_seconds": 604800,
        }
        assert args[1]["max_retries"] == 1

    def test_checkpoint_export_resolves_weaver_path(self):
        tc = _make_async_training_client()
        tc._service.http.get.return_value = {
            "items": [{"id": CHECKPOINT_UUID, "path": "weaver://mdl-123/checkpoints/step-5"}]
        }
        tc._service.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)

        artifact = asyncio.run(
            tc.export_weights(checkpoint="weaver://mdl-123/checkpoints/step-5", force=True)
        )

        assert isinstance(artifact, WeightsArtifact)
        args = tc._service.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/export"
        assert args[1]["json"]["force"] is True

    def test_idempotent_completed_hit_returns_artifact_even_without_wait(self):
        tc = _make_async_training_client()
        tc._service.http.post.return_value = dict(ARTIFACT_PAYLOAD)

        artifact = asyncio.run(tc.export_weights(checkpoint=CHECKPOINT_UUID, wait=False))

        assert isinstance(artifact, WeightsArtifact)
        assert artifact.status == "completed"

    def test_no_wait_returns_async_operation_handle(self):
        tc = _make_async_training_client()
        tc._service.http.post.return_value = {"id": "op-9", "status": "pending"}

        handle = asyncio.run(tc.export_weights(checkpoint=CHECKPOINT_UUID, wait=False))

        assert isinstance(handle, AsyncOperationHandle)
        assert handle.operation_id == "op-9"

    def test_foreign_model_path_raises(self):
        tc = _make_async_training_client()
        with pytest.raises(ValueError, match="belongs to model other"):
            asyncio.run(tc.export_weights(checkpoint="weaver://other/checkpoints/step-5"))


# ---------------------------------------------------------------------------
# CLI: weaver checkpoint export
# ---------------------------------------------------------------------------


class TestCheckpointExportCLI:
    def _client(self) -> MagicMock:
        client = MagicMock()
        client.http.post.return_value = _done_operation(ARTIFACT_PAYLOAD)
        return client

    def test_export_with_checkpoint_id(self, monkeypatch):
        monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
        monkeypatch.delenv("WEAVER_API_KEY", raising=False)
        client = self._client()
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["checkpoint", "export", CHECKPOINT_UUID])

        assert result.exit_code == 0
        client.connect.assert_called_once_with(ensure_session=False)
        args = client.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/export"
        assert args[1]["json"] == {
            "format": "huggingface",
            "merge_adapter": False,
            "ttl_seconds": 604800,
            "force": False,
        }
        assert "Export completed" in result.output
        client.close.assert_called_once_with()

    def test_export_resolves_weaver_uri_and_flags(self, monkeypatch):
        monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
        monkeypatch.delenv("WEAVER_API_KEY", raising=False)
        client = self._client()
        client.http.get.return_value = {
            "items": [{"id": CHECKPOINT_UUID, "path": "weaver://mdl-123/checkpoints/step-5"}]
        }
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli,
                [
                    "checkpoint",
                    "export",
                    "weaver://mdl-123/checkpoints/step-5",
                    "--merge-adapter",
                    "--ttl",
                    "3600",
                    "--force",
                ],
            )

        assert result.exit_code == 0
        client.http.get.assert_called_once_with("/api/v1/models/mdl-123/checkpoints")
        args = client.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/export"
        assert args[1]["json"] == {
            "format": "huggingface",
            "merge_adapter": True,
            "ttl_seconds": 3600,
            "force": True,
        }

    def test_export_no_wait_prints_operation_id(self, monkeypatch):
        monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
        monkeypatch.delenv("WEAVER_API_KEY", raising=False)
        client = MagicMock()
        client.http.post.return_value = {"id": "op-42", "status": "pending"}
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["checkpoint", "export", CHECKPOINT_UUID, "--no-wait"])

        assert result.exit_code == 0
        assert "op-42" in result.output

    def test_export_idempotent_hit_prints_artifact(self, monkeypatch):
        monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
        monkeypatch.delenv("WEAVER_API_KEY", raising=False)
        client = MagicMock()
        client.http.post.return_value = dict(ARTIFACT_PAYLOAD)
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["checkpoint", "export", CHECKPOINT_UUID])

        assert result.exit_code == 0
        assert "already completed" in result.output


class TestExportCLIIdGuard:
    @pytest.mark.parametrize("raw", ["../models/target", "a/b", "ckpt-1"])
    def test_checkpoint_export_rejects_traversal_ids(self, monkeypatch, raw):
        monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
        monkeypatch.delenv("WEAVER_API_KEY", raising=False)
        client = MagicMock()
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["checkpoint", "export", raw, "--no-wait"])
        assert result.exit_code != 0
        client.http.post.assert_not_called()
