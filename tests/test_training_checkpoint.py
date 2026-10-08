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

"""Tests for TrainingClient checkpoint management methods."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

import weaver.async_training_client as async_training_client_module
import weaver.training_client as training_client_module
from weaver._http import WeaverAPIError
from weaver._utils import DEFAULT_SAMPLER_TTL_SECONDS
from weaver.async_training_client import AsyncTrainingClient
from weaver.operations import AsyncOperationHandle, OperationHandle
from weaver.training_client import TrainingClient
from weaver.types.checkpoint import Checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _legacy_checkpoint_capabilities():
    return {"protocol_version": 1, "default_backend": "gpfs", "save_state_backends": ["gpfs"]}


def _make_training_client() -> TrainingClient:
    """Create a TrainingClient with a mocked ServiceClient."""
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    service.http.get.return_value = _legacy_checkpoint_capabilities()
    return TrainingClient(
        service=service,
        model_id="mdl-123",
        base_model="Qwen/Qwen3-8B",
        session_id="sess-abc",
    )


def _make_handle(result: Dict[str, Any] | None = None) -> MagicMock:
    """Create a mock OperationHandle that returns *result*."""
    handle = MagicMock(spec=OperationHandle)
    handle.result.return_value = result
    return handle


def _make_async_training_client() -> AsyncTrainingClient:
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    service.enqueue_operation = AsyncMock()
    service.http.get = AsyncMock(return_value=_legacy_checkpoint_capabilities())
    return AsyncTrainingClient(
        service=service,
        model_id="mdl-123",
        base_model="Qwen/Qwen3-8B",
        session_id="sess-abc",
    )


def _checkpoint_payload(
    checkpoint_id: str,
    *,
    name: str = "step-race",
    status: str = "completed",
) -> Dict[str, Any]:
    return {
        "id": checkpoint_id,
        "path": f"weaver://mdl-123/checkpoints/{name}",
        "name": name,
        "type": "weight",
        "status": status,
    }


# ---------------------------------------------------------------------------
# save_state
# ---------------------------------------------------------------------------


class TestSaveState:
    def test_save_state_returns_checkpoint(self):
        tc = _make_training_client()
        handle = _make_handle(
            {
                "id": "ckpt-1",
                "path": "weaver://mdl-123/checkpoints/after-3-steps",
                "name": "after-3-steps",
                "type": "weight",
            }
        )
        tc._service.enqueue_operation.return_value = handle

        ckpt = tc.save_state(name="after-3-steps")

        assert isinstance(ckpt, Checkpoint)
        assert ckpt.id == "ckpt-1"
        assert ckpt.path == "weaver://mdl-123/checkpoints/after-3-steps"
        assert ckpt.name == "after-3-steps"
        args = tc._service.enqueue_operation.call_args
        assert args[0][0] == "/api/v1/models/mdl-123/checkpoints"
        assert args[0][1] == {"type": "weight", "name": "after-3-steps"}

    def test_save_state_custom_type(self):
        tc = _make_training_client()
        handle = _make_handle(
            {
                "id": "ckpt-2",
                "path": "weaver://mdl-123/checkpoints/ckpt-2",
                "type": "weight_and_optimizer",
            }
        )
        tc._service.enqueue_operation.return_value = handle

        ckpt = tc.save_state(checkpoint_type="weight_and_optimizer")

        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["type"] == "weight_and_optimizer"
        assert "name" not in body
        assert ckpt.checkpoint_type == "weight_and_optimizer"

    def test_save_state_no_name(self):
        tc = _make_training_client()
        handle = _make_handle(
            {
                "id": "ckpt-3",
                "path": "weaver://mdl-123/checkpoints/auto",
                "type": "weight",
            }
        )
        tc._service.enqueue_operation.return_value = handle

        ckpt = tc.save_state()

        body = tc._service.enqueue_operation.call_args[0][1]
        assert "name" not in body
        assert ckpt.path == "weaver://mdl-123/checkpoints/auto"

    def test_save_state_no_wait(self):
        tc = _make_training_client()
        handle = _make_handle()
        tc._service.enqueue_operation.return_value = handle

        result = tc.save_state(name="step-100", wait=False)

        assert result is handle
        handle.result.assert_not_called()

    def test_save_state_recovers_checkpoint_when_operation_projection_races(self):
        tc = _make_training_client()
        tc._service.enqueue_operation.return_value = _make_handle({"saved": True})
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": []},
            {"items": [_checkpoint_payload("ckpt-race")]},
        ]

        checkpoint = tc.save_state(name="step-race")

        assert checkpoint.id == "ckpt-race"
        assert checkpoint.path == "weaver://mdl-123/checkpoints/step-race"

    def test_save_state_ignores_old_checkpoint_regardless_of_listing_order(self):
        tc = _make_training_client()
        tc._service.enqueue_operation.return_value = _make_handle({"saved": True})
        old = _checkpoint_payload("ckpt-old")
        new = _checkpoint_payload("ckpt-new")
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": [old]},
            {"items": [old, new]},
        ]

        checkpoint = tc.save_state(name="step-race")

        assert checkpoint.id == "ckpt-new"

    def test_save_state_rejects_ambiguous_new_checkpoints(self, monkeypatch):
        monkeypatch.setattr(training_client_module, "CHECKPOINT_RECOVERY_DELAYS", (0.0,))
        tc = _make_training_client()
        tc._service.enqueue_operation.return_value = _make_handle({"saved": True})
        old = _checkpoint_payload("ckpt-old")
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": [old]},
            {
                "items": [
                    old,
                    _checkpoint_payload("ckpt-new-1"),
                    _checkpoint_payload("ckpt-new-2"),
                ]
            },
        ]

        with pytest.raises(RuntimeError, match="recovery was ambiguous"):
            tc.save_state(name="step-race")

    def test_save_state_polls_until_new_checkpoint_is_visible(self, monkeypatch):
        monkeypatch.setattr(training_client_module, "CHECKPOINT_RECOVERY_DELAYS", (0.0, 0.0))
        tc = _make_training_client()
        tc._service.enqueue_operation.return_value = _make_handle({"saved": True})
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": []},  # pre-save snapshot
            {"items": []},  # first recovery poll
            {"items": [_checkpoint_payload("ckpt-later")]},
        ]

        checkpoint = tc.save_state(name="step-race")

        assert checkpoint.id == "ckpt-later"
        assert tc._service.http.get.call_count == 4

    def test_save_state_uses_partial_operation_id(self):
        tc = _make_training_client()
        tc._service.enqueue_operation.return_value = _make_handle({"id": "ckpt-new"})
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": [_checkpoint_payload("ckpt-old")]},
            {
                "items": [
                    _checkpoint_payload("ckpt-other"),
                    _checkpoint_payload("ckpt-new"),
                ]
            },
        ]

        checkpoint = tc.save_state(name="step-race")

        assert checkpoint.id == "ckpt-new"

    def test_save_state_never_returns_an_empty_checkpoint(self, monkeypatch):
        monkeypatch.setattr(training_client_module, "CHECKPOINT_RECOVERY_DELAYS", (0.0,))
        tc = _make_training_client()
        tc._service.enqueue_operation.return_value = _make_handle({"saved": True})
        tc._service.http.get.side_effect = lambda path: (
            _legacy_checkpoint_capabilities()
            if path.endswith("/storage-capabilities")
            else {"items": []}
        )

        with pytest.raises(RuntimeError, match="returned no checkpoint metadata"):
            tc.save_state(name="missing")


class TestAsyncSaveState:
    def test_recovers_checkpoint_when_operation_projection_races(self):
        tc = _make_async_training_client()
        handle = MagicMock(spec=AsyncOperationHandle)
        handle.result = AsyncMock(return_value={"saved": True})
        tc._service.enqueue_operation.return_value = handle
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": []},
            {"items": [_checkpoint_payload("ckpt-race")]},
        ]

        checkpoint = asyncio.run(tc.save_state(name="step-race"))

        assert checkpoint.id == "ckpt-race"
        assert checkpoint.path == "weaver://mdl-123/checkpoints/step-race"

    def test_ignores_old_checkpoint_regardless_of_listing_order(self):
        tc = _make_async_training_client()
        handle = MagicMock(spec=AsyncOperationHandle)
        handle.result = AsyncMock(return_value={"saved": True})
        tc._service.enqueue_operation.return_value = handle
        old = _checkpoint_payload("ckpt-old")
        new = _checkpoint_payload("ckpt-new")
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": [old]},
            {"items": [old, new]},
        ]

        checkpoint = asyncio.run(tc.save_state(name="step-race"))

        assert checkpoint.id == "ckpt-new"

    def test_polls_until_new_checkpoint_is_visible(self, monkeypatch):
        monkeypatch.setattr(async_training_client_module, "CHECKPOINT_RECOVERY_DELAYS", (0.0, 0.0))
        tc = _make_async_training_client()
        handle = MagicMock(spec=AsyncOperationHandle)
        handle.result = AsyncMock(return_value={"saved": True})
        tc._service.enqueue_operation.return_value = handle
        tc._service.http.get.side_effect = [
            _legacy_checkpoint_capabilities(),  # New permanent-save preference negotiation.
            {"items": []},
            {"items": []},
            {"items": [_checkpoint_payload("ckpt-later")]},
        ]

        checkpoint = asyncio.run(tc.save_state(name="step-race"))

        assert checkpoint.id == "ckpt-later"
        assert tc._service.http.get.await_count == 4

    def test_recovery_polling_remains_cancellation_responsive(self, monkeypatch):
        monkeypatch.setattr(async_training_client_module, "CHECKPOINT_RECOVERY_DELAYS", (0.0, 60.0))
        tc = _make_async_training_client()
        handle = MagicMock(spec=AsyncOperationHandle)
        handle.result = AsyncMock(return_value={"saved": True})
        tc._service.enqueue_operation.return_value = handle

        async def run():
            second_listing_finished = asyncio.Event()
            listing_count = 0

            async def list_empty(_path):
                nonlocal listing_count
                if _path.endswith("/storage-capabilities"):
                    return _legacy_checkpoint_capabilities()
                listing_count += 1
                if listing_count == 2:
                    second_listing_finished.set()
                return {"items": []}

            tc._service.http.get.side_effect = list_empty
            task = asyncio.create_task(tc.save_state(name="step-race"))
            await second_listing_finished.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

    def test_never_returns_an_empty_checkpoint(self, monkeypatch):
        monkeypatch.setattr(async_training_client_module, "CHECKPOINT_RECOVERY_DELAYS", (0.0,))
        tc = _make_async_training_client()
        handle = MagicMock(spec=AsyncOperationHandle)
        handle.result = AsyncMock(return_value={"saved": True})
        tc._service.enqueue_operation.return_value = handle
        tc._service.http.get.side_effect = lambda path: (
            _legacy_checkpoint_capabilities()
            if path.endswith("/storage-capabilities")
            else {"items": []}
        )

        with pytest.raises(RuntimeError, match="returned no checkpoint metadata"):
            asyncio.run(tc.save_state(name="missing"))


# ---------------------------------------------------------------------------
# load_state
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_load_state_with_checkpoint_object(self):
        tc = _make_training_client()
        handle = _make_handle({"status": "done"})
        tc._service.enqueue_operation.return_value = handle

        ckpt = Checkpoint(id="ckpt-1", path="weaver://mdl-123/checkpoints/step-3")
        result = tc.load_state(ckpt)

        assert result == {"status": "done"}
        args = tc._service.enqueue_operation.call_args
        assert args[0][0] == "/api/v1/models/mdl-123/load"
        body = args[0][1]
        assert body["path"] == "weaver://mdl-123/checkpoints/step-3"
        assert body["include_optimizer"] is False

    def test_load_state_with_string_path(self):
        tc = _make_training_client()
        handle = _make_handle({"status": "done"})
        tc._service.enqueue_operation.return_value = handle

        result = tc.load_state("weaver://mdl-123/checkpoints/step-3")

        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["path"] == "weaver://mdl-123/checkpoints/step-3"
        assert body["include_optimizer"] is False

    def test_load_state_no_wait(self):
        tc = _make_training_client()
        handle = _make_handle()
        tc._service.enqueue_operation.return_value = handle

        result = tc.load_state("weaver://ckpt-path", wait=False)

        assert result is handle
        handle.result.assert_not_called()


# ---------------------------------------------------------------------------
# load_state_with_optimizer
# ---------------------------------------------------------------------------


class TestLoadStateWithOptimizer:
    def test_load_state_with_optimizer_wait(self):
        tc = _make_training_client()
        handle = _make_handle({"status": "done"})
        tc._service.enqueue_operation.return_value = handle

        ckpt = Checkpoint(id="ckpt-1", path="weaver://mdl-123/checkpoints/step-3")
        result = tc.load_state_with_optimizer(ckpt)

        assert result == {"status": "done"}
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["path"] == "weaver://mdl-123/checkpoints/step-3"
        assert body["include_optimizer"] is True

    def test_load_state_with_optimizer_no_wait(self):
        tc = _make_training_client()
        handle = _make_handle()
        tc._service.enqueue_operation.return_value = handle

        result = tc.load_state_with_optimizer("weaver://ckpt-path", wait=False)

        assert result is handle
        handle.result.assert_not_called()


# ---------------------------------------------------------------------------
# list_checkpoints
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    def test_list_checkpoints_returns_typed_list(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = {
            "items": [
                {
                    "id": "ckpt-1",
                    "path": "weaver://ckpt-1",
                    "type": "weight",
                    "status": "completed",
                },
                {
                    "id": "ckpt-2",
                    "path": "weaver://ckpt-2",
                    "type": "weight_and_optimizer",
                },
            ]
        }

        checkpoints = tc.list_checkpoints()

        assert len(checkpoints) == 2
        assert isinstance(checkpoints[0], Checkpoint)
        assert checkpoints[0].id == "ckpt-1"
        assert checkpoints[0].path == "weaver://ckpt-1"
        assert checkpoints[1].checkpoint_type == "weight_and_optimizer"
        tc._service.http.get.assert_called_once_with(
            "/api/v1/models/mdl-123/checkpoints",
        )

    def test_list_checkpoints_empty(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = {"items": []}
        assert tc.list_checkpoints() == []

    def test_list_checkpoints_none_response(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = None
        assert tc.list_checkpoints() == []

    def test_list_checkpoints_with_training_flags(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = {
            "items": [
                {
                    "id": "ckpt-1",
                    "path": "weaver://ckpt-1",
                    "type": "weight",
                    "train_unembed": True,
                    "train_mlp": False,
                    "train_attn": True,
                },
                {
                    "id": "ckpt-2",
                    "path": "weaver://ckpt-2",
                    "type": "weight",
                },
            ]
        }

        checkpoints = tc.list_checkpoints()

        assert checkpoints[0].train_unembed is True
        assert checkpoints[0].train_mlp is False
        assert checkpoints[0].train_attn is True
        assert checkpoints[1].train_unembed is None
        assert checkpoints[1].train_mlp is None
        assert checkpoints[1].train_attn is None


# ---------------------------------------------------------------------------
# Checkpoint type
# ---------------------------------------------------------------------------


class TestCheckpointType:
    def test_from_payload(self):
        payload = {
            "id": "ckpt-x",
            "path": "weaver://ckpt-x",
            "type": "weight",
            "status": "completed",
        }
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.id == "ckpt-x"
        assert ckpt.path == "weaver://ckpt-x"
        assert ckpt.checkpoint_type == "weight"
        assert ckpt.status == "completed"

    def test_from_payload_minimal(self):
        payload = {"id": "ckpt-y", "path": "weaver://ckpt-y"}
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.id == "ckpt-y"
        assert ckpt.name is None
        assert ckpt.checkpoint_type == "weight"
        assert ckpt.status is None
        assert ckpt.train_unembed is None
        assert ckpt.train_mlp is None
        assert ckpt.train_attn is None

    def test_from_payload_with_training_flags(self):
        payload = {
            "id": "ckpt-z",
            "path": "weaver://ckpt-z",
            "type": "weight",
            "train_unembed": True,
            "train_mlp": False,
            "train_attn": True,
        }
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.train_unembed is True
        assert ckpt.train_mlp is False
        assert ckpt.train_attn is True

    def test_from_payload_with_partial_training_flags(self):
        payload = {
            "id": "ckpt-w",
            "path": "weaver://ckpt-w",
            "train_attn": True,
        }
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.train_unembed is None
        assert ckpt.train_mlp is None
        assert ckpt.train_attn is True

    def test_checkpoint_is_frozen(self):
        ckpt = Checkpoint(id="1", path="p")
        with pytest.raises(AttributeError):
            ckpt.id = "2"  # type: ignore[misc]

    def test_from_payload_with_ttl_fields(self):
        payload = {
            "id": "ckpt-t",
            "path": "weaver://ckpt-t",
            "ttl_seconds": 86400,
            "created_at": "2026-04-14T00:00:00Z",
            "expires_at": "2026-04-15T00:00:00Z",
        }
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.ttl_seconds == 86400
        assert ckpt.created_at == "2026-04-14T00:00:00Z"
        assert ckpt.expires_at == "2026-04-15T00:00:00Z"

    def test_from_payload_without_ttl_fields(self):
        payload = {"id": "ckpt-n", "path": "weaver://ckpt-n"}
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.ttl_seconds is None
        assert ckpt.created_at is None
        assert ckpt.expires_at is None


# ---------------------------------------------------------------------------
# save_state TTL
# ---------------------------------------------------------------------------


class TestSaveStateTTL:
    def test_default_no_ttl_in_body(self):
        tc = _make_training_client()
        handle = _make_handle({"id": "ckpt-1", "path": "weaver://ckpt-1"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_state(name="test")
        body = tc._service.enqueue_operation.call_args[0][1]
        assert "ttl_seconds" not in body

    def test_explicit_none_sends_null(self):
        tc = _make_training_client()
        handle = _make_handle({"id": "ckpt-1", "path": "weaver://ckpt-1"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_state(name="test", ttl_seconds=None)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert "ttl_seconds" in body
        assert body["ttl_seconds"] is None

    def test_with_ttl_seconds(self):
        tc = _make_training_client()
        handle = _make_handle({"id": "ckpt-1", "path": "weaver://ckpt-1"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_state(name="test", ttl_seconds=3600)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["ttl_seconds"] == 3600

    def test_sampling_type_defaults_to_sampler_ttl(self):
        # A sampling checkpoint saved without an explicit TTL gets the default
        # sampler TTL so regenerable exports don't accumulate. Weight types keep
        # their permanent default (test_default_no_ttl_in_body).
        tc = _make_training_client()
        handle = _make_handle({"id": "ckpt-s", "path": "weaver://ckpt-s"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_state(name="test", checkpoint_type="sampling")
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["ttl_seconds"] == DEFAULT_SAMPLER_TTL_SECONDS

    def test_sampling_type_explicit_none_stays_permanent(self):
        tc = _make_training_client()
        handle = _make_handle({"id": "ckpt-s", "path": "weaver://ckpt-s"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_state(name="test", checkpoint_type="sampling", ttl_seconds=None)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert "ttl_seconds" in body
        assert body["ttl_seconds"] is None

    def test_sampling_type_explicit_ttl_wins(self):
        tc = _make_training_client()
        handle = _make_handle({"id": "ckpt-s", "path": "weaver://ckpt-s"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_state(name="test", checkpoint_type="sampling", ttl_seconds=3600)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["ttl_seconds"] == 3600


# ---------------------------------------------------------------------------
# save_weights_for_sampler TTL
# ---------------------------------------------------------------------------


class TestSaveWeightsForSamplerTTL:
    def test_default_is_sampler_ttl(self):
        tc = _make_training_client()
        handle = _make_handle({"model_path": "weaver://path"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_weights_for_sampler(name="test")
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["ttl_seconds"] == DEFAULT_SAMPLER_TTL_SECONDS

    def test_explicit_none_no_ttl_in_body(self):
        tc = _make_training_client()
        handle = _make_handle({"model_path": "weaver://path"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_weights_for_sampler(name="test", ttl_seconds=None)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert "ttl_seconds" not in body

    def test_with_ttl_seconds(self):
        tc = _make_training_client()
        handle = _make_handle({"model_path": "weaver://path"})
        tc._service.enqueue_operation.return_value = handle
        tc.save_weights_for_sampler(name="test", ttl_seconds=7200)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["ttl_seconds"] == 7200


# ---------------------------------------------------------------------------
# save_weights_and_get_sampling_client TTL
# ---------------------------------------------------------------------------


class TestSaveWeightsAndGetSamplingClientTTL:
    def test_default_is_sampler_ttl(self):
        tc = _make_training_client()
        handle = _make_handle({"model_path": "weaver://path", "sampling_session_id": "ss-1"})
        tc._service.enqueue_operation.return_value = handle
        tc._service.get_sampling_client.return_value = MagicMock()
        tc.save_weights_and_get_sampling_client()
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["ttl_seconds"] == DEFAULT_SAMPLER_TTL_SECONDS

    def test_explicit_none_no_ttl_in_body(self):
        tc = _make_training_client()
        handle = _make_handle({"model_path": "weaver://path", "sampling_session_id": "ss-1"})
        tc._service.enqueue_operation.return_value = handle
        tc._service.get_sampling_client.return_value = MagicMock()
        tc.save_weights_and_get_sampling_client(ttl_seconds=None)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert "ttl_seconds" not in body

    def test_custom_ttl(self):
        tc = _make_training_client()
        handle = _make_handle({"model_path": "weaver://path", "sampling_session_id": "ss-1"})
        tc._service.enqueue_operation.return_value = handle
        tc._service.get_sampling_client.return_value = MagicMock()
        tc.save_weights_and_get_sampling_client(ttl_seconds=7200)
        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["ttl_seconds"] == 7200


# ---------------------------------------------------------------------------
# set_checkpoint_ttl
# ---------------------------------------------------------------------------


class TestSetCheckpointTTL:
    def test_set_ttl_with_int(self):
        tc = _make_training_client()
        tc._service.http.patch.return_value = {"status": "ok"}
        result = tc.set_checkpoint_ttl("weaver://ckpt-1", ttl_seconds=604800)
        tc._service.http.patch.assert_called_once_with(
            "/api/v1/models/mdl-123/checkpoints/ttl",
            json={"path": "weaver://ckpt-1", "ttl_seconds": 604800},
        )
        assert result == {"status": "ok"}

    def test_set_ttl_none_cancels_expiration(self):
        tc = _make_training_client()
        tc._service.http.patch.return_value = {"status": "ok"}
        tc.set_checkpoint_ttl("weaver://ckpt-1", ttl_seconds=None)
        body = tc._service.http.patch.call_args[1]["json"]
        assert body["ttl_seconds"] is None

    def test_set_ttl_with_checkpoint_object(self):
        tc = _make_training_client()
        tc._service.http.patch.return_value = {"status": "ok"}
        ckpt = Checkpoint(id="ckpt-1", path="weaver://ckpt-1")
        tc.set_checkpoint_ttl(ckpt, ttl_seconds=3600)
        body = tc._service.http.patch.call_args[1]["json"]
        assert body["path"] == "weaver://ckpt-1"
        assert body["ttl_seconds"] == 3600


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("backend", [None, "gpfs", "artifact"])
def test_checkpoint_backend_selection_compatible_defaults(asynchronous, backend):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    client._service.http.get.return_value = {
        "protocol_version": 1,
        "default_backend": "gpfs",
        "save_state_backends": ["gpfs", "artifact"],
    }
    if asynchronous:
        asyncio.run(client.save_state(name="mixed", storage_backend=backend, wait=False))
    else:
        client.save_state(name="mixed", storage_backend=backend, wait=False)
    args = client._service.enqueue_operation.call_args[0]
    assert args[0] == "/api/v1/models/mdl-123/checkpoints" + (
        "/managed" if backend == "artifact" else ""
    )
    body = args[1]
    assert body == (
        {"name": "mixed", "type": "weight"}
        if backend is None
        else {"name": "mixed", "type": "weight", "storage_backend": backend}
    )
    if backend in (None, "artifact"):
        client._service.http.get.assert_called_once_with(
            "/api/v1/models/mdl-123/storage-capabilities"
        )
    else:
        client._service.http.get.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "capabilities",
    [
        None,
        {},
        {"protocol_version": True, "default_backend": "gpfs", "save_state_backends": ["artifact"]},
        {"protocol_version": 1, "default_backend": "gpfs", "save_state_backends": ["gpfs"]},
        "old server response",
    ],
)
def test_managed_checkpoint_requires_explicit_capability(asynchronous, capabilities):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    client._service.http.get.return_value = capabilities
    with pytest.raises(RuntimeError, match="managed checkpoint saves"):
        if asynchronous:
            asyncio.run(client.save_state(name="managed", storage_backend="artifact", wait=False))
        else:
            client.save_state(name="managed", storage_backend="artifact", wait=False)
    client._service.enqueue_operation.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_old_server_rejects_explicit_managed_before_post(asynchronous):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    client._service.http.get.side_effect = WeaverAPIError(404, "not_found", "old server", False)
    with pytest.raises(RuntimeError, match="does not support"):
        if asynchronous:
            asyncio.run(client.save_state(name="managed", storage_backend="artifact", wait=False))
        else:
            client.save_state(name="managed", storage_backend="artifact", wait=False)
    client._service.enqueue_operation.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_storage_rollback_rechecks_capability_each_save(asynchronous):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    client._service.http.get.side_effect = [
        {
            "protocol_version": 1,
            "default_backend": "gpfs",
            "save_state_backends": ["gpfs", "artifact"],
        },
        {"protocol_version": 1, "default_backend": "gpfs", "save_state_backends": ["gpfs"]},
    ]

    def save(name):
        if asynchronous:
            return asyncio.run(client.save_state(name=name, storage_backend="artifact", wait=False))
        return client.save_state(name=name, storage_backend="artifact", wait=False)

    save("before-rollback")
    with pytest.raises(RuntimeError, match="does not currently allow"):
        save("after-rollback")
    assert client._service.enqueue_operation.call_count == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_invalid_checkpoint_backend_has_no_io(asynchronous):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    with pytest.raises(ValueError, match="storage_backend"):
        if asynchronous:
            asyncio.run(client.save_state(storage_backend="unknown", wait=False))
        else:
            client.save_state(storage_backend="unknown", wait=False)
    client._service.http.get.assert_not_called()
    client._service.enqueue_operation.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_new_capability_old_writer_rollout_cannot_silently_write_gpfs(asynchronous):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    client._service.http.get.return_value = {
        "protocol_version": 1,
        "default_backend": "gpfs",
        "save_state_backends": ["gpfs", "artifact"],
    }
    # An old server has no managed route. It must fail before accepting a write,
    # even if capability discovery happened on a different, upgraded server.
    client._service.enqueue_operation.side_effect = WeaverAPIError(
        404, "not_found", "old writer", False
    )
    with pytest.raises(WeaverAPIError):
        if asynchronous:
            asyncio.run(
                client.save_state(name="mixed-rollout", storage_backend="artifact", wait=False)
            )
        else:
            client.save_state(name="mixed-rollout", storage_backend="artifact", wait=False)
    assert client._service.enqueue_operation.call_args[0][0].endswith("/checkpoints/managed")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        "preferred",
        "legacy_response",
        "rollback",
        "explicit_gpfs",
        "ttl",
        "sampling_ttl",
        "sampling_permanent",
        "bad_preference",
        "unsupported",
        "old_server",
        "denied",
        "null_response",
        "empty_response",
    ],
)
def test_permanent_checkpoint_preference_negotiation_and_legacy_compatibility(
    asynchronous, selection
):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    caps = {
        "protocol_version": 1,
        "default_backend": "gpfs",
        "save_state_backends": ["gpfs", "artifact"],
        "preferred_permanent_checkpoint_backend": "artifact",
    }
    kwargs = {"wait": False}
    if selection == "legacy_response":
        caps.pop("preferred_permanent_checkpoint_backend")
    if selection == "rollback":
        caps["preferred_permanent_checkpoint_backend"] = "gpfs"
    if selection == "explicit_gpfs":
        kwargs["storage_backend"] = "gpfs"
    if selection == "ttl":
        kwargs["ttl_seconds"] = 3600
    if selection == "sampling_ttl":
        kwargs["checkpoint_type"] = "sampling"
    if selection == "sampling_permanent":
        kwargs.update(checkpoint_type="sampling", ttl_seconds=None)
    if selection == "bad_preference":
        caps["preferred_permanent_checkpoint_backend"] = True
    if selection == "unsupported":
        caps["save_state_backends"] = ["gpfs"]
    client._service.http.get.return_value = (
        None if selection == "null_response" else {} if selection == "empty_response" else caps
    )
    if selection == "old_server":
        client._service.http.get.side_effect = WeaverAPIError(404, "not_found", "old server", False)
    if selection == "denied":
        client._service.http.get.side_effect = WeaverAPIError(403, "forbidden", "denied", False)

    def save():
        return (
            asyncio.run(client.save_state(**kwargs))
            if asynchronous
            else client.save_state(**kwargs)
        )

    if selection in ("bad_preference", "unsupported", "denied", "null_response", "empty_response"):
        with pytest.raises(WeaverAPIError if selection == "denied" else RuntimeError):
            save()
        client._service.enqueue_operation.assert_not_called()
        return
    save()
    path, body = client._service.enqueue_operation.call_args[0]
    managed = selection in ("preferred", "sampling_permanent")
    assert path.endswith("/checkpoints/managed" if managed else "/checkpoints")
    assert body.get("storage_backend") == (
        "artifact" if managed else "gpfs" if selection == "explicit_gpfs" else None
    )
    if selection == "sampling_permanent":
        assert body["ttl_seconds"] is None
    if selection in ("ttl", "sampling_ttl"):
        assert body["ttl_seconds"] == DEFAULT_SAMPLER_TTL_SECONDS
    assert client._service.http.get.call_count == (
        0 if selection in ("explicit_gpfs", "ttl", "sampling_ttl") else 1
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_checkpoint_preference_refreshes_after_rollback_without_changing_previous_request(
    asynchronous,
):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    base = {
        "protocol_version": 1,
        "default_backend": "gpfs",
        "save_state_backends": ["gpfs", "artifact"],
    }
    client._service.http.get.side_effect = [
        {**base, "preferred_permanent_checkpoint_backend": "artifact"},
        {**base, "preferred_permanent_checkpoint_backend": "gpfs"},
    ]
    for _ in range(2):
        if asynchronous:
            asyncio.run(client.save_state(wait=False))
        else:
            client.save_state(wait=False)
    calls = client._service.enqueue_operation.call_args_list
    assert (
        calls[0].args[0].endswith("/managed") and calls[0].args[1]["storage_backend"] == "artifact"
    )
    assert not calls[1].args[0].endswith("/managed") and "storage_backend" not in calls[1].args[1]
    assert client._service.http.get.call_count == 2
