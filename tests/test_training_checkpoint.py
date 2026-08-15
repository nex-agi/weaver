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

from weaver._utils import DEFAULT_SAMPLER_TTL_SECONDS
from weaver.async_training_client import AsyncTrainingClient
from weaver.operations import AsyncOperationHandle, OperationHandle
from weaver.training_client import TrainingClient
from weaver.types.checkpoint import Checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_handle(result: Dict[str, Any] | None = None) -> MagicMock:
    """Create a mock OperationHandle that returns *result*."""
    handle = MagicMock(spec=OperationHandle)
    handle.result.return_value = result
    return handle


def _make_async_training_client() -> AsyncTrainingClient:
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    service.enqueue_operation = AsyncMock()
    service.http.get = AsyncMock()
    return AsyncTrainingClient(
        service=service,
        model_id="mdl-123",
        base_model="Qwen/Qwen3-8B",
        session_id="sess-abc",
    )


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
        tc._service.http.get.return_value = {
            "items": [
                {
                    "id": "ckpt-race",
                    "path": "weaver://mdl-123/checkpoints/step-race",
                    "name": "step-race",
                    "type": "weight",
                    "status": "completed",
                }
            ]
        }

        checkpoint = tc.save_state(name="step-race")

        assert checkpoint.id == "ckpt-race"
        assert checkpoint.path == "weaver://mdl-123/checkpoints/step-race"

    def test_save_state_never_returns_an_empty_checkpoint(self):
        tc = _make_training_client()
        tc._service.enqueue_operation.return_value = _make_handle({"saved": True})
        tc._service.http.get.return_value = {"items": []}

        with pytest.raises(RuntimeError, match="returned no checkpoint metadata"):
            tc.save_state(name="missing")


class TestAsyncSaveState:
    def test_recovers_checkpoint_when_operation_projection_races(self):
        tc = _make_async_training_client()
        handle = MagicMock(spec=AsyncOperationHandle)
        handle.result = AsyncMock(return_value={"saved": True})
        tc._service.enqueue_operation.return_value = handle
        tc._service.http.get.return_value = {
            "items": [
                {
                    "id": "ckpt-race",
                    "path": "weaver://mdl-123/checkpoints/step-race",
                    "name": "step-race",
                    "type": "weight",
                    "status": "completed",
                }
            ]
        }

        checkpoint = asyncio.run(tc.save_state(name="step-race"))

        assert checkpoint.id == "ckpt-race"
        assert checkpoint.path == "weaver://mdl-123/checkpoints/step-race"

    def test_never_returns_an_empty_checkpoint(self):
        tc = _make_async_training_client()
        handle = MagicMock(spec=AsyncOperationHandle)
        handle.result = AsyncMock(return_value={"saved": True})
        tc._service.enqueue_operation.return_value = handle
        tc._service.http.get.return_value = {"items": []}

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
