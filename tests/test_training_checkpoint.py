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

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from weaver.operations import OperationHandle
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


# ---------------------------------------------------------------------------
# save_state
# ---------------------------------------------------------------------------


class TestSaveState:
    def test_save_state_returns_checkpoint(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = {
            "checkpoint": {
                "id": "ckpt-1",
                "path": "weaver://mdl-123/checkpoints/after-3-steps",
                "name": "after-3-steps",
                "type": "weight",
                "status": "pending",
            },
            "operation": {
                "id": "op-1",
                "Status": "done",
            },
        }

        ckpt = tc.save_state(name="after-3-steps")

        assert isinstance(ckpt, Checkpoint)
        assert ckpt.id == "ckpt-1"
        assert ckpt.path == "weaver://mdl-123/checkpoints/after-3-steps"
        assert ckpt.name == "after-3-steps"
        tc._service.http.post.assert_called_once_with(
            "/api/v1/models/mdl-123/checkpoints",
            json={"type": "weight", "name": "after-3-steps"},
        )

    def test_save_state_custom_type(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = {
            "checkpoint": {
                "id": "ckpt-2",
                "path": "weaver://mdl-123/checkpoints/ckpt-2",
                "type": "weight_and_optimizer",
            },
            "operation": {"id": "op-2", "Status": "done"},
        }

        ckpt = tc.save_state(checkpoint_type="weight_and_optimizer")

        body = tc._service.http.post.call_args[1]["json"]
        assert body["type"] == "weight_and_optimizer"
        assert "name" not in body
        assert ckpt.checkpoint_type == "weight_and_optimizer"

    def test_save_state_no_name(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = {
            "checkpoint": {
                "id": "ckpt-3",
                "path": "weaver://mdl-123/checkpoints/auto",
                "type": "weight",
            },
            "operation": {"id": "op-3", "Status": "done"},
        }

        ckpt = tc.save_state()

        body = tc._service.http.post.call_args[1]["json"]
        assert "name" not in body
        assert ckpt.path == "weaver://mdl-123/checkpoints/auto"

    def test_save_state_no_wait(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = {
            "checkpoint": {"id": "ckpt-4", "path": "weaver://p", "type": "weight"},
            "operation": {"id": "op-4", "Status": "pending"},
        }

        result = tc.save_state(name="step-100", wait=False)

        assert isinstance(result, OperationHandle)
        assert result.operation_id == "op-4"


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

    def test_checkpoint_is_frozen(self):
        ckpt = Checkpoint(id="1", path="p")
        with pytest.raises(AttributeError):
            ckpt.id = "2"  # type: ignore[misc]
