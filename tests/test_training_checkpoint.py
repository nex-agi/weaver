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
from unittest.mock import MagicMock, patch

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
    def test_save_state_wait_returns_path(self):
        tc = _make_training_client()
        handle = _make_handle({"path": "weaver://run/weights/ckpt-001"})
        tc._service.enqueue_operation.return_value = handle

        path = tc.save_state(name="iter-100")

        assert path == "weaver://run/weights/ckpt-001"
        tc._service.enqueue_operation.assert_called_once()
        args = tc._service.enqueue_operation.call_args
        assert args[0][0] == "/api/v1/models/mdl-123/checkpoints"
        body = args[0][1]
        assert body["name"] == "iter-100"
        assert body["checkpoint_type"] == "training"

    def test_save_state_custom_checkpoint_type(self):
        tc = _make_training_client()
        handle = _make_handle({"path": "weaver://run/weights/ckpt-002"})
        tc._service.enqueue_operation.return_value = handle

        path = tc.save_state(checkpoint_type="training_with_optimizer")

        body = tc._service.enqueue_operation.call_args[0][1]
        assert body["checkpoint_type"] == "training_with_optimizer"
        assert "name" not in body
        assert path == "weaver://run/weights/ckpt-002"

    def test_save_state_no_wait_returns_handle(self):
        tc = _make_training_client()
        handle = _make_handle()
        tc._service.enqueue_operation.return_value = handle

        result = tc.save_state(wait=False)

        assert result is handle
        handle.result.assert_not_called()

    def test_save_state_missing_path_raises(self):
        tc = _make_training_client()
        handle = _make_handle({})
        tc._service.enqueue_operation.return_value = handle

        with pytest.raises(RuntimeError, match="missing path"):
            tc.save_state()


# ---------------------------------------------------------------------------
# load_state
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_load_state_wait(self):
        tc = _make_training_client()
        handle = _make_handle({"status": "done"})
        tc._service.enqueue_operation.return_value = handle

        result = tc.load_state("weaver://run/weights/ckpt-001")

        assert result == {"status": "done"}
        args = tc._service.enqueue_operation.call_args
        assert args[0][0] == "/api/v1/models/mdl-123/load"
        body = args[0][1]
        assert body["path"] == "weaver://run/weights/ckpt-001"
        assert body["include_optimizer"] is False

    def test_load_state_no_wait(self):
        tc = _make_training_client()
        handle = _make_handle()
        tc._service.enqueue_operation.return_value = handle

        result = tc.load_state("weaver://run/weights/ckpt-001", wait=False)

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

        result = tc.load_state_with_optimizer("weaver://run/weights/ckpt-001")

        assert result == {"status": "done"}
        args = tc._service.enqueue_operation.call_args
        body = args[0][1]
        assert body["path"] == "weaver://run/weights/ckpt-001"
        assert body["include_optimizer"] is True

    def test_load_state_with_optimizer_no_wait(self):
        tc = _make_training_client()
        handle = _make_handle()
        tc._service.enqueue_operation.return_value = handle

        result = tc.load_state_with_optimizer("weaver://run/weights/ckpt-001", wait=False)

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
                    "path": "weaver://run/weights/ckpt-1",
                    "name": "iter-100",
                    "checkpoint_type": "training",
                    "status": "completed",
                },
                {
                    "id": "ckpt-2",
                    "path": "weaver://run/weights/ckpt-2",
                    "name": None,
                    "checkpoint_type": "training_with_optimizer",
                    "status": "completed",
                },
            ]
        }

        checkpoints = tc.list_checkpoints()

        assert len(checkpoints) == 2
        assert isinstance(checkpoints[0], Checkpoint)
        assert checkpoints[0].id == "ckpt-1"
        assert checkpoints[0].path == "weaver://run/weights/ckpt-1"
        assert checkpoints[0].name == "iter-100"
        assert checkpoints[1].checkpoint_type == "training_with_optimizer"
        tc._service.http.get.assert_called_once_with(
            "/api/v1/models/mdl-123/checkpoints",
        )

    def test_list_checkpoints_empty(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = {"items": []}

        checkpoints = tc.list_checkpoints()

        assert checkpoints == []

    def test_list_checkpoints_none_response(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = None

        checkpoints = tc.list_checkpoints()

        assert checkpoints == []


# ---------------------------------------------------------------------------
# Checkpoint type
# ---------------------------------------------------------------------------


class TestCheckpointType:
    def test_from_payload(self):
        payload = {
            "id": "ckpt-x",
            "path": "weaver://run/weights/ckpt-x",
            "name": "my-ckpt",
            "checkpoint_type": "training",
            "status": "completed",
        }
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.id == "ckpt-x"
        assert ckpt.path == "weaver://run/weights/ckpt-x"
        assert ckpt.name == "my-ckpt"
        assert ckpt.checkpoint_type == "training"
        assert ckpt.status == "completed"

    def test_from_payload_minimal(self):
        payload = {"id": "ckpt-y", "path": "weaver://run/weights/ckpt-y"}
        ckpt = Checkpoint.from_payload(payload)
        assert ckpt.id == "ckpt-y"
        assert ckpt.name is None
        assert ckpt.checkpoint_type == "training"
        assert ckpt.status is None

    def test_checkpoint_is_frozen(self):
        ckpt = Checkpoint(id="1", path="p")
        with pytest.raises(AttributeError):
            ckpt.id = "2"  # type: ignore[misc]
