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

"""Tests for training request metadata passthrough.

Validates that TrainingClient.forward() and forward_backward() pass metadata
as a top-level payload key (not inside datum or loss_fn_inputs), and that it
is omitted when not provided.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from weaver.service_client import ServiceClient
from weaver.training_client import TrainingClient
from weaver.types.datum import Datum
from weaver.types.model_input import ModelInput, ModelInputChunk


def _make_training_client() -> TrainingClient:
    """Create a TrainingClient with a mock service."""
    service = ServiceClient()
    return TrainingClient(
        service=service,
        model_id="model-moe-001",
        base_model="Qwen/Qwen2.5-MoE-A3B",
        session_id="session-rfc0001",
    )


def _make_datum() -> Datum:
    """Create a minimal Datum for testing."""
    model_input = ModelInput(chunks=[ModelInputChunk(type="encoded_text", tokens=[10, 20, 30])])
    return Datum.from_raw(
        model_input=model_input,
        loss_fn_inputs={"target_tokens": [10, 20, 30]},
    )


def _generic_metadata() -> Dict[str, Any]:
    return {"trace_id": "abc", "priority": 3}


def _router_replay_payload() -> Dict[str, Any]:
    """Sample datum-level router_replay metadata contract."""
    return {
        "schema": "weaver.router_replay.datum.v1",
        "mode": "R3",
        "source": "rollout",
        "action": "REPLAY",
        "sample_ref": "weaver://model-a/router-replay/set-1/samples/7",
        "index_set_uri": "weaver://model-a/router-replay/set-1",
        "manifest_uri": "weaver://model-a/router-replay/set-1/manifest.json",
        "fail_fast": True,
    }


class TestForwardMetadata:
    """Test metadata parameter on TrainingClient.forward()."""

    def test_metadata_included_in_payload(self):
        """Metadata appears as top-level 'metadata' key in payload."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            # Return a mock handle whose .result() returns {}
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        meta = _generic_metadata()
        client.forward([_make_datum()], "grpo", metadata=meta)

        assert len(captured_payloads) == 1
        inner_payload = captured_payloads[0]["payload"]
        # metadata is at top level, not inside forward_input
        assert "metadata" in inner_payload
        assert inner_payload["metadata"] == meta
        assert "metadata" not in inner_payload["forward_input"]

    def test_metadata_not_in_datum(self):
        """Metadata does not leak into datum serialization."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        meta = _generic_metadata()
        client.forward([_make_datum()], "grpo", metadata=meta)

        inner_payload = captured_payloads[0]["payload"]
        data_items = inner_payload["forward_input"]["data"]
        for datum_payload in data_items:
            assert "metadata" not in datum_payload
            assert "router_replay" not in datum_payload.get("loss_fn_inputs", {})

    def test_no_metadata_when_none(self):
        """When metadata is None, payload has no 'metadata' key."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        client.forward([_make_datum()], "cross_entropy")

        inner_payload = captured_payloads[0]["payload"]
        assert "metadata" not in inner_payload

    def test_no_metadata_when_empty_dict(self):
        """When metadata is empty dict, payload has no 'metadata' key (falsy)."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        client.forward([_make_datum()], "cross_entropy", metadata={})

        inner_payload = captured_payloads[0]["payload"]
        assert "metadata" not in inner_payload


class TestForwardBackwardMetadata:
    """Test metadata parameter on TrainingClient.forward_backward()."""

    def test_metadata_included_in_payload(self):
        """Metadata appears as top-level 'metadata' key in payload."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        meta = _generic_metadata()
        client.forward_backward([_make_datum()], "grpo", metadata=meta)

        assert len(captured_payloads) == 1
        inner_payload = captured_payloads[0]["payload"]
        assert "metadata" in inner_payload
        assert inner_payload["metadata"] == meta
        assert "metadata" not in inner_payload["forward_backward_input"]

    def test_metadata_not_in_datum(self):
        """Metadata does not leak into datum serialization."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        meta = _generic_metadata()
        client.forward_backward([_make_datum()], "grpo", metadata=meta)

        inner_payload = captured_payloads[0]["payload"]
        data_items = inner_payload["forward_backward_input"]["data"]
        for datum_payload in data_items:
            assert "metadata" not in datum_payload
            assert "router_replay" not in datum_payload.get("loss_fn_inputs", {})

    def test_no_metadata_when_none(self):
        """When metadata is None, payload has no 'metadata' key."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        client.forward_backward([_make_datum()], "cross_entropy")

        inner_payload = captured_payloads[0]["payload"]
        assert "metadata" not in inner_payload

    def test_metadata_with_loss_fn_config(self):
        """Metadata and loss_fn_config coexist without interference."""
        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        meta = _generic_metadata()
        config = {"temperature": 0.7, "clip_ratio": 0.2}
        client.forward_backward([_make_datum()], "grpo", loss_fn_config=config, metadata=meta)

        inner_payload = captured_payloads[0]["payload"]
        # Both are present and independent
        assert inner_payload["metadata"] == meta
        assert inner_payload["forward_backward_input"]["loss_fn_config"] == config
        # metadata is not inside forward_backward_input
        assert "metadata" not in inner_payload["forward_backward_input"]


class TestRouterReplayMetadataType:
    """Test datum-level router replay serialization and request-level rejection."""

    def test_typed_datum_metadata_serialization(self):
        """RouterReplayMetadata.to_payload() can be attached to a Datum."""
        from weaver.types.router_replay import (
            RouterReplayMetadata,
            router_replay_manifest_uri,
            router_replay_sample_uri,
            router_replay_set_uri,
        )

        replay = RouterReplayMetadata.r3_replay(
            sample_ref=router_replay_sample_uri("model-a", "set-1", 7),
            index_set_uri=router_replay_set_uri("model-a", "set-1"),
            manifest_uri=router_replay_manifest_uri("model-a", "set-1"),
        )

        client = _make_training_client()
        captured_payloads = []

        def mock_enqueue(path, payload):
            captured_payloads.append(payload)
            handle = MagicMock()
            handle.result.return_value = {}
            return handle

        client._service.enqueue_operation = mock_enqueue

        datum = _make_datum()
        datum.metadata["router_replay"] = replay.to_payload()

        client.forward_backward([datum], "grpo")

        inner_payload = captured_payloads[0]["payload"]
        assert "metadata" not in inner_payload
        rr = inner_payload["forward_backward_input"]["data"][0]["metadata"]["router_replay"]
        assert rr["mode"] == "R3"
        assert rr["source"] == "rollout"
        assert rr["sample_ref"] == "weaver://model-a/router-replay/set-1/samples/7"
        assert "indices" not in rr
        assert rr["fail_fast"] is True

    def test_forward_rejects_router_replay_argument(self):
        """Request-level router_replay is no longer accepted."""
        from weaver.types.router_replay import RouterReplayMetadata

        client = _make_training_client()

        with pytest.raises(ValueError, match="datum.metadata"):
            client.forward(
                [_make_datum()],
                "forward_logprob",
                router_replay=RouterReplayMetadata.r2_record(),
            )

    def test_forward_backward_rejects_request_metadata_router_replay(self):
        """Top-level metadata.router_replay is no longer accepted."""
        client = _make_training_client()

        with pytest.raises(ValueError, match="datum.metadata"):
            client.forward_backward(
                [_make_datum()],
                "grpo",
                metadata={"router_replay": _router_replay_payload()},
            )
