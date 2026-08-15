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

"""Publication reaches the inference target, or it is not a publication.

The two backends have materially different session lifecycles:

* the checkpoint backends export a checkpoint and then bind a **new** sampling
  session to it; the server's ``sync_weights`` operation behind that session is
  what actually loads the weights into the engine;
* the live collective updates the **existing** session's target in place and
  returns a receipt only after the target committed.

These tests hold each path to its own real sequence. Exporting a checkpoint is
explicitly *not* treated as having published anything.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver._http import WeaverAPIError
from weaver._payloads import (
    requested_weight_sync,
    resolve_session_weight_sync,
    sampling_session_payload,
)
from weaver.async_training_client import AsyncTrainingClient
from weaver.training_client import TrainingClient
from weaver.types.weight_sync import WeightSyncSelection

TRANSACTION = "11111111-1111-4111-8111-111111111111"
SESSION = "22222222-2222-4222-8222-222222222222"
NEXT_SESSION = "33333333-3333-4333-8333-333333333333"


def _live_receipt(expected: str, committed: str) -> dict:
    return {
        "transaction_id": TRANSACTION,
        "expected_weight_version": expected,
        "committed_weight_version": committed,
        "plan_fingerprint": "sha256:" + "a" * 64,
        "target_projection_fingerprint": "sha256:" + "b" * 64,
        "projection_fingerprint": "sha256:" + "c" * 64,
        "target_engine_id": "engine-1",
        "target_engine_generation": "gen-1",
        "trainer_process_generation": "tgen-1",
        "trainer_model_instance_generation": "mgen-1",
        "target_worker_process_generations": [f"w{i}" for i in range(8)],
        "operation_count": 4,
        "canonical_tensor_count": 4,
        "transfer_batch_count": 2,
        "nccl_call_count": 2,
        "transferred_bytes": 4096,
        "wire_transferred_bytes": 4096,
        "transfer_padding_bytes": 0,
        "largest_operation_bytes": 2048,
        "largest_transfer_batch_bytes": 2048,
        "source_workspace_bound_bytes": 4096,
        "target_workspace_bound_bytes": 8192,
        "target_advertised_safe_scratch_bytes": 16384,
        "target_loader_workspace_bound_bytes": 1024,
        "device_completion_token": "token-1",
        "source_peak_allocated_bytes": 1,
        "target_peak_allocated_bytes": 1,
        "phase_timings_ms": {"total_publish": 1.0},
        "plan_reused": True,
        "communicator_reused": True,
        "no_fallback_counters": {
            "sampler_checkpoint_exports": 0,
            "dcp_update_calls": 0,
            "mooncake_operations": 0,
            "fallback_attempts": 0,
        },
        "ready": True,
    }


def _service(result):
    service = MagicMock()
    service.next_operation_seq.return_value = 3
    handle = MagicMock()
    handle.result.return_value = result
    service.enqueue_operation.return_value = handle
    return service


def _trainer(service):
    return TrainingClient(
        service=service, model_id="model-1", base_model="supported/model", session_id="s"
    )


def _sampler(service, selection=None):
    from weaver.sampling_client import SamplingClient

    return SamplingClient(
        service=service,
        sampling_session_id=SESSION,
        base_model="supported/model",
        model_id="model-1",
        weight_sync=selection or WeightSyncSelection(backend="nccl"),
    )


# --- the live collective updates the existing session in place --------------


def test_a_committed_receipt_is_what_advances_the_session_version():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service)
    assert sampler.weight_version == "v0"

    receipt = _trainer(service).publish_live_weights_to_sampler_nccl_v1(
        sampler,
        expected_weight_version="v0",
        proposed_weight_version="v1",
        transaction_id=TRANSACTION,
    )

    assert receipt.committed_weight_version == "v1"
    assert sampler.weight_version == "v1"
    # The same session keeps serving; nothing was exported.
    assert sampler.sampling_session_id == SESSION
    paths = [call.args[0] for call in service.enqueue_operation.call_args_list]
    assert paths == ["/api/v1/models/model-1/publish-live-weights-nccl-v1"]


def test_a_failed_publication_leaves_the_session_version_untouched():
    service = _service(_live_receipt("v0", "v1"))
    service.enqueue_operation.return_value.result.side_effect = WeaverAPIError(
        500, "publish_failed", "target never committed", True
    )
    sampler = _sampler(service)

    with pytest.raises(WeaverAPIError):
        _trainer(service).publish_live_weights_to_sampler_nccl_v1(
            sampler,
            expected_weight_version="v0",
            proposed_weight_version="v1",
            transaction_id=TRANSACTION,
        )

    # No commit, no advance.
    assert sampler.weight_version == "v0"


def test_a_receipt_for_another_transaction_does_not_advance_the_version():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service)
    with pytest.raises(RuntimeError, match="transaction differs from request"):
        _trainer(service).publish_live_weights_to_sampler_nccl_v1(
            sampler,
            expected_weight_version="v0",
            proposed_weight_version="v1",
            transaction_id="44444444-4444-4444-8444-444444444444",
        )
    assert sampler.weight_version == "v0"


def test_the_frozen_selection_is_not_publicly_assignable():
    sampler = _sampler(_service(_live_receipt("v0", "v1")))
    # "Frozen" has to mean caller code cannot change the transport a session
    # runs on, not merely that the SDK chooses not to.
    with pytest.raises(AttributeError):
        sampler.weight_sync = WeightSyncSelection()
    with pytest.raises(AttributeError):
        sampler.weight_version = "v9"


def test_publishing_live_weights_is_refused_on_a_checkpoint_session():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service, WeightSyncSelection())
    with pytest.raises(ValueError, match="was created with backend='default'"):
        _trainer(service).publish_live_weights_to_sampler_nccl_v1(
            sampler, expected_weight_version="v0", proposed_weight_version="v1"
        )
    service.enqueue_operation.assert_not_called()


def test_a_checkpoint_session_tracks_no_weight_version():
    # It is bound to one checkpoint at creation and is not updated in place, so
    # there is no version for the client to advance.
    assert _sampler(_service({}), WeightSyncSelection()).weight_version is None


# --- the checkpoint backends bind a new session, and the server syncs it ----


def _session_service(session_response):
    service = MagicMock()
    service.next_operation_seq.return_value = 3
    export_handle = MagicMock()
    export_handle.result.return_value = {"model_path": "weaver://model-1/checkpoints/v1"}
    service.enqueue_operation.return_value = export_handle
    service.http.post.return_value = session_response
    service.session_id = "session-1"
    service._next_sampling_seq.return_value = 2
    service.get_supported_model_config.return_value = None
    return service


def test_exporting_a_checkpoint_does_not_reach_the_target():
    # This is the trap: /export-sampler completes the checkpoint export and
    # returns its path. It updates no engine, so nothing about it constitutes a
    # publication.
    service = _service({"model_path": "weaver://model-1/checkpoints/v1"})
    path = _trainer(service).save_weights_for_sampler(name="v1")
    assert path == "weaver://model-1/checkpoints/v1"
    paths = [call.args[0] for call in service.enqueue_operation.call_args_list]
    assert paths == ["/api/v1/models/model-1/export-sampler"]
    # Nothing about a sampling session, a router push, or a served version.
    assert not service.http.post.called


def test_binding_a_new_session_is_what_awaits_the_target_update(monkeypatch):
    # The server answers session creation with 202 + a sync_weights operation;
    # the SDK must await that operation, because it is the step that loads the
    # weights into the engine.
    from weaver import service_client as sc

    awaited = []

    class _SyncHandle:
        operation_id = "sync-op-1"

        def wait(self):
            awaited.append("waited")
            return {"status": "done"}

    monkeypatch.setattr(
        sc.OperationHandle, "from_payload", classmethod(lambda cls, http, payload: _SyncHandle())
    )
    service = _session_service(
        {
            "sampling_session": {"ID": NEXT_SESSION, "tokenizer_path": "/models/x"},
            "sync_operation": {"id": "sync-op-1"},
        }
    )
    client = sc.ServiceClient.create_sampling_client(
        service,
        base_model="supported/model",
        model_path="weaver://model-1/checkpoints/v1",
        model_id="model-1",
    )
    assert awaited == ["waited"], "the SDK returned before the target was synced"
    assert client.sampling_session_id == NEXT_SESSION


def test_a_failed_target_sync_fails_session_creation(monkeypatch):
    from weaver import service_client as sc

    class _FailingHandle:
        operation_id = "sync-op-1"

        def wait(self):
            raise WeaverAPIError(500, "weights_sync_failed", "router push failed", True)

    monkeypatch.setattr(
        sc.OperationHandle,
        "from_payload",
        classmethod(lambda cls, http, payload: _FailingHandle()),
    )
    service = _session_service(
        {
            "sampling_session": {"ID": NEXT_SESSION},
            "sync_operation": {"id": "sync-op-1"},
        }
    )
    with pytest.raises(WeaverAPIError, match="router push failed"):
        sc.ServiceClient.create_sampling_client(
            service,
            base_model="supported/model",
            model_path="weaver://model-1/checkpoints/v1",
            model_id="model-1",
        )


# --- session creation carries the configured selection ----------------------


def test_session_body_carries_the_selection_and_the_legacy_spelling():
    body = sampling_session_payload(
        sampling_session_seq_id=1,
        selection=WeightSyncSelection(backend="nccl", update="full"),
        base_model="supported/model",
        model_id="model-1",
    )
    assert body["weight_sync"] == {
        "backend": "nccl",
        "update": "full",
        "debug_checksum": False,
    }
    assert body["weight_sync_mode"] == "nccl_v1"
    assert "model_path" not in body


def test_checkpoint_session_body_is_unchanged_apart_from_the_selection():
    body = sampling_session_payload(
        sampling_session_seq_id=1,
        selection=WeightSyncSelection(),
        base_model="supported/model",
        model_path="weaver://model-1/checkpoints/v1",
        model_id="model-1",
    )
    assert body["base_model"] == "supported/model"
    assert body["model_path"] == "weaver://model-1/checkpoints/v1"
    assert body["model_id"] == "model-1"
    assert "weight_sync_mode" not in body


def test_live_collective_session_refuses_a_checkpoint_path():
    with pytest.raises(ValueError, match="forbids model_path"):
        sampling_session_payload(
            sampling_session_seq_id=1,
            selection=WeightSyncSelection(backend="nccl"),
            base_model="supported/model",
            model_id="model-1",
            model_path="weaver://model-1/checkpoints/v1",
        )


def test_deprecated_flag_maps_to_the_live_collective_backend():
    assert requested_weight_sync(None, True) == WeightSyncSelection(backend="nccl")
    assert requested_weight_sync(None, False) == WeightSyncSelection()
    explicit = WeightSyncSelection(backend="nccl", debug_checksum=True)
    assert requested_weight_sync(explicit, True) is explicit


def test_deprecated_flag_contradicting_an_explicit_selection_is_refused():
    with pytest.raises(ValueError, match="contradicts"):
        requested_weight_sync(WeightSyncSelection(backend="default"), True)


def test_session_creation_fails_when_the_control_plane_froze_another_backend():
    requested = WeightSyncSelection(backend="nccl")
    session = {"id": SESSION, "weight_sync": {"backend": "default", "update": "full"}}
    with pytest.raises(RuntimeError, match="selection mismatch"):
        resolve_session_weight_sync(requested, session)


def test_a_control_plane_that_reports_nothing_keeps_the_requested_selection():
    requested = WeightSyncSelection(backend="nccl")
    assert resolve_session_weight_sync(requested, {"id": SESSION}) == requested


# --- async parity -----------------------------------------------------------


def test_async_publication_advances_only_on_a_committed_receipt():
    async def exercise():
        from weaver.async_sampling_client import AsyncSamplingClient

        service = MagicMock()
        service.next_operation_seq.return_value = 3
        handle = MagicMock()
        handle.result = AsyncMock(return_value=_live_receipt("v0", "v1"))
        service.enqueue_operation = AsyncMock(return_value=handle)
        trainer = AsyncTrainingClient(
            service=service, model_id="model-1", base_model="supported/model", session_id="s"
        )
        sampler = AsyncSamplingClient(
            service=service,
            sampling_session_id=SESSION,
            model_id="model-1",
            weight_sync=WeightSyncSelection(backend="nccl"),
        )
        receipt = await trainer.publish_live_weights_to_sampler_nccl_v1(
            sampler,
            expected_weight_version="v0",
            proposed_weight_version="v1",
            transaction_id=TRANSACTION,
        )
        return receipt, sampler.weight_version

    receipt, tracked = asyncio.run(exercise())
    assert receipt.committed_weight_version == "v1"
    assert tracked == "v1"


def test_async_publication_is_refused_on_a_checkpoint_session():
    async def exercise():
        from weaver.async_sampling_client import AsyncSamplingClient

        service = MagicMock()
        service.enqueue_operation = AsyncMock()
        trainer = AsyncTrainingClient(
            service=service, model_id="model-1", base_model="supported/model", session_id="s"
        )
        sampler = AsyncSamplingClient(
            service=service,
            sampling_session_id=SESSION,
            model_id="model-1",
            weight_sync=WeightSyncSelection(),
        )
        with pytest.raises(ValueError, match="was created with backend='default'"):
            await trainer.publish_live_weights_to_sampler_nccl_v1(
                sampler, expected_weight_version="v0", proposed_weight_version="v1"
            )
        service.enqueue_operation.assert_not_called()

    asyncio.run(exercise())
