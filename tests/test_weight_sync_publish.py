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

"""One caller publishes weights through whichever backend was configured."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver._payloads import (
    requested_weight_sync,
    resolve_session_weight_sync,
    sampling_session_payload,
)
from weaver.async_training_client import AsyncTrainingClient
from weaver.training_client import TrainingClient
from weaver.types.weight_sync import WeightPublication, WeightSyncSelection

TRANSACTION = "11111111-1111-4111-8111-111111111111"
SESSION = "22222222-2222-4222-8222-222222222222"


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


def _sampler(service, selection, *, version=None):
    return SimpleNamespace(
        _service=service,
        model_id="model-1",
        model_path=None,
        sampling_session_id=SESSION,
        weight_sync=selection,
        weight_version=version or ("v0" if selection.is_live_collective else None),
    )


def _enqueued_paths(service):
    return [call.args[0] for call in service.enqueue_operation.call_args_list]


# --- the same call, three configured backends -------------------------------


def test_live_collective_backend_publishes_over_the_collective():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service, WeightSyncSelection(backend="nccl"))

    published = _trainer(service).publish_weights(sampler, version="v1", transaction_id=TRANSACTION)

    assert isinstance(published, WeightPublication)
    assert (published.backend, published.update) == ("nccl", "full")
    assert (published.version, published.base_version) == ("v1", "v0")
    assert published.model_path is None and published.nccl is not None
    paths = _enqueued_paths(service)
    assert paths == ["/api/v1/models/model-1/publish-live-weights-nccl-v1"]
    # No checkpoint was exported on the way.
    assert not any("export-sampler" in path for path in paths)


def test_default_backend_publishes_through_the_established_checkpoint_export():
    service = _service({"model_path": "weaver://model-1/checkpoints/v1"})
    sampler = _sampler(service, WeightSyncSelection())

    published = _trainer(service).publish_weights(sampler, version="v1")

    assert (published.backend, published.update) == ("default", "full")
    assert published.version == "v1"
    assert published.model_path == "weaver://model-1/checkpoints/v1"
    assert published.nccl is None
    paths = _enqueued_paths(service)
    assert paths == ["/api/v1/models/model-1/export-sampler"]
    # The unchanged export body: the version is the checkpoint name.
    assert service.enqueue_operation.call_args.args[1]["path"] == "v1"
    # ...and nothing reached the live collective.
    assert not any("publish-live-weights" in path for path in paths)


def test_mooncake_backend_uses_the_same_export_operation():
    service = _service({"model_path": "weaver://model-1/checkpoints/v1"})
    sampler = _sampler(service, WeightSyncSelection(backend="mooncake"))

    published = _trainer(service).publish_weights(sampler, version="v1")

    assert published.backend == "mooncake"
    assert _enqueued_paths(service) == ["/api/v1/models/model-1/export-sampler"]


def test_default_backend_with_delta_still_uses_the_established_export():
    # The update dimension does not change which operation the SDK issues; the
    # control plane and trainer decide full-vs-delta downstream, unchanged.
    service = _service({"model_path": "weaver://model-1/checkpoints/v1"})
    sampler = _sampler(service, WeightSyncSelection(update="delta"))

    published = _trainer(service).publish_weights(sampler, version="v1")

    assert (published.backend, published.update) == ("default", "delta")
    assert _enqueued_paths(service) == ["/api/v1/models/model-1/export-sampler"]


# --- lineage ----------------------------------------------------------------


def test_live_collective_tracks_the_committed_version_across_publications():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service, WeightSyncSelection(backend="nccl"))
    trainer = _trainer(service)

    assert sampler.weight_version == "v0"
    trainer.publish_weights(sampler, version="v1", transaction_id=TRANSACTION)
    assert sampler.weight_version == "v1"

    # The caller never restates the base: the next publication advances from
    # the version the target actually committed.
    service.enqueue_operation.return_value.result.return_value = _live_receipt("v1", "v2")
    trainer.publish_weights(sampler, version="v2", transaction_id=TRANSACTION)
    assert sampler.weight_version == "v2"
    assert service.enqueue_operation.call_args.args[1]["expected_weight_version"] == "v1"


def test_explicit_base_version_is_honoured_by_the_live_collective():
    service = _service(_live_receipt("v3", "v4"))
    sampler = _sampler(service, WeightSyncSelection(backend="nccl"), version="v3")

    _trainer(service).publish_weights(
        sampler, version="v4", base_version="v3", transaction_id=TRANSACTION
    )

    assert service.enqueue_operation.call_args.args[1]["expected_weight_version"] == "v3"


@pytest.mark.parametrize("backend", ["default", "mooncake"])
def test_base_version_is_refused_where_the_control_plane_owns_the_lineage(backend):
    service = _service({"model_path": "weaver://model-1/checkpoints/v1"})
    sampler = _sampler(service, WeightSyncSelection(backend=backend))

    with pytest.raises(ValueError, match="resolves its own base version"):
        _trainer(service).publish_weights(sampler, version="v1", base_version="v0")
    service.enqueue_operation.assert_not_called()


def test_debug_checksum_selection_reaches_the_transaction():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service, WeightSyncSelection(backend="nccl", debug_checksum=True))
    receipt = _live_receipt("v0", "v1")
    receipt["checksum_algorithm"] = "sha256"
    receipt["checksum_verified_tensor_count"] = 4
    receipt["checksum_aggregate_digest"] = "sha256:" + "d" * 64
    service.enqueue_operation.return_value.result.return_value = receipt

    _trainer(service).publish_weights(sampler, version="v1", transaction_id=TRANSACTION)

    assert service.enqueue_operation.call_args.args[1]["checksum_mode"] == "sha256"


def test_checksum_is_off_by_default_for_the_live_collective():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service, WeightSyncSelection(backend="nccl"))

    _trainer(service).publish_weights(sampler, version="v1", transaction_id=TRANSACTION)

    assert service.enqueue_operation.call_args.args[1]["checksum_mode"] == "off"


# --- the compatibility wrapper ---------------------------------------------


def test_compat_wrapper_refuses_a_session_frozen_on_another_backend():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service, WeightSyncSelection())

    with pytest.raises(ValueError, match="was created with backend='default'"):
        _trainer(service).publish_live_weights_to_sampler_nccl_v1(
            sampler, expected_weight_version="v0", proposed_weight_version="v1"
        )
    service.enqueue_operation.assert_not_called()


def test_compat_wrapper_still_serves_a_live_collective_session():
    service = _service(_live_receipt("v0", "v1"))
    sampler = _sampler(service, WeightSyncSelection(backend="nccl"))

    result = _trainer(service).publish_live_weights_to_sampler_nccl_v1(
        sampler,
        expected_weight_version="v0",
        proposed_weight_version="v1",
        transaction_id=TRANSACTION,
    )

    assert result.committed_weight_version == "v1"


# --- session creation -------------------------------------------------------


def test_session_body_carries_the_selection_and_the_legacy_spelling():
    body = sampling_session_payload(
        sampling_session_seq_id=1,
        selection=WeightSyncSelection(backend="nccl", update="delta"),
        base_model="supported/model",
        model_id="model-1",
    )
    assert body["weight_sync"] == {
        "backend": "nccl",
        "update": "delta",
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
    explicit = WeightSyncSelection(backend="nccl", update="delta")
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


def test_async_publish_matches_the_sync_backend_dispatch():
    async def exercise():
        service = MagicMock()
        service.next_operation_seq.return_value = 3
        handle = MagicMock()
        handle.result = AsyncMock(return_value=_live_receipt("v0", "v1"))
        service.enqueue_operation = AsyncMock(return_value=handle)
        trainer = AsyncTrainingClient(
            service=service, model_id="model-1", base_model="supported/model", session_id="s"
        )
        sampler = _sampler(service, WeightSyncSelection(backend="nccl"))
        published = await trainer.publish_weights(sampler, version="v1", transaction_id=TRANSACTION)
        path, _ = service.enqueue_operation.await_args.args
        return published, path, sampler.weight_version

    published, path, tracked = asyncio.run(exercise())
    assert published.backend == "nccl" and published.version == "v1"
    assert path.endswith("/publish-live-weights-nccl-v1")
    assert tracked == "v1"


def test_async_default_backend_uses_the_export_operation():
    async def exercise():
        service = MagicMock()
        service.next_operation_seq.return_value = 3
        handle = MagicMock()
        handle.result = AsyncMock(return_value={"model_path": "weaver://model-1/checkpoints/v1"})
        service.enqueue_operation = AsyncMock(return_value=handle)
        trainer = AsyncTrainingClient(
            service=service, model_id="model-1", base_model="supported/model", session_id="s"
        )
        sampler = _sampler(service, WeightSyncSelection())
        published = await trainer.publish_weights(sampler, version="v1")
        path, _ = service.enqueue_operation.await_args.args
        return published, path

    published, path = asyncio.run(exercise())
    assert published.backend == "default"
    assert path.endswith("/export-sampler")


def test_async_compat_wrapper_refuses_another_backend():
    async def exercise():
        service = MagicMock()
        service.enqueue_operation = AsyncMock()
        trainer = AsyncTrainingClient(
            service=service, model_id="model-1", base_model="supported/model", session_id="s"
        )
        sampler = _sampler(service, WeightSyncSelection())
        with pytest.raises(ValueError, match="was created with backend='default'"):
            await trainer.publish_live_weights_to_sampler_nccl_v1(
                sampler, expected_weight_version="v0", proposed_weight_version="v1"
            )
        service.enqueue_operation.assert_not_called()

    asyncio.run(exercise())
