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

"""NCCL-v1 SDK surface stays control-only and validates exact receipts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver._payloads import (
    nccl_v1_sampling_session_payload,
    publish_live_weights_nccl_v1_payload,
)
from weaver.async_training_client import AsyncTrainingClient
from weaver.training_client import TrainingClient
from weaver.types.nccl_weight_sync import NCCLWeightSyncV1Result

TRANSACTION = "11111111-1111-4111-8111-111111111111"
GENERATION = "target-generation-1"


def _receipt(**updates):
    value = {
        "transaction_id": TRANSACTION,
        "expected_weight_version": "initial",
        "committed_weight_version": "v0",
        "plan_fingerprint": "sha256:" + "1" * 64,
        "target_projection_fingerprint": "sha256:" + "2" * 64,
        "projection_fingerprint": "sha256:" + "3" * 64,
        "target_engine_id": "engine-1",
        "target_engine_generation": GENERATION,
        "trainer_process_generation": "trainer-process-generation-1",
        "trainer_model_instance_generation": "trainer-model-generation-1",
        "target_worker_process_generations": [
            f"target-process-generation-{rank}" for rank in range(8)
        ],
        "operation_count": 7,
        "canonical_tensor_count": 7,
        "nccl_call_count": 7,
        "transferred_bytes": 4096,
        "largest_operation_bytes": 512,
        "source_workspace_bound_bytes": 2048,
        "target_workspace_bound_bytes": 1536,
        "target_advertised_safe_scratch_bytes": 4096,
        "target_loader_workspace_bound_bytes": 1024,
        "device_completion_token": "complete-1",
        "source_peak_allocated_bytes": 1024,
        "target_peak_allocated_bytes": 2048,
        "phase_timings_ms": {"total_publish": 12.5},
        "plan_reused": False,
        "communicator_reused": False,
        "ready": True,
        "no_fallback_counters": {
            "sampler_checkpoint_exports": 0,
            "dcp_update_calls": 0,
            "mooncake_operations": 0,
            "fallback_attempts": 0,
        },
    }
    value.update(updates)
    return value


def _operation_response(**updates):
    return {
        "status": "succeeded",
        "task_type": "nccl_weight_sync_v1",
        "model_id": "model-1",
        "result": _receipt(**updates),
    }


def test_control_payload_has_no_checkpoint_or_weight_bytes() -> None:
    session = nccl_v1_sampling_session_payload(
        sampling_session_seq_id=1, base_model="supported/model", model_id="model-1"
    )
    payload = publish_live_weights_nccl_v1_payload(
        seq_id=2,
        sampling_session_id="22222222-2222-4222-8222-222222222222",
        expected_weight_version="initial",
        proposed_weight_version="v0",
        transaction_id=TRANSACTION,
    )
    encoded = repr({"session": session, "publish": payload}).lower()
    assert session["weight_sync_mode"] == "nccl_v1"
    assert "target" not in payload
    for forbidden in (
        "model_path",
        "checkpoint",
        "tensor",
        "mooncake",
        "dcp",
        "endpoint",
        "generation",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    ("expected", "proposed"),
    [("initial", "v1"), ("v0", "v2"), ("v01", "v2"), ("v1", "v1")],
)
def test_payload_rejects_skipped_or_noncanonical_versions(expected, proposed) -> None:
    with pytest.raises(ValueError):
        publish_live_weights_nccl_v1_payload(
            seq_id=1,
            sampling_session_id="session-1",
            expected_weight_version=expected,
            proposed_weight_version=proposed,
            transaction_id=TRANSACTION,
        )


def test_sync_publish_returns_only_matching_ready_receipt() -> None:
    service = MagicMock()
    service.next_operation_seq.return_value = 3
    handle = MagicMock()
    handle.result.return_value = _operation_response()
    service.enqueue_operation.return_value = handle
    trainer = TrainingClient(
        service=service, model_id="model-1", base_model="supported/model", session_id="s"
    )
    sampler = SimpleNamespace(
        _service=service,
        model_id="model-1",
        model_path=None,
        sampling_session_id="22222222-2222-4222-8222-222222222222",
    )

    result = trainer.publish_live_weights_to_sampler_nccl_v1(
        sampler,
        expected_weight_version="initial",
        proposed_weight_version="v0",
        transaction_id=TRANSACTION,
    )

    assert isinstance(result, NCCLWeightSyncV1Result)
    assert result.committed_weight_version == "v0"
    path, payload = service.enqueue_operation.call_args.args
    assert path.endswith("/publish-live-weights-nccl-v1")
    assert payload["transaction_id"] == TRANSACTION
    assert "model_path" not in payload


def test_async_publish_returns_matching_ready_receipt() -> None:
    async def exercise() -> NCCLWeightSyncV1Result:
        service = MagicMock()
        service.next_operation_seq.return_value = 4
        handle = MagicMock()
        handle.result = AsyncMock(return_value=_operation_response())
        service.enqueue_operation = AsyncMock(return_value=handle)
        trainer = AsyncTrainingClient(
            service=service,
            model_id="model-1",
            base_model="supported/model",
            session_id="s",
        )
        sampler = SimpleNamespace(
            _service=service,
            model_id="model-1",
            model_path=None,
            sampling_session_id="22222222-2222-4222-8222-222222222222",
        )
        result = await trainer.publish_live_weights_to_sampler_nccl_v1(
            sampler,
            expected_weight_version="initial",
            proposed_weight_version="v0",
            transaction_id=TRANSACTION,
        )
        _, payload = service.enqueue_operation.await_args.args
        assert "model_path" not in payload
        return result

    result = asyncio.run(exercise())
    assert result.transaction_id == TRANSACTION
    assert result.no_fallback_counters["fallback_attempts"] == 0


def test_receipt_requires_zero_fallback_counters() -> None:
    counters = dict(_receipt()["no_fallback_counters"])
    counters["dcp_update_calls"] = 1
    with pytest.raises(RuntimeError, match="zero fallback"):
        NCCLWeightSyncV1Result.from_payload(_receipt(no_fallback_counters=counters))


def test_receipt_requires_exact_call_counts_and_workspace_accounting() -> None:
    with pytest.raises(RuntimeError, match="counts disagree"):
        NCCLWeightSyncV1Result.from_payload(_receipt(nccl_call_count=6))
    with pytest.raises(RuntimeError, match="workspace accounting"):
        NCCLWeightSyncV1Result.from_payload(_receipt(target_workspace_bound_bytes=1535))


def test_receipt_requires_finite_timings_and_exact_target_generations() -> None:
    with pytest.raises(RuntimeError, match="phase timings"):
        NCCLWeightSyncV1Result.from_payload(
            _receipt(phase_timings_ms={"total_publish": float("nan")})
        )
    with pytest.raises(RuntimeError, match="target_worker_process_generations"):
        NCCLWeightSyncV1Result.from_payload(
            _receipt(target_worker_process_generations=["duplicate"] * 8)
        )


def test_receipt_rejects_failed_or_ambiguous_operation_envelope() -> None:
    with pytest.raises(RuntimeError, match="successful result envelope"):
        NCCLWeightSyncV1Result.from_payload({"status": "failed", "result": _receipt()})


# ---------------------------------------------------------------------------
# debug-only checksum mode: request option and conditional result evidence
# ---------------------------------------------------------------------------


SHA256_DIGEST = "sha256:" + "a" * 64


def _publish_payload(**updates):
    kwargs = {
        "seq_id": 1,
        "sampling_session_id": "22222222-2222-4222-8222-222222222222",
        "expected_weight_version": "initial",
        "proposed_weight_version": "v0",
        "transaction_id": TRANSACTION,
    }
    kwargs.update(updates)
    return publish_live_weights_nccl_v1_payload(**kwargs)


def test_checksum_mode_defaults_to_off() -> None:
    assert _publish_payload()["checksum_mode"] == "off"


def test_checksum_mode_accepts_sha256() -> None:
    assert _publish_payload(checksum_mode="sha256")["checksum_mode"] == "sha256"


@pytest.mark.parametrize("mode", ["sha1", "SHA256", "md5", "", " off", None, 1, True, ["sha256"]])
def test_invalid_checksum_mode_fails_before_any_operation(mode) -> None:
    """Rejected in the caller's process -- nothing is enqueued or provisioned."""

    with pytest.raises(ValueError):
        _publish_payload(checksum_mode=mode)


def test_off_receipt_must_not_carry_checksum_evidence() -> None:
    NCCLWeightSyncV1Result.from_payload(_receipt())  # absent fields mean off
    assert NCCLWeightSyncV1Result.from_payload(_receipt()).checksum_algorithm == "off"
    for bad in (
        {"checksum_verified_tensor_count": 7},
        {"checksum_aggregate_digest": SHA256_DIGEST},
    ):
        with pytest.raises(RuntimeError, match="off-mode"):
            NCCLWeightSyncV1Result.from_payload(_receipt(**bad))


def test_sha256_receipt_requires_every_tensor_and_an_aggregate_digest() -> None:
    good = NCCLWeightSyncV1Result.from_payload(
        _receipt(
            checksum_algorithm="sha256",
            checksum_verified_tensor_count=7,  # == canonical_tensor_count
            checksum_aggregate_digest=SHA256_DIGEST,
        )
    )
    assert good.checksum_verified_tensor_count == good.canonical_tensor_count
    assert good.checksum_aggregate_digest == SHA256_DIGEST

    # partial verification is not success
    for bad in (
        {"checksum_verified_tensor_count": 6, "checksum_aggregate_digest": SHA256_DIGEST},
        {"checksum_verified_tensor_count": 0, "checksum_aggregate_digest": SHA256_DIGEST},
        {"checksum_verified_tensor_count": 7},  # no aggregate digest
    ):
        with pytest.raises(RuntimeError, match="checksum"):
            NCCLWeightSyncV1Result.from_payload(_receipt(checksum_algorithm="sha256", **bad))


def test_receipt_rejects_unknown_algorithm_or_malformed_digest() -> None:
    with pytest.raises(RuntimeError, match="checksum_algorithm"):
        NCCLWeightSyncV1Result.from_payload(_receipt(checksum_algorithm="crc32"))
    for digest in ("sha256:zz", "a" * 64, "sha1:" + "a" * 40, 5):
        with pytest.raises(RuntimeError, match="checksum"):
            NCCLWeightSyncV1Result.from_payload(
                _receipt(
                    checksum_algorithm="sha256",
                    checksum_verified_tensor_count=7,
                    checksum_aggregate_digest=digest,
                )
            )


def test_receipt_mode_must_match_the_requested_mode() -> None:
    """A target that silently ignored the request must not look successful."""

    off_receipt = NCCLWeightSyncV1Result.from_payload(_receipt())
    with pytest.raises(RuntimeError, match="checksum mode"):
        off_receipt.validate_request(
            transaction_id=TRANSACTION,
            expected_weight_version="initial",
            proposed_weight_version="v0",
            checksum_mode="sha256",
        )
    on_receipt = NCCLWeightSyncV1Result.from_payload(
        _receipt(
            checksum_algorithm="sha256",
            checksum_verified_tensor_count=7,
            checksum_aggregate_digest=SHA256_DIGEST,
        )
    )
    with pytest.raises(RuntimeError, match="checksum mode"):
        on_receipt.validate_request(
            transaction_id=TRANSACTION,
            expected_weight_version="initial",
            proposed_weight_version="v0",
            checksum_mode="off",
        )
    assert (
        on_receipt.validate_request(
            transaction_id=TRANSACTION,
            expected_weight_version="initial",
            proposed_weight_version="v0",
            checksum_mode="sha256",
        )
        is on_receipt
    )


def test_sync_and_async_publish_send_and_verify_the_requested_mode() -> None:
    receipt = _operation_response(
        checksum_algorithm="sha256",
        checksum_verified_tensor_count=7,
        checksum_aggregate_digest=SHA256_DIGEST,
    )

    service = MagicMock()
    service.next_operation_seq.return_value = 3
    handle = MagicMock()
    handle.result.return_value = receipt
    service.enqueue_operation.return_value = handle
    client = TrainingClient(
        service=service, model_id="model-1", base_model="supported/model", session_id="s"
    )
    sampler = SimpleNamespace(
        _service=service,
        model_id="model-1",
        model_path=None,
        sampling_session_id="22222222-2222-4222-8222-222222222222",
    )
    result = client.publish_live_weights_to_sampler_nccl_v1(
        sampler,
        expected_weight_version="initial",
        proposed_weight_version="v0",
        transaction_id=TRANSACTION,
        checksum_mode="sha256",
    )
    assert service.enqueue_operation.call_args[0][1]["checksum_mode"] == "sha256"
    assert result.checksum_verified_tensor_count == 7

    async_service = MagicMock()
    async_service.next_operation_seq.return_value = 3
    async_handle = MagicMock()
    async_handle.result = AsyncMock(return_value=receipt)
    async_service.enqueue_operation = AsyncMock(return_value=async_handle)
    async_client = AsyncTrainingClient(
        service=async_service,
        model_id="model-1",
        base_model="supported/model",
        session_id="s",
    )
    async_sampler = SimpleNamespace(
        _service=async_service,
        model_id="model-1",
        model_path=None,
        sampling_session_id="22222222-2222-4222-8222-222222222222",
    )
    async_result = asyncio.run(
        async_client.publish_live_weights_to_sampler_nccl_v1(
            async_sampler,
            expected_weight_version="initial",
            proposed_weight_version="v0",
            transaction_id=TRANSACTION,
            checksum_mode="sha256",
        )
    )
    assert async_service.enqueue_operation.call_args[0][1]["checksum_mode"] == "sha256"
    assert async_result.checksum_aggregate_digest == SHA256_DIGEST
