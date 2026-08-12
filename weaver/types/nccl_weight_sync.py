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

"""Typed result for the experimental control-only NCCL-v1 operation."""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping

_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^v(0|[1-9][0-9]*)$")

#: Debug-only payload-integrity modes for one NCCL-v1 publication.
#:
#: ``off`` is the production default and performs no hashing, no device-to-host
#: byte copy, and no per-tensor digest exchange. ``sha256`` proves that every
#: target worker received exactly the canonical bytes the source produced; it
#: does not prove native-loader shard placement or model semantics, and it
#: necessarily copies and hashes the whole model on the source and on every
#: target rank, so it must never be mixed into a performance measurement.
NCCL_V1_CHECKSUM_MODES = ("off", "sha256")


def normalize_nccl_v1_checksum_mode(value: object) -> str:
    """Reject an unsupported mode before any provisioning or transfer happens."""

    if not isinstance(value, str) or value not in NCCL_V1_CHECKSUM_MODES:
        raise ValueError("checksum_mode must be one of " + ", ".join(NCCL_V1_CHECKSUM_MODES))
    return value


def _text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RuntimeError(f"NCCL-v1 response has invalid {name}")
    return value


def _integer(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"NCCL-v1 response has invalid {name}")
    return value


def _text_sequence(
    payload: Mapping[str, Any], name: str, *, expected_length: int
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise RuntimeError(f"NCCL-v1 response has invalid {name}")
    result = tuple(value)
    if any(
        not isinstance(item, str)
        or not item
        or item.strip() != item
        or any(character.isspace() for character in item)
        for item in result
    ) or len(set(result)) != len(result):
        raise RuntimeError(f"NCCL-v1 response has invalid {name}")
    return result


@dataclass(frozen=True, slots=True)
class NCCLWeightSyncV1Result:
    """Proof returned only after the target is committed, closed, and ready."""

    transaction_id: str
    expected_weight_version: str
    committed_weight_version: str
    plan_fingerprint: str
    target_projection_fingerprint: str
    projection_fingerprint: str
    target_engine_id: str
    target_engine_generation: str
    trainer_process_generation: str
    trainer_model_instance_generation: str
    target_worker_process_generations: tuple[str, ...]
    operation_count: int
    canonical_tensor_count: int
    nccl_call_count: int
    transferred_bytes: int
    largest_operation_bytes: int
    source_workspace_bound_bytes: int
    target_workspace_bound_bytes: int
    target_advertised_safe_scratch_bytes: int
    target_loader_workspace_bound_bytes: int
    device_completion_token: str
    source_peak_allocated_bytes: int
    target_peak_allocated_bytes: int
    phase_timings_ms: Mapping[str, float]
    plan_reused: bool
    communicator_reused: bool
    no_fallback_counters: Mapping[str, int]
    checksum_algorithm: str
    checksum_verified_tensor_count: int
    checksum_aggregate_digest: str | None

    @classmethod
    def from_payload(cls, value: object) -> "NCCLWeightSyncV1Result":
        if not isinstance(value, dict):
            raise RuntimeError("NCCL-v1 response must be an object")
        payload: Mapping[str, Any] = value
        # Operation.Response stores the Trainer callback envelope. Accept only
        # its explicit success/result shape; never treat an arbitrary nested
        # object as a committed transaction receipt.
        if "transaction_id" not in payload:
            nested_result = payload.get("result")
            if payload.get("status") != "succeeded" or not isinstance(nested_result, dict):
                raise RuntimeError("NCCL-v1 operation response omits a successful result envelope")
            payload = nested_result
        timings = payload.get("phase_timings_ms")
        counters = payload.get("no_fallback_counters")
        if not isinstance(timings, dict) or any(
            not isinstance(name, str)
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or duration < 0
            for name, duration in timings.items()
        ):
            raise RuntimeError("NCCL-v1 response has invalid phase timings")
        expected_counters = {
            "sampler_checkpoint_exports",
            "dcp_update_calls",
            "mooncake_operations",
            "fallback_attempts",
        }
        if (
            not isinstance(counters, dict)
            or set(counters) != expected_counters
            or any(
                not isinstance(count, int) or isinstance(count, bool) or count != 0
                for count in counters.values()
            )
        ):
            raise RuntimeError("NCCL-v1 response does not prove zero fallback activity")
        if payload.get("ready") is not True:
            raise RuntimeError("NCCL-v1 response does not prove READY")
        if not isinstance(payload.get("plan_reused"), bool) or not isinstance(
            payload.get("communicator_reused"), bool
        ):
            raise RuntimeError("NCCL-v1 response has invalid reuse markers")
        if not timings or "total_publish" not in timings:
            raise RuntimeError("NCCL-v1 response omits total publish timing")
        # Checksum evidence is conditional but never ambiguous: an absent
        # algorithm means the publication ran in the default off mode, and the
        # count/digest must agree with that. A receipt claiming verification
        # without a count, or a count without an aggregate digest, is rejected
        # rather than read as a partial success.
        checksum_algorithm = payload.get("checksum_algorithm", "off")
        if checksum_algorithm is None:
            checksum_algorithm = "off"
        if checksum_algorithm not in NCCL_V1_CHECKSUM_MODES:
            raise RuntimeError("NCCL-v1 response has invalid checksum_algorithm")
        checksum_verified_tensor_count = _integer(
            {"checksum_verified_tensor_count": payload.get("checksum_verified_tensor_count", 0)},
            "checksum_verified_tensor_count",
        )
        checksum_aggregate_digest = payload.get("checksum_aggregate_digest")
        if checksum_aggregate_digest is not None and (
            not isinstance(checksum_aggregate_digest, str)
            or _FINGERPRINT_RE.fullmatch(checksum_aggregate_digest) is None
        ):
            raise RuntimeError("NCCL-v1 response has invalid checksum_aggregate_digest")
        result = cls(
            transaction_id=_text(payload, "transaction_id"),
            expected_weight_version=_text(payload, "expected_weight_version"),
            committed_weight_version=_text(payload, "committed_weight_version"),
            plan_fingerprint=_text(payload, "plan_fingerprint"),
            target_projection_fingerprint=_text(payload, "target_projection_fingerprint"),
            projection_fingerprint=_text(payload, "projection_fingerprint"),
            target_engine_id=_text(payload, "target_engine_id"),
            target_engine_generation=_text(payload, "target_engine_generation"),
            trainer_process_generation=_text(payload, "trainer_process_generation"),
            trainer_model_instance_generation=_text(payload, "trainer_model_instance_generation"),
            target_worker_process_generations=_text_sequence(
                payload, "target_worker_process_generations", expected_length=8
            ),
            operation_count=_integer(payload, "operation_count"),
            canonical_tensor_count=_integer(payload, "canonical_tensor_count"),
            nccl_call_count=_integer(payload, "nccl_call_count"),
            transferred_bytes=_integer(payload, "transferred_bytes"),
            largest_operation_bytes=_integer(payload, "largest_operation_bytes"),
            source_workspace_bound_bytes=_integer(payload, "source_workspace_bound_bytes"),
            target_workspace_bound_bytes=_integer(payload, "target_workspace_bound_bytes"),
            target_advertised_safe_scratch_bytes=_integer(
                payload, "target_advertised_safe_scratch_bytes"
            ),
            target_loader_workspace_bound_bytes=_integer(
                payload, "target_loader_workspace_bound_bytes"
            ),
            device_completion_token=_text(payload, "device_completion_token"),
            source_peak_allocated_bytes=_integer(payload, "source_peak_allocated_bytes"),
            target_peak_allocated_bytes=_integer(payload, "target_peak_allocated_bytes"),
            phase_timings_ms={str(name): float(duration) for name, duration in timings.items()},
            plan_reused=payload["plan_reused"],
            communicator_reused=payload["communicator_reused"],
            no_fallback_counters={str(name): int(count) for name, count in counters.items()},
            checksum_algorithm=checksum_algorithm,
            checksum_verified_tensor_count=checksum_verified_tensor_count,
            checksum_aggregate_digest=checksum_aggregate_digest,
        )
        try:
            if str(uuid.UUID(result.transaction_id)) != result.transaction_id:
                raise ValueError
        except ValueError as error:
            raise RuntimeError("NCCL-v1 response transaction_id is not canonical") from error
        for name in (
            "plan_fingerprint",
            "target_projection_fingerprint",
            "projection_fingerprint",
        ):
            if _FINGERPRINT_RE.fullmatch(getattr(result, name)) is None:
                raise RuntimeError(f"NCCL-v1 response has invalid {name}")
        if result.operation_count <= 0 or result.transferred_bytes <= 0:
            raise RuntimeError("NCCL-v1 response must prove positive tensor/byte counts")
        if not result.canonical_tensor_count == result.nccl_call_count == result.operation_count:
            raise RuntimeError("NCCL-v1 response tensor/NCCL counts disagree")
        if (
            result.largest_operation_bytes <= 0
            or result.largest_operation_bytes > result.transferred_bytes
            or result.source_workspace_bound_bytes < result.largest_operation_bytes
            or result.target_workspace_bound_bytes
            < result.largest_operation_bytes + result.target_loader_workspace_bound_bytes
            or result.target_advertised_safe_scratch_bytes < result.target_workspace_bound_bytes
        ):
            raise RuntimeError("NCCL-v1 response workspace accounting is inconsistent")
        if result.checksum_algorithm == "off":
            if (
                result.checksum_verified_tensor_count != 0
                or result.checksum_aggregate_digest is not None
            ):
                raise RuntimeError("NCCL-v1 off-mode response must not carry checksum evidence")
        elif (
            result.checksum_verified_tensor_count != result.canonical_tensor_count
            or result.checksum_aggregate_digest is None
        ):
            # Partial verification is not a success: every canonical tensor must
            # have been verified on every target worker, or the receipt is a lie.
            raise RuntimeError(
                "NCCL-v1 checksum response must verify every canonical tensor "
                "and carry an aggregate digest"
            )
        return result

    def validate_request(
        self,
        *,
        transaction_id: str,
        expected_weight_version: str,
        proposed_weight_version: str,
        checksum_mode: str = "off",
    ) -> "NCCLWeightSyncV1Result":
        """Reject a successful-looking receipt for any other transaction."""

        if self.transaction_id != transaction_id:
            raise RuntimeError("NCCL-v1 response transaction differs from request")
        if self.checksum_algorithm != checksum_mode:
            # The mode is transaction-scoped, so a receipt reporting a different
            # algorithm than the caller asked for is answering a different
            # request -- including a target that silently ignored the request.
            raise RuntimeError("NCCL-v1 response checksum mode differs from request")
        if self.expected_weight_version != expected_weight_version:
            raise RuntimeError("NCCL-v1 response expected version differs from request")
        if self.committed_weight_version != proposed_weight_version:
            raise RuntimeError("NCCL-v1 response committed version differs from request")
        proposed = _VERSION_RE.fullmatch(self.committed_weight_version)
        expected = _VERSION_RE.fullmatch(self.expected_weight_version)
        if proposed is None:
            raise RuntimeError("NCCL-v1 response version transition is invalid")
        if self.expected_weight_version == "initial":
            if self.committed_weight_version != "v0":
                raise RuntimeError("NCCL-v1 response version transition is invalid")
        elif expected is None or int(proposed.group(1)) != int(expected.group(1)) + 1:
            raise RuntimeError("NCCL-v1 response version transition is invalid")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}
