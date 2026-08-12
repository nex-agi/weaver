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

"""Pure request/response helpers shared by the sync and async clients.

Keeping these IO-free and tokenizer-free means the synchronous and asyncio
client implementations build identical payloads from a single source of truth.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from .types import Datum
from .types.nccl_weight_sync import normalize_nccl_v1_checksum_mode

if TYPE_CHECKING:
    import torch

    from .types.router_replay import RouterReplayMetadata


_NORMALIZED_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")
_NCCL_V1_VERSION_RE = re.compile(r"^v(0|[1-9][0-9]*)$")


def _normalized_nccl_v1_text(name: str, value: object) -> str:
    if not isinstance(value, str) or value.strip() != value or not value:
        raise ValueError(f"{name} must be non-empty normalized text")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} must not contain whitespace")
    return value


def nccl_v1_sampling_session_payload(
    *, sampling_session_seq_id: int, base_model: str, model_id: str
) -> Dict[str, Any]:
    """Build the explicit model-bound session used by live NCCL-v1.

    There is deliberately no ``model_path`` field: including one would enter
    Weaver's durable checkpoint/DCP synchronization path.
    """

    if (
        not isinstance(sampling_session_seq_id, int)
        or isinstance(sampling_session_seq_id, bool)
        or sampling_session_seq_id <= 0
    ):
        raise ValueError("sampling_session_seq_id must be a positive integer")
    return {
        "sampling_session_seq_id": sampling_session_seq_id,
        "base_model": _normalized_nccl_v1_text("base_model", base_model),
        "model_id": _normalized_nccl_v1_text("model_id", model_id),
        "weight_sync_mode": "nccl_v1",
    }


def publish_live_weights_nccl_v1_payload(
    *,
    seq_id: int,
    sampling_session_id: str,
    expected_weight_version: str,
    proposed_weight_version: str,
    transaction_id: str | None = None,
    checksum_mode: str = "off",
) -> Dict[str, Any]:
    """Build the small control-only payload for one live NCCL transaction."""

    if not isinstance(seq_id, int) or isinstance(seq_id, bool) or seq_id <= 0:
        raise ValueError("seq_id must be a positive integer")
    sampling_session_id = _normalized_nccl_v1_text("sampling_session_id", sampling_session_id)
    expected = _normalized_nccl_v1_text("expected_weight_version", expected_weight_version)
    proposed = _normalized_nccl_v1_text("proposed_weight_version", proposed_weight_version)
    if not _NORMALIZED_VERSION_RE.fullmatch(expected) or not _NORMALIZED_VERSION_RE.fullmatch(
        proposed
    ):
        raise ValueError("weight versions contain unsupported characters")
    if expected == proposed:
        raise ValueError("expected and proposed weight versions must differ")
    expected_match = _NCCL_V1_VERSION_RE.fullmatch(expected)
    proposed_match = _NCCL_V1_VERSION_RE.fullmatch(proposed)
    if proposed_match is None:
        raise ValueError("NCCL-v1 versions must use initial/v0/v1/... identities")
    if expected != "initial" and expected_match is None:
        raise ValueError("NCCL-v1 versions must use initial/v0/v1/... identities")
    if expected == "initial":
        if proposed != "v0":
            raise ValueError("the first NCCL-v1 publication must be initial→v0")
    else:
        assert expected_match is not None
        if int(proposed_match.group(1)) != int(expected_match.group(1)) + 1:
            raise ValueError("NCCL-v1 versions must advance exactly once")

    if transaction_id is None:
        transaction_id = str(uuid.uuid4())
    else:
        transaction_id = _normalized_nccl_v1_text("transaction_id", transaction_id)
        try:
            transaction_id = str(uuid.UUID(transaction_id))
        except ValueError as error:
            raise ValueError("transaction_id must be a canonical UUID") from error

    # Validated here so an unsupported mode fails in the caller's process,
    # before any operation is enqueued, provisioned, or transferred.
    checksum_mode = normalize_nccl_v1_checksum_mode(checksum_mode)

    return {
        "seq_id": seq_id,
        "sampling_session_id": sampling_session_id,
        "transaction_id": transaction_id,
        "expected_weight_version": expected,
        "proposed_weight_version": proposed,
        "checksum_mode": checksum_mode,
    }


def build_request_metadata(
    metadata: Mapping[str, Any] | None,
    router_replay: "RouterReplayMetadata | Mapping[str, Any] | None",
) -> Dict[str, Any] | None:
    """Validate and normalize top-level request metadata.

    Router replay metadata must be attached per-:class:`~weaver.types.Datum`
    (``datum.metadata["router_replay"]``); passing it at the request level is a
    hard error.
    """
    if router_replay is not None:
        raise ValueError(
            "router_replay= is no longer accepted at request level. "
            "Attach router replay metadata to each Datum via "
            "datum.metadata['router_replay']."
        )
    if not metadata and router_replay is None:
        return None

    payload = dict(metadata or {})
    if "router_replay" in payload:
        raise ValueError(
            "metadata['router_replay'] is no longer accepted at request level. "
            "Attach router replay metadata to each Datum via "
            "datum.metadata['router_replay']."
        )
    return payload or None


def serialize_data(data: Sequence[Datum]) -> List[Dict[str, Any]]:
    return [datum.to_payload() for datum in data]


def forward_payload(
    *,
    model_id: str,
    seq_id: int,
    data: Sequence[Datum],
    loss_fn: str,
    loss_fn_config: Mapping[str, Any] | None,
    request_metadata: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_id": model_id,
        "seq_id": seq_id,
        "forward_input": {"loss_fn": loss_fn, "data": serialize_data(data)},
    }
    if loss_fn_config:
        payload["forward_input"]["loss_fn_config"] = dict(loss_fn_config)
    if request_metadata:
        payload["metadata"] = request_metadata
    return payload


def forward_backward_payload(
    *,
    model_id: str,
    seq_id: int,
    data: Sequence[Datum],
    loss_fn: str,
    loss_fn_config: Mapping[str, Any] | None,
    request_metadata: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_id": model_id,
        "seq_id": seq_id,
        "forward_backward_input": {"loss_fn": loss_fn, "data": serialize_data(data)},
    }
    if loss_fn_config:
        payload["forward_backward_input"]["loss_fn_config"] = dict(loss_fn_config)
    if request_metadata:
        payload["metadata"] = request_metadata
    return payload


def parse_logprob_tensors(
    fwd_result: Dict[str, Any], data: Sequence[Datum]
) -> List["torch.Tensor"]:
    """Extract per-datum logprob tensors (``requires_grad=True``) from a forward result."""
    import torch

    outputs = fwd_result.get("result", {}).get("loss_fn_outputs", [])
    if not outputs:
        raise ValueError("Forward pass returned no loss_fn_outputs")
    if len(outputs) != len(data):
        raise ValueError(f"Expected {len(data)} loss_fn_outputs, got {len(outputs)}")

    logprob_tensors: List[torch.Tensor] = []
    for output in outputs:
        lp = output.get("logprobs") or output.get("Logprobs")
        if isinstance(lp, dict):
            lp = lp["data"]
        if lp is None:
            raise ValueError("Missing logprobs in forward/backward output")
        logprob_tensors.append(torch.tensor(lp, dtype=torch.float32).requires_grad_(True))
    return logprob_tensors


def build_surrogate_data(
    data: Sequence[Datum], logprob_tensors: Sequence["torch.Tensor"]
) -> List[Datum]:
    """Build surrogate :class:`~weaver.types.Datum` objects carrying gradient weights."""
    surrogate_data: List[Datum] = []
    for i, (datum, logprob_tensor) in enumerate(zip(data, logprob_tensors)):
        if logprob_tensor.grad is None:
            raise ValueError(f"logprob_tensors[{i}] has no gradient after backward")

        raw_targets = datum.loss_fn_inputs.get("target_tokens")
        if raw_targets is None:
            resolved_targets: List[Any] = datum.model_input.to_ints()
        elif hasattr(raw_targets, "tolist"):
            resolved_targets = raw_targets.tolist()
        else:
            resolved_targets = list(raw_targets)

        loss_fn_inputs: Dict[str, Any] = dict(datum.loss_fn_inputs)
        loss_fn_inputs["target_tokens"] = resolved_targets
        loss_fn_inputs["surrogate_weights"] = logprob_tensor.grad.detach().tolist()
        surrogate_data.append(
            Datum.from_raw(model_input=datum.model_input, loss_fn_inputs=loss_fn_inputs)
        )
    return surrogate_data
