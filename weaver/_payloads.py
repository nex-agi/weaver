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

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from .types import Datum

if TYPE_CHECKING:
    import torch

    from .types.router_replay import RouterReplayMetadata


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


def build_router_replay_manifest_body(
    model_id: str,
    replay_set_id: str,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the request body for persisting a router-replay manifest server-side.

    NexRL assembles the manifest (it owns the training-batch framing) but
    delegates the GPFS write to the server, so both client stacks post identical
    bytes. The server derives the write path from ``model_id`` / ``replay_set_id``
    and treats ``manifest`` as opaque content.
    """
    model_id = str(model_id or "").strip().strip("/")
    replay_set_id = str(replay_set_id or "").strip().strip("/")
    if not model_id or not replay_set_id:
        raise ValueError("router replay manifest requires non-empty model_id and replay_set_id")
    if not isinstance(manifest, Mapping):
        raise ValueError("router replay manifest must be a mapping")
    return {
        "model_id": model_id,
        "replay_set_id": replay_set_id,
        "manifest": dict(manifest),
    }
