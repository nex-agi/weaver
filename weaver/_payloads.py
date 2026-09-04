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

"""Request/response helpers shared by the sync and async clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, BinaryIO, Dict, List, Mapping, Sequence

from .config import TensorCompression, TensorTransport
from .tensor_transport import PreparedOperationBody, serialize_training_data
from .training_outputs import align_training_outputs
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
    from .types.datum import normalize_mixed_datum_ids

    return [datum.to_payload() for datum in normalize_mixed_datum_ids(data)]


def validate_sample_ref_operation(
    data: Sequence[Datum],
    *,
    operation: str,
    loss_fn: str,
    loss_fn_config: Mapping[str, Any] | None = None,
    tensor_transport: TensorTransport = "default",
) -> None:
    """Enforce the phase-one SFT-only boundary for managed samples."""

    managed = [(index, datum) for index, datum in enumerate(data) if datum.is_sample_ref]
    if not managed:
        return
    if operation != "forward_backward" or loss_fn != "cross_entropy":
        raise ValueError("SampleRef data only supports built-in cross_entropy forward_backward")
    if loss_fn_config:
        raise ValueError("SampleRef cross_entropy forward_backward requires empty loss_fn_config")
    if tensor_transport != "default":
        raise ValueError(
            "SampleRef cross_entropy forward_backward requires default JSON tensor transport"
        )
    for index, datum in managed:
        if datum.metadata:
            raise ValueError(
                f"datum {index}: SampleRef cross_entropy forward_backward requires empty metadata"
            )


def _prepare_training_operation(
    *,
    input_key: str,
    model_id: str,
    seq_id: int,
    data: Sequence[Datum],
    loss_fn: str,
    loss_fn_config: Mapping[str, Any] | None,
    request_metadata: Dict[str, Any] | None,
    tensor_transport: TensorTransport,
    tensor_compression: TensorCompression,
) -> PreparedOperationBody:
    operation = "forward_backward" if input_key == "forward_backward_input" else "forward"
    validate_sample_ref_operation(
        data,
        operation=operation,
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
        tensor_transport=tensor_transport,
    )
    serialized = serialize_training_data(
        data,
        loss_fn=loss_fn,
        transport=tensor_transport,
        compression=tensor_compression,
    )
    payload: Dict[str, Any] = {
        "model_id": model_id,
        "seq_id": seq_id,
        "tensor_transport": tensor_transport,
        input_key: {
            "loss_fn": loss_fn,
            "data": serialized.data,
        },
    }
    if tensor_transport == "http-binary":
        payload["tensor_compression"] = tensor_compression
    if loss_fn_config:
        payload[input_key]["loss_fn_config"] = dict(loss_fn_config)
    if request_metadata:
        payload["metadata"] = request_metadata
    return PreparedOperationBody({"payload": payload}, serialized.tensor_pack)


def prepare_forward_operation(
    *,
    model_id: str,
    seq_id: int,
    data: Sequence[Datum],
    loss_fn: str,
    loss_fn_config: Mapping[str, Any] | None,
    request_metadata: Dict[str, Any] | None,
    tensor_transport: TensorTransport,
    tensor_compression: TensorCompression = "zstd",
) -> PreparedOperationBody:
    """Prepare a forward operation and optional HTTP tensor attachment."""

    return _prepare_training_operation(
        input_key="forward_input",
        model_id=model_id,
        seq_id=seq_id,
        data=data,
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
        request_metadata=request_metadata,
        tensor_transport=tensor_transport,
        tensor_compression=tensor_compression,
    )


def prepare_forward_backward_operation(
    *,
    model_id: str,
    seq_id: int,
    data: Sequence[Datum],
    loss_fn: str,
    loss_fn_config: Mapping[str, Any] | None,
    request_metadata: Dict[str, Any] | None,
    tensor_transport: TensorTransport,
    tensor_compression: TensorCompression = "zstd",
) -> PreparedOperationBody:
    """Prepare a forward/backward operation and optional HTTP tensor attachment."""

    return _prepare_training_operation(
        input_key="forward_backward_input",
        model_id=model_id,
        seq_id=seq_id,
        data=data,
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
        request_metadata=request_metadata,
        tensor_transport=tensor_transport,
        tensor_compression=tensor_compression,
    )


def forward_payload(
    *,
    model_id: str,
    seq_id: int,
    data: Sequence[Datum],
    loss_fn: str,
    loss_fn_config: Mapping[str, Any] | None,
    request_metadata: Dict[str, Any] | None,
) -> Dict[str, Any]:
    from .types.datum import validate_sample_ref_loss_inputs

    validate_sample_ref_operation(
        data,
        operation="forward",
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
    )
    validate_sample_ref_loss_inputs(data, loss_fn)
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
    from .types.datum import validate_sample_ref_loss_inputs

    validate_sample_ref_operation(
        data,
        operation="forward_backward",
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
    )
    validate_sample_ref_loss_inputs(data, loss_fn)
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
    fwd_result: Dict[str, Any],
    data: Sequence[Datum],
    *,
    tensor_pack: BinaryIO | None = None,
) -> List["torch.Tensor"]:
    """Extract per-datum logprob tensors (``requires_grad=True``) from a forward result."""
    import torch

    validate_sample_ref_operation(data, operation="forward", loss_fn="forward_logprob")
    parsed_result: Mapping[str, Any] = fwd_result
    if tensor_pack is not None:
        from .tensor_transport import materialize_http_tensor_payloads

        parsed_result = materialize_http_tensor_payloads(fwd_result, tensor_pack)
    outputs = align_training_outputs(data, parsed_result)
    if not outputs:
        raise ValueError("Forward pass returned no loss_fn_outputs")

    logprob_tensors: List[torch.Tensor] = []
    for output in outputs:
        if not isinstance(output, Mapping):
            raise ValueError("SampleRef outputs cannot be used for custom training")
        lp = output.get("logprobs")
        if lp is None:
            lp = output.get("Logprobs")
        if isinstance(lp, dict) and "$tensor" in lp:
            if tensor_pack is None:
                raise ValueError("HTTP tensor logprobs require the operation tensor pack")
            from .tensor_transport import materialize_http_tensor

            lp = materialize_http_tensor(lp, tensor_pack)
        elif isinstance(lp, dict):
            lp = lp["data"]
        if lp is None:
            raise ValueError("Missing logprobs in forward/backward output")
        tensor = torch.as_tensor(lp, dtype=torch.float32).detach().clone()
        logprob_tensors.append(tensor.requires_grad_(True))
    return logprob_tensors


def build_surrogate_data(
    data: Sequence[Datum], logprob_tensors: Sequence["torch.Tensor"]
) -> List[Datum]:
    """Build surrogate :class:`~weaver.types.Datum` objects carrying gradient weights."""
    validate_sample_ref_operation(data, operation="forward_backward", loss_fn="surrogate")
    surrogate_data: List[Datum] = []
    for i, (datum, logprob_tensor) in enumerate(zip(data, logprob_tensors)):
        if logprob_tensor.grad is None:
            raise ValueError(f"logprob_tensors[{i}] has no gradient after backward")
        if logprob_tensor.grad.shape != logprob_tensor.shape:
            raise ValueError(
                f"logprob_tensors[{i}] gradient shape must match its target-position shape"
            )

        raw_targets = datum.loss_fn_inputs.get("target_tokens")
        if raw_targets is None:
            assert datum.model_input is not None
            resolved_targets: List[Any] = datum.model_input.to_ints()
        elif hasattr(raw_targets, "tolist"):
            resolved_targets = raw_targets.tolist()
        else:
            resolved_targets = list(raw_targets)

        loss_fn_inputs: Dict[str, Any] = dict(datum.loss_fn_inputs)
        loss_fn_inputs["target_tokens"] = resolved_targets
        loss_fn_inputs["surrogate_weights"] = logprob_tensor.grad.detach().tolist()
        surrogate_data.append(datum.with_loss_fn_inputs(loss_fn_inputs))
    return surrogate_data
