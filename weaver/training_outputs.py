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

"""Safe parsing and correlation helpers for per-datum training outputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types.datum import Datum, normalize_mixed_datum_ids
from .types.managed_dataset import (
    SampleRefOutput,
    _datum_id,
    _is_token_identity_field,
    _one_dimensional_values,
)

AlignedTrainingOutput = SampleRefOutput | Mapping[str, Any]
_MISSING = object()


def _validated_loss_fn_outputs(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("training result must contain a loss_fn_outputs array")
    if not all(isinstance(output, Mapping) for output in value):
        raise ValueError("every loss_fn_outputs entry must be an object")
    return list(value)


def _same_loss_fn_outputs(left: list[Mapping[str, Any]], right: list[Mapping[str, Any]]) -> bool:
    if left is right:
        return True
    try:
        equivalent = left == right
    except (TypeError, ValueError, RuntimeError):
        return False
    return isinstance(equivalent, bool) and equivalent


def _loss_fn_outputs(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract V1-wrapped or V2-direct outputs without guessing precedence."""

    direct = result.get("loss_fn_outputs", _MISSING)
    nested_container = result.get("result", _MISSING)
    nested = _MISSING
    if nested_container is not _MISSING:
        if not isinstance(nested_container, Mapping):
            if direct is not _MISSING:
                raise ValueError("training result contains ambiguous result envelopes")
            raise ValueError("training result result field must be an object")
        nested = nested_container.get("loss_fn_outputs", _MISSING)

    if direct is _MISSING and nested is _MISSING:
        raise ValueError("training result must contain a loss_fn_outputs array")

    direct_outputs = _validated_loss_fn_outputs(direct) if direct is not _MISSING else None
    nested_outputs = _validated_loss_fn_outputs(nested) if nested is not _MISSING else None
    if direct_outputs is not None and nested_outputs is not None:
        if not _same_loss_fn_outputs(direct_outputs, nested_outputs):
            raise ValueError("training result contains ambiguous loss_fn_outputs arrays")
        return direct_outputs
    if direct_outputs is not None:
        return direct_outputs
    assert nested_outputs is not None
    return nested_outputs


def align_training_outputs(
    data: Sequence[Datum], result: Mapping[str, Any]
) -> list[AlignedTrainingOutput]:
    """Validate and align per-datum outputs without consuming redacted tokens.

    Both V1's nested ``result`` envelope and V2's direct operation response are
    accepted. If both shapes are present, they must carry equivalent output
    arrays. Legacy batches in which every token datum lacks ``datum_id`` retain
    the historical positional behavior. Mixed or ID-bearing batches require a
    unique ID on every datum and output, and wire order remains a checked
    invariant in addition to exact ID-set equality. Completed operation handles
    materialize HTTP-binary tensors before this parser runs.
    """

    normalized_data = normalize_mixed_datum_ids(data)
    outputs = _loss_fn_outputs(result)
    if len(outputs) != len(normalized_data):
        raise ValueError(f"Expected {len(normalized_data)} loss_fn_outputs, got {len(outputs)}")

    if all(datum.datum_id is None and not datum.is_sample_ref for datum in normalized_data):
        legacy_outputs: list[AlignedTrainingOutput] = list(outputs)
        return legacy_outputs
    if any(datum.datum_id is None for datum in normalized_data):
        raise ValueError("mixed/new-style output alignment requires datum_id on every datum")

    input_ids = [datum.datum_id for datum in normalized_data]
    assert all(datum_id is not None for datum_id in input_ids)
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("input datum_id values must be unique")

    output_ids: list[str] = []
    aligned: list[AlignedTrainingOutput] = []
    for index, (datum, payload) in enumerate(zip(normalized_data, outputs)):
        try:
            output_id = _datum_id(payload.get("datum_id"))
        except ValueError as exc:
            raise ValueError(f"loss_fn_outputs[{index}] has invalid datum_id: {exc}") from exc
        output_ids.append(output_id)
        if output_id != datum.datum_id:
            raise ValueError(
                f"loss_fn_outputs[{index}] datum_id does not match the input wire order"
            )

        if datum.is_sample_ref:
            managed = SampleRefOutput.from_payload(payload)
            if managed.sample_ref != datum.sample_ref:
                raise ValueError(f"loss_fn_outputs[{index}] sample_ref does not match input")
            aligned.append(managed)
        else:
            if payload.get("kind") == "sample_ref_output":
                raise ValueError(f"loss_fn_outputs[{index}] changed a token Datum into SampleRef")
            aligned.append(payload)

    if len(set(output_ids)) != len(output_ids) or set(output_ids) != set(input_ids):
        raise ValueError("output datum_id values must be a unique exact match for input IDs")
    return aligned


def attach_loss_fn_outputs(
    data: Sequence[Datum],
    result: Mapping[str, Any],
    *,
    field_map: Mapping[str, str] | None = None,
) -> list[Datum]:
    """Copy datums and attach selected aligned numeric outputs as loss inputs.

    Token identity fields are deliberately unaddressable here. Phase-one
    SampleRefs use a closed SFT input contract, so managed outputs cannot be
    reattached as client loss inputs even when their content is public.
    """

    selected = dict(field_map or {"logprobs": "old_logprobs"})
    if not selected:
        return list(data)
    if any(datum.is_sample_ref for datum in data):
        raise ValueError("SampleRef outputs cannot be attached as loss inputs in phase one")
    protected_destinations = {"target_tokens", "loss_mask", "weights", "model_input"}
    if any(_is_token_identity_field(output_name) for output_name in selected) or (
        protected_destinations & set(selected.values())
    ):
        raise ValueError("token-bearing or server-owned fields cannot be attached as loss inputs")

    normalized_data = normalize_mixed_datum_ids(data)
    aligned = align_training_outputs(normalized_data, result)
    copied: list[Datum] = []
    for index, (datum, output) in enumerate(zip(normalized_data, aligned)):
        updates: dict[str, Any] = {}
        for output_name, input_name in selected.items():
            value: Any
            if isinstance(output, SampleRefOutput):
                value = output.get_derived_output(output_name)
                if value is None:
                    raise ValueError(f"loss_fn_outputs[{index}] is missing {output_name}")
            else:
                if output_name not in output:
                    raise ValueError(f"loss_fn_outputs[{index}] is missing {output_name}")
                value = _one_dimensional_values(output[output_name], output_name)
            updates[input_name] = value
        copied.append(datum.with_loss_fn_inputs(updates))
    return copied
