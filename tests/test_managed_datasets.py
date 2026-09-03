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

"""Managed-dataset SDK wire-contract tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from weaver import (
    WEAVER_REDACTED_TOKEN_ID,
    AsyncServiceClient,
    ServiceClient,
    align_training_outputs,
    attach_loss_fn_outputs,
)
from weaver._payloads import (
    build_surrogate_data,
    parse_logprob_tensors,
    prepare_forward_backward_operation,
)
from weaver.async_training_client import AsyncTrainingClient
from weaver.training_client import TrainingClient
from weaver.types import Datum, ModelInput, SampleRef, SampleRefOutput
from weaver.types.tensor import TensorData


def _dataset_payload(**updates):
    payload = {
        "name": "hq-math",
        "version": "2026-08",
        "description": "High-quality math reasoning data",
        "sample_count": 120_000,
        "recommended_ratio": 0.2,
        "compatible_models": ["qwen-*", "llama-*"],
        "status": "published",
        "internal_id": "must-not-be-retained",
        "storage_path": "/gpfs/secret",
    }
    payload.update(updates)
    return payload


def _sample_output(datum: Datum, *, count: int = 2, with_targets: bool = True):
    assert datum.sample_ref is not None
    payload = {
        "kind": "sample_ref_output",
        "datum_id": datum.datum_id,
        "sample_ref": datum.sample_ref.to_payload(),
        "input_token_count": count,
        "logprobs": {"data": [-0.7] * count, "dtype": "float32", "shape": [count]},
        "elementwise_loss": [0.7] * count,
    }
    if with_targets:
        payload["target_tokens"] = [WEAVER_REDACTED_TOKEN_ID] * count
    return payload


def _training_client() -> TrainingClient:
    service = ServiceClient()
    service._http = MagicMock()
    return TrainingClient(
        service=service,
        model_id="model-1",
        base_model="Qwen/Qwen3-8B",
        session_id="session-1",
    )


def _async_training_client() -> AsyncTrainingClient:
    service = AsyncServiceClient()
    service._http = MagicMock()
    service._http.post = AsyncMock()
    return AsyncTrainingClient(
        service=service,
        model_id="model-1",
        base_model="Qwen/Qwen3-8B",
        session_id="session-1",
    )


def test_sample_ref_validation_and_payload():
    datum = Datum.from_sample_ref(
        dataset="hq-math", version="2026-08", sample_idx=3, datum_id=" d-3 "
    )

    assert datum.sample_ref == SampleRef("hq-math", "2026-08", 3)
    assert datum.to_payload() == {
        "kind": "sample_ref",
        "datum_id": "d-3",
        "dataset": "hq-math",
        "version": "2026-08",
        "sample_idx": 3,
        "loss_fn_inputs": {},
    }

    for kwargs in (
        {"dataset": "", "version": "v1", "sample_idx": 0},
        {"dataset": "d", "version": " ", "sample_idx": 0},
        {"dataset": " d", "version": "v1", "sample_idx": 0},
        {"dataset": "d", "version": "v1/part", "sample_idx": 0},
        {"dataset": "d\\part", "version": "v1", "sample_idx": 0},
        {"dataset": ".", "version": "v1", "sample_idx": 0},
        {"dataset": "d", "version": "..", "sample_idx": 0},
        {"dataset": "d\npart", "version": "v1", "sample_idx": 0},
        {"dataset": "d", "version": "v" * 129, "sample_idx": 0},
        {"dataset": "d", "version": "v1", "sample_idx": -1},
        {"dataset": "d", "version": "v1", "sample_idx": True},
    ):
        with pytest.raises(ValueError):
            Datum.from_sample_ref(**kwargs)

    with pytest.raises(ValueError, match="datum_id"):
        Datum.from_sample_ref(dataset="d", version="v1", sample_idx=0, datum_id=" ")
    with pytest.raises(ValueError, match="at most 255"):
        Datum.from_sample_ref(dataset="d", version="v1", sample_idx=0, datum_id="x" * 256)


def test_repeated_sample_refs_get_distinct_occurrence_ids():
    first = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=7)
    second = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=7)

    assert first.sample_ref == second.sample_ref
    assert first.datum_id != second.datum_id


def test_legacy_datum_wire_shape_is_unchanged_and_new_id_is_optional():
    legacy = Datum.from_raw(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [2, 3]},
    )
    identified = Datum.from_raw(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [2, 3]},
        datum_id="local-1",
    )

    assert set(legacy.to_payload()) == {"model_input", "loss_fn_inputs"}
    assert "kind" not in identified.to_payload()
    assert identified.to_payload()["datum_id"] == "local-1"


@pytest.mark.parametrize("field", ["target_tokens", "loss_mask", "weights", "sampling_mask"])
def test_sample_ref_rejects_server_owned_inputs(field):
    with pytest.raises(ValueError, match="server-owned"):
        Datum.from_sample_ref(dataset="d", version="v1", sample_idx=0, loss_fn_inputs={field: [1]})


def test_sample_ref_serializes_inline_tensor_data_like_an_ordinary_datum():
    datum = Datum.from_sample_ref(
        dataset="d",
        version="v1",
        sample_idx=0,
        datum_id="managed-0",
        loss_fn_inputs={"advantages": TensorData(data=[0.25, -0.5], dtype="float32")},
    )

    assert datum.to_payload()["loss_fn_inputs"]["advantages"] == {
        "data": [0.25, -0.5],
        "dtype": "float32",
        "shape": [2],
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.25, 0.25),
        (3, 3),
        (torch.tensor(0.5), 0.5),
        (TensorData.from_array(torch.tensor(2.0)), 2.0),
    ],
)
def test_sample_ref_serializes_numeric_scalars_as_inline_json(value, expected):
    datum = Datum.from_sample_ref(
        dataset="d",
        version="v1",
        sample_idx=0,
        datum_id="managed-0",
        loss_fn_inputs={"coefficient": value},
    )

    assert datum.to_payload()["loss_fn_inputs"]["coefficient"] == expected


def test_token_datum_numeric_scalar_normalization_is_unchanged():
    datum = Datum.from_raw(
        model_input=ModelInput.from_ints([1]),
        loss_fn_inputs={"coefficient": 0.25},
    )

    assert isinstance(datum.loss_fn_inputs["coefficient"], torch.Tensor)
    assert datum.loss_fn_inputs["coefficient"].ndim == 0


def test_sample_ref_rejects_non_vector_non_scalar_loss_input():
    datum = Datum.from_sample_ref(
        dataset="d",
        version="v1",
        sample_idx=0,
        loss_fn_inputs={"coefficient": [[1.0, 2.0]]},
    )

    with pytest.raises(ValueError, match="numeric scalars or one-dimensional"):
        datum.to_payload()


def test_negative_tokens_cannot_reenter_model_or_targets_but_minus_100_remains_ignore_index():
    with pytest.raises(ValueError, match="non-negative"):
        ModelInput.from_ints([1, WEAVER_REDACTED_TOKEN_ID])
    with pytest.raises(ValueError, match="response-only"):
        Datum.from_raw(
            model_input=ModelInput.from_ints([1]),
            loss_fn_inputs={"target_tokens": [WEAVER_REDACTED_TOKEN_ID]},
        )

    datum = Datum.from_raw(
        model_input=ModelInput.from_ints([1]),
        loss_fn_inputs={"target_tokens": [-100]},
    )
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [-100]

    mutable = ModelInput.from_ints([1])
    mutable.chunks[0].tokens.append(WEAVER_REDACTED_TOKEN_ID)
    with pytest.raises(ValueError, match="non-negative"):
        mutable.to_payload()


def test_sync_catalog_returns_only_typed_safe_fields_and_pagination():
    service = ServiceClient()
    service._http = MagicMock()
    service._http.get.return_value = {
        "items": [_dataset_payload()],
        "pagination": {"limit": 20, "offset": 40, "total_count": 101},
    }

    page = service.datasets.list(
        limit=20, offset=40, name="hq", status="published", compatible_model="qwen-*"
    )

    assert page[0].name == "hq-math"
    assert not hasattr(page[0], "internal_id")
    assert not hasattr(page[0], "storage_path")
    assert page.has_more
    service._http.get.assert_called_once_with(
        "/api/v1/managed-datasets",
        params={
            "limit": 20,
            "offset": 40,
            "name": "hq",
            "status": "published",
            "compatible_model": "qwen-*",
        },
    )


def test_catalog_get_quotes_safe_public_path_segments():
    service = ServiceClient()
    service._http = MagicMock()
    service._http.get.return_value = _dataset_payload()

    info = service.datasets.get(name="hq math", version="2026-08")

    assert info.sample_count == 120_000
    service._http.get.assert_called_once_with("/api/v1/managed-datasets/hq%20math/versions/2026-08")

    with pytest.raises(ValueError, match="safe path segment"):
        service.datasets.get(name="hq math", version="2026/08")


def test_async_catalog_has_the_same_contract():
    async def run():
        service = AsyncServiceClient()
        service._http = MagicMock()
        service._http.get = AsyncMock(
            return_value={
                "items": [_dataset_payload()],
                "pagination": {"limit": 100, "offset": 0, "total_count": 1},
            }
        )
        page = await service.datasets.list()
        return page, service._http.get

    page, get = asyncio.run(run())
    assert page[0].version == "2026-08"
    get.assert_awaited_once_with("/api/v1/managed-datasets", params={"limit": 100, "offset": 0})


def test_sync_model_bound_lengths_preserve_and_validate_order():
    client = _training_client()
    refs = [SampleRef("d", "v1", 4), SampleRef("d", "v1", 4), SampleRef("d", "v1", 8)]
    client._service._http.post.return_value = {
        "items": [
            {**refs[0].to_payload(), "input_token_count": 11},
            {**refs[1].to_payload(), "input_token_count": 11},
            {**refs[2].to_payload(), "input_token_count": 17},
        ]
    }

    lengths = client.resolve_sample_ref_lengths(refs)

    assert [item.input_token_count for item in lengths] == [11, 11, 17]
    client._service._http.post.assert_called_once_with(
        "/api/v1/models/model-1/managed-dataset-sample-lengths",
        json={"items": [ref.to_payload() for ref in refs]},
        max_retries=1,
    )


def test_model_bound_lengths_reject_reordering_and_inconsistent_duplicates():
    client = _training_client()
    refs = [SampleRef("d", "v1", 1), SampleRef("d", "v1", 2)]
    client._service._http.post.return_value = {
        "items": [
            {**refs[1].to_payload(), "input_token_count": 4},
            {**refs[0].to_payload(), "input_token_count": 5},
        ]
    }
    with pytest.raises(ValueError, match="request order"):
        client.resolve_sample_ref_lengths(refs)

    duplicate = SampleRef("d", "v1", 1)
    client._service._http.post.return_value = {
        "items": [
            {**duplicate.to_payload(), "input_token_count": 4},
            {**duplicate.to_payload(), "input_token_count": 5},
        ]
    }
    with pytest.raises(ValueError, match="inconsistent"):
        client.resolve_sample_ref_lengths([duplicate, duplicate])


def test_model_bound_lengths_chunk_server_limit_and_preserve_cross_chunk_duplicates(
    monkeypatch,
):
    monkeypatch.setattr("weaver.training_client.MAX_SAMPLE_REF_LENGTH_REQUEST_ITEMS", 2)
    client = _training_client()
    duplicate = SampleRef("d", "v1", 1)
    refs = [duplicate, SampleRef("d", "v1", 2), duplicate]

    def response(*_args, **kwargs):
        return {
            "items": [
                {**item, "input_token_count": 10 + item["sample_idx"]}
                for item in kwargs["json"]["items"]
            ]
        }

    client._service._http.post.side_effect = response

    lengths = client.resolve_sample_ref_lengths(refs)

    assert [item.sample_ref for item in lengths] == refs
    assert [item.input_token_count for item in lengths] == [11, 12, 11]
    assert client._service._http.post.call_count == 2


def test_async_model_bound_lengths_has_the_same_contract():
    async def run():
        client = _async_training_client()
        ref = SampleRef("d", "v1", 2)
        client._service._http.post.return_value = {
            "items": [{**ref.to_payload(), "input_token_count": 9}]
        }
        resolved = await client.resolve_sample_ref_lengths([ref])
        return resolved, ref, client._service._http.post

    resolved, ref, post = asyncio.run(run())
    assert resolved[0].input_token_count == 9
    post.assert_awaited_once_with(
        "/api/v1/models/model-1/managed-dataset-sample-lengths",
        json={"items": [ref.to_payload()]},
        max_retries=1,
    )


def test_managed_output_allows_optional_redacted_tokens_and_checks_all_lengths():
    datum = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2, datum_id="managed-2")
    without_tokens = SampleRefOutput.from_payload(
        _sample_output(datum, count=2, with_targets=False)
    )
    assert without_tokens.is_redacted
    assert without_tokens.target_tokens is None
    assert without_tokens.input_token_count == 2

    wrong_sentinel = _sample_output(datum)
    wrong_sentinel["target_tokens"] = [-8, 3]
    with pytest.raises(ValueError, match="only the -8 sentinel"):
        SampleRefOutput.from_payload(wrong_sentinel)

    wrong_length = _sample_output(datum)
    wrong_length["logprobs"] = [-0.7]
    with pytest.raises(ValueError, match="logprobs length"):
        SampleRefOutput.from_payload(wrong_length)

    for field_name, value in (
        ("logprobs", [float("nan"), -0.7]),
        ("loss", float("inf")),
        ("entropy", [0.1, float("-inf")]),
    ):
        non_finite = _sample_output(datum)
        non_finite[field_name] = value
        with pytest.raises(ValueError, match="finite numeric"):
            SampleRefOutput.from_payload(non_finite)

    with_extra = _sample_output(datum)
    with_extra["entropy"] = [0.1, 0.2]
    with_extra["per_token_kl"] = [0.01, 0.02]
    with_extra["token_losses"] = [0.7, 0.8]
    parsed = SampleRefOutput.from_payload(with_extra)
    assert parsed.get_derived_output("entropy") == (0.1, 0.2)
    assert parsed.get_derived_output("per_token_kl") == (0.01, 0.02)
    assert parsed.get_derived_output("token_losses") == (0.7, 0.8)

    redacted_extra = _sample_output(datum)
    redacted_extra["output-tokens"] = [-8, -8]
    redacted_extra["prompt_tokens"] = 2
    parsed_redacted = SampleRefOutput.from_payload(redacted_extra)
    assert parsed_redacted.redacted_token_outputs["output-tokens"] == (-8, -8)
    assert parsed_redacted.get_derived_output("prompt_tokens") == 2.0

    for identity_field in (
        "token_ids",
        "input_ids",
        "target_ids",
        "output_ids",
        "prompt_ids",
        "generated_ids",
        "teacher_tokens",
        "teacher_labels",
    ):
        identity_leak = _sample_output(datum)
        identity_leak[identity_field] = [1, 2]
        with pytest.raises(ValueError, match="only the -8 sentinel"):
            SampleRefOutput.from_payload(identity_leak)

        redacted_identity = _sample_output(datum)
        redacted_identity[identity_field] = [-8, -8]
        parsed_identity = SampleRefOutput.from_payload(redacted_identity)
        assert parsed_identity.redacted_token_outputs[identity_field] == (-8, -8)

    for forbidden_field, unsafe_value in (
        ("request_ids", [1, 2]),
        ("adapter_id", 17),
        ("decoded_text", [1, 2]),
        ("raw_logits", [0.1, 0.2]),
    ):
        invalid_response = _sample_output(datum)
        invalid_response[forbidden_field] = unsafe_value
        with pytest.raises(ValueError, match="forbidden in a managed output"):
            SampleRefOutput.from_payload(invalid_response)

    too_long_id = _sample_output(datum)
    too_long_id["datum_id"] = "x" * 256
    with pytest.raises(ValueError, match="at most 255"):
        SampleRefOutput.from_payload(too_long_id)


def test_mixed_output_alignment_and_safe_reattachment_preserve_datum_kinds():
    local = Datum.from_raw(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [2, 3]},
        datum_id="local-1",
    )
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2, datum_id="managed-2")
    result = {
        "result": {
            "loss_fn_outputs": [
                {"datum_id": "local-1", "logprobs": {"data": [-0.1, -0.2]}},
                _sample_output(managed),
            ]
        }
    }

    aligned = align_training_outputs([local, managed], result)
    attached = attach_loss_fn_outputs([local, managed], result)

    assert isinstance(aligned[1], SampleRefOutput)
    assert attached[0].model_input is local.model_input
    assert attached[1].sample_ref == managed.sample_ref
    assert attached[1].datum_id == managed.datum_id
    assert "target_tokens" not in attached[1].loss_fn_inputs
    assert attached[1].loss_fn_inputs["old_logprobs"].tolist() == pytest.approx([-0.7, -0.7])

    result["result"]["loss_fn_outputs"][1]["per_token_kl"] = [0.01, 0.02]
    result["result"]["loss_fn_outputs"][1]["token_losses"] = [0.7, 0.8]
    result["result"]["loss_fn_outputs"][0]["per_token_kl"] = [0.03, 0.04]
    result["result"]["loss_fn_outputs"][0]["token_losses"] = [0.5, 0.6]
    attached_derived = attach_loss_fn_outputs(
        [local, managed],
        result,
        field_map={"per_token_kl": "old_kl", "token_losses": "old_token_losses"},
    )
    assert attached_derived[1].loss_fn_inputs["old_kl"].tolist() == pytest.approx([0.01, 0.02])
    assert attached_derived[1].loss_fn_inputs["old_token_losses"].tolist() == pytest.approx(
        [0.7, 0.8]
    )


def test_mixed_output_alignment_requires_ids_on_every_occurrence():
    local = Datum.from_raw(
        model_input=ModelInput.from_ints([1]), loss_fn_inputs={"target_tokens": [2]}
    )
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2)
    result = {
        "result": {
            "loss_fn_outputs": [
                {"logprobs": [0.0]},
                _sample_output(managed, count=1),
            ]
        }
    }
    with pytest.raises(ValueError, match="requires datum_id"):
        align_training_outputs([local, managed], result)


def test_custom_surrogate_preserves_sample_ref_without_synthesizing_targets():
    datum = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2, datum_id="d-2")
    result = {"result": {"loss_fn_outputs": [_sample_output(datum)]}}

    logprobs = parse_logprob_tensors(result, [datum])
    (logprobs[0] * 3.0).sum().backward()
    surrogate = build_surrogate_data([datum], logprobs)

    assert surrogate[0].sample_ref == datum.sample_ref
    assert surrogate[0].datum_id == datum.datum_id
    assert set(surrogate[0].loss_fn_inputs) == {"surrogate_weights"}
    assert surrogate[0].loss_fn_inputs["surrogate_weights"].tolist() == [3.0, 3.0]


def test_http_binary_keeps_sample_refs_inline_and_packs_only_local_datums():
    local = Datum.from_raw(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [2, 3], "weights": [0.0, 1.0]},
        datum_id="local",
    )
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2, datum_id="managed")
    prepared = prepare_forward_backward_operation(
        model_id="model-1",
        seq_id=1,
        data=[managed, local],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        request_metadata=None,
        tensor_transport="http-binary",
    )
    try:
        assert prepared.tensor_pack is not None
        wire = prepared.body["payload"]["forward_backward_input"]["data"]
        assert wire[0] == managed.to_payload()
        assert "$tensor" in wire[1]["model_input"]["chunks"][0]["tokens"]
        assert wire[1]["datum_id"] == "local"
    finally:
        prepared.close()

    managed_only = prepare_forward_backward_operation(
        model_id="model-1",
        seq_id=2,
        data=[managed],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        request_metadata=None,
        tensor_transport="http-binary",
    )
    assert managed_only.tensor_pack is None


def test_create_model_pins_training_max_sequence_length_sync_and_async():
    service = ServiceClient()
    service._session_id = "session-1"
    service._http = MagicMock()
    service._http.post.return_value = {
        "id": "model-1",
        "base_model": "Qwen/Qwen3-8B",
        "tokenizer_path": "/tokenizer",
    }
    service.create_model(
        base_model="Qwen/Qwen3-8B",
        training_max_sequence_length=4096,
    )
    assert service._http.post.call_args.kwargs["json"]["training_max_sequence_length"] == 4096

    async def run():
        async_service = AsyncServiceClient()
        async_service._session_id = "session-1"
        async_service._session = {"id": "session-1"}
        async_service._http = MagicMock()
        async_service._http.post = AsyncMock(
            return_value={
                "id": "model-2",
                "base_model": "Qwen/Qwen3-8B",
                "tokenizer_path": "/tokenizer",
            }
        )
        await async_service.create_model(
            base_model="Qwen/Qwen3-8B", training_max_sequence_length=2048
        )
        return async_service._http.post

    async_post = asyncio.run(run())
    assert async_post.call_args.kwargs["json"]["training_max_sequence_length"] == 2048

    with pytest.raises(ValueError, match=">= 2"):
        service.create_model(base_model="Qwen/Qwen3-8B", training_max_sequence_length=1)


def test_sdk_does_not_close_protocol_over_loss_names_or_client_derived_fields():
    client = _training_client()
    datum = Datum.from_sample_ref(
        dataset="d",
        version="v1",
        sample_idx=0,
        loss_fn_inputs={"future_scalar_signal": [1.0]},
    )
    handle = MagicMock()
    handle.result.return_value = {}
    client._service.enqueue_operation = MagicMock(return_value=handle)

    client.forward([datum], "future_server_loss")

    wire = client._service.enqueue_operation.call_args.args[1]
    inputs = wire["payload"]["forward_input"]["data"][0]["loss_fn_inputs"]
    assert inputs["future_scalar_signal"]["data"] == [1.0]
