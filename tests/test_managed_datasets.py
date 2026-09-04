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

from weaver import WEAVER_REDACTED_TOKEN_ID, AsyncServiceClient, ServiceClient
from weaver import _sampling_utils as _su
from weaver import align_training_outputs, attach_loss_fn_outputs
from weaver._payloads import (
    build_surrogate_data,
    parse_logprob_tensors,
    prepare_forward_backward_operation,
    prepare_forward_operation,
)
from weaver.async_sampling_client import AsyncSamplingClient
from weaver.async_training_client import AsyncTrainingClient
from weaver.sampling_client import SamplingClient
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
        "content_visibility": "protected",
        "internal_id": "must-not-be-retained",
        "storage_path": "/gpfs/secret",
    }
    payload.update(updates)
    return payload


def _sample_output(
    datum: Datum,
    *,
    count: int = 2,
    with_targets: bool = True,
    content_visibility: str = "protected",
):
    assert datum.sample_ref is not None
    payload = {
        "kind": "sample_ref_output",
        "datum_id": datum.datum_id,
        "sample_ref": datum.sample_ref.to_payload(),
        "input_token_count": count,
        "content_visibility": content_visibility,
    }
    if with_targets:
        payload["target_tokens"] = (
            [WEAVER_REDACTED_TOKEN_ID] * count
            if content_visibility == "protected"
            else list(range(101, 101 + count))
        )
    if content_visibility == "public":
        payload["logprobs"] = {
            "data": [-0.7] * count,
            "dtype": "float32",
            "shape": [count],
        }
        payload["elementwise_loss"] = [0.7] * count
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


def test_natural_mixed_sft_batch_assigns_stable_local_occurrence_ids():
    local = Datum.from_raw(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [2, 3], "custom_signal": [0.25, 0.5]},
        metadata={"source": "local"},
    )
    managed = Datum.from_sample_ref(
        dataset="d",
        version="v1",
        sample_idx=2,
        loss_fn_inputs={"coefficient": 0.75},
    )

    def wire_data():
        prepared = prepare_forward_backward_operation(
            model_id="model-1",
            seq_id=7,
            data=[local, managed],
            loss_fn="cross_entropy",
            loss_fn_config=None,
            request_metadata=None,
            tensor_transport="default",
        )
        try:
            return prepared.body["payload"]["forward_backward_input"]["data"]
        finally:
            prepared.close()

    first = wire_data()
    second = wire_data()

    assert local.datum_id is None
    assert first[0]["datum_id"].startswith("d-mixed-")
    assert first[0]["datum_id"] == second[0]["datum_id"]
    assert first[1]["datum_id"] == managed.datum_id
    assert first[0]["loss_fn_inputs"]["target_tokens"]["data"] == [2, 3]
    assert first[0]["loss_fn_inputs"]["custom_signal"]["data"] == [0.25, 0.5]
    assert first[0]["metadata"] == {"source": "local"}
    assert first[1]["loss_fn_inputs"] == {"coefficient": 0.75}


def test_managed_sft_forwards_token_mean_loss_configuration():
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2)
    configured = {"loss_agg_mode": "token-mean", "entropy_coeff": 0.0}

    prepared = prepare_forward_backward_operation(
        model_id="model-1",
        seq_id=7,
        data=[managed],
        loss_fn="cross_entropy",
        loss_fn_config=configured,
        request_metadata=None,
        tensor_transport="default",
    )
    try:
        forwarded = prepared.body["payload"]["forward_backward_input"]
    finally:
        prepared.close()

    assert forwarded["loss_fn_config"] == configured


def test_mixed_batch_preserves_explicit_ids_and_distinguishes_repeated_object_occurrences():
    repeated = Datum.from_raw(
        model_input=ModelInput.from_ints([1]),
        loss_fn_inputs={"target_tokens": [2]},
    )
    identified = Datum.from_raw(
        model_input=ModelInput.from_ints([3]),
        loss_fn_inputs={"target_tokens": [4]},
        datum_id="local-explicit",
    )
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2)

    prepared = prepare_forward_backward_operation(
        model_id="model-1",
        seq_id=7,
        data=[repeated, identified, managed, repeated],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        request_metadata=None,
        tensor_transport="default",
    )
    try:
        wire = prepared.body["payload"]["forward_backward_input"]["data"]
    finally:
        prepared.close()

    assert wire[1]["datum_id"] == "local-explicit"
    assert wire[2]["datum_id"] == managed.datum_id
    assert wire[0]["datum_id"] != wire[3]["datum_id"]
    assert len({datum["datum_id"] for datum in wire}) == len(wire)


@pytest.mark.parametrize("transport", ["default", "http-binary"])
def test_shared_payload_builders_enforce_sample_ref_sft_only(transport):
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2)
    common = {
        "model_id": "model-1",
        "seq_id": 7,
        "data": [managed],
        "loss_fn_config": None,
        "request_metadata": None,
        "tensor_transport": transport,
    }

    with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
        prepare_forward_operation(loss_fn="cross_entropy", **common)
    with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
        prepare_forward_backward_operation(loss_fn="surrogate", **common)


@pytest.mark.parametrize(
    ("prepare", "input_key", "transport"),
    [
        (prepare_forward_operation, "forward_input", "default"),
        (prepare_forward_operation, "forward_input", "http-binary"),
        (prepare_forward_backward_operation, "forward_backward_input", "default"),
        (prepare_forward_backward_operation, "forward_backward_input", "http-binary"),
    ],
)
def test_token_only_batch_retains_legacy_missing_id_wire_shape(prepare, input_key, transport):
    local = Datum.from_raw(
        model_input=ModelInput.from_ints([1]), loss_fn_inputs={"target_tokens": [2]}
    )

    prepared = prepare(
        model_id="model-1",
        seq_id=7,
        data=[local, local],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        request_metadata=None,
        tensor_transport=transport,
    )
    try:
        wire = prepared.body["payload"][input_key]["data"]
    finally:
        prepared.close()

    assert all("datum_id" not in datum for datum in wire)


@pytest.mark.parametrize(
    ("prepare", "input_key"),
    [
        (prepare_forward_operation, "forward_input"),
        (prepare_forward_backward_operation, "forward_backward_input"),
    ],
)
def test_token_only_partially_identified_batch_assigns_missing_occurrence_ids(prepare, input_key):
    missing = Datum.from_raw(
        model_input=ModelInput.from_ints([1]), loss_fn_inputs={"target_tokens": [2]}
    )
    identified = Datum.from_raw(
        model_input=ModelInput.from_ints([3]),
        loss_fn_inputs={"target_tokens": [4]},
        datum_id="local-explicit",
    )

    def wire_data():
        prepared = prepare(
            model_id="model-1",
            seq_id=7,
            data=[missing, identified],
            loss_fn="cross_entropy",
            loss_fn_config=None,
            request_metadata=None,
            tensor_transport="default",
        )
        try:
            return prepared.body["payload"][input_key]["data"]
        finally:
            prepared.close()

    first = wire_data()
    second = wire_data()

    assert missing.datum_id is None
    assert first[0]["datum_id"].startswith("d-mixed-")
    assert first[0]["datum_id"] == second[0]["datum_id"]
    assert first[1]["datum_id"] == "local-explicit"


def test_generated_occurrence_id_anchor_is_unambiguous_for_control_char_ids():
    def local(datum_id=None):
        return Datum.from_raw(
            model_input=ModelInput.from_ints([1]),
            loss_fn_inputs={"target_tokens": [2]},
            datum_id=datum_id,
        )

    def generated_id(data):
        prepared = prepare_forward_backward_operation(
            model_id="model-1",
            seq_id=7,
            data=data,
            loss_fn="cross_entropy",
            loss_fn_config=None,
            request_metadata=None,
            tensor_transport="default",
        )
        try:
            return prepared.body["payload"]["forward_backward_input"]["data"][2]["datum_id"]
        finally:
            prepared.close()

    # Delimiter joining encoded both explicit-ID sets as ``0:a\x1f1:b``.
    delimiter_in_id = generated_id([local("a\x1f1:b"), local(), local()])
    separate_ids = generated_id([local("a"), local("b"), local()])

    assert delimiter_in_id != separate_ids


@pytest.mark.parametrize("field", ["model_input", "target_tokens", "loss_mask", "weights"])
def test_sample_ref_rejects_server_owned_inputs(field):
    with pytest.raises(ValueError, match="server-owned"):
        Datum.from_sample_ref(dataset="d", version="v1", sample_idx=0, loss_fn_inputs={field: [1]})


def test_sample_ref_sft_accepts_caller_loss_inputs():
    mask = [[1, 2], [3]]
    datum = Datum.from_sample_ref(
        dataset="open",
        version="v1",
        sample_idx=0,
        loss_fn_inputs={"sampling_mask": mask},
    )
    assert datum.to_payload()["loss_fn_inputs"]["sampling_mask"] == mask

    client = _training_client()
    handle = MagicMock()
    handle.result.return_value = {}
    client._service.enqueue_operation = MagicMock(return_value=handle)
    client.forward_backward([datum], "cross_entropy")
    client._service.enqueue_operation.assert_called_once()
    client._service._http.get.assert_not_called()


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


def test_sample_ref_serializes_multidimensional_sft_input():
    datum = Datum.from_sample_ref(
        dataset="d",
        version="v1",
        sample_idx=0,
        loss_fn_inputs={"coefficient": [[1.0, 2.0]]},
    )

    assert datum.to_payload()["loss_fn_inputs"]["coefficient"] == {
        "data": [[1.0, 2.0]],
        "dtype": "float32",
        "shape": [1, 2],
    }


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
    assert page[0].content_visibility == "protected"
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
    service._http.get.return_value = _dataset_payload(name="hq math")

    info = service.datasets.get(name="hq math", version="2026-08")

    assert info.sample_count == 120_000
    service._http.get.assert_called_once_with("/api/v1/managed-datasets/hq%20math/versions/2026-08")

    with pytest.raises(ValueError, match="safe path segment"):
        service.datasets.get(name="hq math", version="2026/08")

    for invalid in (None, "", "private", True):
        service._http.get.return_value = _dataset_payload(
            name="hq math", content_visibility=invalid
        )
        with pytest.raises(ValueError, match="content_visibility"):
            service.datasets.get(name="hq math", version="2026-08")


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


def test_managed_dataset_catalog_does_not_expose_download():
    assert not hasattr(ServiceClient().datasets, "download")
    assert not hasattr(AsyncServiceClient().datasets, "download")


def test_sampling_clients_reject_sample_refs_before_enqueue():
    ref = SampleRef("open", "v1", 0)
    sync_service = MagicMock()
    sync_client = SamplingClient(service=sync_service, sampling_session_id="sampling-1")
    with pytest.raises(TypeError, match="prompt must be ModelInput"):
        sync_client.sample(prompt=ref)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prompt must be ModelInput"):
        sync_client.compute_logprobs(prompt=ref)  # type: ignore[arg-type]
    sync_service.enqueue_operation.assert_not_called()

    async def run():
        async_service = MagicMock()
        async_service.enqueue_operation = AsyncMock()
        async_client = AsyncSamplingClient(service=async_service, sampling_session_id="sampling-1")
        with pytest.raises(TypeError, match="prompt must be ModelInput"):
            await async_client.sample(prompt=ref)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="prompt must be ModelInput"):
            await async_client.compute_logprobs(prompt=ref)  # type: ignore[arg-type]
        async_service.enqueue_operation.assert_not_awaited()

    asyncio.run(run())


def test_sampling_model_input_wire_shape_remains_unchanged():
    prompt = ModelInput.from_ints([1, 2, 3])
    assert _su.sampling_prompt_payload(prompt) == prompt.to_payload()


def test_token_in_training_operations_remain_unrestricted():
    client = _training_client()
    handle = MagicMock()
    handle.result.return_value = {}
    client._service.enqueue_operation = MagicMock(return_value=handle)
    datum = Datum.from_raw(
        model_input=ModelInput.from_ints([1]),
        loss_fn_inputs={"target_tokens": [2]},
    )

    client.forward([datum], "future_server_loss")
    client.forward_backward([datum], "surrogate")

    assert client._service.enqueue_operation.call_count == 2


def test_sync_model_bound_lengths_preserve_and_validate_order():
    client = _training_client()
    refs = [SampleRef("d", "v1", 4), SampleRef("d", "v1", 4), SampleRef("d", "v1", 8)]
    client._service._http.post.return_value = {
        "model_data_revision": "mdr1-profile-a",
        "items": [
            {**refs[0].to_payload(), "input_token_count": 11},
            {**refs[1].to_payload(), "input_token_count": 11},
            {**refs[2].to_payload(), "input_token_count": 17},
        ],
    }

    lengths = client.resolve_sample_ref_lengths(refs)

    assert [item.input_token_count for item in lengths] == [11, 11, 17]
    assert {item.model_data_revision for item in lengths} == {"mdr1-profile-a"}
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


def test_model_bound_lengths_reject_invalid_or_inconsistent_data_revision(monkeypatch):
    client = _training_client()
    ref = SampleRef("d", "v1", 1)
    client._service._http.post.return_value = {
        "model_data_revision": "",
        "items": [{**ref.to_payload(), "input_token_count": 11}],
    }
    with pytest.raises(ValueError, match="model_data_revision"):
        client.resolve_sample_ref_lengths([ref])

    client._service._http.post.return_value = {
        "model_data_revision": " mdr1-profile-a",
        "items": [{**ref.to_payload(), "input_token_count": 11}],
    }
    with pytest.raises(ValueError, match="boundary whitespace"):
        client.resolve_sample_ref_lengths([ref])

    monkeypatch.setattr("weaver.training_client.MAX_SAMPLE_REF_LENGTH_REQUEST_ITEMS", 1)
    revisions = iter(["mdr1-profile-a", "mdr1-profile-b"])

    def response(*_args, **kwargs):
        item = kwargs["json"]["items"][0]
        return {
            "model_data_revision": next(revisions),
            "items": [{**item, "input_token_count": 11}],
        }

    client._service._http.post.side_effect = response
    with pytest.raises(ValueError, match="inconsistent model_data_revision"):
        client.resolve_sample_ref_lengths([ref, SampleRef("d", "v1", 2)])


def test_async_model_bound_lengths_has_the_same_contract():
    async def run():
        client = _async_training_client()
        ref = SampleRef("d", "v1", 2)
        client._service._http.post.return_value = {
            "model_data_revision": "mdr1-profile-a",
            "items": [{**ref.to_payload(), "input_token_count": 9}],
        }
        resolved = await client.resolve_sample_ref_lengths([ref])
        return resolved, ref, client._service._http.post

    resolved, ref, post = asyncio.run(run())
    assert resolved[0].input_token_count == 9
    assert resolved[0].model_data_revision == "mdr1-profile-a"
    post.assert_awaited_once_with(
        "/api/v1/models/model-1/managed-dataset-sample-lengths",
        json={"items": [ref.to_payload()]},
        max_retries=1,
    )


def test_protected_managed_output_only_accepts_redaction_and_safe_counts():
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

    null_tokens = _sample_output(datum)
    null_tokens["target_tokens"] = None
    with pytest.raises(ValueError, match="one-dimensional array"):
        SampleRefOutput.from_payload(null_tokens)

    for label_field in (
        "logprobs",
        "elementwise_loss",
        "teacher_logprobs",
        "detached_kl_advantages",
        "per_token_kl",
        "token_losses",
    ):
        leak = _sample_output(datum)
        leak[label_field] = [-0.3, -0.4]
        with pytest.raises(ValueError, match="label-dependent per-token"):
            SampleRefOutput.from_payload(leak)

    for unknown_field, unsafe_value in (
        ("loss", 0.7),
        ("entropy", [0.1, 0.2]),
        ("predictions", [0.2, 0.8]),
        ("debug", "private"),
        ("metadata", {"value": 1}),
    ):
        unknown_output = _sample_output(datum)
        unknown_output[unknown_field] = unsafe_value
        with pytest.raises(ValueError, match="unsupported protected managed output field"):
            SampleRefOutput.from_payload(unknown_output)

    non_string_field = _sample_output(datum)
    non_string_field[7] = [0.1, 0.2]
    with pytest.raises(ValueError, match="field names must be strings"):
        SampleRefOutput.from_payload(non_string_field)

    redacted_extra = _sample_output(datum)
    redacted_extra["output-tokens"] = [-8, -8]
    redacted_extra["prompt_tokens"] = 2
    parsed_redacted = SampleRefOutput.from_payload(redacted_extra)
    assert parsed_redacted.redacted_token_outputs["output_tokens"] == (-8, -8)
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

    missing_visibility = _sample_output(datum)
    del missing_visibility["content_visibility"]
    with pytest.raises(ValueError, match="content_visibility"):
        SampleRefOutput.from_payload(missing_visibility)

    non_finite_count = _sample_output(datum)
    non_finite_count["token_count"] = float("inf")
    with pytest.raises(ValueError, match="finite numeric"):
        SampleRefOutput.from_payload(non_finite_count)


def test_public_managed_output_requires_real_tokens_and_preserves_ordinary_outputs():
    datum = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2, datum_id="managed-2")
    payload = _sample_output(datum, content_visibility="public")
    payload["teacher_logprobs"] = [-0.3, -0.4]
    payload["per_token_kl"] = [0.01, 0.02]
    payload["custom_result"] = {"nested": [1, 2]}
    payload["decoded_text"] = "public content"
    payload["raw_logits"] = [[0.1, 0.2], [0.3, 0.4]]
    payload["output_tokens"] = [201, 202, 203]
    payload["top_k_token_ids"] = [[201, 202], [203, 204]]

    parsed = SampleRefOutput.from_payload(payload)

    assert not parsed.is_redacted
    assert parsed.target_tokens == (101, 102)
    assert parsed.token_outputs["output_tokens"] == [201, 202, 203]
    assert parsed.token_outputs["top_k_token_ids"] == [[201, 202], [203, 204]]
    assert parsed.redacted_token_outputs == {}
    assert parsed.logprobs == (-0.7, -0.7)
    assert parsed.elementwise_loss == (0.7, 0.7)
    assert parsed.get_derived_output("teacher_logprobs") == (-0.3, -0.4)
    assert parsed.get_derived_output("per_token_kl") == (0.01, 0.02)
    assert parsed.get_derived_output("custom_result") == {"nested": [1, 2]}
    assert parsed.get_derived_output("decoded_text") == "public content"
    assert parsed.get_derived_output("raw_logits") == [[0.1, 0.2], [0.3, 0.4]]

    for field_name in ("target_tokens", "output_tokens", "teacher_labels"):
        redacted = _sample_output(datum, content_visibility="public")
        redacted[field_name] = [-8, -8]
        with pytest.raises(ValueError, match="non-negative token IDs"):
            SampleRefOutput.from_payload(redacted)

    for field_name, malformed in (
        (
            "logprobs",
            {"data": [-0.1, -0.2], "dtype": "float32", "shape": [2], "extra": 1},
        ),
        ("logprobs", {"data": [-0.1, -0.2], "dtype": "float32", "shape": [3]}),
        ("logprobs", {"data": [-0.1, -0.2], "dtype": "complex64", "shape": [2]}),
        (
            "output_tokens",
            {"data": [1, 2, 3], "dtype": "float32", "shape": [3]},
        ),
    ):
        malformed_output = _sample_output(datum, content_visibility="public")
        malformed_output[field_name] = malformed
        with pytest.raises(ValueError, match="managed tensor fields|exact|dtype"):
            SampleRefOutput.from_payload(malformed_output)

    non_finite = _sample_output(datum, content_visibility="public")
    non_finite["per_token_kl"] = [0.1, float("-inf")]
    with pytest.raises(ValueError, match="finite numeric"):
        SampleRefOutput.from_payload(non_finite)


def test_mixed_output_alignment_preserves_datum_kinds_but_reattachment_is_forbidden():
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
                _sample_output(managed, content_visibility="public"),
            ]
        }
    }

    aligned = align_training_outputs([local, managed], result)
    assert isinstance(aligned[1], SampleRefOutput)
    with pytest.raises(ValueError, match="cannot be attached as loss inputs"):
        attach_loss_fn_outputs([local, managed], result)

    result["result"]["loss_fn_outputs"][1]["per_token_kl"] = [0.01, 0.02]
    result["result"]["loss_fn_outputs"][1]["token_losses"] = [0.7, 0.8]
    result["result"]["loss_fn_outputs"][0]["per_token_kl"] = [0.03, 0.04]
    result["result"]["loss_fn_outputs"][0]["token_losses"] = [0.5, 0.6]
    with pytest.raises(ValueError, match="cannot be attached as loss inputs"):
        attach_loss_fn_outputs(
            [local, managed],
            result,
            field_map={"per_token_kl": "old_kl", "token_losses": "old_token_losses"},
        )


def test_mixed_output_alignment_uses_generated_local_occurrence_id():
    local = Datum.from_raw(
        model_input=ModelInput.from_ints([1]), loss_fn_inputs={"target_tokens": [2]}
    )
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2)
    prepared = prepare_forward_backward_operation(
        model_id="model-1",
        seq_id=7,
        data=[local, managed],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        request_metadata=None,
        tensor_transport="default",
    )
    try:
        wire = prepared.body["payload"]["forward_backward_input"]["data"]
    finally:
        prepared.close()
    result = {
        "result": {
            "loss_fn_outputs": [
                {"datum_id": wire[0]["datum_id"], "logprobs": [0.0]},
                _sample_output(managed, count=1, content_visibility="public"),
            ]
        }
    }

    aligned = align_training_outputs([local, managed], result)
    assert aligned[0]["datum_id"] == wire[0]["datum_id"]
    with pytest.raises(ValueError, match="cannot be attached as loss inputs"):
        attach_loss_fn_outputs([local, managed], result)


def test_token_only_identified_batch_rejects_duplicate_explicit_ids_before_submission():
    first = Datum.from_raw(
        model_input=ModelInput.from_ints([1]),
        loss_fn_inputs={"target_tokens": [2]},
        datum_id="duplicate",
    )
    second = Datum.from_raw(
        model_input=ModelInput.from_ints([3]),
        loss_fn_inputs={"target_tokens": [4]},
        datum_id="duplicate",
    )

    with pytest.raises(ValueError, match="duplicate datum_id"):
        prepare_forward_backward_operation(
            model_id="model-1",
            seq_id=7,
            data=[first, second],
            loss_fn="cross_entropy",
            loss_fn_config=None,
            request_metadata=None,
            tensor_transport="default",
        )


def test_shared_custom_and_surrogate_helpers_reject_sample_refs():
    datum = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2, datum_id="d-2")
    result = {"result": {"loss_fn_outputs": [_sample_output(datum, content_visibility="public")]}}

    with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
        parse_logprob_tensors(result, [datum])
    logprobs = torch.tensor([-0.7, -0.7], requires_grad=True)
    logprobs.sum().backward()
    with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
        build_surrogate_data([datum], [logprobs])


def test_http_binary_rejects_mixed_and_managed_only_sample_ref_batches():
    local = Datum.from_raw(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [2, 3], "weights": [0.0, 1.0]},
        datum_id="local",
    )
    managed = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=2, datum_id="managed")
    for data in ([managed, local], [managed]):
        with pytest.raises(ValueError, match="default JSON tensor transport"):
            prepare_forward_backward_operation(
                model_id="model-1",
                seq_id=1,
                data=data,
                loss_fn="cross_entropy",
                loss_fn_config=None,
                request_metadata=None,
                tensor_transport="http-binary",
            )


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


def test_sample_ref_sft_accepts_non_server_owned_loss_inputs():
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
    client.forward_backward([datum], "cross_entropy")
    client._service.enqueue_operation.assert_called_once()


def test_sync_training_client_enforces_sample_ref_sft_only_without_catalog_preflight():
    client = _training_client()
    handle = MagicMock()
    handle.result.return_value = {}
    client._service.enqueue_operation = MagicMock(return_value=handle)
    managed = Datum.from_sample_ref(dataset="secret", version="v1", sample_idx=0)
    client.forward_backward([managed], "cross_entropy")
    client._service._http.get.assert_not_called()

    for operation in (
        lambda: client.forward([managed], "cross_entropy"),
        lambda: client.forward([managed], "forward_logprob"),
        lambda: client.forward_backward([managed], "surrogate"),
    ):
        with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
            operation()
    client.forward_backward(
        [managed], "cross_entropy", loss_fn_config={"loss_agg_mode": "token-mean"}
    )
    with_inputs = Datum.from_sample_ref(
        dataset="secret", version="v1", sample_idx=0, loss_fn_inputs={"coefficient": 0.5}
    )
    client.forward_backward([with_inputs], "cross_entropy")
    with_metadata = Datum.from_sample_ref(
        dataset="secret", version="v1", sample_idx=0, metadata={"source": "managed"}
    )
    with pytest.raises(ValueError, match="empty metadata"):
        client.forward_backward([with_metadata], "cross_entropy")
    client._service._config.tensor_transport = "http-binary"
    with pytest.raises(ValueError, match="default JSON tensor transport"):
        client.forward_backward([managed], "cross_entropy")
    assert client._service.enqueue_operation.call_count == 3


def test_async_training_client_matches_sample_ref_sft_only_policy():
    async def run():
        client = _async_training_client()
        handle = MagicMock()
        handle.result = AsyncMock(return_value={})
        client._service.enqueue_operation = AsyncMock(return_value=handle)
        managed = Datum.from_sample_ref(dataset="open", version="v1", sample_idx=0)
        await client.forward_backward([managed], "cross_entropy")
        for operation in (
            lambda: client.forward([managed], "cross_entropy"),
            lambda: client.forward([managed], "forward_logprob"),
            lambda: client.forward_backward([managed], "surrogate"),
        ):
            with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
                await operation()
        await client.forward_backward(
            [managed], "cross_entropy", loss_fn_config={"loss_agg_mode": "token-mean"}
        )
        with_inputs = Datum.from_sample_ref(
            dataset="open", version="v1", sample_idx=0, loss_fn_inputs={"coefficient": 0.5}
        )
        await client.forward_backward([with_inputs], "cross_entropy")
        with_metadata = Datum.from_sample_ref(
            dataset="open", version="v1", sample_idx=0, metadata={"source": "managed"}
        )
        with pytest.raises(ValueError, match="empty metadata"):
            await client.forward_backward([with_metadata], "cross_entropy")
        client._service._config.tensor_transport = "http-binary"
        with pytest.raises(ValueError, match="default JSON tensor transport"):
            await client.forward_backward([managed], "cross_entropy")
        return client

    client = asyncio.run(run())
    assert client._service.enqueue_operation.await_count == 3
    client._service._http.get.assert_not_called()


def test_sync_custom_training_rejects_all_sample_refs():
    client = _training_client()
    datum = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=0, datum_id="d-0")
    client._service.enqueue_operation = MagicMock()
    with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
        client.forward_backward_custom([datum], lambda _data, logprobs: (logprobs[0].sum(), {}))
    client._service.enqueue_operation.assert_not_called()


def test_async_custom_training_rejects_all_sample_refs():
    async def run():
        client = _async_training_client()
        datum = Datum.from_sample_ref(dataset="d", version="v1", sample_idx=0, datum_id="d-0")
        client._service.enqueue_operation = AsyncMock()
        with pytest.raises(ValueError, match="only supports built-in cross_entropy"):
            await client.forward_backward_custom(
                [datum], lambda _data, logprobs: (logprobs[0].sum(), {})
            )
        client._service.enqueue_operation.assert_not_awaited()

    asyncio.run(run())
