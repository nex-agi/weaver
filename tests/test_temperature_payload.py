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

from __future__ import annotations

from types import MethodType
from typing import Any
from unittest.mock import MagicMock

import torch

from weaver.training_client import TrainingClient
from weaver.types import Datum, LogprobsParams, ModelInput


def _make_training_client() -> TrainingClient:
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    return TrainingClient(
        service=service,
        model_id="model-123",
        base_model="Qwen/Qwen3-8B",
        session_id="session-123",
    )


def test_forward_backward_sends_temperature_in_loss_fn_config() -> None:
    client = _make_training_client()
    handle = MagicMock()
    client._service.enqueue_operation.return_value = handle

    result = client.forward_backward(
        [], "cross_entropy", loss_fn_config={"temperature": 0.7}, wait=False
    )

    assert result is handle
    body = client._service.enqueue_operation.call_args[0][1]
    assert body["payload"]["forward_backward_input"]["loss_fn_config"]["temperature"] == 0.7


def test_forward_sends_temperature_in_loss_fn_config() -> None:
    client = _make_training_client()
    handle = MagicMock()
    client._service.enqueue_operation.return_value = handle

    result = client.forward([], "cross_entropy", loss_fn_config={"temperature": 0.7}, wait=False)

    assert result is handle
    path, body = client._service.enqueue_operation.call_args[0]
    assert path == "/api/v1/models/model-123/forward-passes"
    assert body["payload"]["forward_input"]["loss_fn_config"]["temperature"] == 0.7
    assert body["payload"]["forward_input"]["loss_fn"] == "cross_entropy"
    assert "forward_backward_input" not in body["payload"]


def test_forward_waits_for_result_by_default() -> None:
    client = _make_training_client()
    handle = MagicMock()
    handle.result.return_value = {"result": {"loss": 1.23}}
    client._service.enqueue_operation.return_value = handle

    result = client.forward([], "cross_entropy")

    assert result == {"result": {"loss": 1.23}}
    handle.result.assert_called_once_with()


def test_forward_backward_custom_uses_forward_and_reuses_loss_fn_config() -> None:
    client = _make_training_client()
    datum = Datum.from_raw(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": [2],
            "loss_mask": [1],
            "sampling_mask": [[2]],
            "surrogate_weights": [99.0],
        },
    )
    calls: list[dict[str, Any]] = []
    surrogate_data: list[Datum] = []

    def fake_forward(
        self: TrainingClient,
        data: list[Datum],
        loss_fn: str,
        *,
        loss_fn_config: dict[str, float] | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        calls.append({"loss_fn": loss_fn, "loss_fn_config": loss_fn_config})
        if loss_fn == "forward_logprob":
            return {"result": {"loss_fn_outputs": [{"logprobs": [1.0]}]}}
        return {"result": {}}

    def fake_forward_backward(
        self: TrainingClient,
        data: list[Datum],
        loss_fn: str,
        *,
        loss_fn_config: dict[str, float] | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        calls.append({"loss_fn": loss_fn, "loss_fn_config": loss_fn_config})
        surrogate_data.extend(data)
        return {"result": {}}

    client.forward = MethodType(fake_forward, client)
    client.forward_backward = MethodType(fake_forward_backward, client)

    def loss_fn(_data: list[Datum], logprob_tensors: list[torch.Tensor]):
        return logprob_tensors[0].sum(), {}

    client.forward_backward_custom([datum], loss_fn, loss_fn_config={"temperature": 0.7})

    assert calls == [
        {"loss_fn": "forward_logprob", "loss_fn_config": {"temperature": 0.7}},
        {"loss_fn": "surrogate", "loss_fn_config": {"temperature": 0.7}},
    ]
    assert len(surrogate_data) == 1
    assert torch.equal(surrogate_data[0].loss_fn_inputs["target_tokens"], torch.tensor([2]))
    assert torch.equal(surrogate_data[0].loss_fn_inputs["loss_mask"], torch.tensor([1]))
    assert torch.equal(surrogate_data[0].loss_fn_inputs["sampling_mask"], torch.tensor([[2]]))
    assert torch.equal(
        surrogate_data[0].loss_fn_inputs["surrogate_weights"],
        torch.tensor([1.0]),
    )
    assert torch.equal(datum.loss_fn_inputs["surrogate_weights"], torch.tensor([99.0]))


def test_logprobs_params_sends_loss_fn_config() -> None:
    payload = LogprobsParams(loss_fn_config={"temperature": 0.7}).to_payload()
    assert payload["loss_fn_config"] == {"temperature": 0.7}
