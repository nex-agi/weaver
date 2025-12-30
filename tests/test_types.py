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

"""Tests for type definitions and conversions."""

import torch

from weaver.types.datum import Datum
from weaver.types.model_input import ModelInput
from weaver.types.tensor import TensorData, tensor_payload


def test_tensor_payload_from_list():
    """Test creating tensor payload from a list."""
    data = [1, 2, 3, 4]
    payload = tensor_payload(data)
    assert isinstance(payload, TensorData)
    assert payload.dtype == "int64"
    assert len(payload.data) == 4


def test_tensor_payload_from_torch_tensor():
    """Test creating tensor payload from torch tensor."""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    payload = tensor_payload(tensor)
    assert isinstance(payload, TensorData)
    assert payload.dtype == "float32"
    assert len(payload.data) == 3


def test_tensor_payload_multidimensional():
    """Test creating tensor payload from multidimensional data."""
    tensor = torch.randn(2, 3, 4)
    payload = tensor_payload(tensor)
    assert isinstance(payload, TensorData)
    # Verify it's a nested list structure
    assert len(payload.data) == 2


def test_tensor_data_to_tensor():
    """Test converting TensorData back to torch tensor."""
    original = torch.tensor([1, 2, 3, 4])
    payload = tensor_payload(original)
    reconstructed = payload.to_tensor()
    assert torch.equal(reconstructed, original)


def test_tensor_data_to_dict():
    """Test converting TensorData to dictionary."""
    tensor = torch.tensor([1, 2, 3])
    payload = tensor_payload(tensor)
    as_dict = payload.to_dict()
    assert "data" in as_dict
    assert "dtype" in as_dict
    assert as_dict["dtype"] == "int64"


def test_model_input_creation():
    """Test ModelInput creation."""
    model_input = ModelInput.from_ints([1, 2, 3])
    assert len(model_input.chunks) == 1
    assert model_input.chunks[0].tokens == [1, 2, 3]


def test_model_input_to_ints():
    """Test ModelInput conversion to ints."""
    model_input = ModelInput.from_ints([1, 2, 3, 4, 5])
    tokens = model_input.to_ints()
    assert tokens == [1, 2, 3, 4, 5]


def test_model_input_extend_tokens():
    """Test extending tokens in ModelInput."""
    model_input = ModelInput.from_ints([1, 2, 3])
    model_input.extend_tokens([4, 5, 6])
    assert model_input.to_ints() == [1, 2, 3, 4, 5, 6]


def test_model_input_to_payload():
    """Test ModelInput serialization to payload."""
    model_input = ModelInput.from_ints([1, 2, 3])
    payload = model_input.to_payload()
    assert isinstance(payload, dict)
    assert "chunks" in payload
    assert len(payload["chunks"]) == 1


def test_datum_creation():
    """Test Datum creation."""
    model_input = ModelInput.from_ints([1, 2, 3])
    labels = torch.tensor([1, 2, 3])
    datum = Datum(model_input=model_input, loss_fn_inputs={"labels": labels})
    assert datum.model_input is not None
    assert "labels" in datum.loss_fn_inputs


def test_datum_normalizes_tensors():
    """Test that Datum normalizes loss_fn_inputs to tensors."""
    model_input = ModelInput.from_ints([1, 2, 3])
    labels = [1, 2, 3]  # Pass as list
    datum = Datum(model_input=model_input, loss_fn_inputs={"labels": labels})
    assert isinstance(datum.loss_fn_inputs["labels"], torch.Tensor)


def test_datum_to_payload():
    """Test Datum serialization to payload."""
    model_input = ModelInput.from_ints([1, 2, 3])
    labels = torch.tensor([1, 2, 3])
    datum = Datum(model_input=model_input, loss_fn_inputs={"labels": labels})
    payload = datum.to_payload()
    assert isinstance(payload, dict)
    assert "model_input" in payload
    assert "loss_fn_inputs" in payload


def test_datum_from_raw():
    """Test Datum.from_raw factory method."""
    model_input = ModelInput.from_ints([1, 2, 3])
    labels = [1, 2, 3]
    datum = Datum.from_raw(model_input=model_input, loss_fn_inputs={"labels": labels})
    assert isinstance(datum, Datum)
    assert "labels" in datum.loss_fn_inputs
