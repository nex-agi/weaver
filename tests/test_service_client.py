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

"""Tests for the ServiceClient."""

import pytest

from weaver.service_client import ServiceClient


def test_service_client_initialization():
    """Test ServiceClient can be initialized."""
    client = ServiceClient(
        base_url="https://test.example.com",
        api_key="sk-test-key",
    )
    assert client._config.base_url == "https://test.example.com"
    assert client._config.api_key == "sk-test-key"


def test_service_client_default_tags():
    """Test default tags are set."""
    client = ServiceClient()
    assert "weaver-sdk" in client._default_tags


def test_service_client_custom_tags():
    """Test custom default tags."""
    client = ServiceClient(default_tags=["custom", "tags"])
    assert client._default_tags == ["custom", "tags"]


def test_service_client_not_connected_raises():
    """Test accessing http before connect raises error."""
    client = ServiceClient()
    with pytest.raises(RuntimeError, match="ServiceClient is not connected"):
        _ = client.http


def test_service_client_session_id_without_session_raises():
    """Test accessing session_id before initialization raises error."""
    client = ServiceClient()
    with pytest.raises(RuntimeError, match="Session not initialized yet"):
        _ = client.session_id


def test_next_model_seq_is_monotonic():
    """Test model seq counter increments monotonically."""
    client = ServiceClient()
    seq1 = client._next_model_seq()
    seq2 = client._next_model_seq()
    seq3 = client._next_model_seq()
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3


def test_next_sampling_seq_is_monotonic():
    """Test sampling seq counter increments monotonically."""
    client = ServiceClient()
    seq1 = client._next_sampling_seq()
    seq2 = client._next_sampling_seq()
    seq3 = client._next_sampling_seq()
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3


def test_next_operation_seq_per_model():
    """Test operation seq is tracked per model."""
    client = ServiceClient()
    model1_seq1 = client.next_operation_seq("model-1")
    model1_seq2 = client.next_operation_seq("model-1")
    model2_seq1 = client.next_operation_seq("model-2")

    assert model1_seq1 == 1
    assert model1_seq2 == 2
    assert model2_seq1 == 1


def test_next_operation_seq_requires_model_id():
    """Test operation seq raises without model_id."""
    client = ServiceClient()
    with pytest.raises(ValueError, match="model_id is required"):
        client.next_operation_seq("")


def test_service_client_context_manager():
    """Test ServiceClient can be used as context manager."""
    # Note: This will fail without a real server, but tests the structure
    client = ServiceClient(api_key="sk-test-key")
    assert client._http is None
    # We can't actually enter/exit without a real server
    # Just test that __enter__ and __exit__ methods exist
    assert hasattr(client, "__enter__")
    assert hasattr(client, "__exit__")


def test_service_client_close_is_idempotent():
    """Test close can be called multiple times safely."""
    client = ServiceClient()
    client.close()
    client.close()  # Should not raise
    assert client._closed is True
