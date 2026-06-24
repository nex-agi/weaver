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

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_create_model_passes_debug_info():
    """Test create_model extracts debug_info from response and passes to TrainingClient."""
    debug_info = {
        "debug_mode": "manual",
        "model_id": "abc-123",
        "job_name": "user-trainer-full_ft-abc-123",
        "namespace": "qiji",
        "kubectl_exec": "kubectl exec -it user-trainer-full_ft-abc-123-master-0 -n qiji -- /bin/bash",
        "config_file": "/tmp/trainer.env",
    }
    mock_response = {
        "id": "abc-123",
        "base_model": "Qwen/Qwen3-8B",
        "debug_info": debug_info,
    }

    client = ServiceClient(api_key="sk-test-key")
    client._session_id = "session-1"
    client._http = MagicMock()
    client._http.post.return_value = mock_response
    client._http.get.return_value = {"items": []}  # for get_supported_model_config

    training = client.create_model(base_model="Qwen/Qwen3-8B", training_mode="full_ft")

    assert training.debug_info == debug_info
    assert training.debug_info["kubectl_exec"].startswith("kubectl exec")
    assert training.model_id == "abc-123"


def _build_atexit_script(marker_path: str, exit_code: str = "") -> str:
    """Build a subprocess script that verifies atexit calls close().

    Patches APIClient so connect() proceeds and registers atexit.
    Uses _http.close() as the marker hook since atexit holds the original
    bound method.
    """
    return f"""
from unittest.mock import MagicMock, patch
from weaver.service_client import ServiceClient

mock_http = MagicMock()
mock_http.close.side_effect = lambda: open("{marker_path}", "w").write("closed")

with patch("weaver.service_client.APIClient", return_value=mock_http):
    client = ServiceClient(api_key="sk-test")
    client.ensure_session = lambda **kw: None
    client._start_heartbeat = lambda: None
    client.connect()
{exit_code}
"""


def test_atexit_calls_close_on_normal_exit():
    """Test that atexit handler calls close() when process exits normally."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / "closed.marker"
        result = subprocess.run(
            [sys.executable, "-c", _build_atexit_script(str(marker))],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert marker.exists(), "close() was not called on normal exit"


def test_atexit_calls_close_on_exception_exit():
    """Test that atexit handler calls close() when process exits via unhandled exception."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / "closed.marker"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _build_atexit_script(str(marker), 'raise RuntimeError("simulated crash")'),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert marker.exists(), "close() was not called on exception exit"


def test_create_model_debug_info_none_when_absent():
    """Test create_model sets debug_info=None when server response has no debug_info."""
    mock_response = {
        "id": "abc-123",
        "base_model": "Qwen/Qwen3-8B",
    }

    client = ServiceClient(api_key="sk-test-key")
    client._session_id = "session-1"
    client._http = MagicMock()
    client._http.post.return_value = mock_response
    client._http.get.return_value = {"items": []}

    training = client.create_model(base_model="Qwen/Qwen3-8B", training_mode="full_ft")

    assert training.debug_info is None


def _make_create_model_client():
    """Build a ServiceClient with a mocked http layer for create_model payload tests."""
    client = ServiceClient(api_key="sk-test-key")
    client._session_id = "session-1"
    client._http = MagicMock()
    client._http.post.return_value = {"id": "abc-123", "base_model": "Qwen/Qwen3-8B"}
    client._http.get.return_value = {"items": []}
    return client


def test_create_model_passes_workload_hints():
    """Workload hints are forwarded in the create-model request body when provided."""
    client = _make_create_model_client()

    client.create_model(
        base_model="Qwen/Qwen3-8B",
        training_mode="full_ft",
        max_seq_len=8192,
        training_batch_size=32,
        rollout_batch_size=64,
    )

    _, kwargs = client._http.post.call_args
    payload = kwargs["json"]
    assert payload["max_seq_len"] == 8192
    assert payload["training_batch_size"] == 32
    assert payload["rollout_batch_size"] == 64


def test_create_model_omits_workload_hints_when_absent():
    """Hints are not present in the payload when omitted, preserving prior behavior."""
    client = _make_create_model_client()

    client.create_model(base_model="Qwen/Qwen3-8B", training_mode="full_ft")

    _, kwargs = client._http.post.call_args
    payload = kwargs["json"]
    assert "max_seq_len" not in payload
    assert "training_batch_size" not in payload
    assert "rollout_batch_size" not in payload
