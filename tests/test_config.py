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

"""Tests for the configuration module."""

import os

import pytest

from weaver.config import WeaverConfig


def test_config_defaults():
    """Test that config has sensible defaults."""
    config = WeaverConfig()
    assert config.base_url == "https://weaver-console.nex-agi.cn"
    assert config.api_key is None
    assert config.tensor_transport == "default"
    assert config.tensor_compression == "zstd"


def test_config_from_kwargs():
    """Test config initialization with kwargs."""
    config = WeaverConfig.from_env(
        base_url="https://custom.example.com",
        api_key="sk-test-key",
    )
    assert config.base_url == "https://custom.example.com"
    assert config.api_key == "sk-test-key"


def test_config_from_env(monkeypatch):
    """Test config loading from environment variables."""
    monkeypatch.setenv("WEAVER_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("WEAVER_API_KEY", "sk-env-key")
    monkeypatch.setenv("WEAVER_TENSOR_TRANSPORT", "http-binary")
    monkeypatch.setenv("WEAVER_TENSOR_COMPRESSION", "zstd")

    config = WeaverConfig.from_env()
    assert config.base_url == "https://env.example.com"
    assert config.api_key == "sk-env-key"
    assert config.tensor_transport == "http-binary"
    assert config.tensor_compression == "zstd"


def test_config_kwargs_override_env(monkeypatch):
    """Test that kwargs override environment variables."""
    monkeypatch.setenv("WEAVER_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("WEAVER_API_KEY", "sk-env-key")
    monkeypatch.setenv("WEAVER_TENSOR_TRANSPORT", "http-binary")
    monkeypatch.setenv("WEAVER_TENSOR_COMPRESSION", "zstd")

    config = WeaverConfig.from_env(
        base_url="https://override.example.com",
        api_key="sk-override-key",
        tensor_transport="default",
        tensor_compression="raw",
    )
    assert config.base_url == "https://override.example.com"
    assert config.api_key == "sk-override-key"
    assert config.tensor_transport == "default"
    assert config.tensor_compression == "raw"


def test_config_rejects_unknown_tensor_transport(monkeypatch):
    monkeypatch.setenv("WEAVER_TENSOR_TRANSPORT", "magic")

    with pytest.raises(ValueError, match="Unsupported tensor transport"):
        WeaverConfig.from_env()

    with pytest.raises(ValueError, match="Unsupported tensor transport"):
        WeaverConfig(tensor_transport="magic")  # type: ignore[arg-type]


def test_config_rejects_unknown_tensor_compression(monkeypatch):
    monkeypatch.setenv("WEAVER_TENSOR_COMPRESSION", "magic")

    with pytest.raises(ValueError, match="Unsupported tensor compression"):
        WeaverConfig.from_env()

    with pytest.raises(ValueError, match="Unsupported tensor compression"):
        WeaverConfig(tensor_compression="magic")  # type: ignore[arg-type]


def test_http_binary_defaults_to_zstd():
    config = WeaverConfig(tensor_transport="http-binary")

    assert config.tensor_compression == "zstd"


def test_require_auth_with_credentials():
    """Test require_auth passes with valid credentials."""
    config = WeaverConfig(
        base_url="https://example.com",
        api_key="sk-test-key",
    )
    # Should not raise
    config.require_auth()


def test_require_auth_without_credentials():
    """Test require_auth raises without credentials."""
    config = WeaverConfig(base_url="https://example.com")
    with pytest.raises(RuntimeError, match="Weaver credentials missing"):
        config.require_auth()
