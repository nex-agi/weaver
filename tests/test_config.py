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
    assert config.base_url == "https://weaver-console.nex-agi.com"
    assert config.api_key is None


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

    config = WeaverConfig.from_env()
    assert config.base_url == "https://env.example.com"
    assert config.api_key == "sk-env-key"


def test_config_kwargs_override_env(monkeypatch):
    """Test that kwargs override environment variables."""
    monkeypatch.setenv("WEAVER_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("WEAVER_API_KEY", "sk-env-key")

    config = WeaverConfig.from_env(
        base_url="https://override.example.com",
        api_key="sk-override-key",
    )
    assert config.base_url == "https://override.example.com"
    assert config.api_key == "sk-override-key"


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
