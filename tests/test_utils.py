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

"""Tests for utility functions."""

import pytest

from weaver._utils import extract_id, lookup_case_insensitive


def test_lookup_case_insensitive_exact_match():
    """Test exact case match."""
    data = {"user_id": "123"}
    result = lookup_case_insensitive(data, "user_id")
    assert result == "123"


def test_lookup_case_insensitive_lower():
    """Test lowercase match."""
    data = {"userid": "123"}
    result = lookup_case_insensitive(data, "user_id")
    assert result == "123"


def test_lookup_case_insensitive_upper():
    """Test uppercase match."""
    data = {"USER_ID": "123"}
    result = lookup_case_insensitive(data, "user_id")
    assert result == "123"


def test_lookup_case_insensitive_capitalize():
    """Test capitalized match."""
    data = {"User_id": "123"}
    result = lookup_case_insensitive(data, "user_id")
    assert result == "123"


def test_lookup_case_insensitive_camel_case():
    """Test camelCase match."""
    data = {"UserId": "123"}
    result = lookup_case_insensitive(data, "user_id")
    assert result == "123"


def test_lookup_case_insensitive_not_found():
    """Test that None is returned when key is not found."""
    data = {"other_key": "123"}
    result = lookup_case_insensitive(data, "user_id")
    assert result is None


def test_extract_id_success():
    """Test successful ID extraction."""
    data = {"id": "model-123"}
    result = extract_id(data)
    assert result == "model-123"


def test_extract_id_case_insensitive():
    """Test ID extraction with different cases."""
    data = {"ID": "model-456"}
    result = extract_id(data)
    assert result == "model-456"


def test_extract_id_missing():
    """Test that ValueError is raised when ID is missing."""
    data = {"name": "test"}
    with pytest.raises(ValueError, match="Payload missing id field"):
        extract_id(data)


def test_extract_id_converts_to_string():
    """Test that numeric IDs are converted to strings."""
    data = {"id": 789}
    result = extract_id(data)
    assert result == "789"
    assert isinstance(result, str)
