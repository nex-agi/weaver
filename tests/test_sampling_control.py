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

"""Tests for the sampling engine control primitives.

Covers ``pause_generation`` / ``continue_generation`` plus the result-schema
passthrough of ``weight_version`` and pause(abort) partial output, across both
the sync and async sampling clients (issue #84).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver import _sampling_utils as _su
from weaver.async_sampling_client import AsyncSamplingClient
from weaver.sampling_client import SamplingClient
from weaver.types import ModelInput, PauseMode


def _make_sync_client() -> tuple[SamplingClient, MagicMock]:
    mock_service = MagicMock()
    client = SamplingClient(
        service=mock_service,
        sampling_session_id="sess-001",
        base_model="Qwen/Qwen3-8B",
    )
    return client, mock_service


def _make_async_client() -> tuple[AsyncSamplingClient, MagicMock]:
    mock_service = MagicMock()
    mock_service.http.post = AsyncMock()
    client = AsyncSamplingClient(
        service=mock_service,
        sampling_session_id="sess-001",
        base_model="Qwen/Qwen3-8B",
    )
    return client, mock_service


# --------------------------------------------------------------------------- #
# PauseMode enum / coercion                                                    #
# --------------------------------------------------------------------------- #


class TestPauseMode:
    def test_members(self):
        assert PauseMode.ABORT == "abort"
        assert PauseMode.RETRACT == "retract"
        assert PauseMode.IN_PLACE == "in_place"

    def test_coerce_accepts_enum_and_string(self):
        assert _su.coerce_pause_mode(PauseMode.ABORT) == "abort"
        assert _su.coerce_pause_mode("retract") == "retract"

    def test_coerce_rejects_unknown(self):
        with pytest.raises(ValueError):
            _su.coerce_pause_mode("freeze")

    def test_build_body_defaults_and_validates(self):
        assert _su.build_pause_generation_body(PauseMode.ABORT) == {"mode": "abort"}
        assert _su.build_pause_generation_body("in_place") == {"mode": "in_place"}
        with pytest.raises(ValueError):
            _su.build_pause_generation_body("nope")


# --------------------------------------------------------------------------- #
# Sync client                                                                  #
# --------------------------------------------------------------------------- #


class TestSyncClient:
    def test_pause_generation_default_abort(self):
        client, mock_service = _make_sync_client()
        mock_service.http.post.return_value = {"ok": True}

        result = client.pause_generation()

        path, kwargs = mock_service.http.post.call_args[0], mock_service.http.post.call_args[1]
        assert path[0] == "/api/v1/sampling-sessions/sess-001/pause-generation"
        assert kwargs["json"] == {"mode": "abort"}
        assert result == {"ok": True}

    def test_pause_generation_explicit_mode(self):
        client, mock_service = _make_sync_client()
        mock_service.http.post.return_value = {}
        client.pause_generation(mode=PauseMode.RETRACT)
        assert mock_service.http.post.call_args[1]["json"] == {"mode": "retract"}

    def test_pause_generation_invalid_mode(self):
        client, _ = _make_sync_client()
        with pytest.raises(ValueError):
            client.pause_generation(mode="freeze")

    def test_continue_generation(self):
        client, mock_service = _make_sync_client()
        mock_service.http.post.return_value = {"ok": True}
        result = client.continue_generation()
        assert (
            mock_service.http.post.call_args[0][0]
            == "/api/v1/sampling-sessions/sess-001/continue-generation"
        )
        assert result == {"ok": True}


# --------------------------------------------------------------------------- #
# Async client                                                                 #
# --------------------------------------------------------------------------- #


class TestAsyncClient:
    def test_pause_generation_default_abort(self):
        client, mock_service = _make_async_client()
        mock_service.http.post.return_value = {"ok": True}

        result = asyncio.run(client.pause_generation())

        assert (
            mock_service.http.post.call_args[0][0]
            == "/api/v1/sampling-sessions/sess-001/pause-generation"
        )
        assert mock_service.http.post.call_args[1]["json"] == {"mode": "abort"}
        assert result == {"ok": True}

    def test_continue_generation(self):
        client, mock_service = _make_async_client()
        mock_service.http.post.return_value = {"ok": True}
        asyncio.run(client.continue_generation())
        assert (
            mock_service.http.post.call_args[0][0]
            == "/api/v1/sampling-sessions/sess-001/continue-generation"
        )

    def test_pause_generation_invalid_mode(self):
        client, _ = _make_async_client()
        with pytest.raises(ValueError):
            asyncio.run(client.pause_generation(mode="freeze"))


# --------------------------------------------------------------------------- #
# Result schema: weight_version + pause(abort) partial                         #
# --------------------------------------------------------------------------- #


class TestResultSchema:
    def test_weight_version_top_level_passthrough(self):
        payload = {
            "result": {
                "weight_version": "v42",
                "sequences": [{"tokens": [1, 2], "text": "hi", "stop_reason": "stop"}],
            }
        }
        out = _su.normalize_sample_result(payload, MagicMock())
        assert out["weight_version"] == "v42"

    def test_weight_version_per_sequence_passthrough(self):
        payload = {
            "result": {
                "sequences": [
                    {"tokens": [1], "text": "a", "stop_reason": "stop", "weight_version": "v7"}
                ]
            }
        }
        out = _su.normalize_sample_result(payload, MagicMock())
        assert out["sequences"][0]["weight_version"] == "v7"

    def test_abort_partial_is_returned(self):
        payload = {
            "result": {
                "sequences": [
                    {"tokens": [9, 8, 7], "text": "par", "stop_reason": "abort"},
                ]
            }
        }
        out = _su.normalize_sample_result(payload, MagicMock())
        assert out["sequences"][0]["stop_reason"] == "abort"
        assert out["sequences"][0]["tokens"] == [9, 8, 7]

    def test_empty_aborted_sequence_is_preserved(self):
        # A pause(abort) may cut a request before any token is emitted; the
        # aborted marker must survive rather than being filtered out.
        payload = {
            "result": {
                "sequences": [
                    {"tokens": [], "text": "", "stop_reason": "abort"},
                    {"tokens": [], "text": "", "stop_reason": "stop"},
                ]
            }
        }
        out = _su.normalize_sample_result(payload, MagicMock())
        kept = out["sequences"]
        assert len(kept) == 1
        assert kept[0]["stop_reason"] == "abort"
