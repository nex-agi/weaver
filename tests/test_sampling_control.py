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

MODEL_ID = "11111111-2222-3333-4444-555555555555"


def _make_sync_client(
    training_mode: str = "full_ft", **overrides
) -> tuple[SamplingClient, MagicMock]:
    mock_service = MagicMock()
    mock_service.get_model.return_value = {"training_mode": training_mode}
    kwargs = {
        "sampling_session_id": "sess-001",
        "base_model": "Qwen/Qwen3-8B",
        "model_id": MODEL_ID,
    }
    kwargs.update(overrides)
    return SamplingClient(service=mock_service, **kwargs), mock_service


def _make_async_client(
    training_mode: str = "full_ft", **overrides
) -> tuple[AsyncSamplingClient, MagicMock]:
    mock_service = MagicMock()
    mock_service.http.post = AsyncMock()
    mock_service.get_model = AsyncMock(return_value={"training_mode": training_mode})
    kwargs = {
        "sampling_session_id": "sess-001",
        "base_model": "Qwen/Qwen3-8B",
        "model_id": MODEL_ID,
    }
    kwargs.update(overrides)
    return AsyncSamplingClient(service=mock_service, **kwargs), mock_service


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

    def test_paused_resumes_on_success(self):
        client, mock_service = _make_sync_client()
        mock_service.http.post.return_value = {"ok": True}

        with client.paused() as result:
            assert result == {"ok": True}

        paths = [call[0][0] for call in mock_service.http.post.call_args_list]
        assert paths == [
            "/api/v1/sampling-sessions/sess-001/pause-generation",
            "/api/v1/sampling-sessions/sess-001/continue-generation",
        ]

    def test_paused_resumes_on_exception(self):
        """The whole point of the context manager: a frozen engine must not
        survive an error inside the block, since nothing auto-resumes it."""
        client, mock_service = _make_sync_client()
        mock_service.http.post.return_value = {}

        with pytest.raises(RuntimeError, match="boom"):
            with client.paused():
                raise RuntimeError("boom")

        paths = [call[0][0] for call in mock_service.http.post.call_args_list]
        assert paths[-1] == "/api/v1/sampling-sessions/sess-001/continue-generation"


class TestSyncFullFTRestriction:
    def test_lora_model_is_rejected_without_calling_the_server(self):
        client, mock_service = _make_sync_client(training_mode="lora")

        with pytest.raises(ValueError, match="full fine-tuning"):
            client.pause_generation()

        mock_service.http.post.assert_not_called()

    def test_continue_is_restricted_too(self):
        client, mock_service = _make_sync_client(training_mode="lora")
        with pytest.raises(ValueError, match="full fine-tuning"):
            client.continue_generation()
        mock_service.http.post.assert_not_called()

    def test_unbound_client_is_rejected_without_any_request(self):
        """No model_id and a model_path that carries none: a bare base-model or
        shared-pool client, which has no engine of its own to freeze."""
        client, mock_service = _make_sync_client(model_id=None, model_path=None)

        with pytest.raises(ValueError, match="not bound"):
            client.pause_generation()

        mock_service.get_model.assert_not_called()
        mock_service.http.post.assert_not_called()

    def test_model_id_recovered_from_checkpoint_path(self):
        client, mock_service = _make_sync_client(
            model_id=None, model_path=f"weaver://{MODEL_ID}/checkpoints/step-42"
        )
        mock_service.http.post.return_value = {}

        client.pause_generation()

        mock_service.get_model.assert_called_once_with(MODEL_ID)

    def test_eligibility_is_checked_once(self):
        client, mock_service = _make_sync_client()
        mock_service.http.post.return_value = {}

        client.pause_generation()
        client.continue_generation()
        client.pause_generation()

        assert mock_service.get_model.call_count == 1


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

    def test_paused_resumes_on_success(self):
        client, mock_service = _make_async_client()
        mock_service.http.post.return_value = {"ok": True}

        async def _run():
            async with client.paused() as result:
                assert result == {"ok": True}

        asyncio.run(_run())

        paths = [call[0][0] for call in mock_service.http.post.call_args_list]
        assert paths == [
            "/api/v1/sampling-sessions/sess-001/pause-generation",
            "/api/v1/sampling-sessions/sess-001/continue-generation",
        ]

    def test_paused_resumes_on_exception(self):
        client, mock_service = _make_async_client()
        mock_service.http.post.return_value = {}

        async def _run():
            async with client.paused():
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(_run())

        paths = [call[0][0] for call in mock_service.http.post.call_args_list]
        assert paths[-1] == "/api/v1/sampling-sessions/sess-001/continue-generation"


class TestAsyncFullFTRestriction:
    def test_lora_model_is_rejected_without_calling_the_server(self):
        client, mock_service = _make_async_client(training_mode="lora")

        with pytest.raises(ValueError, match="full fine-tuning"):
            asyncio.run(client.pause_generation())

        mock_service.http.post.assert_not_called()

    def test_unbound_client_is_rejected_without_any_request(self):
        client, mock_service = _make_async_client(model_id=None, model_path=None)

        with pytest.raises(ValueError, match="not bound"):
            asyncio.run(client.continue_generation())

        mock_service.get_model.assert_not_called()
        mock_service.http.post.assert_not_called()

    def test_eligibility_is_checked_once(self):
        client, mock_service = _make_async_client()
        mock_service.http.post.return_value = {}

        async def _run():
            await client.pause_generation()
            await client.continue_generation()

        asyncio.run(_run())
        assert mock_service.get_model.call_count == 1


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #


class TestWeaverPathParsing:
    def test_extracts_model_id(self):
        assert (
            _su.parse_model_id_from_weaver_path(f"weaver://{MODEL_ID}/checkpoints/step-42")
            == MODEL_ID
        )

    def test_returns_none_for_non_weaver_paths(self):
        assert _su.parse_model_id_from_weaver_path(None) is None
        assert _su.parse_model_id_from_weaver_path("") is None
        assert _su.parse_model_id_from_weaver_path("/gpfs/checkpoints/step-42") is None
        assert _su.parse_model_id_from_weaver_path("weaver://") is None

    def test_ensure_full_ft_rejects_other_modes(self):
        _su.ensure_full_ft_for_control("full_ft", model_id=MODEL_ID)
        for mode in ("lora", "", None):
            with pytest.raises(ValueError, match="full fine-tuning"):
                _su.ensure_full_ft_for_control(mode, model_id=MODEL_ID)


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
