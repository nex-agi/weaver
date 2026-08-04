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

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver.async_service_client import AsyncServiceClient
from weaver.async_training_client import AsyncTrainingClient
from weaver.service_client import ServiceClient
from weaver.training_client import TrainingClient


def test_service_client_creates_session_in_project_with_constructor_metadata() -> None:
    client = ServiceClient(
        organization_id="organization-1",
        project_id="project-1",
        user_metadata={"recipe": "grpo"},
    )
    client._http = MagicMock()
    client._http.post.return_value = {"id": "session-1"}

    client.ensure_session()

    args, kwargs = client._http.post.call_args
    assert args[0] == "/api/v1/sessions"
    assert kwargs["json"]["organization_id"] == "organization-1"
    assert kwargs["json"]["project_id"] == "project-1"
    assert kwargs["json"]["user_metadata"] == {"recipe": "grpo"}


def test_empty_scope_ids_are_omitted_so_server_fallback_applies(monkeypatch) -> None:
    monkeypatch.setenv("WEAVER_ORGANIZATION_ID", "organization-from-env")
    monkeypatch.setenv("WEAVER_PROJECT_ID", "project-from-env")
    client = ServiceClient(organization_id="  ", project_id="")
    client._http = MagicMock()
    client._http.post.return_value = {"id": "session-default"}

    client.ensure_session()

    payload = client._http.post.call_args.kwargs["json"]
    assert "organization_id" not in payload
    assert "project_id" not in payload


def test_scope_ids_use_environment_when_constructor_values_are_absent(monkeypatch) -> None:
    monkeypatch.setenv("WEAVER_ORGANIZATION_ID", " organization-from-env ")
    monkeypatch.setenv("WEAVER_PROJECT_ID", " project-from-env ")
    client = ServiceClient()
    client._http = MagicMock()
    client._http.post.return_value = {"id": "session-env"}

    client.ensure_session()

    payload = client._http.post.call_args.kwargs["json"]
    assert payload["organization_id"] == "organization-from-env"
    assert payload["project_id"] == "project-from-env"


def test_scope_ids_are_omitted_when_parameter_and_environment_are_absent(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_ORGANIZATION_ID", raising=False)
    monkeypatch.delenv("WEAVER_PROJECT_ID", raising=False)
    client = ServiceClient()
    client._http = MagicMock()
    client._http.post.return_value = {"id": "session-default"}

    client.ensure_session()

    payload = client._http.post.call_args.kwargs["json"]
    assert "organization_id" not in payload
    assert "project_id" not in payload


def test_project_only_scope_is_sent_for_server_side_organization_resolution(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_ORGANIZATION_ID", raising=False)
    monkeypatch.delenv("WEAVER_PROJECT_ID", raising=False)
    client = ServiceClient(project_id="project-1")
    client._http = MagicMock()
    client._http.post.return_value = {"id": "session-project"}

    client.ensure_session()

    payload = client._http.post.call_args.kwargs["json"]
    assert payload["project_id"] == "project-1"
    assert "organization_id" not in payload


def test_project_discovery_falls_back_to_first_organization(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_ORGANIZATION_ID", raising=False)
    client = ServiceClient(organization_id="")
    client._http = MagicMock()
    client._http.get.side_effect = [
        [{"id": "organization-default", "name": "Default"}],
        [{"id": "project-default", "name": "Default Project", "is_default": True}],
    ]

    projects = client.list_projects("")

    assert projects[0]["id"] == "project-default"
    assert client._http.get.call_args_list[0].args[0] == "/api/v1/organizations"
    assert (
        client._http.get.call_args_list[1].args[0]
        == "/api/v1/organizations/organization-default/projects"
    )


def test_training_client_logs_custom_metrics_and_exposes_training_run_id() -> None:
    service = ServiceClient()
    service._http = MagicMock()
    client = TrainingClient(
        service=service,
        model_id="run-1",
        base_model="test/model",
        session_id="session-1",
    )
    occurred_at = datetime(2026, 8, 3, tzinfo=timezone.utc)

    client.log_metrics(
        {"eval/reward": 0.75, "eval/pass_rate": 0.5},
        step=12,
        occurred_at=occurred_at,
        labels={"split": "eval"},
    )

    assert client.training_run_id == client.model_id == "run-1"
    args, kwargs = service._http.post.call_args
    assert args[0] == "/api/v1/sessions/session-1/metrics"
    assert kwargs["json"]["metrics"][0] == {
        "model_id": "run-1",
        "name": "eval/reward",
        "value": 0.75,
        "step": 12,
        "occurred_at": occurred_at.isoformat(),
        "labels": {"split": "eval"},
    }


def test_training_client_rejects_non_finite_metrics() -> None:
    service = ServiceClient()
    service._http = MagicMock()
    client = TrainingClient(
        service=service,
        model_id="run-1",
        base_model="test/model",
        session_id="session-1",
    )
    with pytest.raises(ValueError, match="must be finite"):
        client.log_metrics({"train/loss": float("nan")}, step=1)
    service._http.post.assert_not_called()


def test_async_console_protocol_matches_sync_client() -> None:
    async def run() -> None:
        service = AsyncServiceClient(
            organization_id="organization-2",
            project_id="project-2",
            user_metadata={"recipe": "sft"},
        )
        service._http = MagicMock()
        service._http.post = AsyncMock(return_value={"id": "session-2"})
        await service.ensure_session()
        _, session_kwargs = service._http.post.call_args
        assert session_kwargs["json"]["organization_id"] == "organization-2"
        assert session_kwargs["json"]["project_id"] == "project-2"
        assert session_kwargs["json"]["user_metadata"] == {"recipe": "sft"}

        training = AsyncTrainingClient(
            service=service,
            model_id="run-2",
            base_model="test/model",
            session_id="session-2",
        )
        await training.log_metrics({"eval/reward": 0.8}, step=3)
        assert training.training_run_id == "run-2"
        args, metric_kwargs = service._http.post.call_args
        assert args[0] == "/api/v1/sessions/session-2/metrics"
        assert metric_kwargs["json"]["metrics"][0]["step"] == 3

    asyncio.run(run())


def test_async_empty_scope_ids_are_omitted(monkeypatch) -> None:
    monkeypatch.setenv("WEAVER_ORGANIZATION_ID", "organization-from-env")
    monkeypatch.setenv("WEAVER_PROJECT_ID", "project-from-env")

    async def run() -> None:
        service = AsyncServiceClient(organization_id=" ", project_id="")
        service._http = MagicMock()
        service._http.post = AsyncMock(return_value={"id": "session-default"})

        await service.ensure_session()

        payload = service._http.post.call_args.kwargs["json"]
        assert "organization_id" not in payload
        assert "project_id" not in payload

    asyncio.run(run())


def test_async_project_only_scope_is_sent(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_ORGANIZATION_ID", raising=False)
    monkeypatch.delenv("WEAVER_PROJECT_ID", raising=False)

    async def run() -> None:
        service = AsyncServiceClient(project_id="project-async")
        service._http = MagicMock()
        service._http.post = AsyncMock(return_value={"id": "session-project"})

        await service.ensure_session()

        payload = service._http.post.call_args.kwargs["json"]
        assert payload["project_id"] == "project-async"
        assert "organization_id" not in payload

    asyncio.run(run())
