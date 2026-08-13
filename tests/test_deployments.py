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

"""Tests for checkpoint deployments: Deployment type, clients, and CLI."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from weaver._deployments import (
    build_create_deployment_body,
    deployment_error_guidance,
    deployment_items,
    next_page_offset,
    translate_deployment_error,
    validate_deployment_name,
)
from weaver._http import WeaverAPIError
from weaver.async_service_client import AsyncServiceClient
from weaver.async_training_client import AsyncTrainingClient
from weaver.cli import cli
from weaver.operations import AsyncOperationHandle, OperationHandle
from weaver.service_client import ServiceClient
from weaver.training_client import TrainingClient
from weaver.types.checkpoint import Checkpoint
from weaver.types.deployment import Deployment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mirrors DeploymentService.PublicView in weaver-server
# (internal/services/deployments.go): a closed allowlist that deliberately
# omits provisioner metadata and the native workload id.
DEPLOYMENT_UUID = "11111111-2222-4333-8444-555555555555"
CHECKPOINT_UUID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

DEPLOYMENT_PAYLOAD: Dict[str, Any] = {
    "id": "dep-1",
    "checkpoint_id": CHECKPOINT_UUID,
    "artifact_id": "art-1",
    "model_id": "mdl-123",
    "name": "my-chat-model",
    "status": "running",
    "endpoint": "https://northgate.example.com/v1",
    "northgate_model_id": "ng-77",
    "gpu_type": "H800",
    "replicas": 2,
    "gpus_per_replica": 4,
    "error": None,
    "created_at": "2026-08-12T00:00:00Z",
    "updated_at": "2026-08-12T00:30:00Z",
}


def _make_training_client() -> TrainingClient:
    """Create a TrainingClient with a mocked ServiceClient."""
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    return TrainingClient(
        service=service,
        model_id="mdl-123",
        base_model="Qwen/Qwen3-8B",
        session_id="sess-abc",
    )


def _make_async_training_client() -> AsyncTrainingClient:
    """Create an AsyncTrainingClient with a mocked AsyncServiceClient."""
    service = MagicMock()
    service.next_operation_seq.return_value = 1
    service.http.post = AsyncMock()
    service.http.get = AsyncMock()
    return AsyncTrainingClient(
        service=service,
        model_id="mdl-123",
        base_model="Qwen/Qwen3-8B",
        session_id="sess-abc",
    )


def _make_service_client() -> ServiceClient:
    client = ServiceClient(base_url="https://test.example.com", api_key="sk-test")
    client._http = MagicMock()
    return client


def _make_async_service_client() -> AsyncServiceClient:
    client = AsyncServiceClient(base_url="https://test.example.com", api_key="sk-test")
    client._http = MagicMock()
    client._http.get = AsyncMock()
    client._http.delete = AsyncMock()
    return client


def _done_operation(response: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": "op-1", "status": "done", "response": response}


def _page(items: list, total: int) -> Dict[str, Any]:
    return {"items": items, "pagination": {"total_count": total}}


def _api_error(status: int, code: str, message: str) -> WeaverAPIError:
    return WeaverAPIError(status, code=code, message=message, retryable=False)


# ---------------------------------------------------------------------------
# Deployment type
# ---------------------------------------------------------------------------


class TestDeploymentType:
    def test_from_payload(self):
        deployment = Deployment.from_payload(DEPLOYMENT_PAYLOAD)
        assert deployment.id == "dep-1"
        assert deployment.checkpoint_id == CHECKPOINT_UUID
        assert deployment.artifact_id == "art-1"
        assert deployment.model_id == "mdl-123"
        assert deployment.name == "my-chat-model"
        assert deployment.status == "running"
        assert deployment.endpoint == "https://northgate.example.com/v1"
        assert deployment.northgate_model_id == "ng-77"
        assert deployment.gpu_type == "H800"
        assert deployment.replicas == 2
        assert deployment.gpus_per_replica == 4
        assert deployment.error is None
        assert deployment.created_at == "2026-08-12T00:00:00Z"
        assert deployment.updated_at == "2026-08-12T00:30:00Z"

    def test_from_payload_pending_nulls(self):
        # The server nulls endpoint/artifact/northgate id until they exist.
        deployment = Deployment.from_payload(
            {
                "id": "dep-2",
                "checkpoint_id": "ckpt-2",
                "model_id": "mdl-9",
                "name": "pending-model",
                "status": "pending",
                "artifact_id": None,
                "endpoint": None,
                "northgate_model_id": None,
                "gpu_type": None,
                "replicas": 1,
                "gpus_per_replica": 0,
                "error": None,
            }
        )
        assert deployment.status == "pending"
        assert deployment.artifact_id is None
        assert deployment.endpoint is None
        assert deployment.northgate_model_id is None
        assert deployment.gpu_type is None
        assert deployment.gpus_per_replica == 0

    def test_from_payload_minimal(self):
        deployment = Deployment.from_payload({"id": "dep-3"})
        assert deployment.id == "dep-3"
        assert deployment.name is None
        assert deployment.replicas is None

    def test_from_payload_failed_carries_error_code(self):
        deployment = Deployment.from_payload(
            {"id": "dep-4", "status": "failed", "error": "provisioning_failed"}
        )
        assert deployment.status == "failed"
        assert deployment.error == "provisioning_failed"

    def test_is_frozen(self):
        deployment = Deployment(id="dep-1")
        with pytest.raises(AttributeError):
            deployment.id = "dep-2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pure helpers (_deployments)
# ---------------------------------------------------------------------------


class TestValidateDeploymentName:
    @pytest.mark.parametrize(
        "name",
        ["a", "my-chat-model", "Model.v1_2", "a" * 63, "qwen3-8b.sft-2026"],
    )
    def test_accepts_valid_names(self, name):
        assert validate_deployment_name(name) == name

    def test_strips_surrounding_whitespace(self):
        assert validate_deployment_name("  my-model  ") == "my-model"

    @pytest.mark.parametrize(
        "name,match",
        [
            ("", "must not be empty"),
            ("   ", "must not be empty"),
            ("a" * 64, "at most 63 characters"),
            ("-leading", "invalid deployment name"),
            ("trailing-", "invalid deployment name"),
            ("has space", "invalid deployment name"),
            ("has/slash", "invalid deployment name"),
            ("has:colon", "invalid deployment name"),
        ],
    )
    def test_rejects_invalid_names(self, name, match):
        with pytest.raises(ValueError, match=match):
            validate_deployment_name(name)


class TestBuildCreateDeploymentBody:
    def test_defaults_omit_unset_sizing(self):
        body = build_create_deployment_body(name="my-model")
        assert body == {"name": "my-model", "overwrite": False, "replicas": 1}

    def test_all_fields(self):
        body = build_create_deployment_body(
            name="my-model",
            gpu_type="H800",
            replicas=3,
            gpus_per_replica=8,
            overwrite=True,
        )
        assert body == {
            "name": "my-model",
            "overwrite": True,
            "replicas": 3,
            "gpus_per_replica": 8,
            "gpu_type": "H800",
        }

    @pytest.mark.parametrize("replicas", [0, -1, 9])
    def test_rejects_out_of_range_replicas(self, replicas):
        with pytest.raises(ValueError, match="replicas must be between 1 and 8"):
            build_create_deployment_body(name="m", replicas=replicas)

    @pytest.mark.parametrize("gpus", [0, -1, 17])
    def test_rejects_out_of_range_gpus_per_replica(self, gpus):
        with pytest.raises(ValueError, match="gpus_per_replica must be between 1 and 16"):
            build_create_deployment_body(name="m", gpus_per_replica=gpus)

    def test_rejects_blank_gpu_type(self):
        with pytest.raises(ValueError, match="gpu_type must not be blank"):
            build_create_deployment_body(name="m", gpu_type="  ")

    def test_invalid_name_propagates(self):
        with pytest.raises(ValueError, match="invalid deployment name"):
            build_create_deployment_body(name="bad name")


class TestListingHelpers:
    def test_deployment_items_raises_on_malformed_entries(self):
        # Pagination advances by item count; silently dropping malformed
        # entries would undercount the consumed page (duplicates / early stop).
        with pytest.raises(ValueError, match="malformed deployment list entry"):
            deployment_items({"items": [{"id": "a"}, "junk", None]})

    def test_deployment_items_on_garbage(self):
        assert deployment_items(None) == []
        assert deployment_items({"items": "nope"}) == []

    def test_next_page_offset_advances_until_total(self):
        assert next_page_offset(_page([{}] * 100, 250), 0, 100) == 100
        assert next_page_offset(_page([{}] * 100, 250), 100, 100) == 200
        assert next_page_offset(_page([{}] * 50, 250), 200, 50) is None

    def test_next_page_offset_stops_on_empty_page(self):
        assert next_page_offset(_page([], 250), 0, 0) is None

    def test_next_page_offset_stops_without_usable_total(self):
        assert next_page_offset({"items": [{}]}, 0, 1) is None
        assert next_page_offset(_page([{}], "many"), 0, 1) is None


class TestDeploymentErrorTranslation:
    def test_permission_denied_names_the_gate(self):
        error = _api_error(403, "forbidden", "insufficient capability")
        translated = translate_deployment_error(error)
        assert translated.status_code == 403
        assert translated.code == "forbidden"
        assert "deployment.publish" in translated.message
        assert "allowed_biz_codes" in translated.message
        assert "SSO" in translated.message

    def test_feature_disabled_names_the_flag(self):
        error = _api_error(
            503, "deployment_unavailable", "checkpoint deployment is not enabled on this server"
        )
        translated = translate_deployment_error(error)
        assert translated.status_code == 503
        assert "administrator" in translated.message
        assert "deny-by-default" in translated.message

    def test_name_taken_explains_overwrite_does_not_help(self):
        translated = translate_deployment_error(
            _api_error(409, "name_taken", "deployment name is already in use")
        )
        assert "overwrite=True only replaces a registration on the gateway" in translated.message

    def test_quota_points_at_delete(self):
        translated = translate_deployment_error(
            _api_error(409, "deployment_limit_reached", "deployment limit reached")
        )
        assert "delete_deployment()" in translated.message

    def test_unrelated_forbidden_is_left_alone(self):
        # The create route also answers 403 for a checkpoint the caller cannot
        # write to; that is a different problem with different advice.
        error = _api_error(403, "forbidden", "insufficient access to checkpoint")
        assert translate_deployment_error(error) is error
        assert deployment_error_guidance(error) is None

    def test_other_errors_pass_through_unchanged(self):
        error = _api_error(404, "not_found", "deployment not found")
        assert translate_deployment_error(error) is error

    def test_structured_fields_survive_translation(self):
        error = WeaverAPIError(
            403,
            code="forbidden",
            message="insufficient capability",
            retryable=False,
            request_id="req-9",
            details={"hint": "x"},
        )
        translated = translate_deployment_error(error)
        assert translated.request_id == "req-9"
        assert translated.details == {"hint": "x"}
        assert translated.retryable is False


# ---------------------------------------------------------------------------
# TrainingClient.deploy_checkpoint (sync)
# ---------------------------------------------------------------------------


class TestDeployCheckpoint:
    def test_deploy_with_raw_checkpoint_id(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)

        deployment = tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-chat-model")

        assert isinstance(deployment, Deployment)
        assert deployment.id == "dep-1"
        assert deployment.endpoint == "https://northgate.example.com/v1"
        args = tc._service.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/deployments"
        assert args[1]["json"] == {
            "name": "my-chat-model",
            "overwrite": False,
            "replicas": 1,
        }
        assert args[1]["max_retries"] == 1

    def test_deploy_with_all_options(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)

        tc.deploy_checkpoint(
            CHECKPOINT_UUID,
            name="my-chat-model",
            gpu_type="H800",
            replicas=2,
            gpus_per_replica=4,
            overwrite=True,
        )

        assert tc._service.http.post.call_args[1]["json"] == {
            "name": "my-chat-model",
            "overwrite": True,
            "replicas": 2,
            "gpus_per_replica": 4,
            "gpu_type": "H800",
        }

    def test_deploy_with_checkpoint_object(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)

        ckpt = Checkpoint(id=CHECKPOINT_UUID, path="weaver://mdl-123/checkpoints/step-5")
        tc.deploy_checkpoint(ckpt, name="my-chat-model")

        assert (
            tc._service.http.post.call_args[0][0]
            == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/deployments"
        )

    def test_deploy_resolves_weaver_path(self):
        tc = _make_training_client()
        tc._service.http.get.return_value = {
            "items": [
                {
                    "id": "00000000-0000-4000-8000-000000000000",
                    "path": "weaver://mdl-123/checkpoints/step-1",
                },
                {"id": CHECKPOINT_UUID, "path": "weaver://mdl-123/checkpoints/step-5"},
            ]
        }
        tc._service.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)

        tc.deploy_checkpoint("weaver://mdl-123/checkpoints/step-5", name="my-chat-model")

        tc._service.http.get.assert_called_once_with("/api/v1/models/mdl-123/checkpoints")
        assert (
            tc._service.http.post.call_args[0][0]
            == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/deployments"
        )

    def test_deploy_foreign_model_path_raises(self):
        tc = _make_training_client()
        with pytest.raises(ValueError, match="belongs to model other"):
            tc.deploy_checkpoint("weaver://other/checkpoints/step-5", name="my-model")
        tc._service.http.post.assert_not_called()

    def test_no_wait_returns_operation_handle(self):
        tc = _make_training_client()
        tc._service.http.post.return_value = {"id": "op-9", "status": "pending"}

        handle = tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-model", wait=False)

        assert isinstance(handle, OperationHandle)
        assert handle.operation_id == "op-9"

    def test_invalid_name_fails_before_any_request(self):
        tc = _make_training_client()
        with pytest.raises(ValueError, match="invalid deployment name"):
            tc.deploy_checkpoint(CHECKPOINT_UUID, name="not a valid name")
        tc._service.http.post.assert_not_called()
        tc._service.http.get.assert_not_called()

    def test_out_of_range_replicas_fails_before_any_request(self):
        tc = _make_training_client()
        with pytest.raises(ValueError, match="replicas must be between 1 and 8"):
            tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-model", replicas=99)
        tc._service.http.post.assert_not_called()

    def test_permission_error_is_translated(self):
        tc = _make_training_client()
        tc._service.http.post.side_effect = _api_error(403, "forbidden", "insufficient capability")

        with pytest.raises(WeaverAPIError) as excinfo:
            tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-model")

        assert excinfo.value.status_code == 403
        assert "deployment.publish" in excinfo.value.message

    def test_feature_disabled_error_is_translated(self):
        tc = _make_training_client()
        tc._service.http.post.side_effect = _api_error(
            503, "deployment_unavailable", "checkpoint deployment is not enabled on this server"
        )

        with pytest.raises(WeaverAPIError) as excinfo:
            tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-model")

        assert excinfo.value.code == "deployment_unavailable"
        assert "administrator" in excinfo.value.message


# ---------------------------------------------------------------------------
# AsyncTrainingClient.deploy_checkpoint (async twin)
# ---------------------------------------------------------------------------


class TestAsyncDeployCheckpoint:
    def test_deploy_with_raw_checkpoint_id(self):
        tc = _make_async_training_client()
        tc._service.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)

        deployment = asyncio.run(tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-chat-model"))

        assert isinstance(deployment, Deployment)
        assert deployment.id == "dep-1"
        args = tc._service.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/deployments"
        assert args[1]["json"] == {"name": "my-chat-model", "overwrite": False, "replicas": 1}
        assert args[1]["max_retries"] == 1

    def test_deploy_resolves_weaver_path(self):
        tc = _make_async_training_client()
        tc._service.http.get.return_value = {
            "items": [{"id": CHECKPOINT_UUID, "path": "weaver://mdl-123/checkpoints/step-5"}]
        }
        tc._service.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)

        deployment = asyncio.run(
            tc.deploy_checkpoint(
                "weaver://mdl-123/checkpoints/step-5", name="my-model", gpu_type="H800"
            )
        )

        assert isinstance(deployment, Deployment)
        args = tc._service.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/deployments"
        assert args[1]["json"]["gpu_type"] == "H800"

    def test_no_wait_returns_async_operation_handle(self):
        tc = _make_async_training_client()
        tc._service.http.post.return_value = {"id": "op-9", "status": "pending"}

        handle = asyncio.run(tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-model", wait=False))

        assert isinstance(handle, AsyncOperationHandle)
        assert handle.operation_id == "op-9"

    def test_invalid_name_fails_before_any_request(self):
        tc = _make_async_training_client()
        with pytest.raises(ValueError, match="invalid deployment name"):
            asyncio.run(tc.deploy_checkpoint(CHECKPOINT_UUID, name="bad name"))
        tc._service.http.post.assert_not_called()

    def test_permission_error_is_translated(self):
        tc = _make_async_training_client()
        tc._service.http.post.side_effect = _api_error(403, "forbidden", "insufficient capability")

        with pytest.raises(WeaverAPIError) as excinfo:
            asyncio.run(tc.deploy_checkpoint(CHECKPOINT_UUID, name="my-model"))

        assert "deployment.publish" in excinfo.value.message


# ---------------------------------------------------------------------------
# ServiceClient deployment reads and teardown (sync)
# ---------------------------------------------------------------------------


class TestServiceClientDeployments:
    def test_list_deployments(self):
        client = _make_service_client()
        client._http.get.return_value = _page([DEPLOYMENT_PAYLOAD], 1)

        deployments = client.list_deployments()

        assert [d.id for d in deployments] == ["dep-1"]
        assert isinstance(deployments[0], Deployment)
        client._http.get.assert_called_once_with(
            "/api/v1/deployments", params={"limit": 100, "offset": 0}
        )

    def test_list_deployments_walks_every_page(self):
        client = _make_service_client()
        first = [dict(DEPLOYMENT_PAYLOAD, id=f"dep-{i}") for i in range(100)]
        second = [dict(DEPLOYMENT_PAYLOAD, id="dep-100")]
        client._http.get.side_effect = [_page(first, 101), _page(second, 101)]

        deployments = client.list_deployments()

        assert len(deployments) == 101
        assert deployments[-1].id == "dep-100"
        assert client._http.get.call_args_list[1][1]["params"] == {"limit": 100, "offset": 100}

    def test_list_deployments_translates_unavailable(self):
        client = _make_service_client()
        client._http.get.side_effect = _api_error(
            503, "deployment_unavailable", "checkpoint deployment is not enabled on this server"
        )

        with pytest.raises(WeaverAPIError) as excinfo:
            client.list_deployments()

        assert "administrator" in excinfo.value.message

    def test_get_deployment(self):
        client = _make_service_client()
        client._http.get.return_value = DEPLOYMENT_PAYLOAD

        deployment = client.get_deployment(DEPLOYMENT_UUID)

        assert isinstance(deployment, Deployment)
        assert deployment.northgate_model_id == "ng-77"
        client._http.get.assert_called_once_with(f"/api/v1/deployments/{DEPLOYMENT_UUID}")

    def test_delete_deployment_waits_and_returns_stopped(self):
        client = _make_service_client()
        stopped = dict(DEPLOYMENT_PAYLOAD, status="stopped", endpoint=None)
        client._http.delete.return_value = _done_operation(stopped)

        deployment = client.delete_deployment(DEPLOYMENT_UUID)

        assert isinstance(deployment, Deployment)
        assert deployment.status == "stopped"
        client._http.delete.assert_called_once_with(f"/api/v1/deployments/{DEPLOYMENT_UUID}")

    def test_delete_deployment_no_wait_returns_handle(self):
        client = _make_service_client()
        client._http.delete.return_value = {"id": "op-5", "status": "pending"}

        handle = client.delete_deployment(DEPLOYMENT_UUID, wait=False)

        assert isinstance(handle, OperationHandle)
        assert handle.operation_id == "op-5"

    def test_delete_already_stopped_passes_through(self):
        client = _make_service_client()
        client._http.delete.side_effect = _api_error(
            409, "already_stopped", "deployment is already stopped"
        )

        with pytest.raises(WeaverAPIError) as excinfo:
            client.delete_deployment(DEPLOYMENT_UUID)

        assert excinfo.value.code == "already_stopped"


# ---------------------------------------------------------------------------
# AsyncServiceClient deployment reads and teardown (async twin)
# ---------------------------------------------------------------------------


class TestAsyncServiceClientDeployments:
    def test_list_deployments(self):
        client = _make_async_service_client()
        client._http.get.return_value = _page([DEPLOYMENT_PAYLOAD], 1)

        deployments = asyncio.run(client.list_deployments())

        assert [d.id for d in deployments] == ["dep-1"]
        client._http.get.assert_called_once_with(
            "/api/v1/deployments", params={"limit": 100, "offset": 0}
        )

    def test_list_deployments_walks_every_page(self):
        client = _make_async_service_client()
        first = [dict(DEPLOYMENT_PAYLOAD, id=f"dep-{i}") for i in range(100)]
        client._http.get.side_effect = [
            _page(first, 101),
            _page([dict(DEPLOYMENT_PAYLOAD, id="dep-100")], 101),
        ]

        deployments = asyncio.run(client.list_deployments())

        assert len(deployments) == 101
        assert deployments[-1].id == "dep-100"

    def test_get_deployment(self):
        client = _make_async_service_client()
        client._http.get.return_value = DEPLOYMENT_PAYLOAD

        deployment = asyncio.run(client.get_deployment(DEPLOYMENT_UUID))

        assert isinstance(deployment, Deployment)
        client._http.get.assert_called_once_with(f"/api/v1/deployments/{DEPLOYMENT_UUID}")

    def test_delete_deployment_waits_and_returns_stopped(self):
        client = _make_async_service_client()
        client._http.delete.return_value = _done_operation(
            dict(DEPLOYMENT_PAYLOAD, status="stopped")
        )

        deployment = asyncio.run(client.delete_deployment(DEPLOYMENT_UUID))

        assert deployment.status == "stopped"
        client._http.delete.assert_called_once_with(f"/api/v1/deployments/{DEPLOYMENT_UUID}")

    def test_delete_deployment_no_wait_returns_handle(self):
        client = _make_async_service_client()
        client._http.delete.return_value = {"id": "op-5", "status": "pending"}

        handle = asyncio.run(client.delete_deployment(DEPLOYMENT_UUID, wait=False))

        assert isinstance(handle, AsyncOperationHandle)
        assert handle.operation_id == "op-5"

    def test_permission_error_is_translated(self):
        client = _make_async_service_client()
        client._http.get.side_effect = _api_error(
            503, "deployment_unavailable", "checkpoint deployment is not enabled on this server"
        )

        with pytest.raises(WeaverAPIError) as excinfo:
            asyncio.run(client.list_deployments())

        assert "administrator" in excinfo.value.message


# ---------------------------------------------------------------------------
# CLI: weaver deployment
# ---------------------------------------------------------------------------


@pytest.fixture(name="clean_env")
def _clean_env(monkeypatch):
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)


class TestDeploymentCLI:
    def test_create_with_checkpoint_id(self, clean_env):
        client = MagicMock()
        client.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli, ["deployment", "create", CHECKPOINT_UUID, "--name", "my-chat-model"]
            )

        assert result.exit_code == 0, result.output
        client.connect.assert_called_once_with(ensure_session=False)
        args = client.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/deployments"
        assert args[1]["json"] == {"name": "my-chat-model", "overwrite": False, "replicas": 1}
        assert args[1]["max_retries"] == 1
        assert "Deployment ready" in result.output
        client.close.assert_called_once_with()

    def test_create_resolves_weaver_uri_and_flags(self, clean_env):
        client = MagicMock()
        client.http.post.return_value = _done_operation(DEPLOYMENT_PAYLOAD)
        client.http.get.return_value = {
            "items": [{"id": CHECKPOINT_UUID, "path": "weaver://mdl-123/checkpoints/step-5"}]
        }
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli,
                [
                    "deployment",
                    "create",
                    "weaver://mdl-123/checkpoints/step-5",
                    "--name",
                    "my-chat-model",
                    "--gpu-type",
                    "H800",
                    "--replicas",
                    "2",
                    "--gpus-per-replica",
                    "4",
                    "--overwrite",
                ],
            )

        assert result.exit_code == 0, result.output
        client.http.get.assert_called_once_with("/api/v1/models/mdl-123/checkpoints")
        args = client.http.post.call_args
        assert args[0][0] == f"/api/v1/checkpoints/{CHECKPOINT_UUID}/deployments"
        assert args[1]["json"] == {
            "name": "my-chat-model",
            "overwrite": True,
            "replicas": 2,
            "gpus_per_replica": 4,
            "gpu_type": "H800",
        }

    def test_create_no_wait_prints_operation_id(self, clean_env):
        client = MagicMock()
        client.http.post.return_value = {"id": "op-42", "status": "pending"}
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli, ["deployment", "create", CHECKPOINT_UUID, "--name", "m", "--no-wait"]
            )

        assert result.exit_code == 0, result.output
        assert "op-42" in result.output

    def test_create_invalid_name_exits_without_request(self, clean_env):
        client = MagicMock()
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli, ["deployment", "create", CHECKPOINT_UUID, "--name", "bad name"]
            )

        assert result.exit_code == 1
        assert "invalid deployment name" in result.output
        client.http.post.assert_not_called()

    def test_create_permission_denied_prints_guidance(self, clean_env):
        client = MagicMock()
        client.http.post.side_effect = _api_error(403, "forbidden", "insufficient capability")
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli, ["deployment", "create", CHECKPOINT_UUID, "--name", "my-model"]
            )

        assert result.exit_code == 1
        assert "deployment.publish" in result.output

    def test_create_feature_disabled_prints_guidance(self, clean_env):
        client = MagicMock()
        client.http.post.side_effect = _api_error(
            503, "deployment_unavailable", "checkpoint deployment is not enabled on this server"
        )
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(
                cli, ["deployment", "create", CHECKPOINT_UUID, "--name", "my-model"]
            )

        assert result.exit_code == 1
        assert "administrator" in result.output

    def test_list_table(self, clean_env):
        client = MagicMock()
        client.list_deployments.return_value = [Deployment.from_payload(DEPLOYMENT_PAYLOAD)]
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["deployment", "list"])

        assert result.exit_code == 0, result.output
        # Rich elides long cells at the 80-column test width, so assert on the
        # short columns and the summary line rather than on the full name.
        assert "dep-1" in result.output
        assert "running" in result.output
        assert "H800" in result.output
        assert "Showing 1 deployment(s)" in result.output

    def test_list_json(self, clean_env):
        client = MagicMock()
        client.list_deployments.return_value = [Deployment.from_payload(DEPLOYMENT_PAYLOAD)]
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["deployment", "list", "--format", "json"])

        assert result.exit_code == 0, result.output
        assert "northgate_model_id" in result.output

    def test_get(self, clean_env):
        client = MagicMock()
        client.get_deployment.return_value = Deployment.from_payload(DEPLOYMENT_PAYLOAD)
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["deployment", "get", "dep-1"])

        assert result.exit_code == 0, result.output
        client.get_deployment.assert_called_once_with("dep-1")
        assert "northgate.example.com" in result.output

    def test_delete_waits_by_default(self, clean_env):
        client = MagicMock()
        client.delete_deployment.return_value = Deployment.from_payload(
            dict(DEPLOYMENT_PAYLOAD, status="stopped")
        )
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["deployment", "delete", "dep-1"])

        assert result.exit_code == 0, result.output
        client.delete_deployment.assert_called_once_with("dep-1")
        assert "Deployment stopped" in result.output

    def test_delete_no_wait_prints_operation_id(self, clean_env):
        client = MagicMock()
        handle = MagicMock()
        handle.operation_id = "op-77"
        client.delete_deployment.return_value = handle
        with patch("weaver.cli.ServiceClient", return_value=client):
            result = CliRunner().invoke(cli, ["deployment", "delete", "dep-1", "--no-wait"])

        assert result.exit_code == 0, result.output
        client.delete_deployment.assert_called_once_with("dep-1", wait=False)
        assert "op-77" in result.output


class TestDeploymentIdValidation:
    """A raw deployment id becomes a URL path segment; dot-segment tricks must
    be rejected before any request is built (httpx normalizes `..`, so
    `../checkpoints/<uuid>` would otherwise reroute to the checkpoints API)."""

    CHECKPOINT_UUID = "99999999-8888-4777-a666-555555555555"

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../checkpoints/99999999-8888-4777-a666-555555555555",
            "../models/99999999-8888-4777-a666-555555555555",
            "a/b",
            "..",
            "%2e%2e%2fcheckpoints",
            "dep-1",
            "",
            "11111111222243338444555555555555",  # unhyphenated alias
            "urn:uuid:11111111-2222-4333-8444-555555555555",
        ],
    )
    def test_get_and_delete_reject_non_uuid_ids(self, bad_id):
        client = _make_service_client()
        with pytest.raises(ValueError, match="deployment id must be a"):
            client.get_deployment(bad_id)
        with pytest.raises(ValueError, match="deployment id must be a"):
            client.delete_deployment(bad_id)
        client._http.get.assert_not_called()
        client._http.delete.assert_not_called()

    def test_async_twins_reject_traversal(self):
        client = _make_async_service_client()
        with pytest.raises(ValueError, match="deployment id must be a"):
            asyncio.run(client.get_deployment("../checkpoints/x"))
        with pytest.raises(ValueError, match="deployment id must be a"):
            asyncio.run(client.delete_deployment("../checkpoints/x"))
        client._http.get.assert_not_called()
        client._http.delete.assert_not_called()

    def test_uppercase_uuid_normalized(self):
        client = _make_service_client()
        client._http.get.return_value = dict(DEPLOYMENT_PAYLOAD)
        client.get_deployment(DEPLOYMENT_UUID.upper())
        client._http.get.assert_called_once_with(f"/api/v1/deployments/{DEPLOYMENT_UUID}")

    def test_checkpoint_raw_id_must_be_uuid(self):
        # export_weights / deploy_checkpoint share _resolve_checkpoint_id: a
        # non-weaver:// string is a raw checkpoint id and gets the same guard.
        client = _make_training_client()
        with pytest.raises(ValueError, match="checkpoint id must be a"):
            client.deploy_checkpoint("../models/x", name="n")
        client._service.http.post.assert_not_called()
