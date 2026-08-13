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

"""Deployment type for checkpoints published as NorthGate endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Deployment:
    """One checkpoint published as a standalone, publicly callable endpoint.

    Created by
    :meth:`~weaver.training_client.TrainingClient.deploy_checkpoint`, which
    converts the checkpoint to HuggingFace format, launches a dedicated
    inference workload for it, and registers that workload on the NorthGate
    gateway as an OpenAI-compatible model.

    A deployment is independent of the training inference instance and of
    every other deployment of the same model: it owns its own GPUs, and it
    keeps its weights alive (neither the source checkpoint nor the exported
    artifact can be garbage-collected while the deployment is live).

    Attributes:
        id: Unique deployment identifier (UUID, server-generated).
        checkpoint_id: Identifier of the published checkpoint.
        artifact_id: Identifier of the HuggingFace artifact the workload
            serves from, or ``None`` until the conversion has produced one.
        model_id: Identifier of the model the checkpoint belongs to.
        name: Public model name. It is simultaneously the served model name,
            the gateway's ``model_name``, and a Kubernetes label value, so it
            is limited to 63 characters of letters, digits, ``.``, ``-`` and
            ``_``, starting and ending alphanumerically.
        status: Lifecycle status — ``"pending"``, ``"converting"``,
            ``"provisioning"``, ``"onboarding"``, ``"running"``, ``"failed"``,
            or ``"stopped"``. Only ``running`` serves traffic; ``failed`` and
            ``stopped`` are terminal.
        endpoint: OpenAI-compatible URL the gateway serves this model on, or
            ``None`` before onboarding completes.
        northgate_model_id: The gateway's own id for the onboarded model, or
            ``None`` before onboarding completes.
        gpu_type: GPU type the workload runs on, or ``None`` when the server
            chose its configured default.
        replicas: Number of serving replicas.
        gpus_per_replica: GPUs per replica (``0`` when the launcher's default
            applies).
        error: Stable failure code when ``status == "failed"`` — one of
            ``"conversion_failed"``, ``"provisioning_failed"``,
            ``"onboarding_failed"``, ``"teardown_failed"``,
            ``"deleted_during_deploy"``, or ``"gateway_name_taken"``.
            Detailed diagnostics stay in the operation and the server logs.
        created_at: ISO 8601 creation timestamp (server-generated).
        updated_at: ISO 8601 last-update timestamp (server-generated).
    """

    id: str
    checkpoint_id: str | None = None
    artifact_id: str | None = None
    model_id: str | None = None
    name: str | None = None
    status: str | None = None
    endpoint: str | None = None
    northgate_model_id: str | None = None
    gpu_type: str | None = None
    replicas: int | None = None
    gpus_per_replica: int | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Deployment:
        """Create a Deployment from a server response dict.

        Args:
            payload: Raw JSON dict from the Weaver API.

        Returns:
            A ``Deployment`` instance.
        """
        from .._utils import lookup_case_insensitive

        return cls(
            id=str(lookup_case_insensitive(payload, "id") or ""),
            checkpoint_id=_str_or_none(lookup_case_insensitive(payload, "checkpoint_id")),
            artifact_id=_str_or_none(lookup_case_insensitive(payload, "artifact_id")),
            model_id=_str_or_none(lookup_case_insensitive(payload, "model_id")),
            name=_str_or_none(lookup_case_insensitive(payload, "name")),
            status=_str_or_none(lookup_case_insensitive(payload, "status")),
            endpoint=_str_or_none(lookup_case_insensitive(payload, "endpoint")),
            northgate_model_id=_str_or_none(lookup_case_insensitive(payload, "northgate_model_id")),
            gpu_type=_str_or_none(lookup_case_insensitive(payload, "gpu_type")),
            replicas=_int_or_none(lookup_case_insensitive(payload, "replicas")),
            gpus_per_replica=_int_or_none(lookup_case_insensitive(payload, "gpus_per_replica")),
            error=_str_or_none(lookup_case_insensitive(payload, "error")),
            created_at=_str_or_none(lookup_case_insensitive(payload, "created_at")),
            updated_at=_str_or_none(lookup_case_insensitive(payload, "updated_at")),
        )


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None
