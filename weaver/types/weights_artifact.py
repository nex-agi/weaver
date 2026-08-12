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

"""WeightsArtifact type for HuggingFace weights export management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WeightsArtifact:
    """Represents an exported HuggingFace-format weights artifact.

    An artifact is produced from a checkpoint by
    :meth:`~weaver.training_client.TrainingClient.export_weights` and lives in
    object storage independently of its source checkpoint (deleting or
    expiring the checkpoint does not cascade to the artifact).

    Attributes:
        id: Unique artifact identifier (UUID, server-generated).
        checkpoint_id: Identifier of the source checkpoint.
        model_id: Identifier of the model that produced the checkpoint.
        kind: ``"hf_model"`` (full HF model directory) or ``"hf_adapter"``
            (HF PEFT adapter directory).
        status: Current status (``"pending"``, ``"completed"``, ``"error"``,
            or ``"deleted"``).
        uri: Server-generated artifact URI
            (``weaver://{model_id}/checkpoints/{name}/artifacts/{kind}``).
            Read-only — accepted by
            :meth:`~weaver.service_client.ServiceClient.download_weights`.
        size_bytes: Total size of the exported files, or ``None`` if unknown.
        manifest: Export manifest (file names, sizes, sha256 digests), or
            ``None`` while the export is still pending.
        error: Failure detail when ``status == "error"``.
        ttl_seconds: Time-to-live in seconds, or ``None`` to follow the
            source checkpoint's retention.
        expires_at: ISO 8601 expiration timestamp (server-generated), or
            ``None`` if permanent.
        created_at: ISO 8601 creation timestamp (server-generated).
        updated_at: ISO 8601 last-update timestamp (server-generated).
    """

    id: str
    checkpoint_id: str | None = None
    model_id: str | None = None
    kind: str = "hf_model"
    status: str | None = None
    uri: str | None = None
    size_bytes: int | None = None
    manifest: dict[str, Any] | None = None
    error: str | None = None
    ttl_seconds: int | None = None
    expires_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WeightsArtifact:
        """Create a WeightsArtifact from a server response dict.

        Args:
            payload: Raw JSON dict from the Weaver API.

        Returns:
            A ``WeightsArtifact`` instance.
        """
        from .._utils import lookup_case_insensitive

        manifest = lookup_case_insensitive(payload, "manifest")
        return cls(
            id=str(lookup_case_insensitive(payload, "id") or ""),
            checkpoint_id=_str_or_none(lookup_case_insensitive(payload, "checkpoint_id")),
            model_id=_str_or_none(lookup_case_insensitive(payload, "model_id")),
            kind=str(lookup_case_insensitive(payload, "kind") or "hf_model"),
            status=_str_or_none(lookup_case_insensitive(payload, "status")),
            uri=_str_or_none(lookup_case_insensitive(payload, "uri")),
            size_bytes=_int_or_none(lookup_case_insensitive(payload, "size_bytes")),
            manifest=manifest if isinstance(manifest, dict) else None,
            error=_str_or_none(lookup_case_insensitive(payload, "error")),
            ttl_seconds=_int_or_none(lookup_case_insensitive(payload, "ttl_seconds")),
            expires_at=_str_or_none(lookup_case_insensitive(payload, "expires_at")),
            created_at=_str_or_none(lookup_case_insensitive(payload, "created_at")),
            updated_at=_str_or_none(lookup_case_insensitive(payload, "updated_at")),
        )


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None
