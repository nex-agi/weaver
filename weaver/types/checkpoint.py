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

"""Checkpoint type for checkpoint management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Checkpoint:
    """Represents a saved model checkpoint.

    Attributes:
        id: Unique checkpoint identifier (UUID, server-generated).
        path: Server-generated storage path
            (e.g. ``weaver://{model_id}/checkpoints/step-100``).
            Read-only — never used as an API input.
        name: Human-readable label provided at save time.
        checkpoint_type: ``"weight"`` or ``"weight_and_optimizer"``.
        status: Current status of the checkpoint (e.g. ``"completed"``).
        train_unembed: Whether the unembedding layer was trained (``None`` if
            the server did not report this field).
        train_mlp: Whether MLP layers were trained.
        train_attn: Whether attention layers were trained.
        ttl_seconds: Time-to-live in seconds, or ``None`` for permanent.
        created_at: ISO 8601 creation timestamp (server-generated).
        expires_at: ISO 8601 expiration timestamp (server-generated),
            or ``None`` if permanent.
    """

    id: str
    path: str
    name: str | None = None
    checkpoint_type: str = "weight"
    status: str | None = None
    train_unembed: bool | None = None
    train_mlp: bool | None = None
    train_attn: bool | None = None
    ttl_seconds: int | None = None
    created_at: str | None = None
    expires_at: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Checkpoint:
        """Create a Checkpoint from a server response dict.

        Args:
            payload: Raw JSON dict from the Weaver API.

        Returns:
            A ``Checkpoint`` instance.
        """
        from .._utils import lookup_case_insensitive

        return cls(
            id=str(lookup_case_insensitive(payload, "id") or ""),
            path=str(lookup_case_insensitive(payload, "path") or ""),
            name=_str_or_none(lookup_case_insensitive(payload, "name")),
            checkpoint_type=str(
                lookup_case_insensitive(payload, "checkpoint_type")
                or lookup_case_insensitive(payload, "type")
                or "weight"
            ),
            status=_str_or_none(lookup_case_insensitive(payload, "status")),
            train_unembed=_bool_or_none(lookup_case_insensitive(payload, "train_unembed")),
            train_mlp=_bool_or_none(lookup_case_insensitive(payload, "train_mlp")),
            train_attn=_bool_or_none(lookup_case_insensitive(payload, "train_attn")),
            ttl_seconds=_int_or_none(lookup_case_insensitive(payload, "ttl_seconds")),
            created_at=_str_or_none(lookup_case_insensitive(payload, "created_at")),
            expires_at=_str_or_none(lookup_case_insensitive(payload, "expires_at")),
        )


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _bool_or_none(value: Any) -> bool | None:
    return bool(value) if value is not None else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None
