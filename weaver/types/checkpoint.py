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
        id: Unique checkpoint identifier.
        path: Checkpoint path (e.g. ``weaver://run-id/weights/checkpoint-001``).
        name: Optional human-readable name.
        checkpoint_type: ``"training"`` or ``"training_with_optimizer"``.
        status: Current status of the checkpoint (e.g. ``"completed"``).
    """

    id: str
    path: str
    name: str | None = None
    checkpoint_type: str = "training"
    status: str | None = None

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
                or "training"
            ),
            status=_str_or_none(lookup_case_insensitive(payload, "status")),
        )


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None
