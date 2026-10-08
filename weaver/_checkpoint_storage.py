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

"""Capability checks for explicitly selected checkpoint storage backends."""

from __future__ import annotations

from typing import Any


def validate_storage_backend(backend: str | None) -> None:
    if backend is not None and backend not in ("gpfs", "artifact"):
        raise ValueError("storage_backend must be gpfs or artifact")


def verify_checkpoint_storage_capability(capabilities: Any) -> None:
    if (
        not isinstance(capabilities, dict)
        or type(capabilities.get("protocol_version")) is not int
        or capabilities.get("protocol_version") != 1
        or capabilities.get("default_backend") != "gpfs"
        or not isinstance(capabilities.get("save_state_backends"), list)
        or "artifact" not in capabilities["save_state_backends"]
    ):
        raise RuntimeError("Server does not currently allow managed checkpoint saves")


def preferred_permanent_checkpoint_backend(capabilities: Any) -> str | None:
    """Negotiate upgraded saves without changing historical omitted payloads."""
    if (
        not isinstance(capabilities, dict)
        or type(capabilities.get("protocol_version")) is not int
        or capabilities["protocol_version"] != 1
        or capabilities.get("default_backend") != "gpfs"
        or not isinstance(capabilities.get("save_state_backends"), list)
        or "gpfs" not in capabilities["save_state_backends"]
    ):
        raise RuntimeError("Server checkpoint storage preference is invalid")
    field = "preferred_permanent_checkpoint_backend"
    if field not in capabilities:
        return None  # Older capability responses preserve the historical default.
    if (
        type(capabilities.get("protocol_version")) is not int
        or capabilities["protocol_version"] != 1
        or capabilities.get("default_backend") != "gpfs"
        or capabilities[field] not in ("gpfs", "artifact")
    ):
        raise RuntimeError("Server checkpoint storage preference is invalid")
    return "artifact" if capabilities[field] == "artifact" else None
