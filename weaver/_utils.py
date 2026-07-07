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

"""Shared helper utilities."""

from __future__ import annotations

from typing import Any, Dict


class _UnsetType:
    """Sentinel to distinguish 'parameter not passed' from explicit ``None``."""

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"

    def __bool__(self) -> bool:
        return False


UNSET = _UnsetType()


# Default time-to-live for sampling checkpoints created through the SDK. Sampler
# exports are regenerable, short-lived RL weight-sync artifacts, so the SDK
# stamps a bounded TTL by default instead of letting them accumulate forever on
# shared storage. Callers can pass an explicit ``ttl_seconds`` (including
# ``None`` for permanent retention) to override it.
DEFAULT_SAMPLER_TTL_SECONDS = 3600  # 1 hour


def lookup_case_insensitive(payload: Dict[str, Any], name: str) -> Any:
    variants = {
        name,
        name.lower(),
        name.upper(),
        name.capitalize(),
        name.replace("_", "").lower(),
    }
    for variant in variants:
        if variant in payload:
            return payload[variant]
    snake = name
    camel = snake.replace("_", "")
    upper_camel = "".join(part.capitalize() for part in snake.split("_"))
    for variant in (upper_camel, camel):
        if variant in payload:
            return payload[variant]
    return None


def extract_id(payload: Dict[str, Any]) -> str:
    identifier = lookup_case_insensitive(payload, "id")
    if identifier is None:
        raise ValueError("Payload missing id field")
    return str(identifier)
