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
