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

"""Shared non-sensitive request and readiness policy for pre-model data preparation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._http import WeaverAPIError
from .types.managed_dataset import SampleRef

PREPARATION_CHUNK_ITEMS = 256


def preparation_request(
    refs: Sequence[SampleRef], *, base_model: str, training_max_sequence_length: int
) -> dict[str, Any]:
    """Build the same exact-budget preparation request for both client stacks."""
    if not isinstance(base_model, str) or not base_model or base_model != base_model.strip():
        raise ValueError("base_model must be a non-empty model name")
    if (
        not isinstance(training_max_sequence_length, int)
        or isinstance(training_max_sequence_length, bool)
        or training_max_sequence_length < 2
    ):
        raise ValueError("training_max_sequence_length must be an integer of at least 2")
    if not all(isinstance(ref, SampleRef) for ref in refs):
        raise TypeError("refs must contain only SampleRef values")
    if not 1 <= len(refs) <= PREPARATION_CHUNK_ITEMS:
        raise ValueError("sample preparation requests must be bounded")
    return {
        "base_model": base_model,
        "training_max_sequence_length": training_max_sequence_length,
        "items": [ref.to_payload() for ref in refs],
    }


def validate_preparation_wait(*, wait: bool, timeout: float, poll_interval: float) -> None:
    if not isinstance(wait, bool):
        raise TypeError("wait must be a boolean")
    for name, value in (("timeout", timeout), ("poll_interval", poll_interval)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")


def preparation_retry_delay(
    error: WeaverAPIError, *, wait: bool, remaining: float, poll_interval: float
) -> float:
    """Only readiness/infrastructure codes can wait; ACL/protocol failures propagate."""
    if (
        not wait
        or not error.retryable
        or error.code not in {"managed_dataset_preparing", "managed_datasets_unavailable"}
    ):
        raise error
    if remaining <= 0:
        raise TimeoutError("Timed out preparing managed samples") from error
    try:
        hint = float(error.retry_after) if error.retry_after is not None else poll_interval
    except (TypeError, ValueError):
        hint = poll_interval
    if not math.isfinite(hint) or hint <= 0:
        hint = poll_interval
    return min(remaining, max(poll_interval, min(hint, 30.0)))


def preparation_remaining(deadline: float, now: float) -> float:
    """Reject an expired budget before starting another HTTP request/chunk."""
    remaining = deadline - now
    if remaining <= 0:
        raise TimeoutError("Timed out preparing managed samples")
    return remaining
