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

"""Sampling engine control-plane types shared across Weaver clients."""

from __future__ import annotations

from enum import Enum


class PauseMode(str, Enum):
    """How the sampling engine should pause in-flight generation.

    All three modes freeze the engine until generation is continued; they
    differ in what happens to requests that are already in flight:

    - ``ABORT``: abort waiting + running requests on the spot and return their
      partial output to the caller (``stop_reason="abort"``). This is the mode
      used for partial/async rollout weight swaps
      (abort -> drain -> sync_weights -> continue).
    - ``RETRACT``: retract running requests back into the waiting queue so they
      resume from scratch after ``continue``.
    - ``IN_PLACE``: freeze requests where they are and resume them in place
      after ``continue``.
    """

    ABORT = "abort"
    RETRACT = "retract"
    IN_PLACE = "in_place"


def coerce_pause_mode(mode: "PauseMode | str") -> str:
    """Validate and normalize a pause mode to its wire string.

    Args:
        mode: A :class:`PauseMode` member or its string value.

    Returns:
        The canonical wire string (e.g. ``"abort"``).

    Raises:
        ValueError: If ``mode`` is not one of the supported pause modes.
    """
    if isinstance(mode, PauseMode):
        return mode.value
    try:
        return PauseMode(mode).value
    except ValueError as exc:
        valid = ", ".join(m.value for m in PauseMode)
        raise ValueError(f"invalid pause mode {mode!r}: expected one of {valid}") from exc
