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

"""Identity-safe recovery helpers for checkpoint operation projections."""

from __future__ import annotations

from collections.abc import Iterable, Set

from .types.checkpoint import Checkpoint

CHECKPOINT_RECOVERY_DELAYS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)


def select_recovered_checkpoint(
    candidates: Iterable[Checkpoint],
    *,
    existing_ids: Set[str],
    partial: Checkpoint,
    name: str | None,
    checkpoint_type: str,
) -> Checkpoint | None:
    """Return the one checkpoint that can be tied to the completed save.

    A public ID is authoritative. A public ``weaver://`` path is the next-best
    identity. Raw trainer paths are deliberately ignored because they are
    internal filesystem paths, not the public path exposed by checkpoint rows.
    Without either identity, only IDs absent from the pre-save snapshot qualify.
    """

    partial_id = partial.id or None
    partial_path = partial.path if partial.path.startswith("weaver://") else None
    matches: list[Checkpoint] = []
    for candidate in candidates:
        if candidate.checkpoint_type != checkpoint_type:
            continue
        if name is not None and candidate.name != name:
            continue
        if partial_id is not None and candidate.id != partial_id:
            continue
        if partial_path is not None and candidate.path != partial_path:
            continue
        if partial_id is None and partial_path is None:
            if not candidate.id or candidate.id in existing_ids:
                continue
        matches.append(candidate)

    if len(matches) > 1:
        raise RuntimeError(
            "Save completed but checkpoint recovery was ambiguous; "
            "multiple newly created checkpoints matched the request"
        )
    if not matches:
        return None

    candidate = matches[0]
    status = (candidate.status or "").lower()
    if status in {"error", "deleted"}:
        raise RuntimeError(f"Save completed but the recovered checkpoint entered status {status!r}")
    if status not in {"", "completed"}:
        return None
    if not candidate.id or not candidate.path:
        return None
    return candidate
