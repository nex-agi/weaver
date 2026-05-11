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

"""Router Replay contract types shared across Weaver clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

RouterReplayMode = str
RouterReplaySource = str
RouterReplayFormat = str
RouterReplayTokenAlignment = str

ROUTER_REPLAY_MODE_R2: RouterReplayMode = "R2"
ROUTER_REPLAY_MODE_R3: RouterReplayMode = "R3"
ROUTER_REPLAY_SOURCE_RECOMPUTE: RouterReplaySource = "recompute"
ROUTER_REPLAY_SOURCE_ROLLOUT: RouterReplaySource = "rollout"
ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK: RouterReplayFormat = "token_layer_topk"
ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED: RouterReplayTokenAlignment = "target_aligned"


@dataclass(slots=True)
class RouterReplayIndices:
    """Token-layer-topk router replay indices payload.

    RFC-0001: This is the normalized, backend-neutral contract used by the SDK
    and downstream trainer metadata.
    """

    value: Sequence[Sequence[Sequence[int]]]
    num_layers: int
    topk: int
    format: RouterReplayFormat = ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK
    token_alignment: RouterReplayTokenAlignment = ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED

    def to_payload(self) -> dict[str, object]:
        return {
            "format": self.format,
            "value": _nested_list(self.value),
            "token_alignment": self.token_alignment,
            "num_layers": self.num_layers,
            "topk": self.topk,
        }


@dataclass(slots=True)
class RouterReplayMetadata:
    """Training-side router replay metadata envelope.

    RFC-0001: Training requests carry this object at top-level metadata, not in
    loss_fn_inputs, so the replay execution context stays separate from loss data.
    """

    mode: RouterReplayMode
    source: RouterReplaySource
    indices: RouterReplayIndices
    fail_fast: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "source": self.source,
            "indices": self.indices.to_payload(),
            "fail_fast": self.fail_fast,
        }


@dataclass(slots=True)
class RouterReplayModelConfig:
    """Model-registration router replay toggle.

    RFC-0001: This mirrors supported-models.config.router_replay and keeps the
    enable/mode/shape contract explicit for clients that introspect model config.
    """

    enabled: bool
    mode: RouterReplayMode | None = None
    num_layers: int | None = None
    topk: int | None = None
    fail_fast: bool = True

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"enabled": self.enabled, "fail_fast": self.fail_fast}
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.num_layers is not None:
            payload["num_layers"] = self.num_layers
        if self.topk is not None:
            payload["topk"] = self.topk
        return payload


def _nested_list(value: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            result.append(_nested_list(item))
        else:
            result.append(item)
    return result
