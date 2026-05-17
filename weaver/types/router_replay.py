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
from typing import Any, Mapping, Sequence

from .payload_ref import materialize_payload_ref

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

    This is the normalized, backend-neutral contract used by the SDK and
    downstream trainer metadata.
    """

    num_layers: int
    topk: int
    value: Sequence[Sequence[Sequence[int]]] | None = None
    format: RouterReplayFormat = ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK
    token_alignment: RouterReplayTokenAlignment = ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED
    shards: Sequence[Mapping[str, Any]] | None = None
    transport: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": self.format,
            "token_alignment": self.token_alignment,
            "num_layers": self.num_layers,
            "topk": self.topk,
        }
        if self.value is not None:
            payload["value"] = _nested_list(self.value)
        if self.shards is not None:
            payload["shards"] = [dict(shard) for shard in self.shards]
        if self.transport is not None:
            payload["transport"] = self.transport
        return payload


@dataclass(slots=True)
class RouterReplayMetadata:
    """Training-side router replay metadata envelope.

    Training requests carry this object at top-level metadata, not in
    loss_fn_inputs, so the replay execution context stays separate from loss data.
    """

    mode: RouterReplayMode
    source: RouterReplaySource
    indices: RouterReplayIndices | Mapping[str, Any] | None = None
    fail_fast: bool = True
    action: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode,
            "source": self.source,
            "fail_fast": self.fail_fast,
        }
        if self.indices is not None:
            payload["indices"] = (
                self.indices.to_payload()
                if isinstance(self.indices, RouterReplayIndices)
                else dict(self.indices)
            )
        if self.action is not None:
            payload["action"] = self.action
        return payload

    @classmethod
    def r2_record(cls, *, fail_fast: bool = True) -> "RouterReplayMetadata":
        """Build per-call R2 RECORD metadata for ``forward`` recompute calls."""

        return cls(
            mode=ROUTER_REPLAY_MODE_R2,
            source=ROUTER_REPLAY_SOURCE_RECOMPUTE,
            indices=None,
            fail_fast=fail_fast,
            action="RECORD",
        )

    @classmethod
    def r2_replay(
        cls,
        indices: RouterReplayIndices | Mapping[str, Any],
        *,
        fail_fast: bool = True,
    ) -> "RouterReplayMetadata":
        return cls(
            mode=ROUTER_REPLAY_MODE_R2,
            source=ROUTER_REPLAY_SOURCE_RECOMPUTE,
            indices=indices,
            fail_fast=fail_fast,
        )

    @classmethod
    def r3_replay(
        cls,
        indices: RouterReplayIndices | Mapping[str, Any],
        *,
        fail_fast: bool = True,
    ) -> "RouterReplayMetadata":
        return cls(
            mode=ROUTER_REPLAY_MODE_R3,
            source=ROUTER_REPLAY_SOURCE_ROLLOUT,
            indices=indices,
            fail_fast=fail_fast,
        )


@dataclass(slots=True)
class RouterReplayModelConfig:
    """Model-registration router replay toggle.

    This mirrors supported-models.config.router_replay and keeps the
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


def materialize_router_replay_indices(envelope: Mapping[str, Any]) -> list[Any]:
    """Materialize a router replay indices envelope into inspectable lists.

    Inline envelopes return their ``value`` directly. Sharded envelopes return
    per-sample values when ``sample_indices`` metadata is available, otherwise a
    concatenated token-level list.
    """

    value = envelope.get("value")
    if isinstance(value, list):
        return value

    shards = envelope.get("shards")
    if not isinstance(shards, (list, tuple)):
        return []

    per_sample_parts: dict[int, dict[int, list[Any]]] = {}
    concatenated: list[Any] = []
    saw_sample_layout = False
    seen_sample_pp: set[tuple[int, int]] = set()
    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        shard_value = shard.get("value")
        if shard_value is None and isinstance(shard.get("value_ref"), Mapping):
            shard_value = materialize_payload_ref(shard["value_ref"], field="indices")
            if hasattr(shard_value, "tolist"):
                shard_value = shard_value.tolist()
        if not isinstance(shard_value, list):
            continue
        sample_indices = shard.get("sample_indices")
        tokens_per_sample = int(shard.get("local_tokens_per_sample") or 0)
        if isinstance(sample_indices, list) and tokens_per_sample > 0:
            saw_sample_layout = True
            pp_rank = int(shard.get("pp_rank") or 0)
            for offset, sample_idx in enumerate(sample_indices):
                sample_idx = int(sample_idx)
                sample_pp_key = (sample_idx, pp_rank)
                if sample_pp_key in seen_sample_pp:
                    # TP shards carry duplicate user-visible token/layer rows for
                    # inspection purposes; one TP copy per PP stage is enough.
                    continue
                seen_sample_pp.add(sample_pp_key)
                start = offset * tokens_per_sample
                end = start + tokens_per_sample
                per_sample_parts.setdefault(sample_idx, {})[pp_rank] = shard_value[start:end]
        else:
            concatenated.extend(shard_value)

    if saw_sample_layout:
        materialized: list[Any] = []
        for sample_idx in sorted(per_sample_parts):
            pp_parts = per_sample_parts[sample_idx]
            if not pp_parts:
                continue
            ordered_parts = [pp_parts[pp_rank] for pp_rank in sorted(pp_parts)]
            sample_tokens = ordered_parts[0]
            for extra_part in ordered_parts[1:]:
                sample_tokens = [
                    list(token_layers) + list(extra_part[token_idx])
                    for token_idx, token_layers in enumerate(sample_tokens)
                    if token_idx < len(extra_part)
                ]
            materialized.append(sample_tokens)
        return materialized
    return concatenated
