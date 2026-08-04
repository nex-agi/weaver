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

import json
import os
from dataclasses import dataclass
from pathlib import Path
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
ROUTER_REPLAY_DATUM_SCHEMA = "weaver.router_replay.datum.v1"
ROUTER_REPLAY_INDEX_SET_SCHEMA = "weaver.router_replay.index_set.v1"


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
    """Datum-level router replay metadata envelope.

    Training requests carry this object in ``Datum.metadata["router_replay"]``.
    Batch-level router replay metadata is intentionally no longer canonical.
    """

    mode: RouterReplayMode
    source: RouterReplaySource
    fail_fast: bool = True
    action: str | None = None
    schema: str = ROUTER_REPLAY_DATUM_SCHEMA

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "mode": self.mode,
            "source": self.source,
            "fail_fast": self.fail_fast,
        }
        if self.action is not None:
            payload["action"] = self.action
        return payload

    @classmethod
    def r2_record(
        cls,
        *,
        fail_fast: bool = True,
    ) -> "RouterReplayMetadata":
        """Build datum-level R2 RECORD metadata for ``forward`` recompute calls."""

        return cls(
            mode=ROUTER_REPLAY_MODE_R2,
            source=ROUTER_REPLAY_SOURCE_RECOMPUTE,
            fail_fast=fail_fast,
            action="RECORD",
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


def materialize_router_replay_indices(
    envelope: Mapping[str, Any],
    *,
    trusted: bool = False,
) -> Any:
    """Materialize a router replay indices envelope into inspectable lists.

    Inline envelopes return their ``value`` directly. Sharded envelopes return
    per-sample values when ``sample_indices`` metadata is available, otherwise a
    concatenated token-level list.
    """
    if not _trusted_router_replay_debug(trusted):
        return _router_replay_ref_summary(envelope)

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
            sample_rows = _split_shard_rows_by_sample(
                shard_value=shard_value,
                sample_indices=[int(item) for item in sample_indices],
                tokens_per_sample=tokens_per_sample,
                row_layout=str(shard.get("row_layout") or ""),
                microbatch_sizes=shard.get("microbatch_sizes"),
            )
            for sample_idx, rows in sample_rows.items():
                sample_idx = int(sample_idx)
                sample_pp_key = (sample_idx, pp_rank)
                if sample_pp_key in seen_sample_pp:
                    # TP shards carry duplicate user-visible token/layer rows for
                    # inspection purposes; one TP copy per PP stage is enough.
                    continue
                seen_sample_pp.add(sample_pp_key)
                per_sample_parts.setdefault(sample_idx, {})[pp_rank] = rows
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


def _split_shard_rows_by_sample(
    *,
    shard_value: list[Any],
    sample_indices: list[int],
    tokens_per_sample: int,
    row_layout: str,
    microbatch_sizes: Any,
) -> dict[int, list[Any]]:
    if row_layout != "seq_major_microbatch":
        return {
            int(sample_idx): shard_value[
                offset * tokens_per_sample : (offset + 1) * tokens_per_sample
            ]
            for offset, sample_idx in enumerate(sample_indices)
            if (offset + 1) * tokens_per_sample <= len(shard_value)
        }

    if isinstance(microbatch_sizes, list) and microbatch_sizes:
        mb_sizes = [int(item) for item in microbatch_sizes]
    else:
        mb_sizes = [len(sample_indices)]
    if sum(mb_sizes) != len(sample_indices):
        return {}

    per_sample: dict[int, list[Any]] = {}
    sample_offset = 0
    row_base = 0
    for mb_size in mb_sizes:
        mb_samples = sample_indices[sample_offset : sample_offset + mb_size]
        for pos, sample_idx in enumerate(mb_samples):
            rows: list[Any] = []
            for token_idx in range(tokens_per_sample):
                row_idx = row_base + token_idx * mb_size + pos
                if row_idx < len(shard_value):
                    rows.append(shard_value[row_idx])
            if rows:
                per_sample[int(sample_idx)] = rows
        sample_offset += mb_size
        row_base += mb_size * tokens_per_sample
    return per_sample


def router_replay_set_uri(model_id: str, replay_set_id: str) -> str:
    return f"weaver://{model_id}/router-replay/{replay_set_id}"


def router_replay_manifest_uri(model_id: str, replay_set_id: str) -> str:
    return f"{router_replay_set_uri(model_id, replay_set_id)}/manifest.json"


def router_replay_sample_uri(model_id: str, replay_set_id: str, sample_index: int) -> str:
    return f"{router_replay_set_uri(model_id, replay_set_id)}/samples/{int(sample_index)}"


def router_replay_shard_uri(
    model_id: str,
    replay_set_id: str,
    *,
    dp_rank: int,
    tp_rank: int,
    pp_rank: int,
) -> str:
    return (
        f"{router_replay_set_uri(model_id, replay_set_id)}/shards/"
        f"dp{int(dp_rank)}-tp{int(tp_rank)}-pp{int(pp_rank)}.pt"
    )


def materialize_router_replay_index(
    uri_or_datum: Any,
    *,
    trusted: bool = False,
) -> Any:
    """Materialize router replay index content for explicit user inspection.

    Normal training flows should pass router replay refs opaquely. This helper is
    for debugging.  By default it returns only a ref summary; pass
    ``trusted=True`` or set ``WEAVER_ROUTER_REPLAY_TRUSTED_MATERIALIZE=1`` from a
    trainer/control-plane process to resolve manifests or shard tensors.
    """

    allow_materialize = _trusted_router_replay_debug(trusted)
    if isinstance(uri_or_datum, Mapping):
        if not allow_materialize:
            return _router_replay_ref_summary(uri_or_datum)
        return _materialize_router_replay_mapping(uri_or_datum)
    if not isinstance(uri_or_datum, str):
        raise TypeError("Expected a weaver:// URI, datum mapping, or indices envelope.")
    if not allow_materialize:
        return _router_replay_uri_summary(uri_or_datum)
    return _materialize_router_replay_uri(uri_or_datum)


def _trusted_router_replay_debug(trusted: bool) -> bool:
    return trusted or os.environ.get("WEAVER_ROUTER_REPLAY_TRUSTED_MATERIALIZE") in (
        "1",
        "true",
        "TRUE",
        "yes",
        "on",
    )


def _router_replay_ref_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    replay = value.get("router_replay")
    if replay is None and isinstance(value.get("metadata"), Mapping):
        replay = value["metadata"].get("router_replay")
    if isinstance(replay, Mapping):
        value = replay
    return {
        "kind": "router_replay_ref",
        "materialized": False,
        "mode": value.get("mode"),
        "source": value.get("source"),
        "action": value.get("action"),
        "sample_ref": value.get("sample_ref"),
        "index_set_uri": value.get("index_set_uri") or value.get("uri"),
        "manifest_uri": value.get("manifest_uri"),
        "has_internal_indices": isinstance(value.get("indices"), Mapping),
        "has_internal_shards": isinstance(value.get("shards"), (list, tuple)),
    }


def _router_replay_uri_summary(uri: str) -> dict[str, Any]:
    if not uri.startswith("weaver://"):
        raise ValueError(f"Unsupported router replay URI: {uri!r}")
    kind = "index_set"
    if uri.endswith("/manifest.json"):
        kind = "manifest"
    elif "/samples/" in uri:
        kind = "sample"
    elif "/shards/" in uri:
        kind = "shard"
    return {
        "kind": f"router_replay_{kind}_ref",
        "materialized": False,
        "uri": uri,
    }


def _materialize_router_replay_mapping(uri_or_datum: Mapping[str, Any]) -> Any:
    replay = uri_or_datum.get("router_replay")
    if replay is None and isinstance(uri_or_datum.get("metadata"), Mapping):
        replay = uri_or_datum["metadata"].get("router_replay")
    if isinstance(replay, Mapping):
        result = _materialize_router_replay_payload(replay)
        if result is not None:
            return result
    if uri_or_datum.get("value") is not None or uri_or_datum.get("shards") is not None:
        return materialize_router_replay_indices(uri_or_datum, trusted=True)
    raise ValueError("Mapping does not contain router replay metadata or indices.")


def _materialize_router_replay_payload(replay: Mapping[str, Any]) -> Any | None:
    indices = replay.get("indices")
    if isinstance(indices, Mapping) and indices.get("value") is not None:
        raise ValueError("Inline router_replay.indices.value payloads are no longer supported.")
    if replay.get("sample_index") is not None or replay.get("index_uri") is not None:
        raise ValueError("sample_index/index_uri are no longer supported; use sample_ref.")
    sample_ref = replay.get("sample_ref")
    manifest_uri = replay.get("manifest_uri")
    if isinstance(manifest_uri, str) and isinstance(sample_ref, str):
        _, sample = sample_ref.rsplit("/samples/", 1)
        manifest = _load_manifest_from_uri(manifest_uri)
        return _materialize_sample_from_manifest(manifest, int(sample.strip("/")))
    return None


def _materialize_router_replay_uri(uri: str) -> Any:
    if not uri.startswith("weaver://"):
        raise ValueError(f"Unsupported router replay URI: {uri!r}")
    if uri.endswith("/manifest.json"):
        return _load_manifest_from_uri(uri)
    if "/samples/" in uri:
        prefix, sample = uri.rsplit("/samples/", 1)
        manifest = _load_manifest_from_uri(f"{prefix}/manifest.json")
        return _materialize_sample_from_manifest(manifest, int(sample.strip("/")))
    if "/shards/" in uri:
        path = _resolve_weaver_uri_path(uri)
        if path.suffix == ".pt":
            return materialize_payload_ref({"storage": "gpfs", "format": "torch.save", "uri": uri})
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported router replay URI kind: {uri!r}")


def _load_manifest_from_uri(uri: str) -> Mapping[str, Any]:
    path = _resolve_weaver_uri_path(uri)
    if not path.exists():
        raise FileNotFoundError(f"Router replay manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_weaver_uri_path(uri: str) -> Path:
    relative = uri.removeprefix("weaver://").lstrip("/")
    if not relative:
        raise ValueError("weaver:// URI must include a relative component.")
    for env_key in ("WEAVER_PAYLOAD_REF_ROOT", "WEAVER_ROUTER_REPLAY_REF_ROOT"):
        root = os.environ.get(env_key)
        if root:
            return Path(root) / relative
    return Path(relative)


def _materialize_sample_from_manifest(
    manifest: Mapping[str, Any],
    sample_index: int,
) -> list[Any]:
    raw_shards = manifest.get("shards", [])
    shards = raw_shards if isinstance(raw_shards, list) else []
    envelope = {
        "format": manifest.get("format", ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK),
        "token_alignment": manifest.get(
            "token_alignment", ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED
        ),
        "num_layers": manifest.get("num_layers"),
        "topk": manifest.get("topk"),
        "transport": manifest.get("transport"),
        "shards": shards,
    }
    sample_order = sorted(
        {
            int(sample_idx)
            for shard in shards
            if isinstance(shard, Mapping)
            for sample_idx in _sample_indices_from_shard(shard)
        }
    )
    if sample_index in sample_order:
        values = materialize_router_replay_indices(envelope, trusted=True)
        if values:
            return values[sample_order.index(sample_index)]
    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        sample_indices = shard.get("sample_indices")
        if not isinstance(sample_indices, list) or sample_index not in sample_indices:
            continue
        values = materialize_router_replay_indices({"shards": [shard]}, trusted=True)
        sample_order = sorted({int(item) for item in sample_indices})
        if values and sample_index in sample_order:
            return values[sample_order.index(sample_index)]
    return []


def _sample_indices_from_shard(shard: Mapping[str, Any]) -> list[Any]:
    sample_indices = shard.get("sample_indices")
    return sample_indices if isinstance(sample_indices, list) else []
