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

"""Tests for Router Replay contract helpers."""

import json

from weaver.types.payload_ref import materialize_payload_ref
from weaver.types.router_replay import (
    ROUTER_REPLAY_DATUM_SCHEMA,
    ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK,
    ROUTER_REPLAY_MODE_R2,
    ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED,
    RouterReplayIndices,
    RouterReplayMetadata,
    RouterReplayModelConfig,
    materialize_router_replay_index,
    materialize_router_replay_indices,
)


def test_router_replay_indices_to_payload():
    indices = RouterReplayIndices(
        value=[[[1, 2], [3, 4]]],
        num_layers=2,
        topk=2,
    )

    assert indices.to_payload() == {
        "format": ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK,
        "value": [[[1, 2], [3, 4]]],
        "token_alignment": ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED,
        "num_layers": 2,
        "topk": 2,
    }


def test_router_replay_r2_record_has_no_indices():
    metadata = RouterReplayMetadata.r2_record()

    assert metadata.to_payload() == {
        "schema": ROUTER_REPLAY_DATUM_SCHEMA,
        "mode": "R2",
        "source": "recompute",
        "fail_fast": True,
        "action": "RECORD",
    }


def test_router_replay_indices_support_ref_shards():
    indices = RouterReplayIndices(
        num_layers=2,
        topk=2,
        shards=[
            {
                "sample_indices": [0],
                "local_tokens_per_sample": 1,
                "value_ref": {
                    "storage": "gpfs",
                    "format": "torch.save",
                    "relative_path": "op/dp0.pt",
                },
            }
        ],
        transport="gpfs_torch_save",
    )

    payload = indices.to_payload()
    assert "value" not in payload
    assert payload["transport"] == "gpfs_torch_save"
    assert payload["shards"][0]["value_ref"]["relative_path"] == "op/dp0.pt"


def test_materialize_router_replay_indices_returns_inline_value():
    value = [[[1, 2]]]

    summary = materialize_router_replay_indices({"value": value})
    assert summary["kind"] == "router_replay_ref"
    assert summary["materialized"] is False
    assert materialize_router_replay_indices({"value": value}, trusted=True) == value


def test_materialize_router_replay_indices_combines_pipeline_shards():
    envelope = {
        "transport": "gpfs_torch_save",
        "format": "token_layer_topk",
        "shards": [
            {
                "dp_rank": 0,
                "tp_rank": 0,
                "pp_rank": 0,
                "sample_indices": [0],
                "local_tokens_per_sample": 2,
                "value": [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ],
            },
            {
                "dp_rank": 0,
                "tp_rank": 0,
                "pp_rank": 1,
                "sample_indices": [0],
                "local_tokens_per_sample": 2,
                "value": [
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                ],
            },
            {
                "dp_rank": 0,
                "tp_rank": 1,
                "pp_rank": 0,
                "sample_indices": [0],
                "local_tokens_per_sample": 2,
                "value": [
                    [[101, 102], [103, 104]],
                    [[105, 106], [107, 108]],
                ],
            },
        ],
    }

    assert materialize_router_replay_indices(envelope, trusted=True) == [
        [
            [[1, 2], [3, 4], [9, 10], [11, 12]],
            [[5, 6], [7, 8], [13, 14], [15, 16]],
        ]
    ]


def test_materialize_router_replay_indices_handles_seq_major_microbatch():
    envelope = {
        "format": "token_layer_topk",
        "shards": [
            {
                "dp_rank": 0,
                "tp_rank": 0,
                "pp_rank": 0,
                "sample_indices": [0, 2, 4],
                "local_tokens_per_sample": 3,
                "microbatch_sizes": [3],
                "row_layout": "seq_major_microbatch",
                "value": [
                    ["s0-t0"],
                    ["s2-t0"],
                    ["s4-t0"],
                    ["s0-t1"],
                    ["s2-t1"],
                    ["s4-t1"],
                    ["s0-t2"],
                    ["s2-t2"],
                    ["s4-t2"],
                ],
            }
        ],
    }

    assert materialize_router_replay_indices(envelope, trusted=True) == [
        [["s0-t0"], ["s0-t1"], ["s0-t2"]],
        [["s2-t0"], ["s2-t1"], ["s2-t2"]],
        [["s4-t0"], ["s4-t1"], ["s4-t2"]],
    ]


def test_materialize_router_replay_index_selects_sample_from_manifest(tmp_path, monkeypatch):
    manifest = {
        "format": "token_layer_topk",
        "token_alignment": "target_aligned",
        "num_layers": 2,
        "topk": 1,
        "shards": [
            {
                "pp_rank": 0,
                "sample_indices": [3, 1],
                "local_tokens_per_sample": 1,
                "value": [[["s3-pp0"]], [["s1-pp0"]]],
            },
            {
                "pp_rank": 1,
                "sample_indices": [3, 1],
                "local_tokens_per_sample": 1,
                "value": [[["s3-pp1"]], [["s1-pp1"]]],
            },
        ],
    }
    manifest_path = tmp_path / "model-a" / "router-replay" / "set-1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("WEAVER_PAYLOAD_REF_ROOT", str(tmp_path))

    summary = materialize_router_replay_index("weaver://model-a/router-replay/set-1/samples/3")
    assert summary["materialized"] is False
    assert summary["kind"] == "router_replay_sample_ref"

    assert materialize_router_replay_index(
        "weaver://model-a/router-replay/set-1/samples/3",
        trusted=True,
    ) == [[["s3-pp0"], ["s3-pp1"]]]


def test_materialize_payload_ref_resolves_relative_path_with_root(tmp_path, monkeypatch):
    ref_root = tmp_path / "payload_refs"
    ref_root.mkdir()
    payload_path = ref_root / "op-1" / "payload.json"
    payload_path.parent.mkdir()
    payload_path.write_text('{"indices": [1, 2, 3]}')
    monkeypatch.setenv("WEAVER_PAYLOAD_REF_ROOT", str(ref_root))

    assert materialize_payload_ref(
        {
            "storage": "gpfs",
            "format": "json",
            "relative_path": "op-1/payload.json",
        },
        field="indices",
    ) == [1, 2, 3]


def test_router_replay_model_config_to_payload():
    config = RouterReplayModelConfig(
        enabled=True,
        mode=ROUTER_REPLAY_MODE_R2,
        num_layers=64,
        topk=8,
    )

    assert config.to_payload() == {
        "enabled": True,
        "mode": "R2",
        "num_layers": 64,
        "topk": 8,
        "fail_fast": True,
    }


def test_router_replay_model_config_to_payload_preserves_fail_fast_false():
    config = RouterReplayModelConfig(enabled=False, fail_fast=False)

    assert config.to_payload() == {"enabled": False, "fail_fast": False}
