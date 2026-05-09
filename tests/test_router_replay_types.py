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

from weaver.types.router_replay import (
    ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK,
    ROUTER_REPLAY_MODE_R2,
    ROUTER_REPLAY_SOURCE_RECOMPUTE,
    ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED,
    RouterReplayIndices,
    RouterReplayMetadata,
    RouterReplayModelConfig,
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


def test_router_replay_metadata_to_payload():
    metadata = RouterReplayMetadata(
        mode=ROUTER_REPLAY_MODE_R2,
        source=ROUTER_REPLAY_SOURCE_RECOMPUTE,
        indices=RouterReplayIndices(value=[[[7, 8]]], num_layers=1, topk=2),
    )

    assert metadata.to_payload() == {
        "mode": "R2",
        "source": "recompute",
        "indices": {
            "format": ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK,
            "value": [[[7, 8]]],
            "token_alignment": ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED,
            "num_layers": 1,
            "topk": 2,
        },
        "fail_fast": True,
    }


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
