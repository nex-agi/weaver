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

"""Public type helpers re-exported for ergonomic imports."""

from .checkpoint import Checkpoint
from .datum import Datum
from .logprobs import LogprobsParams
from .lora_config import LoraConfig
from .model_input import ModelInput, ModelInputChunk
from .optim import AdamParams
from .payload_ref import PayloadRef, PayloadRefMaterializationError, materialize_payload_ref
from .router_replay import (
    ROUTER_REPLAY_DATUM_SCHEMA,
    ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK,
    ROUTER_REPLAY_INDEX_SET_SCHEMA,
    ROUTER_REPLAY_MODE_R2,
    ROUTER_REPLAY_MODE_R3,
    ROUTER_REPLAY_SOURCE_RECOMPUTE,
    ROUTER_REPLAY_SOURCE_ROLLOUT,
    ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED,
    RouterReplayIndices,
    RouterReplayMetadata,
    RouterReplayModelConfig,
    materialize_router_replay_index,
    materialize_router_replay_indices,
    router_replay_manifest_uri,
    router_replay_sample_uri,
    router_replay_set_uri,
    router_replay_shard_uri,
)
from .sampling import SamplingParams
from .sampling_control import PauseMode, coerce_pause_mode
from .tensor import TensorData

__all__ = [
    "AdamParams",
    "Checkpoint",
    "Datum",
    "LoraConfig",
    "LogprobsParams",
    "ModelInput",
    "ModelInputChunk",
    "PauseMode",
    "PayloadRef",
    "PayloadRefMaterializationError",
    "ROUTER_REPLAY_FORMAT_TOKEN_LAYER_TOPK",
    "ROUTER_REPLAY_MODE_R2",
    "ROUTER_REPLAY_MODE_R3",
    "ROUTER_REPLAY_SOURCE_RECOMPUTE",
    "ROUTER_REPLAY_SOURCE_ROLLOUT",
    "ROUTER_REPLAY_TOKEN_ALIGNMENT_TARGET_ALIGNED",
    "ROUTER_REPLAY_DATUM_SCHEMA",
    "ROUTER_REPLAY_INDEX_SET_SCHEMA",
    "RouterReplayIndices",
    "RouterReplayMetadata",
    "RouterReplayModelConfig",
    "SamplingParams",
    "TensorData",
    "coerce_pause_mode",
    "materialize_payload_ref",
    "materialize_router_replay_index",
    "materialize_router_replay_indices",
    "router_replay_manifest_uri",
    "router_replay_sample_uri",
    "router_replay_set_uri",
    "router_replay_shard_uri",
]
