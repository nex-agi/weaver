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
from .deployment import Deployment
from .logprobs import LogprobsParams
from .lora_config import LoraConfig
from .managed_dataset import (
    WEAVER_REDACTED_TOKEN_ID,
    ManagedDatasetInfo,
    ManagedDatasetPage,
    SampleRef,
    SampleRefLength,
    SampleRefOutput,
)
from .model_input import ModelInput, ModelInputChunk
from .optim import AdamParams

# Router-replay / payload-ref types are an internal protocol shared by NexRL and
# weaver-trainer, not part of the SDK's public surface. They are re-exported here
# only so those privileged consumers keep working, and are deliberately excluded
# from __all__ below so a general SDK user never sees router-replay / ref
# machinery. New consumers should import them from the submodule directly
# (weaver.types.router_replay / weaver.types.payload_ref).
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
from .supported_model import SupportedModel, SupportedModelPrice, SupportedTrainingMode
from .tensor import TensorData
from .weights_artifact import WeightsArtifact

# Public API only. Router-replay / payload-ref symbols are imported above for
# internal consumers (NexRL, weaver-trainer) but intentionally omitted here so
# they are not part of the SDK's public surface.
__all__ = [
    "AdamParams",
    "Checkpoint",
    "Datum",
    "Deployment",
    "LoraConfig",
    "LogprobsParams",
    "ManagedDatasetInfo",
    "ManagedDatasetPage",
    "ModelInput",
    "ModelInputChunk",
    "PauseMode",
    "SamplingParams",
    "SampleRef",
    "SampleRefLength",
    "SampleRefOutput",
    "SupportedModel",
    "SupportedModelPrice",
    "SupportedTrainingMode",
    "TensorData",
    "WEAVER_REDACTED_TOKEN_ID",
    "WeightsArtifact",
    "coerce_pause_mode",
]
