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

"""LoRA configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.

    This follows the tinker design with simple, high-level controls.
    """

    rank: int
    """LoRA rank (dimension of low-rank matrices)"""

    seed: Optional[int] = None
    """Seed used for initialization of LoRA weights.

    Useful if you need deterministic or reproducible initialization of weights.
    """

    train_unembed: bool = True
    """Whether to add LoRA to the unembedding layer"""

    train_mlp: bool = True
    """Whether to add LoRAs to the MLP layers (including MoE layers)"""

    train_attn: bool = True
    """Whether to add LoRAs to the attention layers"""

    def to_payload(self) -> dict[str, object]:
        """Convert to API payload format."""
        payload: dict[str, object] = {
            "rank": self.rank,
            "train_unembed": self.train_unembed,
            "train_mlp": self.train_mlp,
            "train_attn": self.train_attn,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload
