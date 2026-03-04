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

"""Logprobs related helper types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LogprobsParams:
    """Parameters for compute_logprobs requests."""

    return_rollout_token_expert: bool = False

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.return_rollout_token_expert:
            payload["return_rollout_token_expert"] = True
        return payload
