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

"""Tests for sequence ID management across clients."""

from weaver.service_client import ServiceClient
from weaver.training_client import TrainingClient


def test_training_client_seq_ids_are_monotonic():
    training = TrainingClient(
        service=ServiceClient(),
        model_id="model-123",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        session_id="session-1",
    )

    assert [training._next_seq() for _ in range(3)] == [1, 2, 3]


def test_service_client_seq_counters_are_monotonic():
    service = ServiceClient()

    model_seq_values = [service._next_model_seq() for _ in range(3)]
    sampling_seq_values = [service._next_sampling_seq() for _ in range(3)]

    assert model_seq_values == [1, 2, 3]
    assert sampling_seq_values == [1, 2, 3]
