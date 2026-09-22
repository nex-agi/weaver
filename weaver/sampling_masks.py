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

"""Validate opaque sampling support handles without loading candidate tensors."""
from __future__ import annotations

import hashlib
import struct
from typing import Any


def validate_mask_sequence(sequence: dict[str, Any]) -> None:
    ref = sequence.get("sampling_mask_ref")
    if not isinstance(ref, dict) or ref.get("schema") != "weaver.sampling_mask.v1":
        raise ValueError("Sampler did not return the requested sampling mask reference")
    if "sampling_masks" in sequence:
        raise ValueError("Sampler must not return both inline and referenced sampling masks")
    tokens = sequence.get("tokens")
    if not isinstance(tokens, list) or any(
        not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < 2**31 for t in tokens
    ):
        raise ValueError("Invalid sampling mask response tokens")
    if ref.get("token_count") != len(tokens) or ref.get("empty_row") != "full-vocabulary":
        raise ValueError("Sampling mask reference dimensions/semantics differ from the response")
    if not ref.get("weight_version") or ref["weight_version"] != sequence.get("weight_version"):
        raise ValueError("Sampling mask reference weight version differs from the response")
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<i", token))
    if digest.hexdigest() != ref.get("tokens_sha256"):
        raise ValueError("Sampling mask reference tokens differ from the response")


def validate_mask_result(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Sampler result must be an object")
    result = payload.get("result", payload)
    sequences = result.get("sequences") if isinstance(result, dict) else None
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("Sampling mask result is missing sequences")
    for sequence in sequences:
        if not isinstance(sequence, dict):
            raise ValueError("Sampler sequence must be an object")
        validate_mask_sequence(sequence)
