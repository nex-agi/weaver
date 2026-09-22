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

"""Validation of rollout-time distributions used by score centering."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

SCHEMA = "behavior/unfiltered/v1"
REF_SCHEMA = "weaver.sampler_distribution.v1"
REF_FIELD = "sampler_distribution_ref"
SAMPLER_FIELDS = (
    "sampler_logprobs",
    "sampler_topk_ids",
    "sampler_topk_logprobs",
    "sampler_topk_mask",
    "sampler_distribution",
)


def validate_sampler_sequence(
    sequence: dict[str, Any], k: int, transport: str | None = None
) -> None:
    """Reject missing/truncated distributions rather than silently disabling SC."""
    if not isinstance(sequence, dict):
        raise ValueError("Sampler sequence must be an object")
    if REF_FIELD in sequence:
        if transport == "inline":
            raise ValueError("Expected inline sampler statistics, received a reference")
        validate_sampler_ref_sequence(sequence, k)
        return
    if transport == "ref":
        raise ValueError("Sampler did not return the requested distribution reference")
    if any(key not in sequence for key in SAMPLER_FIELDS):
        raise ValueError("Sampler does not support topk_output_logprobs: missing sampler fields")
    distribution = sequence["sampler_distribution"]
    if not isinstance(distribution, dict) or distribution.get("schema") != SCHEMA:
        raise ValueError("Unsupported sampler_distribution schema")
    tokens = sequence.get("tokens", [])
    for key in SAMPLER_FIELDS[:-1]:
        if not isinstance(sequence[key], list) or len(sequence[key]) != len(tokens):
            raise ValueError(f"{key} must align with generated tokens")
    for index, token in enumerate(tokens):
        ids = sequence["sampler_topk_ids"][index]
        probs = sequence["sampler_topk_logprobs"][index]
        mask = sequence["sampler_topk_mask"][index]
        if any(not isinstance(row, list) or len(row) != k for row in (ids, probs, mask)):
            raise ValueError("Sampler top-k width differs from the requested width")
        if any(type(v) is not int or v < 0 for v in ids) or len(set(ids)) != k:
            raise ValueError("Sampler top-k IDs must be unique nonnegative integers")
        if any(v != 1 for v in mask):
            raise ValueError("Unfiltered sampler head mask must be all ones")
        sampled = sequence["sampler_logprobs"][index]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v > 0
            for v in [sampled, *probs]
        ):
            raise ValueError("Sampler log probabilities must be finite and nonpositive")
        if sum(math.exp(v) for v in probs) > 1 + 1e-5:
            raise ValueError("Sampler head probability mass exceeds one")
        if token in ids and abs(probs[ids.index(token)] - sampled) > 1e-4:
            raise ValueError("Sampler head and sampled-token probabilities disagree")


def validate_sampler_result(payload: Any, k: int, transport: str | None = None) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Sampler result must be an object")
    result = payload.get("result", payload)
    sequences = result.get("sequences") if isinstance(result, dict) else None
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("Sampler result is missing sequences for topk_output_logprobs")
    for sequence in sequences:
        validate_sampler_sequence(sequence, k, transport)


def validate_sampler_ref_sequence(sequence: dict[str, Any], k: int) -> None:
    """Validate a returned handle without downloading its distribution tensors."""
    if any(
        name in sequence
        for name in ("sampler_topk_ids", "sampler_topk_logprobs", "sampler_topk_mask")
    ):
        raise ValueError("Sampler must return either a distribution reference or inline heads")
    ref = sequence.get(REF_FIELD)
    distribution = sequence.get("sampler_distribution")
    tokens = sequence.get("tokens")
    sampled = sequence.get("sampler_logprobs")
    if not isinstance(ref, dict) or ref.get("schema") != REF_SCHEMA:
        raise ValueError("Unsupported sampler distribution reference")
    if not isinstance(distribution, dict) or distribution.get("schema") != SCHEMA:
        raise ValueError("Unsupported sampler_distribution schema")
    if not isinstance(tokens, list) or any(
        type(t) is not int or not 0 <= t < 2**31 for t in tokens
    ):
        raise ValueError("Invalid referenced response tokens")
    if ref.get("top_k") != k or ref.get("token_count") != len(tokens):
        raise ValueError("Distribution reference dimensions differ from the response")
    if not ref.get("weight_version") or ref.get("weight_version") != distribution.get(
        "weight_version"
    ):
        raise ValueError("Distribution reference weight version differs from the response")
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<i", token))
    if digest.hexdigest() != ref.get("tokens_sha256"):
        raise ValueError("Distribution reference tokens differ from the response")
    if (
        not isinstance(sampled, list)
        or len(sampled) != len(tokens)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v > 0
            for v in sampled
        )
    ):
        raise ValueError("Sampler log probabilities must be finite and response-aligned")


def sampler_distribution_data(sequence: dict[str, Any]) -> dict[str, Any]:
    """Preserve either inline statistics or an opaque distribution handle."""
    fields = (
        ("sampler_logprobs", "sampler_distribution", REF_FIELD)
        if REF_FIELD in sequence
        else SAMPLER_FIELDS
    )
    return {name: sequence[name] for name in fields}
