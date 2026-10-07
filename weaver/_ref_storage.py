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

"""Public sampling storage selection and opaque result validation."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from math import isfinite
from time import time
from typing import Any
from uuid import UUID

from . import _sampling_utils as _su
from ._checkpoint_storage import validate_storage_backend

_MANAGED = "weaver-artifact/v1"
_FORMATS = {
    "sampling_mask_ref": "safetensors-csr-chunks",
    "sampler_distribution_ref": "safetensors-chunks",
    "moe_topk_indices_ref": "safetensors",
}


def prepare_ref_storage(
    body: dict[str, Any], backend: str | None, model_id: str | None, model_path: str | None
) -> tuple[str | None, str | None]:
    validate_storage_backend(backend)
    kinds = []
    if body.get("sampling_mask_transport") == "ref":
        kinds.append("sampling_mask_ref")
    if body.get("score_centering", {}).get("transport") == "ref":
        kinds.append("sampler_distribution_ref")
    if body.get("return_moe_topk_indices"):
        kinds.append("moe_topk_indices_ref")
    kind = kinds[0] if len(kinds) == 1 else None
    if backend is not None:
        body["ref_storage_backend"] = backend
    if backend != "artifact":
        return kind, None
    if kind is None:
        raise ValueError("Managed sampling requires one supported ref output")
    scope = model_id or _su.parse_model_id_from_weaver_path(model_path)
    try:
        if not isinstance(scope, str) or str(UUID(scope)) != scope or UUID(scope).int == 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Managed sampling requires a canonical model UUID") from exc
    return kind, scope


def verify_ref_storage_capability(capabilities: Any, kind: str) -> None:
    if (
        not isinstance(capabilities, dict)
        or not isinstance(capabilities.get("protocol_version"), int)
        or isinstance(capabilities.get("protocol_version"), bool)
        or capabilities["protocol_version"] != 1
        or capabilities.get("default_backend") != "gpfs"
    ):
        raise RuntimeError("Server does not currently allow this managed sampling ref")
    if (
        not isinstance(capabilities.get("sampling_ref_backends"), list)
        or "artifact" not in capabilities["sampling_ref_backends"]
        or not isinstance(capabilities.get("managed_sampling_ref_kinds"), list)
        or kind not in capabilities["managed_sampling_ref_kinds"]
    ):
        raise RuntimeError("Server does not currently allow this managed sampling ref")


def _validate_backend(
    ref: dict[str, Any], kind: str, backend: str | None, model_id: str | None
) -> None:
    if backend != "artifact":
        # Historical GPFS envelopes may omit storage; their existing semantic
        # validators remain authoritative. An explicit locator cannot masquerade
        # as a historical envelope.
        if ref.get("storage", "gpfs") != "gpfs" or "artifact_ref" in ref:
            raise ValueError("Sampler returned a ref from a different storage backend")
        return
    if ref.get("storage") != _MANAGED or ref.get("model_id") != model_id:
        raise ValueError("Sampler returned a ref from a different backend or model")
    if {"path", "uri", "endpoint", "token", "cache_domain"} & ref.keys():
        raise ValueError("Managed ref cannot select a local path or workload configuration")
    locator = ref.get("artifact_ref")
    prefix = f"artifact://refs/{model_id}/"
    try:
        if not isinstance(locator, str) or not locator.startswith(prefix):
            raise ValueError
        identity = locator[len(prefix) :]
        artifact_id, generation = identity.split("@")
        if (
            str(UUID(artifact_id)) != artifact_id
            or str(UUID(generation)) != generation
            or UUID(artifact_id).int == 0
            or UUID(generation).int == 0
        ):
            raise ValueError
    except ValueError as exc:
        raise ValueError("Managed ref requires a fixed locator in the model namespace") from exc
    path = ref.get("relative_path")
    if (
        ref.get("format") != _FORMATS[kind]
        or not isinstance(path, str)
        or not 0 < len(path) <= 4096
    ):
        raise ValueError("Invalid managed ref payload path or encoding")
    if (
        "\\" in path
        or any(ord(c) < 32 for c in path)
        or any(part in ("", ".", "..") for part in path.split("/"))
        or path == ".weaver-ref-envelope.json"
    ):
        raise ValueError("Invalid managed ref payload path or encoding")


def sample_result_validator(
    body: dict[str, Any], backend: str | None, model_id: str | None
) -> Callable[[Any], None] | None:
    sc = body.get("score_centering", {})
    mask_ref = body.get("sampling_mask_transport") == "ref"
    distribution_ref = sc.get("transport") == "ref"
    router_ref = bool(body.get("return_moe_topk_indices"))
    if not sc and not mask_ref and not router_ref:
        return None

    def validate(payload: Any) -> None:
        if sc:
            from .score_centering import validate_sampler_result

            validate_sampler_result(payload, sc["head_size"], sc.get("transport", "inline"))
        if mask_ref:
            from .sampling_masks import validate_mask_result

            validate_mask_result(payload)
        if mask_ref or distribution_ref:
            result = payload.get("result", payload)
            kind = "sampling_mask_ref" if mask_ref else "sampler_distribution_ref"
            for sequence in result["sequences"]:
                _validate_backend(sequence[kind], kind, backend, model_id)
        if router_ref:
            result = payload.get("result", payload) if isinstance(payload, dict) else {}
            if not isinstance(result, dict):
                if backend != "artifact":
                    return
                raise ValueError("Sampler omitted router sequences")
            sequences = result.get("sequences")
            if not isinstance(sequences, list):
                if backend != "artifact":
                    return  # Keep historical normalization authoritative.
                raise ValueError("Sampler omitted router sequences")
            if backend == "artifact" and not sequences:
                raise ValueError("Sampler omitted router sequences")
            for sequence in sequences:
                if not isinstance(sequence, dict):
                    raise ValueError("Sampler returned an invalid router sequence")
                ref = sequence.get("moe_topk_indices_ref")
                if backend != "artifact" and ref is None:
                    continue  # Historical results may carry inline routing.
                if not isinstance(ref, dict):
                    raise ValueError("Sampler omitted its router ref")
                _validate_backend(ref, "moe_topk_indices_ref", backend, model_id)
                if backend == "artifact":
                    _validate_managed_router(ref, sequence, body)

    return validate


def _validate_managed_router(
    ref: dict[str, Any], sequence: dict[str, Any], body: dict[str, Any]
) -> None:
    tokens, shape = sequence.get("tokens"), ref.get("shape")
    prompt_count = sum(len(chunk.get("tokens", [])) for chunk in body["prompt"]["chunks"])
    if ref.get("schema") != "weaver.router_replay.r2_recorded_indices.v1" or ref.get(
        "dtype"
    ) not in ("int16", "int32"):
        raise ValueError("Managed router ref encoding is invalid")
    if (
        not isinstance(tokens, list)
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(not _integer_in_range(value, 0, 2147483647) for value in shape)
    ):
        raise ValueError("Managed router ref dimensions are invalid")
    if shape[0] != max(0, prompt_count + len(tokens) - 1) or shape[1] == 0 or shape[2] == 0:
        raise ValueError("Managed router ref shape disagrees with the generated sequence")
    if (
        not isinstance(sequence.get("weight_version"), str)
        or not sequence["weight_version"]
        or ref.get("weight_version") != sequence["weight_version"]
    ):
        raise ValueError("Managed router ref behavior version disagrees with the sequence")
    if (
        not _integer_in_range(ref.get("size_bytes"), 1, 9007199254740991)
        or "moe_topk_indices" in sequence
        or sequence.get("moe_topk_indices_error") is not None
    ):
        raise ValueError("Managed router ref disagrees with the generated sequence")
    expiry = ref.get("expires_at")
    if isinstance(expiry, str):
        try:
            if not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", expiry
            ):
                raise ValueError
            expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise ValueError("Managed router ref expiration is invalid") from exc
    if not isinstance(expiry, int | float) or isinstance(expiry, bool):
        raise ValueError("Managed router ref expiration is invalid")
    if not isfinite(expiry) or not expiry > time():
        raise ValueError("Managed router ref is expired")


def _integer_in_range(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
