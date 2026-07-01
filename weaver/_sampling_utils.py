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

"""Sampling request/response helpers shared by the sync and async clients.

Result normalization needs a tokenizer only to decode token ids or encode
choice text, so these helpers take a ``get_tokenizer`` callable rather than
owning one. The callable is invoked lazily, only when decoding is required.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from transformers.tokenization_utils import PreTrainedTokenizer

from ._utils import lookup_case_insensitive
from .types import LogprobsParams, ModelInput, SamplingParams
from .types.sampling_control import PauseMode, coerce_pause_mode

TokenizerProvider = Callable[[], PreTrainedTokenizer]


def build_sample_body(
    *,
    prompt: ModelInput,
    sampling_params: SamplingParams | None,
    num_samples: int,
    include_prompt_logprobs: bool,
    topk_prompt_logprobs: int,
    return_sampling_mask: bool,
    return_old_logprob: bool,
    return_moe_topk_indices: bool,
) -> Dict[str, Any]:
    params = sampling_params or SamplingParams()
    body: Dict[str, Any] = {
        "prompt": prompt.to_payload(),
        "sampling_params": params.to_payload(),
        "num_samples": num_samples,
        "prompt_logprobs": include_prompt_logprobs,
        "topk_prompt_logprobs": topk_prompt_logprobs,
    }
    if return_sampling_mask:
        body["return_sampling_mask"] = True
    if return_old_logprob:
        body["return_old_logprob"] = True
    if return_moe_topk_indices:
        body["return_moe_topk_indices"] = True
    return body


def build_logprobs_body(
    prompt: ModelInput, logprobs_params: LogprobsParams | None
) -> Dict[str, Any]:
    params = logprobs_params or LogprobsParams()
    return {"prompt": prompt.to_payload(), **params.to_payload()}


def build_pause_generation_body(mode: "PauseMode | str") -> Dict[str, Any]:
    """Build the ``pause-generation`` request body, validating ``mode``."""
    return {"mode": coerce_pause_mode(mode)}


def sanitize_tokens(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    tokens: List[int] = []
    for item in value:
        try:
            tokens.append(int(item))
        except (TypeError, ValueError):
            continue
    return tokens


def choice_text(choice: Any) -> str | None:
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return None


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def result_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result = lookup_case_insensitive(payload, "result")
    if isinstance(result, dict):
        return result
    return {}


def coerce_prompt_logprob_list(value: Any, expected: int) -> List[float | None] | None:
    if not isinstance(value, list):
        return None
    normalized: List[float | None] = []
    for item in value:
        if item is None:
            normalized.append(None)
            continue
        normalized.append(coerce_float(item))
    if expected > 0 and len(normalized) == expected:
        return normalized
    if normalized:
        return normalized
    return None


def prompt_tokens(prompt: ModelInput) -> List[int]:
    try:
        return prompt.to_ints()
    except ValueError:
        tokens: List[int] = []
        for chunk in prompt.chunks:
            tokens.extend(int(token) for token in chunk.tokens)
        return tokens


def sequences_from_result(
    result: Dict[str, Any], get_tokenizer: TokenizerProvider
) -> List[Dict[str, Any]]:
    existing_sequences = result.get("sequences")
    sequences: List[Dict[str, Any]] = []
    if isinstance(existing_sequences, list):
        tokenizer: Optional[PreTrainedTokenizer] = None
        for raw in existing_sequences:
            if not isinstance(raw, dict):
                continue
            tokens = sanitize_tokens(raw.get("tokens"))
            text = raw.get("text")
            if text is None and tokens:
                tokenizer = tokenizer or get_tokenizer()
                text = tokenizer.decode(tokens, skip_special_tokens=False)
            sequence: Dict[str, Any] = {
                "tokens": tokens,
                "text": text,
                "stop_reason": raw.get("stop_reason"),
            }
            if "logprobs" in raw and isinstance(raw["logprobs"], list):
                sequence["logprobs"] = raw["logprobs"]
            if "old_logprobs" in raw and isinstance(raw["old_logprobs"], list):
                sequence["old_logprobs"] = raw["old_logprobs"]
            if "sampling_masks" in raw and raw["sampling_masks"] is not None:
                sequence["sampling_masks"] = raw["sampling_masks"]
            if "moe_topk_indices" in raw and raw["moe_topk_indices"] is not None:
                sequence["moe_topk_indices"] = raw["moe_topk_indices"]
            weight_version = lookup_case_insensitive(raw, "weight_version")
            if weight_version is not None:
                sequence["weight_version"] = weight_version
            sequences.append(sequence)
        # Keep aborted sequences even when empty: a pause(mode="abort") may cut a
        # request before any token is emitted, and NexRL still needs the partial
        # (stop_reason="abort") signal rather than a silently dropped sequence.
        return [seq for seq in sequences if seq["tokens"] or seq.get("stop_reason") == "abort"]

    choices = result.get("choices")
    if not isinstance(choices, list):
        return []
    tokenizer = get_tokenizer()
    for choice in choices:
        text = choice_text(choice)
        if text is None:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        stop_reason = choice.get("finish_reason") or choice.get("finishReason")
        sequences.append({"tokens": tokens, "text": text, "stop_reason": stop_reason})
    return sequences


def normalize_sample_result(payload: Any, get_tokenizer: TokenizerProvider) -> Any:
    if not isinstance(payload, dict):
        return payload
    if "sequences" in payload:
        return payload
    result = lookup_case_insensitive(payload, "result")
    if not isinstance(result, dict):
        return payload
    sequences = sequences_from_result(result, get_tokenizer)
    normalized = dict(payload)
    if sequences:
        normalized["sequences"] = sequences
    normalized["raw_result"] = result
    usage = result.get("usage")
    if usage:
        normalized["usage"] = usage
    # Surface the engine weight version so NexRL can compute staleness /
    # off-policy masks without digging into raw_result.
    weight_version = lookup_case_insensitive(result, "weight_version")
    if weight_version is not None:
        normalized["weight_version"] = weight_version
    return normalized


def normalize_prompt_logprobs(prompt: ModelInput, payload: Any) -> List[float | None]:
    tokens = prompt_tokens(prompt)
    if not tokens:
        return []
    result = result_payload(payload)
    prompt_values = coerce_prompt_logprob_list(result.get("prompt_logprobs"), len(tokens))
    if prompt_values is None:
        raise RuntimeError("trainer response missing prompt_logprobs")
    return prompt_values
