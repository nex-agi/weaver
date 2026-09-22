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

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from transformers import PreTrainedTokenizer

from ._utils import lookup_case_insensitive
from .score_centering import REF_FIELD, SAMPLER_FIELDS
from .types import LogprobsParams, ModelInput, SamplingParams
from .types.sampling_control import coerce_pause_mode
from .types.score_centering import ScoreCenteringConfig

if TYPE_CHECKING:
    from .types import PauseMode

TokenizerProvider = Callable[[], PreTrainedTokenizer]


def sampling_prompt_payload(prompt: ModelInput) -> Dict[str, Any]:
    """Serialize a token prompt."""

    if isinstance(prompt, ModelInput):
        return prompt.to_payload()
    raise TypeError("prompt must be ModelInput")


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
    score_centering: ScoreCenteringConfig | None = None,
    topk_output_logprobs: int = 0,
    sampler_distribution_transport: str = "inline",
    sampling_mask_transport: str = "inline",
) -> Dict[str, Any]:
    if sampling_mask_transport not in ("inline", "ref"):
        raise ValueError("sampling_mask_transport must be inline or ref")
    if sampling_mask_transport == "ref" and not return_sampling_mask:
        raise ValueError("sampling mask refs require return_sampling_mask=True")
    params = sampling_params or SamplingParams()
    if score_centering is not None:
        if topk_output_logprobs != 0 or sampler_distribution_transport != "inline":
            raise ValueError("score_centering cannot be combined with legacy SC options")
        if not isinstance(score_centering, dict) or set(score_centering) - {
            "head_size",
            "transport",
        }:
            raise ValueError("score_centering accepts only head_size and transport")
        head_size = score_centering.get("head_size")
        if (
            not isinstance(head_size, int)
            or isinstance(head_size, bool)
            or not 1 <= head_size <= 128
        ):
            raise ValueError("score_centering.head_size must be an integer in [1,128]")
        topk_output_logprobs = head_size
        sampler_distribution_transport = score_centering.get("transport", "inline")
    if sampler_distribution_transport not in ("inline", "ref"):
        raise ValueError("sampler_distribution_transport must be inline or ref")
    if sampler_distribution_transport == "ref" and not topk_output_logprobs:
        raise ValueError("distribution refs require topk_output_logprobs > 0")
    if (
        not isinstance(topk_output_logprobs, int)
        or isinstance(topk_output_logprobs, bool)
        or not 0 <= topk_output_logprobs <= 128
    ):
        raise ValueError("topk_output_logprobs must be an integer between 0 and 128")
    if topk_output_logprobs:
        if params.temperature != 1 or params.top_p != 1 or params.top_k != -1:
            raise ValueError("score_centering requires temperature=1, top_p=1, top_k=-1")
        if return_sampling_mask or return_old_logprob or return_moe_topk_indices:
            raise ValueError(
                "score_centering does not yet support sampling masks, old logprobs or router replay"
            )
        if include_prompt_logprobs or topk_prompt_logprobs:
            raise ValueError("score_centering is decode-only; prompt logprobs must be disabled")
    body: Dict[str, Any] = {
        "prompt": sampling_prompt_payload(prompt),
        "sampling_params": params.to_payload(),
        "num_samples": num_samples,
        "prompt_logprobs": include_prompt_logprobs,
        "topk_prompt_logprobs": topk_prompt_logprobs,
    }
    if topk_output_logprobs:
        body["score_centering"] = {
            "head_size": topk_output_logprobs,
            "transport": sampler_distribution_transport,
        }
    if return_sampling_mask:
        body["return_sampling_mask"] = True
        if sampling_mask_transport == "ref":
            body["sampling_mask_transport"] = "ref"
    if return_old_logprob:
        body["return_old_logprob"] = True
    if return_moe_topk_indices:
        body["return_moe_topk_indices"] = True
    return body


def build_logprobs_body(
    prompt: ModelInput, logprobs_params: LogprobsParams | None
) -> Dict[str, Any]:
    params = logprobs_params or LogprobsParams()
    return {"prompt": sampling_prompt_payload(prompt), **params.to_payload()}


def build_pause_generation_body(mode: "PauseMode | str") -> Dict[str, Any]:
    """Build the ``pause-generation`` request body, validating ``mode``."""
    return {"mode": coerce_pause_mode(mode)}


# Checkpoint URIs are ``weaver://<model-id>/checkpoints/<name>``; the model id is
# always the first path segment.
_WEAVER_PATH_RE = re.compile(r"^weaver://([^/]+)/")


def parse_model_id_from_weaver_path(path: str | None) -> str | None:
    """Extract the model id embedded in a ``weaver://`` checkpoint URI.

    A sampling client created from a checkpoint path alone still knows which
    model produced it, because the id is part of the URI. This recovers it so
    the client does not have to round-trip to the server for something it is
    already holding.

    Note this is *not* :func:`weaver._http.extract_model_id_from_path`, which
    parses API request paths (``/api/v1/models/<id>/...``), not storage URIs.

    Args:
        path: A ``weaver://<model-id>/checkpoints/<name>`` URI, or None.

    Returns:
        The model id, or None if ``path`` is absent or not a weaver URI.
    """
    if not path:
        return None
    match = _WEAVER_PATH_RE.match(path.strip())
    if not match:
        return None
    return match.group(1) or None


def ensure_full_ft_for_control(training_mode: Any, *, model_id: str | None) -> None:
    """Reject generation control against anything but a full fine-tuning model.

    pause/continue act on a whole inference engine, not on one sampling session.
    Full fine-tuning is the only mode with a dedicated engine; LoRA adapters are
    served from a single shared engine per base model, so pausing there would
    abort in-flight generation for every other tenant on that base model.

    Args:
        training_mode: The model's ``training_mode`` as reported by the server.
        model_id: The model the control primitive resolved to, for the message.

    Raises:
        ValueError: If ``training_mode`` is not ``"full_ft"``.
    """
    if training_mode == "full_ft":
        return
    raise ValueError(
        f"generation control is only supported for full fine-tuning models, but model "
        f"{model_id} has training_mode={training_mode!r}. LoRA adapters share one "
        f"inference engine per base model, so pausing it would abort in-flight "
        f"generation for unrelated tenants."
    )


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
            if "sampling_mask_ref" in raw:
                sequence["sampling_mask_ref"] = raw["sampling_mask_ref"]
            # The server offloads the large routing-index tensor to a GPFS
            # safetensors shard and returns an opaque ref instead of the inline
            # array; surface it verbatim so NexRL can attach it as a datum ref
            # without the indices ever crossing the wire. The ref is
            # authoritative: when it is present we ignore any inline
            # moe_topk_indices a malformed response may also carry, so the
            # consumer never receives both.
            if "moe_topk_indices_ref" in raw and raw["moe_topk_indices_ref"] is not None:
                sequence["moe_topk_indices_ref"] = raw["moe_topk_indices_ref"]
            elif "moe_topk_indices" in raw and raw["moe_topk_indices"] is not None:
                sequence["moe_topk_indices"] = raw["moe_topk_indices"]
            for field in (*SAMPLER_FIELDS, REF_FIELD):
                if field in raw:
                    sequence[field] = raw[field]
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
    expected = len(tokens)
    result = result_payload(payload)
    prompt_values = coerce_prompt_logprob_list(result.get("prompt_logprobs"), expected)
    if prompt_values is None:
        raise RuntimeError("trainer response missing prompt_logprobs")
    return prompt_values
