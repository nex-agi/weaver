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

"""Sampling client for inference requests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers.tokenization_utils import PreTrainedTokenizer

from ._utils import lookup_case_insensitive
from .operations import OperationHandle
from .service_client import ServiceClient
from .types import ModelInput, SamplingParams


class SamplingClient:
    def __init__(
        self,
        *,
        service: ServiceClient,
        sampling_session_id: str,
        base_model: str | None = None,
        model_path: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._service = service
        self.sampling_session_id = sampling_session_id
        self.base_model = base_model
        self.model_path = model_path
        self.model_id = model_id
        self._tokenizer: PreTrainedTokenizer | None = None

    def sample(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams | None = None,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        wait: bool = True,
    ) -> OperationHandle | Dict[str, Any]:
        params = sampling_params or SamplingParams()
        body = {
            "prompt": prompt.to_payload(),
            "sampling_params": params.to_payload(),
            "num_samples": num_samples,
            "prompt_logprobs": include_prompt_logprobs,
            "topk_prompt_logprobs": topk_prompt_logprobs,
        }
        handle = self._service.enqueue_operation(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/samples",
            body,
        )
        if not wait:
            return handle
        raw_result = handle.result()
        return self._normalize_sample_result(raw_result)  # type: ignore[return-value]

    def compute_logprobs(self, *, prompt: ModelInput) -> List[float | None]:
        body = {"prompt": prompt.to_payload()}
        handle = self._service.enqueue_operation(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/logprobs",
            body,
        )
        payload = handle.result()
        return self._normalize_prompt_logprobs(prompt, payload)

    def _ensure_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        base_model = self._ensure_base_model()
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        return self._tokenizer

    def _ensure_base_model(self) -> str:
        if self.base_model:
            return self.base_model
        session = self._service.http.get(f"/api/v1/sampling-sessions/{self.sampling_session_id}")
        base_model = lookup_case_insensitive(session, "base_model") or lookup_case_insensitive(
            session, "base_model_name"
        )
        if not base_model:
            raise RuntimeError("sampling session is missing base_model")
        self.base_model = str(base_model)
        return self.base_model

    def _normalize_sample_result(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        if "sequences" in payload:
            return payload
        result = lookup_case_insensitive(payload, "result")
        if not isinstance(result, dict):
            return payload
        sequences = self._sequences_from_result(result)
        normalized = dict(payload)
        if sequences:
            normalized["sequences"] = sequences
        normalized["raw_result"] = result
        usage = result.get("usage")
        if usage:
            normalized["usage"] = usage
        return normalized

    def _sequences_from_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        existing_sequences = result.get("sequences")
        sequences: List[Dict[str, Any]] = []
        if isinstance(existing_sequences, list):
            tokenizer: Optional[PreTrainedTokenizer] = None
            for raw in existing_sequences:
                if not isinstance(raw, dict):
                    continue
                tokens = self._sanitize_tokens(raw.get("tokens"))
                text = raw.get("text")
                if text is None and tokens:
                    tokenizer = tokenizer or self._ensure_tokenizer()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                sequence = {
                    "tokens": tokens,
                    "text": text,
                    "stop_reason": raw.get("stop_reason"),
                }
                if "logprobs" in raw and isinstance(raw["logprobs"], list):
                    sequence["logprobs"] = raw["logprobs"]
                sequences.append(sequence)
            return [seq for seq in sequences if seq["tokens"]]

        choices = result.get("choices")
        if not isinstance(choices, list):
            return []
        tokenizer = self._ensure_tokenizer()
        for choice in choices:
            text = self._choice_text(choice)
            if text is None:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            stop_reason = choice.get("finish_reason") or choice.get("finishReason")
            sequences.append(
                {
                    "tokens": tokens,
                    "text": text,
                    "stop_reason": stop_reason,
                }
            )
        return sequences

    def _sanitize_tokens(self, value: Any) -> List[int]:
        if not isinstance(value, list):
            return []
        tokens: List[int] = []
        for item in value:
            try:
                tokens.append(int(item))
            except (TypeError, ValueError):
                continue
        return tokens

    def _choice_text(self, choice: Any) -> str | None:
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

    def _normalize_prompt_logprobs(self, prompt: ModelInput, payload: Any) -> List[float | None]:
        tokens = self._prompt_tokens(prompt)
        if not tokens:
            return []
        result_payload = self._result_payload(payload)
        prompt_values = self._coerce_prompt_logprob_list(
            result_payload.get("prompt_logprobs"), len(tokens)
        )
        if prompt_values is None:
            raise RuntimeError("trainer response missing prompt_logprobs")
        return prompt_values

    def _prompt_tokens(self, prompt: ModelInput) -> List[int]:
        try:
            return prompt.to_ints()
        except ValueError:
            tokens: List[int] = []
            for chunk in prompt.chunks:
                tokens.extend(int(token) for token in chunk.tokens)
            return tokens

    def _result_payload(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        result = lookup_case_insensitive(payload, "result")
        if isinstance(result, dict):
            return result
        return {}

    def _coerce_prompt_logprob_list(
        self,
        value: Any,
        expected: int,
    ) -> List[float | None] | None:
        if not isinstance(value, list):
            return None
        normalized: List[float | None] = []
        for item in value:
            if item is None:
                normalized.append(None)
                continue
            normalized.append(self._coerce_float(item))
        if expected > 0 and len(normalized) == expected:
            return normalized
        if normalized:
            return normalized
        return None

    def _coerce_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
