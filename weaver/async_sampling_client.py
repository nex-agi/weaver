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

"""Asyncio-native sampling client for inference requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, overload

from transformers.tokenization_utils import PreTrainedTokenizer

from . import _sampling_utils as _su
from ._utils import lookup_case_insensitive
from .async_service_client import AsyncServiceClient
from .operations import AsyncOperationHandle
from .types import LogprobsParams, ModelInput, SamplingParams

if TYPE_CHECKING:
    from typing import Literal


class AsyncSamplingClient:
    def __init__(
        self,
        *,
        service: AsyncServiceClient,
        sampling_session_id: str,
        base_model: str | None = None,
        model_path: str | None = None,
        model_id: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        self._service = service
        self.sampling_session_id = sampling_session_id
        self.base_model = base_model
        self.model_path = model_path
        self.model_id = model_id
        self.tokenizer_path = tokenizer_path
        self._tokenizer: PreTrainedTokenizer | None = None

    @overload
    async def sample(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams | None = None,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_sampling_mask: bool = False,
        return_old_logprob: bool = False,
        return_moe_topk_indices: bool = False,
        wait: "Literal[True]" = True,
    ) -> Dict[str, Any]: ...

    @overload
    async def sample(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams | None = None,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_sampling_mask: bool = False,
        return_old_logprob: bool = False,
        return_moe_topk_indices: bool = False,
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def sample(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams | None = None,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_sampling_mask: bool = False,
        return_old_logprob: bool = False,
        return_moe_topk_indices: bool = False,
        wait: bool = True,
    ) -> AsyncOperationHandle | Dict[str, Any]:
        body = _su.build_sample_body(
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=num_samples,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            return_sampling_mask=return_sampling_mask,
            return_old_logprob=return_old_logprob,
            return_moe_topk_indices=return_moe_topk_indices,
        )
        handle = await self._service.enqueue_operation(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/samples",
            body,
        )
        if not wait:
            return handle
        raw_result = await handle.result()
        # Resolve the tokenizer source up front so result normalization (which
        # may need to decode token ids) stays synchronous.
        await self._ensure_tokenizer_source()
        return _su.normalize_sample_result(raw_result, self._ensure_tokenizer)  # type: ignore[return-value]

    async def compute_logprobs(
        self,
        *,
        prompt: ModelInput,
        logprobs_params: LogprobsParams | None = None,
    ) -> List[float | None] | Dict[str, Any]:
        """Compute log-probabilities for the given prompt.

        See :meth:`weaver.sampling_client.SamplingClient.compute_logprobs`.
        """
        params = logprobs_params or LogprobsParams()
        body = _su.build_logprobs_body(prompt, params)
        handle = await self._service.enqueue_operation(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/logprobs",
            body,
        )
        payload = await handle.result()
        logprobs = _su.normalize_prompt_logprobs(prompt, payload)
        if not params.return_rollout_token_expert:
            return logprobs
        result = _su.result_payload(payload)
        return {
            "logprobs": logprobs,
            "return_rollout_token_expert_data": result.get("return_rollout_token_expert_data"),
        }

    async def _ensure_tokenizer_source(self) -> None:
        """Make sure a tokenizer path or base_model is known (may fetch the session)."""
        if self.tokenizer_path or self.base_model:
            return
        session = await self._service.http.get(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}"
        )
        base_model = lookup_case_insensitive(session, "base_model") or lookup_case_insensitive(
            session, "base_model_name"
        )
        if not base_model:
            raise RuntimeError("sampling session is missing base_model")
        self.base_model = str(base_model)

    def _ensure_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        model_name_or_path = self.tokenizer_path or self.base_model
        if not model_name_or_path:
            raise RuntimeError(
                "tokenizer source unresolved; base_model or tokenizer_path is required"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        return self._tokenizer
