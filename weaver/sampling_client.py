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

from typing import Any, Dict, List

from transformers.tokenization_utils import PreTrainedTokenizer

from . import _sampling_utils as _su
from ._utils import lookup_case_insensitive
from .operations import OperationHandle
from .service_client import ServiceClient
from .types import LogprobsParams, ModelInput, PauseMode, SamplingParams


class SamplingClient:
    def __init__(
        self,
        *,
        service: ServiceClient,
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

    def sample(
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
    ) -> OperationHandle | Dict[str, Any]:
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
        handle = self._service.enqueue_operation(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/samples",
            body,
        )
        if not wait:
            return handle
        raw_result = handle.result()
        return _su.normalize_sample_result(raw_result, self._ensure_tokenizer)  # type: ignore[return-value]

    def compute_logprobs(
        self,
        *,
        prompt: ModelInput,
        logprobs_params: LogprobsParams | None = None,
    ) -> List[float | None] | Dict[str, Any]:
        """Compute log-probabilities for the given prompt.

        The sampling-client contract is prompt-token-aligned: the returned list
        has length ``len(prompt_tokens)``, and index 0 is ``None`` because the
        first token has no previous token context to score. This differs from
        trainer-side ``forward_logprob`` tasks, which score explicit
        ``target_tokens`` and return one logprob per target token, with no
        leading placeholder.

        Args:
            prompt: The model input (tokens) to compute logprobs for.
            logprobs_params: Optional parameters (e.g. return_rollout_token_expert for MoE router replay).
                When None, uses defaults.

        Returns:
            When logprobs_params.return_rollout_token_expert=False: List[float|None] of per-token logprobs.
            When logprobs_params.return_rollout_token_expert=True: Dict with "logprobs" and
                "return_rollout_token_expert_data" (None if not MoE or not available).
        """
        params = logprobs_params or LogprobsParams()
        body = _su.build_logprobs_body(prompt, params)
        handle = self._service.enqueue_operation(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/logprobs",
            body,
        )
        payload = handle.result()
        logprobs = _su.normalize_prompt_logprobs(prompt, payload)
        if not params.return_rollout_token_expert:
            return logprobs
        result = _su.result_payload(payload)
        return {
            "logprobs": logprobs,
            "return_rollout_token_expert_data": result.get("return_rollout_token_expert_data"),
        }

    def pause_generation(self, *, mode: PauseMode | str = PauseMode.ABORT) -> Dict[str, Any]:
        """Pause in-flight generation on the engines behind this session.

        This is a stateless control primitive: it freezes the engine until
        :meth:`continue_generation` is called. With the default
        ``mode="abort"`` the waiting + running requests are aborted on the spot
        and their partial output is returned to the callers of :meth:`sample`
        (with ``stop_reason="abort"``) — this is the recovery shape used for
        partial/async rollout weight swaps
        (abort -> drain -> sync_weights -> continue).

        Args:
            mode: How to treat in-flight requests. One of :class:`PauseMode`
                (``abort`` / ``retract`` / ``in_place``) or its string value.

        Returns:
            The server response payload.

        Raises:
            ValueError: If ``mode`` is not a supported pause mode.
        """
        body = _su.build_pause_generation_body(mode)
        return self._service.http.post(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/pause-generation",
            json=body,
        )

    def continue_generation(self) -> Dict[str, Any]:
        """Resume generation after a :meth:`pause_generation` call.

        Returns:
            The server response payload.
        """
        return self._service.http.post(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/continue-generation",
        )

    def _normalize_sample_result(self, payload: Any) -> Any:
        return _su.normalize_sample_result(payload, self._ensure_tokenizer)

    def _ensure_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        # Use custom tokenizer_path if provided, otherwise use base_model
        if self.tokenizer_path:
            model_name_or_path = self.tokenizer_path
        else:
            model_name_or_path = self._ensure_base_model()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
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
