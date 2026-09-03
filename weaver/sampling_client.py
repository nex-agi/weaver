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

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from transformers.tokenization_utils import PreTrainedTokenizer

from . import _sampling_utils as _su
from ._utils import lookup_case_insensitive
from .operations import OperationHandle
from .service_client import ServiceClient
from .types import LogprobsParams, ModelInput, PauseMode, SampleRef, SamplingParams


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
        # Cached result of the generation-control eligibility check; a model's
        # training mode is fixed at creation, so one confirmation is enough.
        self._is_full_ft = False
        # Managed dataset visibility is immutable for a (name, version), so it
        # is safe to cache for client-side UX. The server remains authoritative.
        self._managed_dataset_visibility_cache: Dict[tuple[str, str], str] = {}

    def _ensure_sample_ref_is_public(self, prompt: ModelInput | SampleRef) -> None:
        """Fail before enqueue when sampling is attempted with protected data."""

        if not isinstance(prompt, SampleRef):
            return
        source = (prompt.dataset, prompt.version)
        visibility = self._managed_dataset_visibility_cache.get(source)
        if visibility is None:
            info = self._service.datasets.get(name=source[0], version=source[1])
            visibility = info.content_visibility
            self._managed_dataset_visibility_cache[source] = visibility
        if visibility != "public":
            raise ValueError(
                f"managed dataset {source[0]!r} version {source[1]!r} is protected; "
                "sampling and compute_logprobs only support public SampleRef data"
            )

    def sample(
        self,
        *,
        prompt: ModelInput | SampleRef,
        sampling_params: SamplingParams | None = None,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_sampling_mask: bool = False,
        return_old_logprob: bool = False,
        return_moe_topk_indices: bool = False,
        wait: bool = True,
    ) -> OperationHandle | Dict[str, Any]:
        self._ensure_sample_ref_is_public(prompt)
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
        prompt: ModelInput | SampleRef,
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
            prompt: The model input, or a public managed-sample reference, to
                compute logprobs for. A reference resolves to the complete
                server-rendered model input.
            logprobs_params: Optional parameters (e.g. return_rollout_token_expert for MoE router replay).
                When None, uses defaults.

        Returns:
            When logprobs_params.return_rollout_token_expert=False: List[float|None] of per-token logprobs.
            When logprobs_params.return_rollout_token_expert=True: Dict with "logprobs" and
                "return_rollout_token_expert_data" (None if not MoE or not available).
        """
        self._ensure_sample_ref_is_public(prompt)
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

    def _ensure_full_ft(self) -> None:
        """Verify this client's model is full fine-tuning, once, then cache it.

        The check runs client-side so an unsupported call fails immediately and
        with a precise message, instead of travelling to the server to be
        rejected there. The server enforces the same rule independently.

        Raises:
            ValueError: If the client is not bound to a full fine-tuning model.
        """
        if self._is_full_ft:
            return
        model_id = self.model_id or _su.parse_model_id_from_weaver_path(self.model_path)
        if not model_id:
            raise ValueError(
                "generation control requires a full fine-tuning model, but this sampling "
                "client is not bound to one (no model_id, and model_path carries no model "
                "id). Sampling against a bare base model or a LoRA adapter on the shared "
                "base-model pool cannot be paused: the engine is shared with other tenants."
            )
        model = self._service.get_model(model_id)
        _su.ensure_full_ft_for_control(
            lookup_case_insensitive(model, "training_mode"), model_id=model_id
        )
        self._is_full_ft = True

    def pause_generation(self, *, mode: PauseMode | str = PauseMode.ABORT) -> Dict[str, Any]:
        """Pause in-flight generation on the engine serving this model.

        This is a stateless control primitive: it freezes the engine until
        :meth:`continue_generation` is called. With the default
        ``mode="abort"`` the waiting + running requests are aborted on the spot
        and their partial output is returned to the callers of :meth:`sample`
        (with ``stop_reason="abort"``) — this is the recovery shape used for
        partial/async rollout weight swaps
        (abort -> drain -> sync_weights -> continue).

        **Scope**: the pause is engine-wide, not scoped to this sampling
        session. It freezes every in-flight request on the engine, including
        ones issued through an earlier sampling session of the same model —
        which is exactly what a weight swap needs, since the requests to abort
        are the ones belonging to the previous weight epoch.

        **Full fine-tuning only.** A LoRA adapter is served from one shared
        engine per base model, so pausing it would abort generation for
        unrelated tenants; such a call is rejected before any request is sent.

        Prefer :meth:`paused` over calling this directly: a pause that is never
        continued leaves the engine frozen indefinitely, and there is no
        server-side auto-resume.

        Args:
            mode: How to treat in-flight requests. One of :class:`PauseMode`
                (``abort`` / ``retract`` / ``in_place``) or its string value.

        Returns:
            An acknowledgement, ``{"ok": True, "status": "paused", "mode": ...}``.
            The engine's own reply is deliberately not surfaced: it carries
            replica topology and backend details, and varies by engine version.

        Raises:
            ValueError: If ``mode`` is not a supported pause mode, or this
                client is not bound to a full fine-tuning model.
        """
        body = _su.build_pause_generation_body(mode)
        self._ensure_full_ft()
        return self._service.http.post(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/pause-generation",
            json=body,
        )

    def continue_generation(self) -> Dict[str, Any]:
        """Resume generation after a :meth:`pause_generation` call.

        Engine-wide and full-fine-tuning-only, like :meth:`pause_generation`.

        Returns:
            An acknowledgement, ``{"ok": True, "status": "running"}``.

        Raises:
            ValueError: If this client is not bound to a full fine-tuning model.
        """
        self._ensure_full_ft()
        return self._service.http.post(
            f"/api/v1/sampling-sessions/{self.sampling_session_id}/continue-generation",
        )

    @contextmanager
    def paused(self, *, mode: PauseMode | str = PauseMode.ABORT) -> Iterator[Dict[str, Any]]:
        """Pause the engine for the duration of the block, then always resume.

        A bare :meth:`pause_generation` that never reaches its
        :meth:`continue_generation` — because the block raised, or the caller
        returned early — leaves the engine frozen for good: nothing on the
        server auto-resumes it. This pairs the two so the resume survives errors.

        The resume is issued on *this* client even if the block replaced it with
        a new one, which is correct: both address the same engine.

        Example:
            >>> with sampling_client.paused(mode=PauseMode.ABORT):
            ...     path = training_client.save_weights_for_sampler(name="step-42")
            ...     new_client = service.create_sampling_client(
            ...         model_path=path, model_id=model_id, base_model=base_model
            ...     )

        Args:
            mode: How to treat in-flight requests, as in :meth:`pause_generation`.

        Yields:
            The :meth:`pause_generation` response.

        Raises:
            ValueError: If ``mode`` is invalid or this client is not bound to a
                full fine-tuning model. Nothing is paused in that case.
        """
        result = self.pause_generation(mode=mode)
        try:
            yield result
        finally:
            self.continue_generation()

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
