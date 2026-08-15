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

"""Asyncio-native training client built on top of AsyncServiceClient.

Every operation is awaited. Pass ``wait=False`` to get an
:class:`~weaver.operations.AsyncOperationHandle` back immediately and await it
later, which lets several server-side operations overlap::

    fb = await tc.forward_backward(data, "cross_entropy", wait=False)
    opt = await tc.optim_step(params, wait=False)
    await fb            # both already submitted; the waits overlap
    await opt
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence, Tuple, overload

from ._payloads import (
    build_request_metadata,
    build_surrogate_data,
    forward_backward_payload,
    forward_payload,
    parse_logprob_tensors,
    publish_live_weights_nccl_v1_payload,
)
from ._utils import DEFAULT_SAMPLER_TTL_SECONDS, UNSET, _UnsetType, lookup_case_insensitive
from .async_service_client import AsyncServiceClient
from .operations import AsyncOperationHandle
from .types import AdamParams, Datum
from .types.checkpoint import Checkpoint
from .types.nccl_weight_sync import NCCLWeightSyncV1Result

if TYPE_CHECKING:
    from typing import Literal

    import torch

    from .async_sampling_client import AsyncSamplingClient
    from .types.router_replay import RouterReplayMetadata

logger = logging.getLogger(__name__)


class AsyncTrainingClient:
    def __init__(
        self,
        *,
        service: AsyncServiceClient,
        model_id: str,
        base_model: str,
        session_id: str,
        tokenizer_path: str | None = None,
        debug_info: Dict[str, Any] | None = None,
    ) -> None:
        self._service = service
        self.model_id = model_id
        self.base_model = base_model
        self.session_id = session_id
        self.tokenizer_path = tokenizer_path
        self.debug_info = debug_info
        self._tokenizer: Any = None

    @property
    def training_run_id(self) -> str:
        """Canonical identifier for this Training Run; model_id is retained."""

        return self.model_id

    async def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int,
        occurred_at: datetime | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist application-side scalar metrics in Weaver."""

        if step < 0:
            raise ValueError("step must be non-negative")
        if len(metrics) > 1000:
            raise ValueError("at most 1000 metrics may be logged per call")
        at = occurred_at or datetime.now(timezone.utc)
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        points = []
        for name, raw_value in metrics.items():
            metric_name = str(name).strip()
            if not metric_name:
                raise ValueError("metric names must not be empty")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"metric {metric_name!r} must be finite")
            points.append(
                {
                    "model_id": self.model_id,
                    "name": metric_name,
                    "value": value,
                    "step": step,
                    "occurred_at": at.isoformat(),
                    "labels": dict(labels or {}),
                }
            )
        if points:
            await self._service.http.post(
                f"/api/v1/sessions/{self.session_id}/metrics", json={"metrics": points}
            )

    def _next_seq(self) -> int:
        return self._service.next_operation_seq(self.model_id)

    @overload
    async def forward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        loss_fn_config: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: "Literal[True]" = True,
    ) -> Dict[str, Any]: ...

    @overload
    async def forward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        loss_fn_config: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def forward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        loss_fn_config: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: bool = True,
    ) -> AsyncOperationHandle | Dict[str, Any]:
        """Compute a forward pass without accumulating gradients.

        Args:
            data: Sequence of training data.
            loss_fn: Name of the loss function to use.
            loss_fn_config: Optional loss function configuration.
            metadata: Optional top-level request metadata. Router replay metadata
                must be attached to each Datum as ``datum.metadata["router_replay"]``.
            router_replay: Deprecated request-level Router Replay envelope; passing
                it raises ``ValueError``.
            wait: If True (default), awaits completion and returns the result dict;
                if False, returns an ``AsyncOperationHandle`` immediately.
        """
        payload = forward_payload(
            model_id=self.model_id,
            seq_id=self._next_seq(),
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            request_metadata=build_request_metadata(metadata, router_replay),
        )
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/forward-passes",
            {"payload": payload},
        )
        return await handle.result() if wait else handle

    @overload
    async def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: "Literal[True]" = True,
    ) -> Dict[str, Any]: ...

    @overload
    async def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: bool = True,
    ) -> AsyncOperationHandle | Dict[str, Any]:
        """Compute a forward and backward pass, accumulating gradients.

        Args:
            data: Sequence of training data.
            loss_fn: Name of the loss function to use.
            loss_fn_config: Optional loss function configuration.
            metadata: Optional top-level request metadata. Router replay metadata
                must be attached to each Datum as ``datum.metadata["router_replay"]``.
            router_replay: Deprecated request-level Router Replay envelope; passing
                it raises ``ValueError``.
            wait: If True (default), awaits completion and returns the result dict;
                if False, returns an ``AsyncOperationHandle`` immediately.
        """
        payload = forward_backward_payload(
            model_id=self.model_id,
            seq_id=self._next_seq(),
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            request_metadata=build_request_metadata(metadata, router_replay),
        )
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/forward-backward-passes",
            {"payload": payload},
        )
        return await handle.result() if wait else handle

    async def forward_backward_custom(
        self,
        data: Sequence[Datum],
        loss_fn: Callable[
            [Sequence[Datum], List["torch.Tensor"]], Tuple["torch.Tensor", Dict[str, Any]]
        ],
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Run a custom loss function with surrogate-based gradient propagation.

        Orchestrates two sequential server calls: a forward pass to obtain
        per-token logprobs, then a surrogate backward pass that applies the
        user-computed gradients. See
        :meth:`weaver.training_client.TrainingClient.forward_backward_custom`.
        """
        # Step A: forward pass to get logprobs
        fwd_result = await self.forward(
            data, "forward_logprob", loss_fn_config=loss_fn_config, wait=True
        )

        # Step B: parse logprobs from response
        logprob_tensors = parse_logprob_tensors(fwd_result, data)

        # Step C: run user's loss function
        try:
            loss, metrics = loss_fn(data, logprob_tensors)
        except Exception as exc:
            raise RuntimeError(f"User loss_fn failed: {exc}") from exc

        if loss.dim() != 0:
            raise ValueError(f"loss_fn must return a scalar loss, got shape {loss.shape}")

        # Step D: backprop through user graph into logprob tensors
        loss.backward()

        # Step E: build surrogate Datum objects from the propagated gradients
        surrogate_data = build_surrogate_data(data, logprob_tensors)

        # Step F: surrogate backward pass
        await self.forward_backward(
            surrogate_data, "surrogate", loss_fn_config=loss_fn_config, wait=True
        )

        # Step G: return loss and metrics
        return {"loss": loss.detach(), "metrics": metrics}

    @overload
    async def optim_step(
        self, params: AdamParams, *, wait: "Literal[True]" = True
    ) -> Dict[str, Any]: ...

    @overload
    async def optim_step(
        self, params: AdamParams, *, wait: "Literal[False]"
    ) -> AsyncOperationHandle: ...

    async def optim_step(
        self, params: AdamParams, *, wait: bool = True
    ) -> AsyncOperationHandle | Dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "seq_id": self._next_seq(),
            "adam_params": params.to_payload(),
        }
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/optimizer-steps",
            {"payload": payload},
        )
        return await handle.result() if wait else handle

    @overload
    async def publish_live_weights_to_sampler_nccl_v1(
        self,
        sampling_client: "AsyncSamplingClient",
        *,
        expected_weight_version: str,
        proposed_weight_version: str,
        transaction_id: str | None = None,
        checksum_mode: str = "off",
        wait: "Literal[True]" = True,
    ) -> NCCLWeightSyncV1Result: ...

    @overload
    async def publish_live_weights_to_sampler_nccl_v1(
        self,
        sampling_client: "AsyncSamplingClient",
        *,
        expected_weight_version: str,
        proposed_weight_version: str,
        transaction_id: str | None = None,
        checksum_mode: str = "off",
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def publish_live_weights_to_sampler_nccl_v1(
        self,
        sampling_client: "AsyncSamplingClient",
        *,
        expected_weight_version: str,
        proposed_weight_version: str,
        transaction_id: str | None = None,
        checksum_mode: str = "off",
        wait: bool = True,
    ) -> NCCLWeightSyncV1Result | AsyncOperationHandle:
        """Async twin of the live-collective publication."""

        selection = getattr(sampling_client, "weight_sync", None)
        if selection is not None and not selection.is_live_collective:
            raise ValueError(
                "this sampling session was created with "
                f"backend={selection.backend!r}; publishing live weights would "
                "silently use a transport the session was not configured for"
            )
        return await self._publish_live_weights(
            sampling_client,
            expected_weight_version=expected_weight_version,
            proposed_weight_version=proposed_weight_version,
            transaction_id=transaction_id,
            checksum_mode=checksum_mode,
            wait=wait,
        )

    async def _publish_live_weights(
        self,
        sampling_client: "AsyncSamplingClient",
        *,
        expected_weight_version: str,
        proposed_weight_version: str,
        transaction_id: str | None,
        checksum_mode: str,
        wait: bool,
    ) -> NCCLWeightSyncV1Result | AsyncOperationHandle:
        """Run one live-collective transaction."""

        if getattr(sampling_client, "_service", None) is not self._service:
            raise ValueError("sampling client belongs to another Weaver service")
        if sampling_client.model_id != self.model_id:
            raise ValueError("sampling client is not bound to this training model")
        if sampling_client.model_path:
            raise ValueError("NCCL-v1 sampling client must not carry a model_path")
        payload = publish_live_weights_nccl_v1_payload(
            seq_id=self._next_seq(),
            sampling_session_id=sampling_client.sampling_session_id,
            expected_weight_version=expected_weight_version,
            proposed_weight_version=proposed_weight_version,
            transaction_id=transaction_id,
            checksum_mode=checksum_mode,
        )
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/publish-live-weights-nccl-v1",
            payload,
        )
        if not wait:
            return handle
        receipt = NCCLWeightSyncV1Result.from_payload(await handle.result()).validate_request(
            transaction_id=payload["transaction_id"],
            expected_weight_version=payload["expected_weight_version"],
            proposed_weight_version=payload["proposed_weight_version"],
            checksum_mode=payload["checksum_mode"],
        )
        # Only now: the receipt proves the target committed this version,
        # closed the transaction and resumed serving.
        binder = getattr(sampling_client, "_bind_committed_weight_version", None)
        if binder is not None:
            binder(receipt.committed_weight_version)
        return receipt

    @overload
    async def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = DEFAULT_SAMPLER_TTL_SECONDS,
        wait: "Literal[True]" = True,
    ) -> str: ...

    @overload
    async def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = DEFAULT_SAMPLER_TTL_SECONDS,
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = DEFAULT_SAMPLER_TTL_SECONDS,
        wait: bool = True,
    ) -> str | AsyncOperationHandle:
        """Export model weights for sampling.

        See :meth:`weaver.training_client.TrainingClient.save_weights_for_sampler`.
        Returns the model path (str) when *wait* is True, else an
        ``AsyncOperationHandle``.
        """
        body: Dict[str, Any] = {"seq_id": self._next_seq()}
        if name:
            body["path"] = name
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/export-sampler",
            body,
        )
        if not wait:
            return handle
        result = await handle.result()
        model_path = lookup_case_insensitive(result or {}, "model_path") or lookup_case_insensitive(
            result or {}, "path"
        )
        if not model_path:
            raise RuntimeError("Export response missing model path")
        return str(model_path)

    @overload
    async def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = DEFAULT_SAMPLER_TTL_SECONDS,
        wait: "Literal[True]" = True,
    ) -> "AsyncSamplingClient": ...

    @overload
    async def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = DEFAULT_SAMPLER_TTL_SECONDS,
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = DEFAULT_SAMPLER_TTL_SECONDS,
        wait: bool = True,
    ) -> "AsyncSamplingClient | AsyncOperationHandle":
        """Export model weights and create an async sampling client.

        See :meth:`weaver.training_client.TrainingClient.save_weights_and_get_sampling_client`.
        """
        body: Dict[str, Any] = {"seq_id": self._next_seq()}
        if name:
            body["path"] = name
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/export-sampler",
            body,
        )
        if not wait:
            return handle
        result = await handle.result()
        sampling_session_id = lookup_case_insensitive(result or {}, "sampling_session_id")
        model_path = lookup_case_insensitive(result or {}, "model_path") or lookup_case_insensitive(
            result or {}, "path"
        )
        if sampling_session_id:
            return await self._service.get_sampling_client(
                model_path=model_path or "",
                base_model=self.base_model,
                model_id=self.model_id,
                sampling_session_id=sampling_session_id,
                tokenizer_path=self.tokenizer_path,
            )
        if model_path:
            return await self._service.get_sampling_client(
                model_path=str(model_path),
                base_model=self.base_model,
                model_id=self.model_id,
                tokenizer_path=self.tokenizer_path,
            )
        raise RuntimeError("Export response missing sampling session id or model path")

    @property
    def tokenizer(self):  # type: ignore[misc]
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_name_or_path = self.tokenizer_path if self.tokenizer_path else self.base_model
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        return self._tokenizer

    def get_tokenizer(self):  # Backwards compatible accessor
        return self.tokenizer

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    @overload
    async def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
        ttl_seconds: int | None | _UnsetType = ...,
        wait: "Literal[True]" = True,
    ) -> Checkpoint: ...

    @overload
    async def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
        ttl_seconds: int | None | _UnsetType = ...,
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
        ttl_seconds: int | None | _UnsetType = UNSET,
        wait: bool = True,
    ) -> Checkpoint | AsyncOperationHandle:
        """Save the current model weights as a checkpoint.

        See :meth:`weaver.training_client.TrainingClient.save_state`. Returns a
        :class:`~weaver.types.Checkpoint` when *wait* is True, else an
        ``AsyncOperationHandle``.
        """
        body: Dict[str, Any] = {"type": checkpoint_type}
        if name is not None:
            body["name"] = name
        if not isinstance(ttl_seconds, _UnsetType):
            body["ttl_seconds"] = ttl_seconds
        elif checkpoint_type == "sampling":
            # Regenerable sampling checkpoints default to a bounded TTL so they
            # don't accumulate on shared storage; weight checkpoints stay
            # permanent unless an explicit ttl_seconds is given.
            body["ttl_seconds"] = DEFAULT_SAMPLER_TTL_SECONDS
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/checkpoints",
            body,
        )
        if not wait:
            return handle
        result = await handle.result()
        return Checkpoint.from_payload(result if isinstance(result, dict) else {})

    @overload
    async def load_state(
        self, path: str | Checkpoint, *, wait: "Literal[True]" = True
    ) -> Dict[str, Any]: ...

    @overload
    async def load_state(
        self, path: str | Checkpoint, *, wait: "Literal[False]"
    ) -> AsyncOperationHandle: ...

    async def load_state(
        self,
        path: str | Checkpoint,
        *,
        wait: bool = True,
    ) -> AsyncOperationHandle | Dict[str, Any]:
        """Restore model weights from a checkpoint (optimizer state is **not** restored)."""
        return await self._load_checkpoint(path, include_optimizer=False, wait=wait)

    @overload
    async def load_state_with_optimizer(
        self, path: str | Checkpoint, *, wait: "Literal[True]" = True
    ) -> Dict[str, Any]: ...

    @overload
    async def load_state_with_optimizer(
        self, path: str | Checkpoint, *, wait: "Literal[False]"
    ) -> AsyncOperationHandle: ...

    async def load_state_with_optimizer(
        self,
        path: str | Checkpoint,
        *,
        wait: bool = True,
    ) -> AsyncOperationHandle | Dict[str, Any]:
        """Restore model weights **and** optimizer state from a checkpoint."""
        return await self._load_checkpoint(path, include_optimizer=True, wait=wait)

    async def _load_checkpoint(
        self,
        path: str | Checkpoint,
        *,
        include_optimizer: bool,
        wait: bool,
    ) -> AsyncOperationHandle | Dict[str, Any]:
        checkpoint_path = path.path if isinstance(path, Checkpoint) else path
        body: Dict[str, Any] = {
            "path": checkpoint_path,
            "include_optimizer": include_optimizer,
        }
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/load",
            body,
        )
        return await handle.result() if wait else handle

    async def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints for this model."""
        response = await self._service.http.get(
            f"/api/v1/models/{self.model_id}/checkpoints",
        )
        items = (response or {}).get("items", []) if isinstance(response, dict) else []
        return [Checkpoint.from_payload(item) for item in items if isinstance(item, dict)]

    async def set_checkpoint_ttl(
        self,
        path: str | Checkpoint,
        ttl_seconds: int | None,
    ) -> Dict[str, Any]:
        """Set or cancel the TTL (time-to-live) for a checkpoint."""
        checkpoint_path = path.path if isinstance(path, Checkpoint) else path
        body: Dict[str, Any] = {"path": checkpoint_path, "ttl_seconds": ttl_seconds}
        return await self._service.http.patch(
            f"/api/v1/models/{self.model_id}/checkpoints/ttl",
            json=body,
        )

    async def terminate(self, instance_types: list[str] | None = None) -> Dict[str, Any]:
        """Terminate trainer and/or inference instances for this model."""
        return await self._service.terminate_model(self.model_id, instance_types)
