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

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence, Tuple, overload

from ._artifacts import DEFAULT_EXPORT_TTL_SECONDS, is_artifact_payload, validate_resource_id
from ._async_http import _await_blocking_io
from ._checkpoint_recovery import CHECKPOINT_RECOVERY_DELAYS, select_recovered_checkpoint
from ._deployments import build_create_deployment_body, translate_deployment_error
from ._http import WeaverAPIError
from ._payloads import (
    build_request_metadata,
    build_surrogate_data,
    parse_logprob_tensors,
    prepare_forward_backward_operation,
    prepare_forward_operation,
)
from ._sampling_utils import parse_model_id_from_weaver_path
from ._utils import DEFAULT_SAMPLER_TTL_SECONDS, UNSET, _UnsetType, lookup_case_insensitive
from .async_service_client import AsyncServiceClient
from .operations import AsyncOperationHandle, build_async_operation_handle
from .tensor_transport import PreparedOperationBody
from .types import AdamParams, Datum
from .types.checkpoint import Checkpoint
from .types.deployment import Deployment
from .types.managed_dataset import (
    MAX_SAMPLE_REF_LENGTH_REQUEST_ITEMS,
    SampleRef,
    SampleRefLength,
    parse_sample_ref_lengths,
)
from .types.weights_artifact import WeightsArtifact

if TYPE_CHECKING:
    from typing import Literal

    import torch

    from .async_sampling_client import AsyncSamplingClient
    from .types.router_replay import RouterReplayMetadata

logger = logging.getLogger(__name__)


def _close_prepared_payload(task: "asyncio.Task[PreparedOperationBody]") -> None:
    """Release a payload whose background build outlived its caller."""

    try:
        task.result().close()
    except BaseException:
        pass


async def _build_training_payload(
    builder: Callable[..., PreparedOperationBody],
    **kwargs: Any,
) -> PreparedOperationBody:
    """Build an async request without blocking the loop on pack file I/O."""

    if kwargs.get("loss_fn") == "cross_entropy" and kwargs.get("tensor_transport") != "default":
        task = asyncio.create_task(asyncio.to_thread(builder, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(_close_prepared_payload)
            raise
    return builder(**kwargs)


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
        # Dataset-version visibility is immutable. This cache only avoids
        # repeated catalog round trips for SDK-side public-operation checks;
        # the server remains authoritative on every request.
        self._managed_dataset_visibility_cache: Dict[tuple[str, str], str] = {}

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

    async def _ensure_sample_refs_are_public(self, data: Sequence[Datum]) -> None:
        """Fail early when a public-only operation contains a protected ref."""

        sources = {
            (datum.sample_ref.dataset, datum.sample_ref.version)
            for datum in data
            if datum.sample_ref is not None
        }
        for source in sorted(sources):
            visibility = self._managed_dataset_visibility_cache.get(source)
            if visibility is None:
                info = await self._service.datasets.get(name=source[0], version=source[1])
                visibility = info.content_visibility
                self._managed_dataset_visibility_cache[source] = visibility
            if visibility != "public":
                raise ValueError(
                    f"managed dataset {source[0]!r} version {source[1]!r} is protected; "
                    "protected SampleRef data only supports cross_entropy forward_backward "
                    "with empty loss_fn_inputs"
                )

    async def _validate_forward_backward_sample_refs(
        self, data: Sequence[Datum], loss_fn: str
    ) -> None:
        managed = [datum for datum in data if datum.is_sample_ref]
        if not managed:
            return
        protected_safe = loss_fn == "cross_entropy" and all(
            not datum.loss_fn_inputs for datum in managed
        )
        if not protected_safe:
            await self._ensure_sample_refs_are_public(managed)

    async def resolve_sample_ref_lengths(self, refs: Sequence[SampleRef]) -> List[SampleRefLength]:
        """Resolve safe, model-bound input lengths for whole-sample batching."""

        requested = list(refs)
        if not requested:
            return []
        if not all(isinstance(ref, SampleRef) for ref in requested):
            raise TypeError("refs must contain only SampleRef values")
        resolved: List[SampleRefLength] = []
        known_counts: Dict[SampleRef, int] = {}
        known_revision: str | None = None
        for start in range(0, len(requested), MAX_SAMPLE_REF_LENGTH_REQUEST_ITEMS):
            chunk = requested[start : start + MAX_SAMPLE_REF_LENGTH_REQUEST_ITEMS]
            payload = await self._service.http.post(
                f"/api/v1/models/{self.model_id}/managed-dataset-sample-lengths",
                json={"items": [ref.to_payload() for ref in chunk]},
                max_retries=1,
            )
            parsed = parse_sample_ref_lengths(chunk, payload)
            chunk_revision = parsed[0].model_data_revision
            if start == 0:
                known_revision = chunk_revision
            elif chunk_revision != known_revision:
                raise ValueError(
                    "sample length chunks returned inconsistent model_data_revision values"
                )
            for item in parsed:
                previous = known_counts.setdefault(item.sample_ref, item.input_token_count)
                if previous != item.input_token_count:
                    raise ValueError("duplicate SampleRef entries returned inconsistent lengths")
            resolved.extend(parsed)
        return resolved

    async def _enqueue_prepared(
        self, path: str, prepared: PreparedOperationBody
    ) -> AsyncOperationHandle:
        if prepared.tensor_pack is None:
            return await self._service.enqueue_operation(path, prepared.body)
        return await self._service.enqueue_operation(
            path, prepared.body, tensor_pack=prepared.tensor_pack
        )

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
        if any(datum.is_sample_ref for datum in data):
            await self._ensure_sample_refs_are_public(data)
        payload = await _build_training_payload(
            prepare_forward_operation,
            model_id=self.model_id,
            seq_id=self._next_seq(),
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            request_metadata=build_request_metadata(metadata, router_replay),
            tensor_transport=self._service.tensor_transport,
            tensor_compression=self._service.tensor_compression,
        )
        try:
            path = f"/api/v1/models/{self.model_id}/forward-passes"
            handle = await self._enqueue_prepared(path, payload)
        finally:
            payload.close()
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
        await self._validate_forward_backward_sample_refs(data, loss_fn)
        payload = await _build_training_payload(
            prepare_forward_backward_operation,
            model_id=self.model_id,
            seq_id=self._next_seq(),
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            request_metadata=build_request_metadata(metadata, router_replay),
            tensor_transport=self._service.tensor_transport,
            tensor_compression=self._service.tensor_compression,
        )
        try:
            path = f"/api/v1/models/{self.model_id}/forward-backward-passes"
            handle = await self._enqueue_prepared(path, payload)
        finally:
            payload.close()
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
        fwd_handle = await self.forward(
            data, "forward_logprob", loss_fn_config=loss_fn_config, wait=False
        )
        fwd_result = await fwd_handle.result()
        logprob_tensors = await _await_blocking_io(parse_logprob_tensors, fwd_result, data)

        try:
            loss, metrics = loss_fn(data, logprob_tensors)
        except Exception as exc:
            raise RuntimeError(f"User loss_fn failed: {exc}") from exc

        if loss.dim() != 0:
            raise ValueError(f"loss_fn must return a scalar loss, got shape {loss.shape}")

        loss.backward()
        surrogate_data = build_surrogate_data(data, logprob_tensors)
        await self.forward_backward(
            surrogate_data, "surrogate", loss_fn_config=loss_fn_config, wait=True
        )
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
        existing_checkpoint_ids = (
            {checkpoint.id for checkpoint in await self.list_checkpoints() if checkpoint.id}
            if wait
            else set()
        )
        handle = await self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/checkpoints",
            body,
        )
        if not wait:
            return handle
        result = await handle.result()
        checkpoint = Checkpoint.from_payload(result if isinstance(result, dict) else {})
        if checkpoint.id and checkpoint.path.startswith("weaver://"):
            return checkpoint

        # See the synchronous client's save_state. asyncio.sleep keeps this
        # bounded recovery loop cancellation-responsive.
        for delay in CHECKPOINT_RECOVERY_DELAYS:
            if delay:
                await asyncio.sleep(delay)
            candidate = select_recovered_checkpoint(
                await self.list_checkpoints(),
                existing_ids=existing_checkpoint_ids,
                partial=checkpoint,
                name=name,
                checkpoint_type=checkpoint_type,
            )
            if candidate is not None:
                return candidate
        raise RuntimeError(
            "Save completed but the server returned no checkpoint metadata "
            "and no unique completed checkpoint appeared before the recovery timeout"
        )

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

    # ------------------------------------------------------------------
    # HF weights export
    # ------------------------------------------------------------------

    @overload
    async def export_weights(
        self,
        *,
        checkpoint: str | Checkpoint | None = None,
        merge_adapter: bool = False,
        ttl_seconds: int | None = DEFAULT_EXPORT_TTL_SECONDS,
        force: bool = False,
        wait: "Literal[True]" = True,
    ) -> WeightsArtifact: ...

    @overload
    async def export_weights(
        self,
        *,
        checkpoint: str | Checkpoint | None = None,
        merge_adapter: bool = False,
        ttl_seconds: int | None = DEFAULT_EXPORT_TTL_SECONDS,
        force: bool = False,
        wait: "Literal[False]",
    ) -> WeightsArtifact | AsyncOperationHandle: ...

    async def export_weights(
        self,
        *,
        checkpoint: str | Checkpoint | None = None,
        merge_adapter: bool = False,
        ttl_seconds: int | None = DEFAULT_EXPORT_TTL_SECONDS,
        force: bool = False,
        wait: bool = True,
    ) -> WeightsArtifact | AsyncOperationHandle:
        """Export model weights in HuggingFace format.

        See :meth:`weaver.training_client.TrainingClient.export_weights`.
        Returns a :class:`~weaver.types.WeightsArtifact` when *wait* is True
        (or on an idempotent completed hit), else an ``AsyncOperationHandle``.
        """
        body: Dict[str, Any] = {
            "format": "huggingface",
            "merge_adapter": merge_adapter,
            "ttl_seconds": ttl_seconds,
        }
        if checkpoint is None:
            path = f"/api/v1/models/{self.model_id}/export-hf"
        else:
            checkpoint_id = await self._resolve_checkpoint_id(checkpoint)
            body["force"] = force
            path = f"/api/v1/checkpoints/{checkpoint_id}/export"
        # max_retries=1: exports are non-idempotent POSTs (see enqueue_operation).
        response = await self._service.http.post(path, json=body, max_retries=1)
        # A completed idempotent hit answers with the artifact itself instead
        # of an operation envelope; return it directly even when wait=False.
        if is_artifact_payload(response):
            return WeightsArtifact.from_payload(response)
        handle = build_async_operation_handle(
            self._service.http, response if isinstance(response, dict) else {}
        )
        if not wait:
            return handle
        result = await handle.result()
        return WeightsArtifact.from_payload(result if isinstance(result, dict) else {})

    async def _resolve_checkpoint_id(self, checkpoint: str | Checkpoint) -> str:
        """Resolve a checkpoint reference to its server-side id."""
        if isinstance(checkpoint, Checkpoint):
            if not checkpoint.id:
                raise ValueError("Checkpoint object has no id")
            return validate_resource_id(checkpoint.id, kind="checkpoint")
        reference = checkpoint.strip()
        if not reference:
            raise ValueError("checkpoint reference must not be empty")
        if not reference.startswith("weaver://"):
            # A raw id becomes a URL path segment; require a canonical UUID so
            # dot-segment tricks cannot reroute the request.
            return validate_resource_id(reference, kind="checkpoint")
        owner = parse_model_id_from_weaver_path(reference)
        if owner and owner != self.model_id:
            raise ValueError(
                f"Checkpoint path {reference!r} belongs to model {owner}, "
                f"but this client trains model {self.model_id}"
            )
        for existing in await self.list_checkpoints():
            if existing.path == reference:
                return existing.id
        raise ValueError(
            f"No checkpoint with path {reference!r} found for model "
            f"{self.model_id}; pass a Checkpoint from save_state() or "
            "list_checkpoints()"
        )

    # ------------------------------------------------------------------
    # NorthGate deployment
    # ------------------------------------------------------------------

    @overload
    async def deploy_checkpoint(
        self,
        checkpoint: str | Checkpoint,
        *,
        name: str,
        gpu_type: str | None = None,
        replicas: int = 1,
        gpus_per_replica: int | None = None,
        overwrite: bool = False,
        wait: "Literal[True]" = True,
    ) -> Deployment: ...

    @overload
    async def deploy_checkpoint(
        self,
        checkpoint: str | Checkpoint,
        *,
        name: str,
        gpu_type: str | None = None,
        replicas: int = 1,
        gpus_per_replica: int | None = None,
        overwrite: bool = False,
        wait: "Literal[False]",
    ) -> AsyncOperationHandle: ...

    async def deploy_checkpoint(  # pylint: disable=too-many-arguments
        self,
        checkpoint: str | Checkpoint,
        *,
        name: str,
        gpu_type: str | None = None,
        replicas: int = 1,
        gpus_per_replica: int | None = None,
        overwrite: bool = False,
        wait: bool = True,
    ) -> Deployment | AsyncOperationHandle:
        """Publish a checkpoint as a public, OpenAI-compatible endpoint.

        See :meth:`weaver.training_client.TrainingClient.deploy_checkpoint`.
        Returns a :class:`~weaver.types.Deployment` when *wait* is True, else
        an ``AsyncOperationHandle``.
        """
        body = build_create_deployment_body(
            name=name,
            gpu_type=gpu_type,
            replicas=replicas,
            gpus_per_replica=gpus_per_replica,
            overwrite=overwrite,
        )
        checkpoint_id = await self._resolve_checkpoint_id(checkpoint)
        try:
            # max_retries=1: creating a deployment is a non-idempotent POST
            # that launches GPUs and claims a global gateway name.
            response = await self._service.http.post(
                f"/api/v1/checkpoints/{checkpoint_id}/deployments", json=body, max_retries=1
            )
        except WeaverAPIError as exc:
            raise translate_deployment_error(exc) from exc
        handle = build_async_operation_handle(
            self._service.http, response if isinstance(response, dict) else {}
        )
        if not wait:
            return handle
        result = await handle.result()
        return Deployment.from_payload(result if isinstance(result, dict) else {})

    async def terminate(self, instance_types: list[str] | None = None) -> Dict[str, Any]:
        """Terminate trainer and/or inference instances for this model."""
        return await self._service.terminate_model(self.model_id, instance_types)
