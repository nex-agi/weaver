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

"""Training client built on top of the Weaver ServiceClient."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence, Tuple, overload

from ._payloads import (
    build_request_metadata,
    build_surrogate_data,
    forward_backward_payload,
    forward_payload,
    parse_logprob_tensors,
    serialize_data,
)
from ._utils import UNSET, _UnsetType, lookup_case_insensitive
from .operations import OperationHandle
from .service_client import ServiceClient
from .types import AdamParams, Datum
from .types.checkpoint import Checkpoint

if TYPE_CHECKING:
    from typing import Literal

    import torch

    from .sampling_client import SamplingClient
    from .types.router_replay import RouterReplayMetadata

logger = logging.getLogger(__name__)


class TrainingClient:
    def __init__(
        self,
        *,
        service: ServiceClient,
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

    def _next_seq(self) -> int:
        return self._service.next_operation_seq(self.model_id)

    def _serialize_data(
        self,
        data: Sequence[Datum],
        *,
        blob_ctx: Dict[str, Any] | None = None,
    ) -> Sequence[Dict[str, Any]]:
        if blob_ctx is not None:
            from .blob_store import get_blob_store

            store = get_blob_store()
            if store.enabled:
                # Pack the whole batch's offloaded tensors into one file: each
                # datum field becomes a slice rather than its own tiny file.
                pack = store.open_pack(model_id=blob_ctx["model_id"], seq_id=blob_ctx["seq_id"])
                ctx = {**blob_ctx, "pack": pack}
                payload = [datum.to_payload(blob_ctx=ctx, index=i) for i, datum in enumerate(data)]
                pack.commit()
                return payload
        # Offload disabled (or no ctx): defer to the shared serializer so the
        # sync and async stacks emit identical inline bytes.
        return serialize_data(data)

    def _build_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None",
    ) -> Dict[str, Any] | None:
        return build_request_metadata(metadata, router_replay)

    @overload
    def forward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        loss_fn_config: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: Literal[True] = True,
    ) -> Dict[str, Any]: ...

    @overload
    def forward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        loss_fn_config: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def forward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        loss_fn_config: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: bool = True,
    ) -> OperationHandle | Dict[str, Any]:
        """Compute a forward pass without accumulating gradients.

        Args:
            data: Sequence of training data.
            loss_fn: Name of the loss function to use.
            loss_fn_config: Optional loss function configuration.
            metadata: Optional top-level request metadata (e.g. router_replay
                context). Router replay metadata must be attached to each Datum
                as ``datum.metadata["router_replay"]``.
            router_replay: Deprecated request-level Router Replay envelope.
                Passing this argument raises ``ValueError``.
            wait: If True, blocks until the operation completes.
        """
        seq_id = self._next_seq()
        blob_ctx = {"model_id": self.model_id, "seq_id": seq_id}
        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "seq_id": seq_id,
            "forward_input": {
                "loss_fn": loss_fn,
                "data": self._serialize_data(data, blob_ctx=blob_ctx),
            },
        }
        if loss_fn_config:
            payload["forward_input"]["loss_fn_config"] = dict(loss_fn_config)
        request_metadata = self._build_metadata(metadata, router_replay)
        if request_metadata:
            payload["metadata"] = request_metadata
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/forward-passes",
            {"payload": payload},
        )
        return handle.result() if wait else handle

    @overload
    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: Literal[True] = True,
    ) -> Dict[str, Any]: ...

    @overload
    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        router_replay: "RouterReplayMetadata | Mapping[str, Any] | None" = None,
        wait: bool = True,
    ) -> OperationHandle | Dict[str, Any]:
        """Compute a forward and backward pass, accumulating gradients.

        Args:
            data: Sequence of training data.
            loss_fn: Name of the loss function to use.
            loss_fn_config: Optional loss function configuration.
            metadata: Optional top-level request metadata (e.g. router_replay
                context). Router replay metadata must be attached to each Datum
                as ``datum.metadata["router_replay"]``.
            router_replay: Deprecated request-level Router Replay envelope.
                Passing this argument raises ``ValueError``.
            wait: If True, blocks until the operation completes.
        """
        seq_id = self._next_seq()
        blob_ctx = {"model_id": self.model_id, "seq_id": seq_id}
        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "seq_id": seq_id,
            "forward_backward_input": {
                "loss_fn": loss_fn,
                "data": self._serialize_data(data, blob_ctx=blob_ctx),
            },
        }
        if loss_fn_config:
            payload["forward_backward_input"]["loss_fn_config"] = dict(loss_fn_config)
        request_metadata = self._build_metadata(metadata, router_replay)
        if request_metadata:
            payload["metadata"] = request_metadata
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/forward-backward-passes",
            {"payload": payload},
        )
        return handle.result() if wait else handle

    def forward_backward_custom(
        self,
        data: Sequence[Datum],
        loss_fn: Callable[
            [Sequence[Datum], List["torch.Tensor"]], Tuple["torch.Tensor", Dict[str, Any]]
        ],
        *,
        loss_fn_config: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Run a custom loss function with surrogate-based gradient propagation.

        This orchestrates two sequential server calls:
        1. A forward pass to obtain per-token logprobs.
        2. A surrogate backward pass that applies user-computed gradients.

        The user-supplied *loss_fn* receives the original data and the
        logprob tensors (with ``requires_grad=True``), and must return
        ``(scalar_loss, metrics_dict)``.  Gradients flow back through
        ``loss.backward()`` into the logprob tensors, whose ``.grad``
        fields are then sent to the trainer as surrogate weights.

        Args:
            data: Sequence of training data.
            loss_fn: ``(data, logprob_tensors) -> (loss, metrics)``
        """
        # Step A: forward pass to get logprobs
        fwd_result = self.forward(data, "forward_logprob", loss_fn_config=loss_fn_config, wait=True)

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
        self.forward_backward(surrogate_data, "surrogate", loss_fn_config=loss_fn_config, wait=True)

        # Step G: return loss and metrics
        return {"loss": loss.detach(), "metrics": metrics}

    @overload
    def optim_step(self, params: AdamParams, *, wait: Literal[True] = True) -> Dict[str, Any]: ...

    @overload
    def optim_step(self, params: AdamParams, *, wait: Literal[False]) -> OperationHandle: ...

    def optim_step(
        self, params: AdamParams, *, wait: bool = True
    ) -> OperationHandle | Dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "seq_id": self._next_seq(),
            "adam_params": params.to_payload(),
        }
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/optimizer-steps",
            {"payload": payload},
        )
        return handle.result() if wait else handle

    @overload
    def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = 86400,
        wait: Literal[True] = True,
    ) -> str: ...

    @overload
    def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = 86400,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = 86400,
        wait: bool = True,
    ) -> str | OperationHandle:
        """Export model weights for sampling.

        Sampler weights are intended for short-lived RL weight-sync use, so
        the default TTL is **1 day (86400 s)**.  Pass ``ttl_seconds=None`` to
        keep the exported checkpoint permanently (use ``save_state`` if you
        need a durable checkpoint instead).

        Args:
            name: Optional custom path name for the exported weights
            ttl_seconds: Time-to-live in seconds for the exported checkpoint.
                Defaults to ``86400`` (1 day).  Pass ``None`` for permanent.
            wait: If True (default), waits for export to complete and returns path.
                  If False, returns an OperationHandle immediately.

        Returns:
            Model path (str) if wait=True, OperationHandle if wait=False

        Raises:
            RuntimeError: If export response is missing model path
        """
        body: Dict[str, Any] = {"seq_id": self._next_seq()}
        if name:
            body["path"] = name
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/export-sampler",
            body,
        )
        if not wait:
            return handle
        result = handle.result()
        model_path = lookup_case_insensitive(result or {}, "model_path") or lookup_case_insensitive(
            result or {}, "path"
        )
        if not model_path:
            raise RuntimeError("Export response missing model path")
        return str(model_path)

    @overload
    def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = 86400,
        wait: Literal[True] = True,
    ) -> "SamplingClient": ...

    @overload
    def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = 86400,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        ttl_seconds: int | None = 86400,
        wait: bool = True,
    ) -> "SamplingClient" | OperationHandle:
        """Export model weights and create a sampling client.

        This is a convenience method that combines save_weights_for_sampler
        and get_sampling_client. For more control, use those methods separately.

        Because this method is designed for frequent RL weight-sync calls,
        the default TTL is **1 day (86400 s)**.  Pass ``ttl_seconds=None``
        to keep the checkpoint permanently.

        Args:
            name: Optional custom path name for the exported weights
            ttl_seconds: Time-to-live in seconds for the exported checkpoint.
                Defaults to ``86400`` (1 day).  Pass ``None`` for permanent.
            wait: If True (default), waits for export and returns SamplingClient.
                  If False, returns an OperationHandle immediately.

        Returns:
            SamplingClient if wait=True, OperationHandle if wait=False

        Raises:
            RuntimeError: If export response is missing required information
        """
        body: Dict[str, Any] = {"seq_id": self._next_seq()}
        if name:
            body["path"] = name
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/export-sampler",
            body,
        )
        if not wait:
            return handle
        result = handle.result()
        sampling_session_id = lookup_case_insensitive(result or {}, "sampling_session_id")
        model_path = lookup_case_insensitive(result or {}, "model_path") or lookup_case_insensitive(
            result or {}, "path"
        )
        if sampling_session_id:
            return self._service.get_sampling_client(
                model_path=model_path or "",
                base_model=self.base_model,
                model_id=self.model_id,
                sampling_session_id=sampling_session_id,
                tokenizer_path=self.tokenizer_path,
            )
        if model_path:
            return self._service.get_sampling_client(
                model_path=str(model_path),
                base_model=self.base_model,
                model_id=self.model_id,
                tokenizer_path=self.tokenizer_path,
            )
        raise RuntimeError("Export response missing sampling session id or model path")

    @cached_property
    def tokenizer(self):  # type: ignore[misc]
        from transformers import AutoTokenizer

        # Use custom tokenizer_path if provided, otherwise use base_model
        model_name_or_path = self.tokenizer_path if self.tokenizer_path else self.base_model
        return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    def get_tokenizer(self):  # Backwards compatible accessor
        return self.tokenizer

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    @overload
    def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
        ttl_seconds: int | None = ...,
        wait: Literal[True] = True,
    ) -> Checkpoint: ...

    @overload
    def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
        ttl_seconds: int | None = ...,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
        ttl_seconds: int | None | _UnsetType = UNSET,
        wait: bool = True,
    ) -> Checkpoint | OperationHandle:
        """Save the current model weights as a checkpoint.

        The server dispatches an async save task to the trainer, which
        writes weight files to disk at a server-generated path.

        Args:
            name: Human-readable checkpoint label (e.g. ``"step-100"``).
                The server generates the full storage path incorporating
                the model ID automatically.
            checkpoint_type: ``"weight"`` (default) or
                ``"weight_and_optimizer"``.
            ttl_seconds: Time-to-live in seconds for the checkpoint.
                Defaults to ``None`` (permanent, backward-compatible).
                Pass an integer to set auto-expiration, or explicit ``None``
                to ensure permanent retention.
            wait: If True (default), blocks until the save completes and
                returns a :class:`~weaver.types.Checkpoint`.

        Returns:
            A :class:`~weaver.types.Checkpoint` when *wait* is True, else
            an :class:`OperationHandle`.
        """
        body: Dict[str, Any] = {"type": checkpoint_type}
        if name is not None:
            body["name"] = name
        if not isinstance(ttl_seconds, _UnsetType):
            body["ttl_seconds"] = ttl_seconds
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/checkpoints",
            body,
        )
        if not wait:
            return handle
        result = handle.result()
        return Checkpoint.from_payload(result if isinstance(result, dict) else {})

    @overload
    def load_state(
        self,
        path: str | Checkpoint,
        *,
        wait: Literal[True] = True,
    ) -> Dict[str, Any]: ...

    @overload
    def load_state(
        self,
        path: str | Checkpoint,
        *,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def load_state(
        self,
        path: str | Checkpoint,
        *,
        wait: bool = True,
    ) -> OperationHandle | Dict[str, Any]:
        """Restore model weights from a checkpoint (optimizer state is **not** restored).

        Args:
            path: Checkpoint storage path (``weaver://...`` URI returned by
                :meth:`save_state`), or a :class:`~weaver.types.Checkpoint`
                object whose ``.path`` will be used.
            wait: If True (default), blocks until the load completes.

        Returns:
            Server response dict when *wait* is True, else an
            :class:`OperationHandle`.
        """
        return self._load_checkpoint(path, include_optimizer=False, wait=wait)

    @overload
    def load_state_with_optimizer(
        self,
        path: str | Checkpoint,
        *,
        wait: Literal[True] = True,
    ) -> Dict[str, Any]: ...

    @overload
    def load_state_with_optimizer(
        self,
        path: str | Checkpoint,
        *,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def load_state_with_optimizer(
        self,
        path: str | Checkpoint,
        *,
        wait: bool = True,
    ) -> OperationHandle | Dict[str, Any]:
        """Restore model weights **and** optimizer state from a checkpoint.

        This enables true resume-from-checkpoint training where Adam momentum
        and other optimizer statistics are preserved.

        Args:
            path: Checkpoint storage path (``weaver://...`` URI), or a
                :class:`~weaver.types.Checkpoint` object.
            wait: If True (default), blocks until the load completes.

        Returns:
            Server response dict when *wait* is True, else an
            :class:`OperationHandle`.
        """
        return self._load_checkpoint(path, include_optimizer=True, wait=wait)

    def _load_checkpoint(
        self,
        path: str | Checkpoint,
        *,
        include_optimizer: bool,
        wait: bool,
    ) -> OperationHandle | Dict[str, Any]:
        checkpoint_path = path.path if isinstance(path, Checkpoint) else path
        body: Dict[str, Any] = {
            "path": checkpoint_path,
            "include_optimizer": include_optimizer,
        }
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/load",
            body,
        )
        return handle.result() if wait else handle

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints for this model.

        Returns:
            A list of :class:`~weaver.types.Checkpoint` objects.
        """
        response = self._service.http.get(
            f"/api/v1/models/{self.model_id}/checkpoints",
        )
        items = (response or {}).get("items", []) if isinstance(response, dict) else []
        return [Checkpoint.from_payload(item) for item in items if isinstance(item, dict)]

    def set_checkpoint_ttl(
        self,
        path: str | Checkpoint,
        ttl_seconds: int | None,
    ) -> Dict[str, Any]:
        """Set or cancel the TTL (time-to-live) for a checkpoint.

        Args:
            path: Checkpoint storage path (``weaver://...`` URI), or a
                :class:`~weaver.types.Checkpoint` object whose ``.path``
                will be used.
            ttl_seconds: TTL in seconds.  Pass ``None`` to cancel
                expiration (make the checkpoint permanent).

        Returns:
            Server response dict confirming the TTL update.
        """
        checkpoint_path = path.path if isinstance(path, Checkpoint) else path
        body: Dict[str, Any] = {
            "path": checkpoint_path,
            "ttl_seconds": ttl_seconds,
        }
        return self._service.http.patch(
            f"/api/v1/models/{self.model_id}/checkpoints/ttl",
            json=body,
        )

    def terminate(self, instance_types: list[str] | None = None) -> Dict[str, Any]:
        """Terminate trainer and/or inference instances for this model.

        Args:
            instance_types: List of instance types to terminate (e.g., ["trainer", "inference"]).
                          Defaults to both if not specified.

        Returns:
            Dictionary with termination results for each instance type
        """
        return self._service.terminate_model(self.model_id, instance_types)
