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
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence, overload

from ._utils import lookup_case_insensitive
from .operations import OperationHandle, build_operation_handle
from .service_client import ServiceClient
from .types import AdamParams, Datum
from .types.checkpoint import Checkpoint

if TYPE_CHECKING:
    from typing import Literal

    from .sampling_client import SamplingClient

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
    ) -> None:
        self._service = service
        self.model_id = model_id
        self.base_model = base_model
        self.session_id = session_id
        self.tokenizer_path = tokenizer_path

    def _next_seq(self) -> int:
        return self._service.next_operation_seq(self.model_id)

    def _serialize_data(self, data: Sequence[Datum]) -> Sequence[Dict[str, Any]]:
        return [datum.to_payload() for datum in data]

    @overload
    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, float] | None = None,
        wait: Literal[True] = True,
    ) -> Dict[str, Any]: ...

    @overload
    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, float] | None = None,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str,
        *,
        loss_fn_config: Mapping[str, float] | None = None,
        wait: bool = True,
    ) -> OperationHandle | Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "seq_id": self._next_seq(),
            "forward_backward_input": {
                "loss_fn": loss_fn,
                "data": self._serialize_data(data),
            },
        }
        if loss_fn_config:
            payload["forward_backward_input"]["loss_fn_config"] = dict(loss_fn_config)
        handle = self._service.enqueue_operation(
            f"/api/v1/models/{self.model_id}/forward-backward-passes",
            {"payload": payload},
        )
        return handle.result() if wait else handle

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
        wait: Literal[True] = True,
    ) -> str: ...

    @overload
    def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def save_weights_for_sampler(
        self,
        *,
        name: str | None = None,
        wait: bool = True,
    ) -> str | OperationHandle:
        """Export model weights for sampling.

        Args:
            name: Optional custom path name for the exported weights
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
        wait: Literal[True] = True,
    ) -> "SamplingClient": ...

    @overload
    def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        wait: Literal[False],
    ) -> OperationHandle: ...

    def save_weights_and_get_sampling_client(
        self,
        *,
        name: str | None = None,
        wait: bool = True,
    ) -> "SamplingClient" | OperationHandle:
        """Export model weights and create a sampling client.

        This is a convenience method that combines save_weights_for_sampler
        and get_sampling_client. For more control, use those methods separately.

        Args:
            name: Optional custom path name for the exported weights
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
        wait: Literal[True] = True,
    ) -> Checkpoint: ...

    @overload
    def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
        wait: Literal[False],
    ) -> OperationHandle: ...

    def save_state(
        self,
        *,
        name: str | None = None,
        checkpoint_type: str = "weight",
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
            wait: If True (default), blocks until the save completes and
                returns a :class:`~weaver.types.Checkpoint`.

        Returns:
            A :class:`~weaver.types.Checkpoint` when *wait* is True, else
            an :class:`OperationHandle`.
        """
        body: Dict[str, Any] = {"type": checkpoint_type}
        if name is not None:
            body["name"] = name
        response = self._service.http.post(
            f"/api/v1/models/{self.model_id}/checkpoints",
            json=body,
        )
        response = response or {}
        checkpoint_data = lookup_case_insensitive(response, "checkpoint") or {}
        operation_data = lookup_case_insensitive(response, "operation") or response
        handle = build_operation_handle(self._service.http, operation_data)
        if not wait:
            return handle
        handle.wait()
        return Checkpoint.from_payload(checkpoint_data)

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

    def terminate(self, instance_types: list[str] | None = None) -> Dict[str, Any]:
        """Terminate trainer and/or inference instances for this model.

        Args:
            instance_types: List of instance types to terminate (e.g., ["trainer", "inference"]).
                          Defaults to both if not specified.

        Returns:
            Dictionary with termination results for each instance type
        """
        return self._service.terminate_model(self.model_id, instance_types)
