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
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

from ._utils import lookup_case_insensitive
from .operations import OperationHandle
from .service_client import ServiceClient
from .types import AdamParams, Datum

if TYPE_CHECKING:
    from .sampling_client import SamplingClient

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .sampling_client import SamplingClient


class TrainingClient:
    def __init__(
        self, *, service: ServiceClient, model_id: str, base_model: str, session_id: str
    ) -> None:
        self._service = service
        self.model_id = model_id
        self.base_model = base_model
        self.session_id = session_id

    def _next_seq(self) -> int:
        return self._service.next_operation_seq(self.model_id)

    def _serialize_data(self, data: Sequence[Datum]) -> Sequence[Dict[str, Any]]:
        return [datum.to_payload() for datum in data]

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
            )
        if model_path:
            return self._service.get_sampling_client(
                model_path=str(model_path),
                base_model=self.base_model,
                model_id=self.model_id,
            )
        raise RuntimeError("Export response missing sampling session id or model path")

    @cached_property
    def tokenizer(self):  # type: ignore[misc]
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)

    def get_tokenizer(self):  # Backwards compatible accessor
        return self.tokenizer
