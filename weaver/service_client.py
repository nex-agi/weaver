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

"""High-level ServiceClient that manages sessions and child clients."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from . import __version__
from ._http import APIClient
from ._utils import extract_id, lookup_case_insensitive
from .config import WeaverConfig
from .operations import OperationHandle, build_operation_handle
from .types import LoraConfig

if TYPE_CHECKING:
    from .sampling_client import SamplingClient
    from .training_client import TrainingClient


logger = logging.getLogger(__name__)


# Default LoRA configuration
DEFAULT_LORA_CONFIG = LoraConfig(rank=32)


class ServiceClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Initialize ServiceClient.

        Args:
            base_url: Base URL of the Weaver server. Defaults to https://weaver-console.nex-agi.com
            api_key: API key for authentication (starts with 'sk-'). Get from admin UI at /api-keys
            default_tags: Default tags for sessions
            session_id: Optional existing session ID to reuse
            heartbeat_interval: Interval in seconds for session heartbeat
        """
        self._config = WeaverConfig.from_env(
            base_url=base_url,
            api_key=api_key,
        )
        self._default_tags = list(default_tags or ["weaver-sdk"])
        self._session_id = session_id
        self._heartbeat_interval = heartbeat_interval

        self._http: APIClient | None = None
        self._session: Dict[str, Any] | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop_event: threading.Event = threading.Event()
        self._closed = False
        self._model_seq_counter = 1
        self._sampling_seq_counter = 1
        self._operation_seq_by_model: Dict[str, int] = {}

    def __enter__(self) -> "ServiceClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def http(self) -> APIClient:
        if self._http is None:
            raise RuntimeError("ServiceClient is not connected")
        return self._http

    def connect(self) -> None:
        if self._http is not None:
            return
        self._http = APIClient(self._config)
        if self._session_id:
            self._fetch_session(self._session_id)
        else:
            self.ensure_session()
        self._start_heartbeat()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._heartbeat_thread:
            self._heartbeat_stop_event.set()
            self._heartbeat_thread.join(timeout=5.0)
        if self._http is not None:
            self._http.close()
        self._http = None

    def ensure_session(
        self,
        *,
        tags: Optional[Sequence[str]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._session is not None:
            return self._session
        payload = {
            "tags": list(tags or self._default_tags),
            "user_metadata": user_metadata or {},
            "sdk_version": __version__,
        }
        session = self.http.post("/api/v1/sessions", json=payload)
        self._session_id = extract_id(session)
        self._session = session
        return session  # type: ignore[return-value]

    def _fetch_session(self, session_id: str) -> None:
        session = self.http.get(f"/api/v1/sessions/{session_id}")
        self._session_id = extract_id(session)
        self._session = session

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread or not self._session_id:
            return
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        assert self._session_id is not None
        while not self._heartbeat_stop_event.is_set():
            try:
                self.http.post(f"/api/v1/sessions/{self._session_id}/heartbeat")
            except Exception as exc:  # pragma: no cover - best effort heartbeat
                logger.debug("session heartbeat failed: %s", exc)
            time.sleep(self._heartbeat_interval)

    @property
    def session_id(self) -> str:
        if not self._session_id:
            raise RuntimeError("Session not initialized yet")
        return self._session_id

    def create_model(
        self,
        *,
        base_model: str,
        model_seq_id: Optional[int] = None,
        lora_config: Union[LoraConfig, Dict[str, Any]] = DEFAULT_LORA_CONFIG,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> "TrainingClient":
        """Create a training model with LoRA configuration.

        Args:
            base_model: Base model name (e.g., "Qwen/Qwen3-8B")
            model_seq_id: Optional model sequence ID
            lora_config: LoRA configuration (default: LoraConfig(rank=32) with all layers enabled)
            user_metadata: Optional user metadata

        Returns:
            TrainingClient for the created model

        Examples:
            # Use default LoRA (rank=32, all layers enabled)
            client.create_model(base_model="Qwen/Qwen3-8B")

            # Custom LoRA configuration
            client.create_model(
                base_model="Qwen/Qwen3-8B",
                lora_config=LoraConfig(rank=16, seed=42)
            )

            # Attention-only LoRA
            client.create_model(
                base_model="Qwen/Qwen3-8B",
                lora_config=LoraConfig(
                    rank=32,
                    train_attn=True,
                    train_mlp=False,
                    train_unembed=False,
                )
            )
        """
        model_seq_id = model_seq_id or self._next_model_seq()
        payload: Dict[str, Any] = {
            "model_seq_id": model_seq_id,
            "base_model": base_model,
        }

        if isinstance(lora_config, LoraConfig):
            payload["lora_config"] = lora_config.to_payload()
        else:
            payload["lora_config"] = lora_config

        if user_metadata is not None:
            payload["user_metadata"] = user_metadata

        response = self.http.post(
            f"/api/v1/sessions/{self.session_id}/models",
            json=payload,
        )
        model_id = extract_id(response)
        from .training_client import TrainingClient  # avoid circular import

        return TrainingClient(
            service=self,
            model_id=model_id,
            base_model=lookup_case_insensitive(response, "base_model") or base_model,
            session_id=self.session_id,
        )

    def _next_model_seq(self) -> int:
        value = self._model_seq_counter
        self._model_seq_counter += 1
        return value

    def _next_sampling_seq(self) -> int:
        value = self._sampling_seq_counter
        self._sampling_seq_counter += 1
        return value

    def next_operation_seq(self, model_id: str) -> int:
        """Return the next seq_id for a given model, shared across clients."""

        if not model_id:
            raise ValueError("model_id is required to generate seq_id")
        current = self._operation_seq_by_model.get(model_id)
        if current is None:
            current = 1
        self._operation_seq_by_model[model_id] = current + 1
        return current

    def create_sampling_client(
        self,
        *,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None,
        sampling_session_seq_id: Optional[int] = None,
        sampling_session_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> "SamplingClient":
        from .sampling_client import SamplingClient  # local import to avoid cycles

        if sampling_session_id is None:
            if model_id and not model_path:
                raise ValueError("model_path is required when model_id is provided")
            seq_id = sampling_session_seq_id or self._next_sampling_seq()
            body = {
                "sampling_session_seq_id": seq_id,
                "base_model": base_model,
                "model_path": model_path,
            }
            if model_id:
                body["model_id"] = model_id

            session = self.http.post(
                f"/api/v1/sessions/{self.session_id}/sampling-sessions",
                json=body,
            )
            sampling_session_id = extract_id(session)
        return SamplingClient(
            service=self,
            sampling_session_id=sampling_session_id,
            base_model=base_model,
            model_path=model_path,
            model_id=model_id,
        )

    def get_sampling_client(
        self,
        model_path: str,
        *,
        base_model: Optional[str] = None,
        model_id: Optional[str] = None,
        sampling_session_id: Optional[str] = None,
    ) -> "SamplingClient":
        """Create a sampling client from an exported model path.

        This is a convenience wrapper around create_sampling_client with clearer naming
        for the common use case of loading weights from a path.

        Args:
            model_path: Path to the exported model weights
            base_model: Base model name (e.g., "llama-3-8b")
            model_id: Optional model ID to associate with this sampling session
            sampling_session_id: Optional existing sampling session ID to reuse

        Returns:
            Configured SamplingClient ready for inference
        """
        return self.create_sampling_client(
            model_path=model_path,
            base_model=base_model,
            model_id=model_id,
            sampling_session_id=sampling_session_id,
        )

    def enqueue_operation(self, path: str, payload: Dict[str, Any]) -> OperationHandle:
        response = self.http.post(path, json=payload)
        return build_operation_handle(self.http, response)

    def list_supported_models(self) -> List[str]:
        """Return supported model names exposed by the server."""
        payload = self.http.get("/api/v1/supported-models")
        print(f"payload: {payload}")
        if not isinstance(payload, dict):
            return []
        items = payload.get("items")
        names: List[str] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = lookup_case_insensitive(item, "name")
                status = lookup_case_insensitive(item, "status")
                if status and str(status).lower() != "healthy":
                    continue
                if name:
                    names.append(str(name))
        return names

    def list_training_runs(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List training runs with pagination.

        Args:
            limit: Maximum number of items to return (default: 25)
            offset: Number of items to skip (default: 0)

        Returns:
            Dictionary with 'items' (list of training runs) and 'pagination' info
        """
        params = {"limit": limit, "offset": offset}
        return self.http.get("/api/v1/training-runs", params=params)  # type: ignore[return-value]

    def get_training_run(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific training run.

        Args:
            run_id: The training run ID (model ID)

        Returns:
            Dictionary with training run details including checkpoints
        """
        return self.http.get(f"/api/v1/training-runs/{run_id}")  # type: ignore[return-value]

    def list_models(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List models with pagination.

        Args:
            limit: Maximum number of items to return (default: 25)
            offset: Number of items to skip (default: 0)

        Returns:
            Dictionary with 'items' (list of models) and 'pagination' info
        """
        params = {"limit": limit, "offset": offset}
        return self.http.get("/api/v1/models", params=params)  # type: ignore[return-value]

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.

        Args:
            model_id: The model ID

        Returns:
            Dictionary with model details
        """
        return self.http.get(f"/api/v1/models/{model_id}")  # type: ignore[return-value]
