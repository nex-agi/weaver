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

"""Asyncio-native ServiceClient that manages sessions and child clients.

This is the asyncio twin of :class:`weaver.service_client.ServiceClient`. Every
network call is awaited so the event loop is free for other coroutines, and the
session heartbeat runs as an :class:`asyncio.Task` instead of a thread.

Usage::

    async with AsyncServiceClient(api_key="sk-...") as svc:
        tc = await svc.create_model(base_model="Qwen/Qwen3-8B")
        await tc.forward_backward(data, "cross_entropy")
        await tc.optim_step(AdamParams(learning_rate=1e-4))

Event loop model
----------------
The client **owns no event loop**. It never calls ``asyncio.run``,
``new_event_loop`` or ``run_until_complete`` — it is just coroutines plus an
``httpx.AsyncClient`` and runs on whatever loop is active when you ``await`` it.

**One client instance is bound to one event loop and one thread.** The binding
happens on the first call (``connect()``): the ``httpx.AsyncClient`` connection
pool and the heartbeat ``asyncio.Task`` both attach to the loop running then.
Do **not** share a single instance across loops or threads.

Integrating without hangs:

* **Inside an existing async app** (FastAPI / Starlette / aiohttp / Jupyter):
  you already have a running loop, so ``await`` the client directly — do **not**
  call ``asyncio.run`` (it raises "cannot be called from a running event loop").
  Create the client once at startup, reuse it for the app's lifetime on that
  loop, and ``await svc.aclose()`` on shutdown.
* **From synchronous code**: wrap a whole workflow in a single
  ``asyncio.run(main())``. If you must call repeatedly from sync code, either
  keep one long-lived loop on a dedicated thread and submit coroutines with
  ``asyncio.run_coroutine_threadsafe``, or build *and close* a client inside
  each ``asyncio.run`` — do not keep one instance and call ``asyncio.run`` with
  it repeatedly.
* **Always close** via ``async with`` or ``await svc.aclose()`` so the heartbeat
  task is cancelled before the loop closes (otherwise asyncio cancels it at
  shutdown and may log a "Task was destroyed but it is pending" warning).
* **Multiple threads**: give each thread its own loop and its own client, or
  marshal calls onto the owning loop with ``asyncio.run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from . import __version__
from ._async_http import AsyncAPIClient
from ._utils import extract_id, lookup_case_insensitive
from .config import WeaverConfig
from .operations import AsyncOperationHandle, build_async_operation_handle
from .types import LoraConfig

if TYPE_CHECKING:
    from .async_sampling_client import AsyncSamplingClient
    from .async_training_client import AsyncTrainingClient

logger = logging.getLogger(__name__)


# Default LoRA configuration
DEFAULT_LORA_CONFIG = LoraConfig(rank=32)


class AsyncServiceClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Initialize AsyncServiceClient.

        Args:
            base_url: Base URL of the Weaver server. Defaults to https://weaver-console.nex-agi.cn
            api_key: API key for authentication (starts with 'sk-'). Get from admin UI at /api-keys
            default_tags: Default tags for sessions
            session_id: Optional existing session ID to reuse
            heartbeat_interval: Interval in seconds for session heartbeat
        """
        self._config = WeaverConfig.from_env(base_url=base_url, api_key=api_key)
        self._default_tags = list(default_tags or ["weaver-sdk"])
        self._session_id = session_id
        self._heartbeat_interval = heartbeat_interval

        self._http: AsyncAPIClient | None = None
        self._session: Dict[str, Any] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._closed = False
        self._model_seq_counter = 1
        self._sampling_seq_counter = 1
        self._operation_seq_by_model: Dict[str, int] = {}
        self._created_models: List[str] = []

    async def __aenter__(self) -> "AsyncServiceClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @property
    def http(self) -> AsyncAPIClient:
        if self._http is None:
            raise RuntimeError("AsyncServiceClient is not connected")
        return self._http

    async def connect(self) -> None:
        if self._http is not None:
            return
        self._http = AsyncAPIClient(self._config)
        if self._session_id:
            await self._fetch_session(self._session_id)
        else:
            await self.ensure_session()
        self._start_heartbeat()

    async def _ensure_connected(self) -> None:
        if self._http is None:
            await self.connect()

    async def terminate_model(
        self,
        model_id: str,
        instance_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Terminate trainer and/or inference instances for a model."""
        payload: Dict[str, Any] = {}
        if instance_types is not None:
            payload["instance_types"] = instance_types
        return await self.http.post(
            f"/api/v1/models/{model_id}/terminate",
            json=payload if payload else None,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        for model_id in self._created_models:
            try:
                logger.debug("Terminating model %s during cleanup", model_id)
                await self.terminate_model(model_id)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("Failed to terminate model %s: %s", model_id, exc)

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):  # pragma: no cover - best effort
                pass
            self._heartbeat_task = None
        if self._http is not None:
            await self._http.aclose()
        self._http = None

    async def ensure_session(
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
        session = await self.http.post("/api/v1/sessions", json=payload)
        self._session_id = extract_id(session)
        self._session = session
        return session

    async def _fetch_session(self, session_id: str) -> None:
        session = await self.http.get(f"/api/v1/sessions/{session_id}")
        self._session_id = extract_id(session)
        self._session = session

    def _start_heartbeat(self) -> None:
        if self._heartbeat_task is not None or not self._session_id:
            return
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        assert self._session_id is not None
        while True:
            try:
                await self.http.post(f"/api/v1/sessions/{self._session_id}/heartbeat")
            except Exception as exc:  # pragma: no cover - best effort heartbeat
                logger.debug("session heartbeat failed: %s", exc)
            await asyncio.sleep(self._heartbeat_interval)

    @property
    def session_id(self) -> str:
        if not self._session_id:
            raise RuntimeError("Session not initialized yet")
        return self._session_id

    async def create_model(
        self,
        *,
        base_model: str,
        model_seq_id: Optional[int] = None,
        training_mode: Optional[str] = None,
        lora_config: Union[LoraConfig, Dict[str, Any]] = DEFAULT_LORA_CONFIG,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> "AsyncTrainingClient":
        """Create a training model with LoRA or FullFT configuration.

        See :meth:`weaver.service_client.ServiceClient.create_model`. Returns an
        :class:`AsyncTrainingClient`.
        """
        await self._ensure_connected()
        model_seq_id = model_seq_id or self._next_model_seq()
        payload: Dict[str, Any] = {
            "model_seq_id": model_seq_id,
            "base_model": base_model,
        }
        if training_mode is not None:
            payload["training_mode"] = training_mode
        if training_mode is None or training_mode == "lora":
            payload["lora_config"] = (
                lora_config.to_payload() if isinstance(lora_config, LoraConfig) else lora_config
            )
        if user_metadata is not None:
            payload["user_metadata"] = user_metadata

        response = await self.http.post(
            f"/api/v1/sessions/{self.session_id}/models",
            json=payload,
        )
        model_id = extract_id(response)
        self._created_models.append(model_id)

        from .async_training_client import AsyncTrainingClient  # avoid circular import

        tokenizer_path = lookup_case_insensitive(response, "tokenizer_path")
        if not tokenizer_path:
            model_config = await self.get_supported_model_config(base_model)
            if model_config:
                config = model_config.get("config", {})
                resource = config.get("resource", {})
                tokenizer_config = resource.get("tokenizer", {})
                tokenizer_path = tokenizer_config.get("path")

        debug_info = lookup_case_insensitive(response, "debug_info")

        return AsyncTrainingClient(
            service=self,
            model_id=model_id,
            base_model=lookup_case_insensitive(response, "base_model") or base_model,
            session_id=self.session_id,
            tokenizer_path=tokenizer_path,
            debug_info=debug_info,
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

    async def create_sampling_client(
        self,
        *,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None,
        sampling_session_seq_id: Optional[int] = None,
        sampling_session_id: Optional[str] = None,
        model_id: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> "AsyncSamplingClient":
        from .async_sampling_client import AsyncSamplingClient  # local import to avoid cycles

        await self._ensure_connected()
        if sampling_session_id is None:
            if model_id and not model_path:
                raise ValueError("model_path is required when model_id is provided")
            seq_id = sampling_session_seq_id or self._next_sampling_seq()
            body: Dict[str, Any] = {
                "sampling_session_seq_id": seq_id,
                "base_model": base_model,
                "model_path": model_path,
            }
            if model_id:
                body["model_id"] = model_id

            resp = await self.http.post(
                f"/api/v1/sessions/{self.session_id}/sampling-sessions",
                json=body,
            )

            # Handle async sync_weights response (202 Accepted):
            # Response shape: {"sampling_session": {...}, "sync_operation": {...}}
            sync_op_payload = (
                lookup_case_insensitive(resp, "sync_operation") if isinstance(resp, dict) else None
            )
            if sync_op_payload and isinstance(sync_op_payload, dict):
                session = lookup_case_insensitive(resp, "sampling_session") or {}
                sync_handle = AsyncOperationHandle.from_payload(self.http, sync_op_payload)
                logger.info(
                    "Waiting for background weights sync (operation %s)...",
                    sync_handle.operation_id,
                )
                await sync_handle.wait()
                logger.info("Weights sync completed.")
            else:
                session = resp

            sampling_session_id = extract_id(session)
            if tokenizer_path is None:
                tokenizer_path = lookup_case_insensitive(session, "tokenizer_path")
            if tokenizer_path is None and base_model:
                model_config = await self.get_supported_model_config(base_model)
                if model_config:
                    config = model_config.get("config", {})
                    resource = config.get("resource", {})
                    tokenizer_config = resource.get("tokenizer", {})
                    tokenizer_path = tokenizer_config.get("path")
        return AsyncSamplingClient(
            service=self,
            sampling_session_id=sampling_session_id,
            base_model=base_model,
            model_path=model_path,
            model_id=model_id,
            tokenizer_path=tokenizer_path,
        )

    async def get_sampling_client(
        self,
        model_path: str,
        *,
        base_model: Optional[str] = None,
        model_id: Optional[str] = None,
        sampling_session_id: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> "AsyncSamplingClient":
        """Create a sampling client from an exported model path."""
        return await self.create_sampling_client(
            model_path=model_path,
            base_model=base_model,
            model_id=model_id,
            sampling_session_id=sampling_session_id,
            tokenizer_path=tokenizer_path,
        )

    async def enqueue_operation(self, path: str, payload: Dict[str, Any]) -> AsyncOperationHandle:
        # max_retries=1: operations like save_state are non-idempotent POSTs —
        # retrying after a timeout would create duplicate server-side operations.
        response = await self.http.post(path, json=payload, max_retries=1)
        return build_async_operation_handle(self.http, response)

    async def get_supported_model_config(self, base_model: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific supported model."""
        payload = await self.http.get("/api/v1/supported-models")
        if not isinstance(payload, dict):
            return None
        items = payload.get("items")
        if not isinstance(items, list):
            return None
        for item in items:
            if not isinstance(item, dict):
                continue
            name = lookup_case_insensitive(item, "name")
            if name and str(name) == base_model:
                return item
        return None

    async def list_supported_models(self) -> List[str]:
        """Return supported model names exposed by the server."""
        payload = await self.http.get("/api/v1/supported-models")
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

    async def list_training_runs(self, *, limit: int = 25, offset: int = 0) -> Dict[str, Any]:
        """List training runs with pagination."""
        return await self.http.get(
            "/api/v1/training-runs", params={"limit": limit, "offset": offset}
        )

    async def get_training_run(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific training run."""
        return await self.http.get(f"/api/v1/training-runs/{run_id}")

    async def list_models(self, *, limit: int = 25, offset: int = 0) -> Dict[str, Any]:
        """List models with pagination."""
        return await self.http.get("/api/v1/models", params={"limit": limit, "offset": offset})

    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        return await self.http.get(f"/api/v1/models/{model_id}")
