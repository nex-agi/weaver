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
  shutdown and may log a "Task was destroyed but it is pending" warning), and so
  models created through this client are terminated promptly (see below).
* **Multiple threads**: give each thread its own loop and its own client, or
  marshal calls onto the owning loop with ``asyncio.run_coroutine_threadsafe``.

atexit safety net
~~~~~~~~~~~~~~~~~
For parity with the sync ``ServiceClient``, ``connect()`` registers an
``atexit`` handler that terminates any models this client created if the process
exits without an explicit ``aclose()``. Because the owning loop is usually
already closed at interpreter shutdown, the handler creates a **throwaway** event
loop and a fresh ``AsyncAPIClient`` solely to fire the terminate requests — the
library still never runs coroutines on the caller's loop. Prefer ``async with``
or ``await svc.aclose()``: the atexit path is only a best-effort backstop and,
like the sync client, does not cover ``SIGKILL`` (the server-side reaper does).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

from . import __version__
from ._async_http import AsyncAPIClient
from ._utils import extract_id, lookup_case_insensitive, optional_scope_id
from .config import WeaverConfig
from .operations import AsyncOperationHandle, build_async_operation_handle
from .types import LoraConfig

if TYPE_CHECKING:
    from .async_sampling_client import AsyncSamplingClient
    from .async_training_client import AsyncTrainingClient

logger = logging.getLogger(__name__)


# Default LoRA configuration
DEFAULT_LORA_CONFIG = LoraConfig(rank=32)


class AsyncServiceClient:  # pylint: disable=too-many-public-methods
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        labels: Optional[Mapping[str, str]] = None,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Initialize AsyncServiceClient.

        Args:
            base_url: Base URL of the Weaver server. Defaults to https://weaver-console.nex-agi.cn
            api_key: API key for authentication (starts with 'sk-'). Get from admin UI at /api-keys
            default_tags: Default tags for sessions
            session_id: Optional existing session ID to reuse
            name: Optional experiment display name for a newly created Session
            labels: Optional searchable string metadata for a newly created Session
            organization_id: Optional Organization that owns a newly created session
            project_id: Optional Project used for a newly created session
            organization: Optional organization UUID, globally unique slug, or display name
            project: Optional project UUID, organization-local slug, or display name
            user_metadata: Metadata attached when a new session is created
            heartbeat_interval: Interval in seconds for session heartbeat
        """
        self._config = WeaverConfig.from_env(base_url=base_url, api_key=api_key)
        self._default_tags = list(default_tags or ["weaver-sdk"])
        self._session_id = session_id
        self._session_name = name.strip() if name and name.strip() else None
        self._session_labels = dict(labels or {})
        self._organization_id = optional_scope_id(organization_id, "WEAVER_ORGANIZATION_ID")
        self._project_id = optional_scope_id(project_id, "WEAVER_PROJECT_ID")
        self._organization_reference = optional_scope_id(organization, "WEAVER_ORGANIZATION")
        self._project_reference = optional_scope_id(project, "WEAVER_PROJECT")
        self._session_user_metadata = dict(user_metadata or {})
        self._heartbeat_interval = heartbeat_interval

        self._http: AsyncAPIClient | None = None
        self._session: Dict[str, Any] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._closed = False
        self._model_seq_counter = 1
        self._sampling_seq_counter = 1
        self._operation_seq_by_model: Dict[str, int] = {}
        self._created_models: List[str] = []
        self._atexit_registered = False

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

    async def connect(self, *, ensure_session: bool = True) -> None:
        """Connect, optionally without creating or fetching a Session."""

        if self._http is None:
            self._http = AsyncAPIClient(self._config)
        if not ensure_session or self._session is not None:
            return
        if self._session_id:
            await self._fetch_session(self._session_id)
        else:
            await self.ensure_session()
        self._start_heartbeat()
        self._register_atexit()

    async def _ensure_connected(self) -> None:
        if self._http is None or self._session is None:
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

    def _register_atexit(self) -> None:
        """Register a best-effort atexit safety net (sync-client parity).

        Mirrors ``ServiceClient.connect()``'s ``atexit.register(self.close)``: on
        interpreter shutdown, models created through this client are terminated
        even if the caller forgot to ``await aclose()`` or use ``async with``.
        """
        if self._atexit_registered:
            return
        atexit.register(self._atexit_terminate_created_models)
        self._atexit_registered = True

    def _atexit_terminate_created_models(self) -> None:
        """Terminate created models at interpreter shutdown (best-effort).

        Runs from ``atexit``, where the client's owning event loop is typically
        already closed, so we cannot ``await aclose()`` on it. Instead we spin up
        a throwaway loop with a fresh HTTP client and fire the terminate
        requests. This is the async twin of the sync client's
        ``atexit.register(self.close)`` safety net. It does not cover ``SIGKILL``;
        the server-side reaper remains the backstop for that.
        """
        if self._closed or not self._created_models:
            return
        self._closed = True
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._terminate_created_models_isolated())
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.debug("atexit model cleanup failed: %s", exc)

    async def _terminate_created_models_isolated(self) -> None:
        """Terminate every created model on a fresh HTTP client.

        Used only by the atexit safety net: the original ``AsyncAPIClient`` is
        bound to the (now closed) owning loop, so a fresh client is created on
        the throwaway loop to issue the terminate requests. Reuses
        :meth:`terminate_model` so the request logic is not forked.
        """
        self._http = AsyncAPIClient(self._config)
        try:
            for model_id in list(self._created_models):
                try:
                    logger.debug("Terminating model %s during atexit cleanup", model_id)
                    await self.terminate_model(model_id)
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.debug("Failed to terminate model %s: %s", model_id, exc)
        finally:
            try:
                await self._http.aclose()
            finally:
                self._http = None

    async def aclose(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_terminate_created_models)
            self._atexit_registered = False
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
        name: Optional[str] = None,
        labels: Optional[Mapping[str, str]] = None,
        tags: Optional[Sequence[str]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._session is not None:
            return self._session
        organization_id = self._organization_id
        project_id = self._project_id
        if (organization_id is None and self._organization_reference) or (
            project_id is None and self._project_reference
        ):
            resolved = await self.resolve_scope(
                organization_id or self._organization_reference,
                project_id or self._project_reference,
            )
            if organization_id is None:
                organization = resolved.get("organization")
                if not isinstance(organization, dict):
                    raise ValueError("Scope response missing organization")
                organization_id = extract_id(organization)
            if project_id is None:
                project = resolved.get("project")
                if not isinstance(project, dict):
                    raise ValueError("Scope response missing project")
                project_id = extract_id(project)

        payload = {
            "tags": list(tags or self._default_tags),
            "user_metadata": (
                user_metadata if user_metadata is not None else self._session_user_metadata
            ),
            "sdk_version": __version__,
        }
        session_name = self._session_name if name is None else name.strip()
        session_labels = self._session_labels if labels is None else dict(labels)
        if session_name:
            payload["name"] = session_name
        if session_labels:
            payload["labels"] = dict(session_labels)
        if organization_id:
            payload["organization_id"] = organization_id
        if project_id:
            payload["project_id"] = project_id
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
        performance_tier: Optional[str] = None,
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
        if performance_tier is not None:
            payload["performance_tier"] = performance_tier

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

    async def create_training_client(self, **kwargs: Any) -> "AsyncTrainingClient":
        """Create a Training Run client; compatibility alias for create_model."""

        # The required base_model is supplied through kwargs by this compatibility
        # alias; create_model performs the normal runtime validation.
        # pylint: disable-next=missing-kwoa
        return await self.create_model(**kwargs)

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

    async def _supported_model_scope_params(self) -> Dict[str, str]:
        organization_id = self._organization_id
        if organization_id is None and isinstance(self._session, dict):
            raw_id = lookup_case_insensitive(self._session, "organization_id")
            organization_id = str(raw_id).strip() if raw_id else None
        if organization_id is None and (
            self._organization_reference or self._project_id or self._project_reference
        ):
            scope = await self.resolve_scope(
                self._organization_reference,
                self._project_id or self._project_reference,
            )
            organization = scope.get("organization")
            if isinstance(organization, dict):
                raw_id = lookup_case_insensitive(organization, "id")
                organization_id = str(raw_id).strip() if raw_id else None
        return {"organization_id": organization_id} if organization_id else {}

    async def _list_supported_model_records(self) -> List[Dict[str, Any]]:
        """Traverse the role-filtered supported-model collection."""

        records: List[Dict[str, Any]] = []
        limit = 100
        offset = 0
        scope = await self._supported_model_scope_params()
        while True:
            params: Dict[str, Any] = {"limit": limit, "offset": offset, **scope}
            payload = await self.http.get("/api/v1/supported-models", params=params)
            if not isinstance(payload, dict):
                break
            items = payload.get("items")
            if not isinstance(items, list):
                break
            page = [item for item in items if isinstance(item, dict)]
            records.extend(page)
            pagination = payload.get("pagination")
            if not isinstance(pagination, dict):
                break
            total = pagination.get("total_count")
            try:
                total_count = int(str(total))
            except (TypeError, ValueError):
                break
            offset += len(items)
            if not items or offset >= total_count:
                break
        return records

    async def get_supported_model_config(self, base_model: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific supported model."""
        for item in await self._list_supported_model_records():
            name = lookup_case_insensitive(item, "name")
            if name and str(name) == base_model:
                return item
        return None

    async def list_supported_models(self) -> List[str]:
        """Return usable model names exposed to the authenticated role."""

        names: List[str] = []
        for item in await self._list_supported_model_records():
            name = lookup_case_insensitive(item, "name")
            status = lookup_case_insensitive(item, "status")
            if status and str(status).lower() not in {"healthy", "available"}:
                continue
            if name:
                names.append(str(name))
        return names

    async def list_organizations(self) -> List[Dict[str, Any]]:
        """List organizations available to the authenticated user."""

        payload = await self.http.get("/api/v1/organizations")
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    async def resolve_scope(
        self,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve UUID/slug/name references to canonical scope IDs."""

        params: Dict[str, str] = {}
        organization_ref = organization.strip() if organization else None
        project_ref = project.strip() if project else None
        if organization_ref:
            params["organization"] = organization_ref
        if project_ref:
            params["project"] = project_ref
        payload = await self.http.get("/api/v1/scope/resolve", params=params)
        return payload if isinstance(payload, dict) else {}

    async def list_projects(self, organization_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List projects, falling back to the user's stable default organization."""

        resolved = optional_scope_id(organization_id, "WEAVER_ORGANIZATION_ID")
        if resolved is None:
            resolved = self._organization_id
        if resolved is None:
            scope = await self.resolve_scope()
            organization = scope.get("organization")
            if not isinstance(organization, dict):
                return []
            resolved = str(organization.get("id", "")).strip() or None
        if resolved is None:
            return []
        payload = await self.http.get(f"/api/v1/organizations/{resolved}/projects")
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _quota_scope_params(self, organization_id: Optional[str]) -> Dict[str, str]:
        resolved = optional_scope_id(organization_id, "WEAVER_ORGANIZATION_ID")
        if resolved is None:
            resolved = self._organization_id
        return {"org_id": resolved} if resolved is not None else {}

    async def get_quota_balance(self, organization_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the caller's nano-USD quota balance."""

        scope = self._quota_scope_params(organization_id)
        if scope:
            payload = await self.http.get("/api/v1/quota/balance", params=scope)
        else:
            payload = await self.http.get("/api/v1/quota/balance")
        return payload if isinstance(payload, dict) else {}

    async def list_quota_requests(
        self,
        organization_id: Optional[str] = None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List quota requests submitted by the current user."""

        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        params.update(self._quota_scope_params(organization_id))
        payload = await self.http.get("/api/v1/quota/requests", params=params)
        return payload if isinstance(payload, dict) else {}

    async def request_quota(
        self,
        amount_usd: str,
        *,
        reason: str,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request additional USD quota without losing nano-USD precision."""

        payload = await self.http.post(
            "/api/v1/quota/requests",
            json={"amount_usd": str(amount_usd), "reason": reason},
            params=self._quota_scope_params(organization_id) or None,
            max_retries=1,
        )
        return payload if isinstance(payload, dict) else {}

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
