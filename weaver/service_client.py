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

# The client is the SDK's single entry point, so it aggregates every resource
# family (sessions, models, checkpoints, artifacts, deployments) by design.
# pylint: disable=too-many-lines

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    overload,
)

import httpx

from . import __version__
from ._artifacts import (
    ARTIFACT_KINDS,
    DOWNLOAD_MAX_TRANSPORT_RETRIES,
    DOWNLOAD_MAX_URL_REFRESHES,
    ArtifactFile,
    check_downloaded_file,
    check_downloaded_file_at,
    descriptor_files,
    ensure_within_directory,
    is_file_already_complete,
    is_file_already_complete_at,
    parse_download_target,
    resolve_checkpoint_id_from_listing,
    resume_offset_at,
    select_artifact_payload,
    validate_resource_id,
)
from ._deployments import (
    DEPLOYMENT_PAGE_SIZE,
    deployment_items,
    next_page_offset,
    translate_deployment_error,
)
from ._http import (
    APIClient,
    DownloadURLExpiredError,
    WeaverAPIError,
    build_download_client,
    compute_retry_delay,
    stream_download_to_file,
)
from ._safeio import (
    legacy_open_for_write,
    open_for_write,
    open_parent_fd,
    rename_within,
    supports_dir_fd,
)
from ._utils import extract_id, lookup_case_insensitive, optional_scope_id
from .config import TensorCompression, TensorTransport, WeaverConfig
from .operations import OperationHandle, build_operation_handle
from .tensor_transport import TensorPack
from .types import LoraConfig
from .types.deployment import Deployment
from .types.weights_artifact import WeightsArtifact

if TYPE_CHECKING:
    from typing import Literal

    from .sampling_client import SamplingClient
    from .training_client import TrainingClient


logger = logging.getLogger(__name__)


# Default LoRA configuration
DEFAULT_LORA_CONFIG = LoraConfig(rank=32)


class ServiceClient:  # pylint: disable=too-many-public-methods
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
        tensor_transport: TensorTransport | None = None,
        tensor_compression: TensorCompression | None = None,
    ) -> None:
        """Initialize ServiceClient.

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
            tensor_transport: Training tensor transport. Defaults to
                ``WEAVER_TENSOR_TRANSPORT`` or ``"default"``.
            tensor_compression: HTTP tensor-pack compression. Defaults to
                ``WEAVER_TENSOR_COMPRESSION`` or ``"zstd"``.
        """
        self._config = WeaverConfig.from_env(
            base_url=base_url,
            api_key=api_key,
            tensor_transport=tensor_transport,
            tensor_compression=tensor_compression,
        )
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

        self._http: APIClient | None = None
        self._session: Dict[str, Any] | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop_event: threading.Event = threading.Event()
        self._closed = False
        self._model_seq_counter = 1
        self._sampling_seq_counter = 1
        self._operation_seq_by_model: Dict[str, int] = {}
        self._created_models: List[str] = []  # Track created model IDs for cleanup

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

    @property
    def tensor_transport(self) -> TensorTransport:
        """Configured transport for dense training tensors."""

        return self._config.tensor_transport

    @property
    def tensor_compression(self) -> TensorCompression:
        """Configured HTTP tensor-pack compression."""

        return self._config.tensor_compression

    def connect(self, *, ensure_session: bool = True) -> None:
        """Connect, optionally without creating or fetching a Session."""

        if self._http is None:
            self._http = APIClient(self._config)
            atexit.register(self.close)
        if not ensure_session or self._session is not None:
            return
        if self._session_id:
            self._fetch_session(self._session_id)
        else:
            self.ensure_session()
        self._start_heartbeat()

    def terminate_model(
        self,
        model_id: str,
        instance_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Terminate trainer and/or inference instances for a model.

        Args:
            model_id: The model ID to terminate
            instance_types: List of instance types to terminate (e.g., ["trainer", "inference"]).
                          Defaults to both if not specified.

        Returns:
            Dictionary with termination results for each instance type
        """
        payload: Dict[str, Any] = {}
        if instance_types is not None:
            payload["instance_types"] = instance_types

        return self.http.post(
            f"/api/v1/models/{model_id}/terminate",
            json=payload if payload else None,
        )  # type: ignore[return-value]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Terminate all created models before closing
        for model_id in self._created_models:
            try:
                logger.debug("Terminating model %s during cleanup", model_id)
                self.terminate_model(model_id)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("Failed to terminate model %s: %s", model_id, exc)

        if self._heartbeat_thread:
            self._heartbeat_stop_event.set()
            self._heartbeat_thread.join(timeout=5.0)
        if self._http is not None:
            self._http.close()
        self._http = None

    def ensure_session(
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
            resolved = self.resolve_scope(
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
        training_mode: Optional[str] = None,
        lora_config: Union[LoraConfig, Dict[str, Any]] = DEFAULT_LORA_CONFIG,
        user_metadata: Optional[Dict[str, Any]] = None,
        performance_tier: Optional[str] = None,
    ) -> "TrainingClient":
        """Create a training model with LoRA or FullFT configuration.

        Args:
            base_model: Base model name (e.g., "Qwen/Qwen3-8B"). The maximum sequence length
                is encoded in the name: a long-context variant uses a ``:<max_seq_len>`` suffix
                (e.g. "Qwen/Qwen3-8B:262144"). Pick the variant whose context fits your
                workload rather than passing a separate length parameter.
            model_seq_id: Optional model sequence ID
            training_mode: Training mode - "lora" or "full_ft" (default: None -> server defaults to "lora")
            lora_config: LoRA configuration (default: LoraConfig(rank=32) with all layers enabled)
            full_ft_config: Full fine-tuning config dict (optional, for full_ft mode only)
            user_metadata: Optional user metadata
            performance_tier: Optional throughput tier selecting how much parallelism / data
                parallelism the server provisions. Higher tiers deliver proportionally more
                throughput at proportionally higher price (e.g. "fast" ~= 2x the throughput
                and 2x the price of "normal"). Recognized values: "normal", "fast", "flash".
                Defaults to the server default tier when omitted.

        Note:
            ``performance_tier`` is optional: when omitted, behavior is unchanged for existing
            users and the server applies its default tier. The value is validated by the
            server, which rejects unsupported tiers with HTTP 400; such errors surface via
            WeaverAPIError.

        Returns:
            TrainingClient for the created model

        Examples:
            # Use default LoRA (rank=32, all layers enabled)
            client.create_model(base_model="Qwen/Qwen3-8B")

            # Custom LoRA configuration
            client.create_model(
                base_model="Qwen/Qwen3-8B",
                training_mode="lora",
                lora_config=LoraConfig(rank=16, seed=42)
            )

            # Long-context (256k) variant, full fine-tuning, fast throughput tier
            client.create_model(
                base_model="Qwen/Qwen3-8B:262144",
                training_mode="full_ft",
                performance_tier="fast",
            )
        """
        model_seq_id = model_seq_id or self._next_model_seq()
        payload: Dict[str, Any] = {
            "model_seq_id": model_seq_id,
            "base_model": base_model,
        }

        if training_mode is not None:
            payload["training_mode"] = training_mode

        # If training_mode is omitted (None), the server defaults to "lora", so include lora_config.
        if training_mode is None or training_mode == "lora":
            payload["lora_config"] = (
                lora_config.to_payload() if isinstance(lora_config, LoraConfig) else lora_config
            )

        if user_metadata is not None:
            payload["user_metadata"] = user_metadata

        # Optional throughput tier passed through for the server to plan parallelism and
        # pricing. Omitted -> not sent, preserving today's behavior for existing users; the
        # server owns validation (unsupported tier -> HTTP 400 -> WeaverAPIError).
        if performance_tier is not None:
            payload["performance_tier"] = performance_tier

        response = self.http.post(
            f"/api/v1/sessions/{self.session_id}/models",
            json=payload,
        )
        model_id = extract_id(response)

        # Track created models for cleanup
        self._created_models.append(model_id)

        from .training_client import TrainingClient  # avoid circular import

        # Extract tokenizer_path from response if provided by server
        tokenizer_path = lookup_case_insensitive(response, "tokenizer_path")

        # If not in response, try to get it from supported models config
        if not tokenizer_path:
            model_config = self.get_supported_model_config(base_model)
            if model_config:
                config = model_config.get("config", {})
                resource = config.get("resource", {})
                tokenizer_config = resource.get("tokenizer", {})
                tokenizer_path = tokenizer_config.get("path")

        # Extract debug_info for manual debug mode
        debug_info = lookup_case_insensitive(response, "debug_info")

        return TrainingClient(
            service=self,
            model_id=model_id,
            base_model=lookup_case_insensitive(response, "base_model") or base_model,
            session_id=self.session_id,
            tokenizer_path=tokenizer_path,
            debug_info=debug_info,
        )

    def create_training_client(self, **kwargs: Any) -> "TrainingClient":
        """Create a Training Run client.

        This is the canonical product name for :meth:`create_model`. The old
        method remains supported for compatibility with existing recipes.
        """

        return self.create_model(**kwargs)

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
        tokenizer_path: Optional[str] = None,
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

            resp = self.http.post(
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
                created_sampling_session_id = extract_id(session)
                # Wait for the background sync_weights to finish
                sync_handle = OperationHandle.from_payload(self.http, sync_op_payload)
                logger.info(
                    "Waiting for background weights sync (operation %s)...",
                    sync_handle.operation_id,
                )
                try:
                    sync_handle.wait()
                except BaseException:
                    # Creating a sampling client is transactional from the SDK's
                    # point of view.  If waiting is interrupted or the sync
                    # fails, delete the remote sampling session so its pending
                    # sync cannot be redispatched into a later cold-start pool.
                    if created_sampling_session_id:
                        try:
                            self.http.delete(
                                f"/api/v1/sampling-sessions/{created_sampling_session_id}"
                            )
                        except Exception as cleanup_exc:  # pragma: no cover - best effort
                            logger.warning(
                                "Failed to clean up sampling session %s after weights sync "
                                "failure: %s",
                                created_sampling_session_id,
                                cleanup_exc,
                            )
                    raise
                logger.info("Weights sync completed.")
            else:
                # Standard 201 response: body is the SamplingSession directly
                session = resp

            sampling_session_id = extract_id(session)
            # Extract tokenizer_path from response if provided by server
            if tokenizer_path is None:
                tokenizer_path = lookup_case_insensitive(session, "tokenizer_path")

            # If still not found and base_model is provided, try to get it from supported models config
            if tokenizer_path is None and base_model:
                model_config = self.get_supported_model_config(base_model)
                if model_config:
                    config = model_config.get("config", {})
                    resource = config.get("resource", {})
                    tokenizer_config = resource.get("tokenizer", {})
                    tokenizer_path = tokenizer_config.get("path")
        return SamplingClient(
            service=self,
            sampling_session_id=sampling_session_id,
            base_model=base_model,
            model_path=model_path,
            model_id=model_id,
            tokenizer_path=tokenizer_path,
        )

    def get_sampling_client(
        self,
        model_path: str,
        *,
        base_model: Optional[str] = None,
        model_id: Optional[str] = None,
        sampling_session_id: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> "SamplingClient":
        """Create a sampling client from an exported model path.

        This is a convenience wrapper around create_sampling_client with clearer naming
        for the common use case of loading weights from a path.

        Args:
            model_path: Path to the exported model weights
            base_model: Base model name (e.g., "llama-3-8b")
            model_id: Optional model ID to associate with this sampling session
            sampling_session_id: Optional existing sampling session ID to reuse
            tokenizer_path: Optional custom tokenizer path

        Returns:
            Configured SamplingClient ready for inference
        """
        return self.create_sampling_client(
            model_path=model_path,
            base_model=base_model,
            model_id=model_id,
            sampling_session_id=sampling_session_id,
            tokenizer_path=tokenizer_path,
        )

    def enqueue_operation(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        tensor_pack: TensorPack | None = None,
    ) -> OperationHandle:
        if tensor_pack is None:
            response = self.http.post(path, json=payload, max_retries=1)
        else:
            response = self.http.post_tensor_multipart(
                path,
                request=payload,
                tensor_pack=tensor_pack,
            )
        return build_operation_handle(self.http, response)

    def _supported_model_scope_params(self) -> Dict[str, str]:
        organization_id = self._organization_id
        if organization_id is None and isinstance(self._session, dict):
            raw_id = lookup_case_insensitive(self._session, "organization_id")
            organization_id = str(raw_id).strip() if raw_id else None
        if organization_id is None and (
            self._organization_reference or self._project_id or self._project_reference
        ):
            scope = self.resolve_scope(
                self._organization_reference,
                self._project_id or self._project_reference,
            )
            organization = scope.get("organization")
            if isinstance(organization, dict):
                raw_id = lookup_case_insensitive(organization, "id")
                organization_id = str(raw_id).strip() if raw_id else None
        return {"organization_id": organization_id} if organization_id else {}

    def _list_supported_model_records(self) -> List[Dict[str, Any]]:
        """Traverse the role-filtered supported-model collection."""

        records: List[Dict[str, Any]] = []
        limit = 100
        offset = 0
        scope = self._supported_model_scope_params()
        while True:
            params: Dict[str, Any] = {"limit": limit, "offset": offset, **scope}
            payload = self.http.get("/api/v1/supported-models", params=params)
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

    def list_supported_models(self) -> List[str]:
        """Return usable model names exposed to the authenticated role."""

        names: List[str] = []
        for item in self._list_supported_model_records():
            name = lookup_case_insensitive(item, "name")
            status = lookup_case_insensitive(item, "status")
            if status and str(status).lower() not in {"healthy", "available"}:
                continue
            if name:
                names.append(str(name))
        return names

    def list_organizations(self) -> List[Dict[str, Any]]:
        """List organizations available to the authenticated user."""

        payload = self.http.get("/api/v1/organizations")
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def resolve_scope(
        self,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve UUID/slug/name references to canonical organization/project IDs.

        Empty references are omitted, preserving the server's stable personal
        organization and default-project fallback. Ambiguous display names are
        surfaced as the server's 409 ``ambiguous_scope_reference`` error.
        """

        params: Dict[str, str] = {}
        organization_ref = organization.strip() if organization else None
        project_ref = project.strip() if project else None
        if organization_ref:
            params["organization"] = organization_ref
        if project_ref:
            params["project"] = project_ref
        payload = self.http.get("/api/v1/scope/resolve", params=params)
        return payload if isinstance(payload, dict) else {}

    def list_projects(self, organization_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List projects, falling back to the user's stable default organization."""

        resolved = optional_scope_id(organization_id, "WEAVER_ORGANIZATION_ID")
        if resolved is None:
            resolved = self._organization_id
        if resolved is None:
            scope = self.resolve_scope()
            organization = scope.get("organization")
            if not isinstance(organization, dict):
                return []
            resolved = str(organization.get("id", "")).strip() or None
        if resolved is None:
            return []
        payload = self.http.get(f"/api/v1/organizations/{resolved}/projects")
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _quota_scope_params(self, organization_id: Optional[str]) -> Dict[str, str]:
        resolved = optional_scope_id(organization_id, "WEAVER_ORGANIZATION_ID")
        if resolved is None:
            resolved = self._organization_id
        return {"org_id": resolved} if resolved is not None else {}

    def get_quota_balance(self, organization_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the caller's nano-USD quota balance.

        When no organization is supplied by parameter, constructor, or
        environment, the unscoped endpoint lets the server select the user's
        default organization.
        """

        scope = self._quota_scope_params(organization_id)
        if scope:
            payload = self.http.get("/api/v1/quota/balance", params=scope)
        else:
            payload = self.http.get("/api/v1/quota/balance")
        return payload if isinstance(payload, dict) else {}

    def list_quota_requests(
        self,
        organization_id: Optional[str] = None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List quota requests submitted by the current user."""

        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        params.update(self._quota_scope_params(organization_id))
        payload = self.http.get("/api/v1/quota/requests", params=params)
        return payload if isinstance(payload, dict) else {}

    def request_quota(
        self,
        amount_usd: str,
        *,
        reason: str,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request additional USD quota for the current user.

        ``amount_usd`` is sent as a decimal string; callers should not convert
        it to ``float`` because the server accepts up to nano-USD precision.
        """

        payload = self.http.post(
            "/api/v1/quota/requests",
            json={"amount_usd": str(amount_usd), "reason": reason},
            params=self._quota_scope_params(organization_id) or None,
            max_retries=1,
        )
        return payload if isinstance(payload, dict) else {}

    def get_supported_model_config(self, base_model: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific supported model.

        Args:
            base_model: Base model name to look up

        Returns:
            Model configuration dict if found, None otherwise
        """
        for item in self._list_supported_model_records():
            name = lookup_case_insensitive(item, "name")
            if name and str(name) == base_model:
                return item
        return None

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

    # ------------------------------------------------------------------
    # HF weights download
    # ------------------------------------------------------------------

    def download_weights(
        self,
        target: str | WeightsArtifact,
        dest: str | Path,
        *,
        kind: str | None = None,
        verify: bool = True,
        max_concurrency: int = 4,
    ) -> Path:
        """Download an exported HF weights artifact to a local directory.

        Fetches the artifact's download descriptor and streams every file to
        *dest* in parallel. Each file is written atomically (``.part`` then
        rename), resumed with HTTP Range requests on transport errors, and
        re-fetched with fresh presigned URLs when one expires. Downloads never
        trigger a conversion implicitly: if no completed artifact exists, run
        :meth:`~weaver.training_client.TrainingClient.export_weights` first.

        Args:
            target: What to download — a
                :class:`~weaver.types.WeightsArtifact`, an artifact
                ``weaver://.../artifacts/{kind}`` URI, a checkpoint
                ``weaver://`` URI (its single completed artifact is chosen),
                or a raw artifact id.
            dest: Directory the files are written into (created if missing).
            kind: Artifact kind (``"hf_model"`` or ``"hf_adapter"``) to select
                when *target* is a checkpoint URI with several artifacts.
            verify: If True (default), verify each file's sha256 against the
                download manifest.
            max_concurrency: Maximum number of files downloaded in parallel.

        Returns:
            The destination directory as a :class:`~pathlib.Path`.

        Raises:
            ValueError: On an unresolvable target, kind conflict, or
                ambiguous artifact selection.
            RuntimeError: When no completed artifact exists, or on a
                size/sha256 mismatch.
        """
        if kind is not None and kind not in ARTIFACT_KINDS:
            raise ValueError(f"kind must be one of {ARTIFACT_KINDS}, got {kind!r}")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        artifact_id = self._resolve_weights_artifact_id(target, kind)
        dest_dir = Path(dest).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)

        descriptor_path = f"/api/v1/artifacts/{artifact_id}/download"
        files = descriptor_files(self.http.get(descriptor_path))
        urls = {entry.name: entry.url for entry in files}
        urls_lock = threading.Lock()

        def current_url(name: str) -> str:
            with urls_lock:
                return urls[name]

        def refresh_url(name: str) -> str:
            # Presigned URLs live ~15 minutes; re-fetching the descriptor is
            # idempotent and cheap, and refreshes every pending file at once.
            with urls_lock:
                for entry in descriptor_files(self.http.get(descriptor_path)):
                    urls[entry.name] = entry.url
                return urls[name]

        workers = max(1, min(max_concurrency, len(files)))
        with build_download_client() as download_client:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self._download_weights_file,
                        download_client,
                        entry,
                        dest_dir=dest_dir,
                        verify=verify,
                        current_url=current_url,
                        refresh_url=refresh_url,
                    )
                    for entry in files
                ]
                for future in futures:
                    future.result()
        return dest_dir

    def _resolve_weights_artifact_id(self, target: str | WeightsArtifact, kind: str | None) -> str:
        """Resolve a download target to a concrete artifact id."""
        if isinstance(target, WeightsArtifact):
            if not target.id:
                raise ValueError("WeightsArtifact has no id")
            return validate_resource_id(target.id, kind="artifact")
        parsed = parse_download_target(target)
        if parsed.artifact_id:
            return parsed.artifact_id
        if parsed.kind and kind and parsed.kind != kind:
            raise ValueError(f"kind={kind!r} conflicts with the artifact URI kind {parsed.kind!r}")
        effective_kind = parsed.kind or kind
        listing = self.http.get(f"/api/v1/models/{parsed.model_id}/checkpoints")
        items = (listing or {}).get("items", []) if isinstance(listing, dict) else []
        checkpoint_id = resolve_checkpoint_id_from_listing(items, parsed.checkpoint_path or "")
        if checkpoint_id is None:
            raise ValueError(
                f"No checkpoint with path {parsed.checkpoint_path!r} found for "
                f"model {parsed.model_id}"
            )
        artifacts = self.http.get(f"/api/v1/checkpoints/{checkpoint_id}/artifacts")
        artifact_items = (artifacts or {}).get("items", []) if isinstance(artifacts, dict) else []
        selected = select_artifact_payload(artifact_items, effective_kind, context=str(target))
        return extract_id(selected)

    def _download_weights_file(
        self,
        download_client: httpx.Client,
        entry: ArtifactFile,
        *,
        dest_dir: Path,
        verify: bool,
        current_url: Callable[[str], str],
        refresh_url: Callable[[str], str],
    ) -> None:
        """Download one manifest file with resume, URL refresh, and verification.

        Descriptor file names are untrusted, so nothing here is written
        through the composed path ``dest_dir / entry.name``. The destination
        directory is walked one component at a time with no-follow ``openat``
        semantics, and the resume stat, every write, the hash and the final
        publish are all issued relative to the descriptor that walk returns
        (:mod:`weaver._safeio`). Replacing a directory with a symlink after
        the walk is inert: a descriptor names an inode, and no step after the
        walk resolves the path again.
        """
        rel = PurePosixPath(entry.name)
        # Cheap early rejection of an obviously escaping name, before anything
        # is created on disk. The descriptor chain below is the real guarantee.
        ensure_within_directory(dest_dir, (dest_dir / rel).parent)
        if not supports_dir_fd():
            self._download_weights_file_unanchored(
                download_client,
                entry,
                dest_dir=dest_dir,
                verify=verify,
                current_url=current_url,
                refresh_url=refresh_url,
            )
            return
        final_name = rel.name
        part_name = f"{final_name}.part"
        parent_fd = open_parent_fd(dest_dir, rel, create=True)
        try:
            if is_file_already_complete_at(parent_fd, final_name, entry, verify=verify):
                return
            url = current_url(entry.name)
            url_refreshes = 0
            transport_retries = 0
            while True:
                resume_from = resume_offset_at(parent_fd, part_name, entry)
                try:
                    with open_for_write(parent_fd, part_name, append=resume_from > 0) as sink:
                        stream_download_to_file(
                            url, sink, client=download_client, resume_from=resume_from
                        )
                    break
                except DownloadURLExpiredError:
                    url_refreshes += 1
                    if url_refreshes > DOWNLOAD_MAX_URL_REFRESHES:
                        raise
                    url = refresh_url(entry.name)
                except (httpx.TransportError, OSError):
                    # GETs are idempotent and the .part keeps its bytes, so retry
                    # with a Range resume instead of restarting the shard.
                    transport_retries += 1
                    if transport_retries > DOWNLOAD_MAX_TRANSPORT_RETRIES:
                        raise
                    time.sleep(compute_retry_delay(transport_retries))
            check_downloaded_file_at(parent_fd, part_name, entry, verify=verify)
            # Atomic publish through the same descriptor: readers never observe
            # a half-written file, and the rename cannot be redirected.
            rename_within(parent_fd, part_name, final_name)
        finally:
            os.close(parent_fd)

    def _download_weights_file_unanchored(
        self,
        download_client: httpx.Client,
        entry: ArtifactFile,
        *,
        dest_dir: Path,
        verify: bool,
        current_url: Callable[[str], str],
        refresh_url: Callable[[str], str],
    ) -> None:
        """Path-based download for platforms without ``dir_fd`` support.

        Only Windows reaches this branch, and Windows is not a supported
        execution environment for this SDK (see :mod:`weaver._safeio`). It
        keeps the pre-anchoring behaviour — containment check plus a no-follow
        open of the final component — and with it the check-to-use window the
        anchored path above removes.
        """
        final_path = dest_dir / entry.name
        final_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_within_directory(dest_dir, final_path.parent)
        if is_file_already_complete(final_path, entry, verify=verify):
            return
        part_path = final_path.with_name(final_path.name + ".part")
        url = current_url(entry.name)
        url_refreshes = 0
        transport_retries = 0
        while True:
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            if entry.size is not None and resume_from > entry.size:
                # Longer than the manifest says it should be: poisoned partial.
                part_path.unlink()
                resume_from = 0
            try:
                with legacy_open_for_write(part_path, append=resume_from > 0) as sink:
                    stream_download_to_file(
                        url, sink, client=download_client, resume_from=resume_from
                    )
                break
            except DownloadURLExpiredError:
                url_refreshes += 1
                if url_refreshes > DOWNLOAD_MAX_URL_REFRESHES:
                    raise
                url = refresh_url(entry.name)
            except (httpx.TransportError, OSError):
                # GETs are idempotent and the .part keeps its bytes, so retry
                # with a Range resume instead of restarting the shard.
                transport_retries += 1
                if transport_retries > DOWNLOAD_MAX_TRANSPORT_RETRIES:
                    raise
                time.sleep(compute_retry_delay(transport_retries))
        check_downloaded_file(part_path, entry, verify=verify)
        # Atomic publish: readers never observe a half-written file.
        part_path.replace(final_path)

    # ------------------------------------------------------------------
    # NorthGate deployments
    # ------------------------------------------------------------------

    def list_deployments(self) -> List[Deployment]:
        """List the deployments this principal published.

        Deployments are owner-scoped: the listing shows the ones this
        principal created, not every deployment of the models it can access.
        Stopped and failed deployments are included, so the history of an
        endpoint stays visible after it is taken down.

        Returns:
            Every :class:`~weaver.types.Deployment` the caller owns, newest
            first.

        Raises:
            WeaverAPIError: If deployments are disabled on this server (503).
        """
        deployments: List[Deployment] = []
        offset = 0
        while True:
            params = {"limit": DEPLOYMENT_PAGE_SIZE, "offset": offset}
            try:
                payload = self.http.get("/api/v1/deployments", params=params)
            except WeaverAPIError as exc:
                raise translate_deployment_error(exc) from exc
            page = deployment_items(payload)
            deployments.extend(Deployment.from_payload(item) for item in page)
            next_offset = next_page_offset(payload, offset, len(page))
            if next_offset is None:
                return deployments
            offset = next_offset

    def get_deployment(self, deployment_id: str) -> Deployment:
        """Fetch one deployment by id.

        Args:
            deployment_id: The deployment's server-generated id.

        Returns:
            The :class:`~weaver.types.Deployment`, including its endpoint URL
            once it is running.

        Raises:
            WeaverAPIError: If the deployment does not exist or belongs to
                another principal (404 — a deployment owned by someone else
                is reported as missing), or deployments are disabled (503).
        """
        deployment_id = validate_resource_id(deployment_id, kind="deployment")
        try:
            payload = self.http.get(f"/api/v1/deployments/{deployment_id}")
        except WeaverAPIError as exc:
            raise translate_deployment_error(exc) from exc
        return Deployment.from_payload(payload if isinstance(payload, dict) else {})

    @overload
    def delete_deployment(
        self, deployment_id: str, *, wait: Literal[True] = True
    ) -> Deployment: ...

    @overload
    def delete_deployment(self, deployment_id: str, *, wait: Literal[False]) -> OperationHandle: ...

    def delete_deployment(
        self, deployment_id: str, *, wait: bool = True
    ) -> Deployment | OperationHandle:
        """Take a deployment down.

        Offboards the model from the gateway, stops the workload, releases the
        job name, and frees the materialized weights the deployment was
        pinning. The name becomes available again once the deployment reaches
        ``stopped``.

        Deleting deliberately does not need the ``deployment.publish``
        capability: whoever published an endpoint must always be able to take
        it down, even if their grant was revoked afterwards.

        Args:
            deployment_id: The deployment's server-generated id.
            wait: If True (default), blocks until teardown finishes and
                returns the stopped :class:`~weaver.types.Deployment`.

        Returns:
            The stopped :class:`~weaver.types.Deployment` when *wait* is True,
            else an :class:`OperationHandle` whose result is that deployment.

        Raises:
            WeaverAPIError: If the deployment is unknown or owned by someone
                else (404), is already stopped (409 ``already_stopped``), or
                deployments are disabled on this server (503).
        """
        deployment_id = validate_resource_id(deployment_id, kind="deployment")
        try:
            response = self.http.delete(f"/api/v1/deployments/{deployment_id}")
        except WeaverAPIError as exc:
            raise translate_deployment_error(exc) from exc
        handle = build_operation_handle(self.http, response if isinstance(response, dict) else {})
        if not wait:
            return handle
        result = handle.result()
        return Deployment.from_payload(result if isinstance(result, dict) else {})
