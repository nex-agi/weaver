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

"""Configuration-time weight-sync selection shared by every Weaver layer.

The selection answers two independent questions:

``backend``
    *How* updated weights reach the inference target: the durable checkpoint
    path (``default``), the RDMA object pool (``mooncake``), or a live
    cross-job collective (``nccl``).

``update``
    *What* is sent: every value (``full``) or only the values that changed
    since an exact base version (``delta``).

They are separate dimensions on purpose. There is no ``nccl_delta`` or
``mooncake_delta_fast`` mode, and a combination that is not qualified is
rejected here -- in the caller's process, before any operation is enqueued,
provisioned, or transferred -- rather than silently degraded at run time.

The selection is **configuration-time and immutable for one run/session**. The
authoritative value lives in the control plane: the supported-model registry
resolves it and the sampling session freezes it at creation. This type is how
the SDK states an expectation and how it reads the frozen value back; it is
never the authority itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping

if TYPE_CHECKING:
    from .nccl_weight_sync import NCCLWeightSyncV1Result

#: Transport backends. ``default`` is the established durable-checkpoint path.
WEIGHT_SYNC_BACKENDS = ("default", "mooncake", "nccl")

#: Update dimension, orthogonal to :data:`WEIGHT_SYNC_BACKENDS`.
WEIGHT_SYNC_UPDATES = ("full", "delta")

#: Qualified ``(backend, update)`` combinations mapped to why an unqualified
#: one is refused. A missing key is supported; a present key is not, and its
#: value is the exact operator-facing reason.
#:
#: ``mooncake`` uploads whole checkpoint shard files verbatim and has no delta
#: producer, so selecting it disables delta. That is an existing capability gap
#: in the Mooncake implementation, reported rather than worked around: this
#: table refuses the combination instead of quietly running a full sync.
_UNSUPPORTED_COMBINATIONS = {
    ("mooncake", "delta"): (
        "the mooncake backend uploads whole checkpoint shards verbatim and has "
        "no delta producer; use backend='default' for delta, or update='full'"
    ),
}

#: Backends whose transport can verify payload integrity on demand. The debug
#: checksum hashes the exact transferred bytes on both sides, which only the
#: live collective transport implements.
_DEBUG_CHECKSUM_BACKENDS = ("nccl",)

#: Backends whose delta implementation is driven by an explicit publication
#: counter. The durable-checkpoint delta path re-baselines on accumulated byte
#: drift instead and keeps its own established knobs, so a caller must not
#: express its policy through this field.
_REBASELINE_INTERVAL_BACKENDS = ("nccl",)


def normalize_weight_sync_backend(value: object) -> str:
    """Return ``value`` if it names a known backend.

    Args:
        value: Candidate backend name.

    Returns:
        The validated backend name.

    Raises:
        ValueError: If ``value`` is not one of :data:`WEIGHT_SYNC_BACKENDS`.
    """

    if not isinstance(value, str) or value not in WEIGHT_SYNC_BACKENDS:
        raise ValueError("weight sync backend must be one of " + ", ".join(WEIGHT_SYNC_BACKENDS))
    return value


def normalize_weight_sync_update(value: object) -> str:
    """Return ``value`` if it names a known update dimension.

    Args:
        value: Candidate update name.

    Returns:
        The validated update name.

    Raises:
        ValueError: If ``value`` is not one of :data:`WEIGHT_SYNC_UPDATES`.
    """

    if not isinstance(value, str) or value not in WEIGHT_SYNC_UPDATES:
        raise ValueError("weight sync update must be one of " + ", ".join(WEIGHT_SYNC_UPDATES))
    return value


@dataclass(frozen=True, slots=True)
class WeightSyncSelection:
    """One immutable weight-sync configuration.

    Attributes:
        backend: Transport backend, see :data:`WEIGHT_SYNC_BACKENDS`.
        update: Update dimension, see :data:`WEIGHT_SYNC_UPDATES`.
        debug_checksum: Verify transferred payload bytes on both sides. This is
            a debugging facility: it copies and hashes the whole model on the
            source and on every target rank, so it must never be enabled for a
            performance measurement. Off by default.
        rebaseline_interval: For ``nccl`` + ``delta`` only, publish a full
            update every N publications so a target never depends on an
            unbounded delta chain. ``None`` defers to the control plane's
            configured value.
    """

    backend: str = "default"
    update: str = "full"
    debug_checksum: bool = False
    rebaseline_interval: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", normalize_weight_sync_backend(self.backend))
        object.__setattr__(self, "update", normalize_weight_sync_update(self.update))
        if not isinstance(self.debug_checksum, bool):
            raise ValueError("debug_checksum must be a boolean")
        unsupported = _UNSUPPORTED_COMBINATIONS.get((self.backend, self.update))
        if unsupported is not None:
            raise ValueError(
                f"weight sync backend={self.backend!r} update={self.update!r} "
                f"is not supported: {unsupported}"
            )
        if self.debug_checksum and self.backend not in _DEBUG_CHECKSUM_BACKENDS:
            raise ValueError(
                "debug_checksum verifies transferred payload bytes and is only "
                "available for backend " + ", ".join(_DEBUG_CHECKSUM_BACKENDS)
            )
        if self.rebaseline_interval is not None:
            if (
                not isinstance(self.rebaseline_interval, int)
                or isinstance(self.rebaseline_interval, bool)
                or self.rebaseline_interval < 1
            ):
                raise ValueError("rebaseline_interval must be a positive integer")
            if self.backend not in _REBASELINE_INTERVAL_BACKENDS or self.update != "delta":
                raise ValueError(
                    "rebaseline_interval applies only to backend "
                    + ", ".join(_REBASELINE_INTERVAL_BACKENDS)
                    + " with update='delta'"
                )

    @property
    def is_live_collective(self) -> bool:
        """True when weights move over the live cross-job collective."""

        return self.backend == "nccl"

    @property
    def is_delta(self) -> bool:
        """True when only values changed since the base version are sent."""

        return self.update == "delta"

    @classmethod
    def from_payload(cls, value: object) -> "WeightSyncSelection":
        """Read the selection a control plane froze for one session.

        A response that carries no selection at all is the established
        durable-checkpoint configuration, so an older control plane keeps
        working unchanged. The narrow legacy ``weight_sync_mode`` spelling is
        also accepted and resolves to the live collective backend.

        Args:
            value: Session payload, or the nested ``weight_sync`` object.

        Returns:
            The frozen selection.

        Raises:
            RuntimeError: If the payload is not an object, or carries a
                selection this SDK cannot represent.
        """

        if not isinstance(value, dict):
            raise RuntimeError("weight sync selection must be an object")
        payload: Mapping[str, Any] = value
        nested = payload.get("weight_sync")
        if isinstance(nested, dict):
            payload = nested
        elif "backend" not in payload:
            # No structured selection. Fall back to the legacy narrow spelling,
            # then to the established default configuration.
            legacy = payload.get("weight_sync_mode")
            if isinstance(legacy, str) and legacy.strip() == "nccl_v1":
                return cls(backend="nccl", update="full")
            return cls()
        try:
            return cls(
                backend=payload.get("backend", "default"),
                update=payload.get("update", "full"),
                debug_checksum=bool(payload.get("debug_checksum", False)),
                rebaseline_interval=payload.get("rebaseline_interval"),
            )
        except ValueError as error:
            raise RuntimeError(
                f"control plane reported an unusable weight sync selection: {error}"
            ) from error

    def to_payload(self) -> Dict[str, Any]:
        """Serialize the selection for a request body."""

        payload: Dict[str, Any] = {
            "backend": self.backend,
            "update": self.update,
            "debug_checksum": self.debug_checksum,
        }
        if self.rebaseline_interval is not None:
            payload["rebaseline_interval"] = self.rebaseline_interval
        return payload

    def assert_matches(self, resolved: "WeightSyncSelection") -> "WeightSyncSelection":
        """Fail unless the control plane froze what this caller asked for.

        The caller's expectation is an assertion, never an override: if the
        registry resolved a different backend the run must stop here rather
        than proceed against a transport the caller did not intend. Fields the
        caller left unset (``rebaseline_interval=None``) defer to the control
        plane and are not compared.

        Args:
            resolved: The selection the control plane froze for the session.

        Returns:
            ``resolved``, so callers can bind the authoritative value.

        Raises:
            RuntimeError: If any asserted field differs.
        """

        differences = [
            f"{name}: requested {getattr(self, name)!r}, control plane resolved "
            f"{getattr(resolved, name)!r}"
            for name in ("backend", "update", "debug_checksum")
            if getattr(self, name) != getattr(resolved, name)
        ]
        if self.rebaseline_interval is not None and (
            self.rebaseline_interval != resolved.rebaseline_interval
        ):
            differences.append(
                f"rebaseline_interval: requested {self.rebaseline_interval!r}, control "
                f"plane resolved {resolved.rebaseline_interval!r}"
            )
        if differences:
            raise RuntimeError("weight sync selection mismatch -- " + "; ".join(differences))
        return resolved


def reported_selection(value: object) -> WeightSyncSelection | None:
    """Return the selection a control plane reported, or ``None`` if it did not.

    This distinction matters and must not be collapsed into a default. A
    control plane that reports ``backend="default"`` has made a statement the
    caller's expectation can be checked against; one that reports nothing at
    all has made no statement, and asserting against an invented default would
    fail a perfectly valid run.

    Args:
        value: Sampling-session payload.

    Returns:
        The reported selection, or ``None`` when the payload carries no
        selection at all.

    Raises:
        RuntimeError: If a selection is present but unusable.
    """

    if not isinstance(value, dict):
        return None
    if isinstance(value.get("weight_sync"), dict) or "weight_sync_mode" in value:
        return WeightSyncSelection.from_payload(value)
    return None


@dataclass(frozen=True, slots=True)
class WeightPublication:
    """Backend-neutral receipt for one published weight version.

    Every backend answers the same two questions -- which version is now live
    on the target, and what it was published against -- so an RL loop reads
    those fields without knowing which transport ran. Backend-specific proof
    stays in its own field rather than being flattened into the common shape.

    Attributes:
        backend: The backend that ran, as frozen by the control plane.
        update: Whether a full or delta update was published.
        version: The weight version now live on the target.
        base_version: The version this publication was applied against, when
            the backend tracks an explicit lineage.
        sampling_session_id: The session the target serves this version to.
        model_path: Durable checkpoint URI, for checkpoint-based backends only.
        nccl: The committed live-collective receipt, for ``nccl`` only.
    """

    backend: str
    update: str
    version: str
    base_version: str | None = None
    sampling_session_id: str | None = None
    model_path: str | None = None
    nccl: "NCCLWeightSyncV1Result | None" = None
