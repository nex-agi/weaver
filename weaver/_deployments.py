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

"""Deployment helpers shared by the sync and async clients.

Everything in this module is pure (no IO) so both client stacks build
identical requests and interpret identical responses; see
``.claude/rules/async-compatibility.md``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._http import WeaverAPIError
from ._utils import lookup_case_insensitive

# Request sanity bounds. These mirror the server's own guards so an obvious typo
# fails locally instead of after a round trip — the server re-validates and
# remains authoritative.
MAX_DEPLOYMENT_REPLICAS = 8
MAX_DEPLOYMENT_GPUS_PER_REPLICA = 16

# A deployment name lives in three name spaces at once — the served model name,
# a Kubernetes label value, and the gateway's global model name — so it is
# validated against the strictest of the three. Anything outside this shape
# would be silently rewritten downstream and published under a name the caller
# never asked for.
DEPLOYMENT_NAME_MAX_LEN = 63
_DEPLOYMENT_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")


def validate_deployment_name(name: str) -> str:
    """Validate a deployment name and return it stripped.

    Args:
        name: The user-supplied public model name.

    Returns:
        The name with surrounding whitespace removed.

    Raises:
        ValueError: If the name is empty, too long, or contains characters
            that are not valid in all three downstream name spaces.
    """
    normalized = (name or "").strip()
    if not normalized:
        raise ValueError("deployment name must not be empty")
    if len(normalized) > DEPLOYMENT_NAME_MAX_LEN:
        raise ValueError(
            f"deployment name must be at most {DEPLOYMENT_NAME_MAX_LEN} characters, "
            f"got {len(normalized)}"
        )
    if not _DEPLOYMENT_NAME_RE.match(normalized):
        raise ValueError(
            f"invalid deployment name {normalized!r}: it may contain only letters, "
            "digits, '.', '-' and '_', and must start and end with a letter or digit"
        )
    return normalized


def build_create_deployment_body(
    *,
    name: str,
    gpu_type: Optional[str] = None,
    replicas: int = 1,
    gpus_per_replica: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build the ``POST /checkpoints/{id}/deployments`` request body.

    Unset sizing fields are omitted rather than sent as zero, so the server
    applies its configured defaults.

    Raises:
        ValueError: On an invalid name or an out-of-range replica/GPU count.
    """
    body: Dict[str, Any] = {
        "name": validate_deployment_name(name),
        "overwrite": overwrite,
    }
    if type(overwrite) is not bool:  # bool is deliberate: reject "false"/1
        raise ValueError(f"overwrite must be a bool, got {overwrite!r}")
    if type(replicas) is not int:  # exact type: bool is an int subclass, floats compare
        raise ValueError(f"replicas must be an int, got {replicas!r}")
    if replicas < 1 or replicas > MAX_DEPLOYMENT_REPLICAS:
        raise ValueError(
            f"replicas must be between 1 and {MAX_DEPLOYMENT_REPLICAS}, got {replicas}"
        )
    body["replicas"] = replicas
    if gpus_per_replica is not None:
        if type(gpus_per_replica) is not int:
            raise ValueError(f"gpus_per_replica must be an int, got {gpus_per_replica!r}")
        if gpus_per_replica < 1 or gpus_per_replica > MAX_DEPLOYMENT_GPUS_PER_REPLICA:
            raise ValueError(
                f"gpus_per_replica must be between 1 and {MAX_DEPLOYMENT_GPUS_PER_REPLICA}, "
                f"got {gpus_per_replica}"
            )
        body["gpus_per_replica"] = gpus_per_replica
    if gpu_type is not None:
        normalized_gpu_type = gpu_type.strip()
        if not normalized_gpu_type:
            raise ValueError("gpu_type must not be blank; pass None to use the server default")
        body["gpu_type"] = normalized_gpu_type
    return body


def deployment_items(payload: Any) -> List[Dict[str, Any]]:
    """Extract the ``items`` array from a deployment list response.

    Malformed entries raise instead of being dropped: pagination advances the
    offset by the item count, so silently filtering would undercount the
    records consumed from the page — duplicating rows on the next request or
    terminating the walk early.

    Raises:
        ValueError: If an items entry is not an object.
    """
    if not isinstance(payload, dict):
        return []
    items = lookup_case_insensitive(payload, "items")
    if not isinstance(items, list):
        return []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"malformed deployment list entry: {item!r}")
    return list(items)


# The list endpoint pages at 20 by default and caps a page at 100; both stacks
# walk every page so a caller with many stopped deployments still sees them all.
DEPLOYMENT_PAGE_SIZE = 100


def next_page_offset(payload: Any, offset: int, page_len: int) -> Optional[int]:
    """Return the offset of the next page, or ``None`` when the listing is done.

    Stops on an empty page as well as on the reported total, so a server that
    omits or misreports ``total_count`` cannot spin the caller in a loop.
    """
    if page_len == 0:
        return None
    advanced = offset + page_len
    pagination = (
        lookup_case_insensitive(payload, "pagination") if isinstance(payload, dict) else None
    )
    if not isinstance(pagination, dict):
        return None
    try:
        total = int(str(lookup_case_insensitive(pagination, "total_count")))
    except (TypeError, ValueError):
        return None
    return advanced if advanced < total else None


# Actionable guidance for the deployment-specific failures. The server's own
# messages are accurate but terse ("insufficient capability"), and both of the
# gates below are operator-side configuration the caller cannot discover from
# the API — so the SDK names what has to change and who can change it.
_PERMISSION_GUIDANCE = (
    "Publishing requires the 'deployment.publish' capability, which is granted by "
    "principal origin rather than by Weaver role: an SSO session always qualifies, "
    "an API key only when it was minted under an IAM biz_code on the server's "
    "deployment.allowed_biz_codes allowlist, and a service credential never does. "
    "Sign in with SSO, or ask a Weaver administrator to add your biz_code to the "
    "allowlist. Listing, reading and deleting your own deployments do not need it."
)

_UNAVAILABLE_GUIDANCE = (
    "Checkpoint deployment is turned off on this server: the 'deployment' feature "
    "block is disabled or unconfigured. It is deny-by-default and an administrator "
    "must enable it and supply the gateway credentials and the biz_code allowlist."
)

_NAME_TAKEN_GUIDANCE = (
    "Deployment names are unique across every deployment that is not stopped. "
    "Delete the deployment holding the name (deleting releases it) or choose "
    "another name. Note that overwrite=True only replaces a registration on the "
    "gateway — it does not free a name already used inside Weaver."
)

_QUOTA_GUIDANCE = (
    "Each live deployment holds GPUs and pins its weights on shared storage, so "
    "the server caps how many you may run at once. Delete one with "
    "delete_deployment() before publishing another."
)


def deployment_error_guidance(error: WeaverAPIError) -> Optional[str]:
    """Return actionable guidance for a deployment API error, if any.

    Args:
        error: The error raised by the HTTP layer.

    Returns:
        A sentence explaining what to do about it, or ``None`` when the
        server's own message already stands on its own.
    """
    code = (error.code or "").lower()
    if error.status_code == 503 and code == "deployment_unavailable":
        return _UNAVAILABLE_GUIDANCE
    if error.status_code == 403 and "insufficient capability" in (error.message or "").lower():
        # Scoped to the capability message on purpose: the same route can answer
        # 403 "forbidden" for a checkpoint the caller may not write to, which is
        # a different problem with different advice.
        return _PERMISSION_GUIDANCE
    if error.status_code == 409 and code == "name_taken":
        return _NAME_TAKEN_GUIDANCE
    if error.status_code == 409 and code == "deployment_limit_reached":
        return _QUOTA_GUIDANCE
    return None


def translate_deployment_error(error: WeaverAPIError) -> WeaverAPIError:
    """Re-render a deployment API error with actionable guidance appended.

    The class, status code, error code and every structured field are
    preserved, so ``except WeaverAPIError`` handlers and code branching on
    ``error.code`` keep working; only the human-readable message grows.
    """
    guidance = deployment_error_guidance(error)
    if guidance is None:
        return error
    return WeaverAPIError(
        error.status_code,
        code=error.code,
        message=f"{error.message}. {guidance}",
        retryable=error.retryable,
        request_id=error.request_id,
        retry_after=error.retry_after,
        required_nanos=error.required_nanos,
        available_nanos=error.available_nanos,
        required_usd=error.required_usd,
        available_usd=error.available_usd,
        details=error.details,
    )
