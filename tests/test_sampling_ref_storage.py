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

"""Storage coexistence and downgrade rejection through real operation handles."""

from __future__ import annotations

import asyncio
import hashlib
import struct
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from weaver._http import WeaverAPIError
from weaver.async_sampling_client import AsyncSamplingClient
from weaver.operations import AsyncOperationHandle, OperationHandle
from weaver.sampling_client import SamplingClient
from weaver.types import ModelInput, SamplingParams

MODEL = str(uuid4())
KINDS = ["sampling_mask_ref", "sampler_distribution_ref", "moe_topk_indices_ref"]
CAPABILITIES = {
    "protocol_version": 1,
    "default_backend": "gpfs",
    "sampling_ref_backends": ["gpfs", "artifact"],
    "managed_sampling_ref_kinds": KINDS,
}


def payload(kind, storage="weaver-artifact/v1"):
    ref = {
        "storage": storage,
        "model_id": MODEL,
        "artifact_ref": f"artifact://refs/{MODEL}/{uuid4()}@{uuid4()}",
        "relative_path": f"{MODEL}/refs/{uuid4()}/manifest.json",
        "format": "safetensors-csr-chunks" if kind == KINDS[0] else "safetensors-chunks",
        "schema": (
            "weaver.sampling_mask.v1" if kind == KINDS[0] else "weaver.sampler_distribution.v1"
        ),
        "tokens_sha256": hashlib.sha256(struct.pack("<ii", 3, 4)).hexdigest(),
        "token_count": 2,
        "weight_version": "v1",
        "empty_row": "full-vocabulary",
        "top_k": 2,
    }
    if storage == "gpfs":
        del ref["artifact_ref"]
    if kind == KINDS[2]:
        ref.update(
            format="safetensors",
            schema="weaver.router_replay.r2_recorded_indices.v1",
            dtype="int16",
            shape=[3, 2, 2],
            size_bytes=136,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=2)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        )
    seq = {"tokens": [3, 4], "text": "answer", "weight_version": "v1", kind: ref}
    if kind == KINDS[1]:
        seq["sampler_logprobs"] = [-1.0, -2.0]
        seq["sampler_distribution"] = {
            "schema": "behavior/unfiltered/v1",
            "weight_version": "v1",
        }
    return {"sequences": [seq]}


def options(kind):
    common = {"prompt": ModelInput.from_ints([1, 2])}
    if kind == KINDS[0]:
        return common | {"return_sampling_mask": True, "sampling_mask_transport": "ref"}
    if kind == KINDS[2]:
        return common | {"return_moe_topk_indices": True}
    return common | {
        "sampling_params": SamplingParams(score_centering={"head_size": 2, "transport": "ref"})
    }


def client_fixture(async_mode, response, **binding):
    service = MagicMock()
    service.http.get.return_value = deepcopy(CAPABILITIES)
    state = {"id": "op", "status": "done", "response": response}
    if async_mode:
        service.http.get = AsyncMock(return_value=deepcopy(CAPABILITIES))
        handle = AsyncOperationHandle.from_payload(service.http, state)
        service.enqueue_operation = AsyncMock(return_value=handle)
        client = AsyncSamplingClient(service=service, sampling_session_id="s", **binding)
        client._ensure_tokenizer_source = AsyncMock()
    else:
        handle = OperationHandle.from_payload(service.http, state)
        service.enqueue_operation.return_value = handle
        client = SamplingClient(service=service, sampling_session_id="s", **binding)
    return service, handle, client


def call(client, async_mode, kwargs):
    return asyncio.run(client.sample(**kwargs)) if async_mode else client.sample(**kwargs)


def result(handle, async_mode):
    return asyncio.run(handle.result()) if async_mode else handle.result()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("binding", ["model_id", "model_path"])
def test_managed_selection_and_deferred_validation(async_mode, wait, kind, binding):
    response = payload(kind)
    model_binding = {
        binding: MODEL if binding == "model_id" else f"weaver://{MODEL}/checkpoints/training"
    }
    service, handle, client = client_fixture(async_mode, response, **model_binding)
    output = call(
        client,
        async_mode,
        options(kind)
        | {
            "ref_storage_backend": "artifact",
            "wait": wait,
        },
    )
    assert service.enqueue_operation.call_args.args[0].endswith("/samples/managed-refs")
    assert service.enqueue_operation.call_args.args[1]["ref_storage_backend"] == "artifact"
    service.http.get.assert_called_once_with(f"/api/v1/models/{MODEL}/storage-capabilities")
    assert (output if wait else result(output, async_mode))["sequences"][0][kind] == response[
        "sequences"
    ][0][kind]
    assert output is handle if not wait else True


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backend", [None, "gpfs"])
def test_old_gpfs_envelopes_need_no_new_server(async_mode, kind, backend):
    response = payload(kind, "gpfs")
    # Old public responses did not promise new locator/storage metadata.
    ref = response["sequences"][0][kind]
    for field in ["storage", "model_id", "format", "relative_path"]:
        del ref[field]
    service, _, client = client_fixture(async_mode, response)
    output = call(client, async_mode, options(kind) | {"ref_storage_backend": backend})
    assert output["sequences"][0][kind] == ref
    service.http.get.assert_not_called()
    path, body = service.enqueue_operation.call_args.args
    assert path.endswith("/samples")
    assert body.get("ref_storage_backend") == backend
    assert ("ref_storage_backend" in body) == (backend is not None)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "gpfs",
        "inline",
        "scope",
        "locator_scope",
        "mutable_locator",
        "noncanonical",
        "path_escape",
        "absolute_path",
        "reserved_path",
        "format",
        "private_token",
        "uri",
    ],
)
def test_managed_response_rejects_downgrade_or_spoof(async_mode, kind, wait, bad):
    response = payload(kind)
    ref = response["sequences"][0][kind]
    if bad == "gpfs":
        ref["storage"] = "gpfs"
    elif bad == "inline":
        del response["sequences"][0][kind]
    elif bad == "scope":
        ref["model_id"] = str(uuid4())
    elif bad == "locator_scope":
        ref["artifact_ref"] = ref["artifact_ref"].replace(MODEL, str(uuid4()))
    elif bad == "mutable_locator":
        ref["artifact_ref"] = f"artifact://refs/{MODEL}/latest"
    elif bad == "noncanonical":
        ref["artifact_ref"] = ref["artifact_ref"].upper()
    elif bad == "path_escape":
        ref["relative_path"] = "../manifest.json"
    elif bad == "absolute_path":
        ref["relative_path"] = "/data/manifest.json"
    elif bad == "reserved_path":
        ref["relative_path"] = ".weaver-ref-envelope.json"
    elif bad == "format":
        ref["format"] = "unrecognized"
    elif bad == "private_token":
        ref["token"] = "untrusted"
    else:
        ref["uri"] = "file:///data/manifest.json"
    service, _, client = client_fixture(async_mode, response, model_id=MODEL)
    kwargs = options(kind) | {"ref_storage_backend": "artifact", "wait": wait}
    with pytest.raises(ValueError):
        output = call(client, async_mode, kwargs)
        if not wait:
            result(output, async_mode)
    assert service.enqueue_operation.call_count == 1  # no GPFS retry
    if async_mode and wait:
        client._ensure_tokenizer_source.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("bad", ["old_server", "disabled", "unknown_kind", "bool_version"])
def test_checkpoint_capability_does_not_authorize_ref_writes(async_mode, bad):
    service, _, client = client_fixture(async_mode, payload(KINDS[0]), model_id=MODEL)
    caps = {
        "protocol_version": 1,
        "default_backend": "gpfs",
        "save_state_backends": ["gpfs", "artifact"],
    }
    if bad != "old_server":
        caps |= deepcopy(CAPABILITIES)
        if bad == "disabled":
            caps["sampling_ref_backends"] = ["gpfs"]
        elif bad == "unknown_kind":
            caps["managed_sampling_ref_kinds"] = [KINDS[1]]
        else:
            caps["protocol_version"] = True
    service.http.get.return_value = caps
    with pytest.raises(RuntimeError):
        call(client, async_mode, options(KINDS[0]) | {"ref_storage_backend": "artifact"})
    service.enqueue_operation.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", [404, 405, 403, 503])
def test_capability_failure_never_enqueues_gpfs(async_mode, status):
    service, _, client = client_fixture(async_mode, payload(KINDS[0]), model_id=MODEL)
    service.http.get.side_effect = WeaverAPIError(status, "failed", "failed", False)
    expected = RuntimeError if status in (404, 405) else WeaverAPIError
    with pytest.raises(expected):
        call(client, async_mode, options(KINDS[0]) | {"ref_storage_backend": "artifact"})
    service.enqueue_operation.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
def test_capability_is_rechecked_after_rollback(async_mode):
    service, _, client = client_fixture(async_mode, payload(KINDS[0]), model_id=MODEL)
    call(client, async_mode, options(KINDS[0]) | {"ref_storage_backend": "artifact"})
    service.http.get.return_value = CAPABILITIES | {"sampling_ref_backends": ["gpfs"]}
    with pytest.raises(RuntimeError):
        call(client, async_mode, options(KINDS[0]) | {"ref_storage_backend": "artifact"})
    assert service.http.get.call_count == 2
    assert service.enqueue_operation.call_count == 1
    # Explicit GPFS is still admissible and retains the legacy route.
    gpfs_service, _, gpfs_client = client_fixture(async_mode, payload(KINDS[0], "gpfs"))
    call(gpfs_client, async_mode, options(KINDS[0]) | {"ref_storage_backend": "gpfs"})
    gpfs_service.http.get.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_default_gpfs_cannot_accept_managed_result(async_mode, kind):
    service, _, client = client_fixture(async_mode, payload(kind))
    with pytest.raises(ValueError):
        pending = call(client, async_mode, options(kind) | {"wait": False})
        result(pending, async_mode)
    service.http.get.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("bad", ["backend", "no_ref", "router", "no_model", "noncanonical_model"])
def test_invalid_managed_intent_fails_before_io(async_mode, bad):
    binding = {"model_id": MODEL}
    kwargs = options(KINDS[0]) | {"ref_storage_backend": "artifact"}
    if bad == "backend":
        kwargs["ref_storage_backend"] = "unknown"
    elif bad == "no_ref":
        kwargs = {"prompt": ModelInput.from_ints([1]), "ref_storage_backend": "artifact"}
    elif bad == "router":
        kwargs["return_moe_topk_indices"] = True
    elif bad == "no_model":
        binding = {}
    else:
        binding["model_id"] = "model-id"
    service, _, client = client_fixture(async_mode, payload(KINDS[0]), **binding)
    with pytest.raises(ValueError):
        call(client, async_mode, kwargs)
    service.http.get.assert_not_called()
    service.enqueue_operation.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unrecognized"),
        ("dtype", "float32"),
        ("shape", [4, 2, 2]),
        ("shape", [3, 0, 2]),
        ("shape", [3, 2, True]),
        ("shape", [3, 2]),
        ("weight_version", "stale"),
        ("size_bytes", 0),
        ("size_bytes", True),
        ("expires_at", "2000-01-01T00:00:00Z"),
        ("expires_at", "invalid"),
        ("expires_at", float("inf")),
    ],
)
def test_managed_router_semantics_fail_on_immediate_and_deferred_results(
    async_mode, wait, field, value
):
    response = payload(KINDS[2])
    response["sequences"][0][KINDS[2]][field] = value
    service, _, client = client_fixture(async_mode, response, model_id=MODEL)
    with pytest.raises(ValueError):
        output = call(
            client,
            async_mode,
            options(KINDS[2]) | {"ref_storage_backend": "artifact", "wait": wait},
        )
        if not wait:
            result(output, async_mode)
    assert service.enqueue_operation.call_count == 1


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("backend", [None, "gpfs"])
def test_legacy_inline_router_remains_accepted(async_mode, backend):
    response = {"sequences": [{"tokens": [3, 4], "text": "answer", "moe_topk_indices": [[1, 2]]}]}
    service, _, client = client_fixture(async_mode, response)
    output = call(client, async_mode, options(KINDS[2]) | {"ref_storage_backend": backend})
    assert output["sequences"][0]["moe_topk_indices"] == [[1, 2]]
    service.http.get.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("position", [0, 1])
def test_managed_router_validates_every_sequence(async_mode, position):
    response = payload(KINDS[2])
    response["sequences"].append(deepcopy(response["sequences"][0]))
    response["sequences"][position][KINDS[2]]["weight_version"] = "stale"
    _, _, client = client_fixture(async_mode, response, model_id=MODEL)
    with pytest.raises(ValueError):
        call(client, async_mode, options(KINDS[2]) | {"ref_storage_backend": "artifact"})


@pytest.mark.parametrize("async_mode", [False, True])
def test_managed_router_accepts_rfc3339_offset_expiration(async_mode):
    response = payload(KINDS[2])
    response["sequences"][0][KINDS[2]]["expires_at"] = (
        datetime.now(timezone.utc) + timedelta(hours=2)
    ).isoformat()
    _, _, client = client_fixture(async_mode, response, model_id=MODEL)
    call(client, async_mode, options(KINDS[2]) | {"ref_storage_backend": "artifact"})
