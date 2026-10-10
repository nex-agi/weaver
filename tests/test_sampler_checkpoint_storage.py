"""Public sampler storage routing, legacy coexistence and rollback checks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.test_training_checkpoint import _make_async_training_client, _make_training_client
from weaver._http import WeaverAPIError

METHODS = ["save_weights_for_sampler", "save_weights_and_get_sampling_client"]


def capabilities():
    return {
        "protocol_version": 1,
        "default_backend": "gpfs",
        "save_state_backends": ["gpfs", "artifact"],
        "sampler_export_backends": ["gpfs", "artifact"],
        "preferred_permanent_checkpoint_backend": "artifact",
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "case",
    [
        "preferred",
        "explicit_artifact",
        "explicit_gpfs",
        "default_ttl",
        "old_capability",
        "missing_export_capability",
        "rollback",
        "old_route",
        "old_route_explicit",
        "forbidden",
        "unavailable",
        "null_response",
        "malformed_exports",
        "unsupported_explicit",
        "managed_ttl",
    ],
)
def test_sampler_storage_selection_and_rejection(asynchronous, method, case):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    caps = capabilities()
    kwargs = {"name": "sampler", "ttl_seconds": None, "wait": False}
    if case == "explicit_artifact" or case == "old_route_explicit":
        kwargs["storage_backend"] = "artifact"
    if case == "explicit_gpfs":
        kwargs["storage_backend"] = "gpfs"
    if case == "default_ttl":
        kwargs.pop("ttl_seconds")
    if case == "old_capability":
        caps.pop("sampler_export_backends")
        caps.pop("preferred_permanent_checkpoint_backend")
    if case == "missing_export_capability":
        caps.pop("sampler_export_backends")
    if case == "rollback":
        caps["preferred_permanent_checkpoint_backend"] = "gpfs"
        caps["sampler_export_backends"] = ["gpfs"]
    if case == "unsupported_explicit":
        caps["sampler_export_backends"] = ["gpfs"]
        kwargs["storage_backend"] = "artifact"
    if case == "malformed_exports":
        caps["sampler_export_backends"] = True
    if case == "managed_ttl":
        kwargs.update(storage_backend="artifact", ttl_seconds=3600)
    client._service.http.get.return_value = None if case == "null_response" else caps
    if case in ("old_route", "old_route_explicit", "forbidden", "unavailable"):
        code = 404 if case.startswith("old_route") else 403 if case == "forbidden" else 500
        client._service.http.get.side_effect = WeaverAPIError(code, "test", "fixture", False)

    def invoke():
        result = getattr(client, method)(**kwargs)
        return asyncio.run(result) if asynchronous else result

    error = None
    if case in ("old_route_explicit", "null_response", "malformed_exports", "unsupported_explicit"):
        error = RuntimeError
    elif case in ("forbidden", "unavailable"):
        error = WeaverAPIError
    elif case == "managed_ttl":
        error = ValueError
    if error:
        with pytest.raises(error):
            invoke()
        client._service.enqueue_operation.assert_not_called()
        if case == "managed_ttl":
            client._service.http.get.assert_not_called()
        return
    result = invoke()
    assert result is client._service.enqueue_operation.return_value
    route, body = client._service.enqueue_operation.call_args.args
    managed = case in ("preferred", "explicit_artifact")
    assert route.endswith("/export-sampler/managed" if managed else "/export-sampler")
    assert body.get("storage_backend") == (
        "artifact" if managed else "gpfs" if case == "explicit_gpfs" else None
    )
    assert body.get("ttl_seconds") == (3600 if case == "default_ttl" else None)
    if case in ("explicit_gpfs", "default_ttl"):
        client._service.http.get.assert_not_called()
    else:
        client._service.http.get.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_sampler_capability_is_refreshed_after_rollback(asynchronous, method):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    first = capabilities()
    second = {
        **first,
        "preferred_permanent_checkpoint_backend": "gpfs",
        "sampler_export_backends": ["gpfs"],
    }
    client._service.http.get.side_effect = [first, second]
    for _ in range(2):
        result = getattr(client, method)(ttl_seconds=None, wait=False)
        if asynchronous:
            asyncio.run(result)
    calls = client._service.enqueue_operation.call_args_list
    assert calls[0].args[0].endswith("/managed")
    assert calls[0].args[1]["storage_backend"] == "artifact"
    assert not calls[1].args[0].endswith("/managed")
    assert "storage_backend" not in calls[1].args[1]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_managed_sampler_preserves_public_waiting_result(asynchronous, method):
    client = _make_async_training_client() if asynchronous else _make_training_client()
    client._service.http.get.return_value = capabilities()
    payload = {"model_path": "weaver://mdl-123/checkpoints/sampler"}
    handle = MagicMock()
    handle.result = (
        AsyncMock(return_value=payload) if asynchronous else MagicMock(return_value=payload)
    )
    client._service.enqueue_operation.return_value = handle
    sampler = MagicMock()
    client._service.get_sampling_client = (
        AsyncMock(return_value=sampler) if asynchronous else MagicMock(return_value=sampler)
    )
    result = getattr(client, method)(ttl_seconds=None, wait=True)
    result = asyncio.run(result) if asynchronous else result
    assert result == (sampler if method.endswith("sampling_client") else payload["model_path"])
    assert client._service.enqueue_operation.call_args.args[0].endswith("/managed")
