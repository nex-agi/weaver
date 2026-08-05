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

from __future__ import annotations

import httpx
import pytest
from click.testing import CliRunner

from weaver import WeaverAPIError as PublicWeaverAPIError
from weaver._http import WeaverAPIError, raise_for_response
from weaver.cli import cli


def test_api_error_is_part_of_public_sdk_surface() -> None:
    assert PublicWeaverAPIError is WeaverAPIError


def test_error_contract_extracts_request_retry_and_quota_details() -> None:
    response = httpx.Response(
        402,
        headers={"X-Request-ID": "req-header", "Retry-After": "17"},
        json={
            "error": "quota_exceeded",
            "message": "insufficient quota",
            "retryable": False,
            "request_id": "req-body",
            "details": {
                "required_nanos": "2500000000",
                "available_nanos": 1_000_000_000,
                "required_usd": "2.5",
                "available_usd": "1",
            },
        },
    )

    with pytest.raises(WeaverAPIError) as caught:
        raise_for_response(response)

    error = caught.value
    assert error.status_code == 402
    assert error.code == "quota_exceeded"
    assert error.message == "insufficient quota"
    assert error.retryable is False
    assert error.request_id == "req-body"
    assert error.retry_after == "17"
    assert error.required_nanos == 2_500_000_000
    assert error.available_nanos == 1_000_000_000
    assert error.required_usd == "2.5"
    assert error.available_usd == "1"


def test_error_contract_uses_headers_and_top_level_legacy_details() -> None:
    response = httpx.Response(
        429,
        headers={"X-Request-ID": "req-header", "Retry-After": "3"},
        json={
            "error": "rate_limited",
            "message": "slow down",
            "retryable": True,
            "required_nanos": 9,
            "available_nanos": 4,
        },
    )

    with pytest.raises(WeaverAPIError) as caught:
        raise_for_response(response)

    assert caught.value.request_id == "req-header"
    assert caught.value.retry_after == "3"
    assert caught.value.required_nanos == 9
    assert caught.value.available_nanos == 4
    assert caught.value.required_usd == "0.000000009"
    assert caught.value.available_usd == "0.000000004"


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (402, "quota_exceeded", "Quota exceeded"),
        (429, "rate_limited", "Rate limited"),
        (503, "quota_unavailable", "Service temporarily unavailable"),
    ],
)
def test_cli_renders_actionable_capacity_errors(monkeypatch, status, code, expected) -> None:
    def fail(*_args, **_kwargs):
        raise WeaverAPIError(
            status,
            code,
            "capacity error",
            retryable=status != 402,
            request_id="request-123",
            retry_after="11",
            required_nanos=2_000_000_000,
            available_nanos=500_000_000,
            required_usd="2",
            available_usd="0.5",
        )

    monkeypatch.setattr("weaver.service_client.ServiceClient.list_organizations", fail)
    result = CliRunner().invoke(cli, ["organizations", "list", "--format", "json"])

    assert result.exit_code == 1
    assert expected in result.output
    assert "request-123" in result.output
    if status == 402:
        assert "$2" in result.output
        assert "$0.5" in result.output
    if status == 429:
        assert "11" in result.output
