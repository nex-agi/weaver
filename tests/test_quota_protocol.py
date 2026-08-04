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

import asyncio
from unittest.mock import AsyncMock, MagicMock

from weaver.async_service_client import AsyncServiceClient
from weaver.service_client import ServiceClient


def test_sync_quota_uses_default_server_scope_when_org_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_ORGANIZATION_ID", raising=False)
    client = ServiceClient()
    client._http = MagicMock()
    client._http.get.return_value = {"available_nanos": 1_250_000_000, "available_usd": "1.25"}

    result = client.get_quota_balance()

    assert result["available_usd"] == "1.25"
    client._http.get.assert_called_once_with("/api/v1/quota/balance")


def test_sync_quota_uses_explicit_org_and_preserves_decimal_amount() -> None:
    client = ServiceClient(organization_id="org-constructor")
    client._http = MagicMock()
    client._http.post.return_value = {"id": "request-1", "amount_usd": "0.000000001"}

    result = client.request_quota("0.000000001", reason="one nano", organization_id="org-explicit")

    assert result["id"] == "request-1"
    client._http.post.assert_called_once_with(
        "/api/v1/quota/requests",
        json={"amount_usd": "0.000000001", "reason": "one nano"},
        params={"org_id": "org-explicit"},
        max_retries=1,
    )


def test_sync_list_quota_requests_preserves_pagination() -> None:
    client = ServiceClient(organization_id="org-1")
    client._http = MagicMock()
    client._http.get.return_value = {"items": [], "pagination": {"total_count": 0}}

    result = client.list_quota_requests(limit=10, offset=20)

    assert result["items"] == []
    client._http.get.assert_called_once_with(
        "/api/v1/quota/requests",
        params={"limit": 10, "offset": 20, "org_id": "org-1"},
    )


def test_async_quota_methods_match_sync_contract() -> None:
    async def run() -> None:
        client = AsyncServiceClient(organization_id="org-async")
        client._http = MagicMock()
        client._http.get = AsyncMock(
            side_effect=[
                {"available_nanos": 2_000_000_000, "available_usd": "2"},
                {"items": [], "pagination": {"total_count": 0}},
            ]
        )
        client._http.post = AsyncMock(return_value={"id": "request-async"})

        balance = await client.get_quota_balance()
        requests = await client.list_quota_requests(limit=5, offset=0)
        created = await client.request_quota("3.5", reason="more training")

        assert balance["available_nanos"] == 2_000_000_000
        assert requests["items"] == []
        assert created["id"] == "request-async"
        client._http.get.assert_any_await("/api/v1/quota/balance", params={"org_id": "org-async"})
        client._http.post.assert_awaited_once_with(
            "/api/v1/quota/requests",
            json={"amount_usd": "3.5", "reason": "more training"},
            params={"org_id": "org-async"},
            max_retries=1,
        )

    asyncio.run(run())
