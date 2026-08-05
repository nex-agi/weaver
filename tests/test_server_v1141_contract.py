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

"""HTTP wire-contract tests matching weaver-server v1.14.1.

The response bodies and route/query names in this fixture mirror the public
Gin handlers at the v1.14.1 tag. Unlike the narrow MagicMock tests, these tests
exercise URL encoding, JSON serialization, headers and the real httpx layer.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

import pytest
from click.testing import CliRunner

from weaver._http import APIClient, WeaverAPIError
from weaver.cli import cli
from weaver.config import WeaverConfig
from weaver.service_client import ServiceClient

ORG_ID = "11111111-1111-4111-8111-111111111111"
PROJECT_ID = "22222222-2222-4222-8222-222222222222"
SESSION_ID = "33333333-3333-4333-8333-333333333333"


class ServerV1141Fixture:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.error_response: tuple[int, dict[str, Any], dict[str, str]] | None = None
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self.server is not None
        host = str(self.server.server_address[0])
        port = int(self.server.server_address[1])
        return f"http://{host}:{port}"

    def start(self) -> None:
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _request_body(self) -> dict[str, Any] | None:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return None
                return json.loads(self.rfile.read(length))

            def _reply(
                self,
                status: int,
                payload: dict[str, Any] | list[dict[str, Any]],
                headers: dict[str, str] | None = None,
            ) -> None:
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(encoded)

            def _record(
                self, body: dict[str, Any] | None = None
            ) -> tuple[str, dict[str, list[str]]]:
                parsed = urlparse(self.path)
                fixture.requests.append(
                    {
                        "method": self.command,
                        "path": parsed.path,
                        "query": parse_qs(parsed.query),
                        "json": body,
                        "api_key": self.headers.get("X-WEAVER-API-KEY"),
                    }
                )
                return parsed.path, parse_qs(parsed.query)

            def do_GET(self) -> None:  # noqa: N802
                path, _query = self._record()
                if fixture.error_response is not None:
                    status, payload, headers = fixture.error_response
                    self._reply(status, payload, headers)
                    return
                if path == "/api/v1/organizations":
                    self._reply(
                        200,
                        [
                            {
                                "id": ORG_ID,
                                "name": "Research",
                                "slug": "research",
                                "current_user_role": "admin",
                                "created_at": "2026-08-04T00:00:00Z",
                                "updated_at": "2026-08-04T00:00:00Z",
                            }
                        ],
                    )
                    return
                if path == f"/api/v1/organizations/{ORG_ID}/projects":
                    self._reply(
                        200,
                        [
                            {
                                "id": PROJECT_ID,
                                "organization_id": ORG_ID,
                                "name": "Default Project",
                                "slug": "default",
                                "is_default": True,
                                "read_only": False,
                                "current_user_role": "manager",
                                "created_at": "2026-08-04T00:00:00Z",
                                "updated_at": "2026-08-04T00:00:00Z",
                            }
                        ],
                    )
                    return
                if path == "/api/v1/quota/balance":
                    self._reply(
                        200,
                        {
                            "organization_id": ORG_ID,
                            "currency": "USD",
                            "granted_nanos": 5_000_000_000,
                            "reserved_nanos": 1_000_000_000,
                            "settled_nanos": 500_000_000,
                            "available_nanos": 3_500_000_000,
                            "granted_usd": "5",
                            "reserved_usd": "1",
                            "settled_usd": "0.5",
                            "available_usd": "3.5",
                        },
                    )
                    return
                if path == "/api/v1/quota/requests":
                    self._reply(
                        200,
                        {
                            "items": [],
                            "pagination": {"total_count": 0, "limit": 10, "offset": 20},
                        },
                    )
                    return
                self._reply(404, {"error": "not_found", "message": "not found", "retryable": False})

            def do_POST(self) -> None:  # noqa: N802
                body = self._request_body()
                path, _query = self._record(body)
                if path == "/api/v1/sessions":
                    self._reply(
                        201,
                        {
                            "id": SESSION_ID,
                            "organization_id": ORG_ID,
                            "project_id": PROJECT_ID,
                            "tags": body["tags"] if body else [],
                            "user_metadata": body["user_metadata"] if body else {},
                            "sdk_version": body["sdk_version"] if body else "",
                            "created_at": "2026-08-04T00:00:00Z",
                            "updated_at": "2026-08-04T00:00:00Z",
                        },
                    )
                    return
                if path == "/api/v1/quota/requests":
                    self._reply(
                        201,
                        {
                            "id": "44444444-4444-4444-8444-444444444444",
                            "organization_id": ORG_ID,
                            "email": "user@example.com",
                            "amount_nanos": 1,
                            "amount_usd": "0.000000001",
                            "currency": "USD",
                            "reason": body["reason"] if body else "",
                            "status": "pending",
                        },
                    )
                    return
                self._reply(404, {"error": "not_found", "message": "not found", "retryable": False})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


@pytest.fixture()
def server_v1141() -> Iterator[ServerV1141Fixture]:
    fixture = ServerV1141Fixture()
    fixture.start()
    try:
        yield fixture
    finally:
        fixture.close()


def test_cli_org_and_project_discovery_matches_v1141_wire_contract(
    server_v1141: ServerV1141Fixture,
) -> None:
    runner = CliRunner()
    organizations = runner.invoke(
        cli,
        [
            "organizations",
            "list",
            "--format",
            "json",
            "--base-url",
            server_v1141.base_url,
            "--api-key",
            "sk-contract",
        ],
    )
    projects = runner.invoke(
        cli,
        [
            "projects",
            "list",
            "--organization-id",
            ORG_ID,
            "--format",
            "json",
            "--base-url",
            server_v1141.base_url,
            "--api-key",
            "sk-contract",
        ],
    )

    assert organizations.exit_code == 0
    assert json.loads(organizations.output) == [{"id": ORG_ID, "name": "Research", "role": "admin"}]
    assert projects.exit_code == 0
    assert json.loads(projects.output) == [
        {"id": PROJECT_ID, "name": "Default Project", "role": "manager"}
    ]
    assert all(item["api_key"] == "sk-contract" for item in server_v1141.requests)


@pytest.mark.parametrize(
    ("project_id", "expected_scope"),
    [(None, {}), (PROJECT_ID, {"project_id": PROJECT_ID})],
)
def test_session_default_and_project_only_fallback_match_v1141_wire_contract(
    monkeypatch: pytest.MonkeyPatch,
    server_v1141: ServerV1141Fixture,
    project_id: str | None,
    expected_scope: dict[str, str],
) -> None:
    monkeypatch.delenv("WEAVER_ORGANIZATION_ID", raising=False)
    monkeypatch.delenv("WEAVER_PROJECT_ID", raising=False)
    client = ServiceClient(
        base_url=server_v1141.base_url,
        api_key="sk-contract",
        project_id=project_id,
    )
    client.connect(ensure_session=False)
    try:
        created = client.ensure_session()
    finally:
        client.close()

    assert created["id"] == SESSION_ID
    request = server_v1141.requests[-1]
    assert request["path"] == "/api/v1/sessions"
    assert {key: request["json"][key] for key in expected_scope} == expected_scope
    for key in {"organization_id", "project_id"} - expected_scope.keys():
        assert key not in request["json"]


def test_user_quota_methods_match_v1141_paths_queries_and_decimal_body(
    server_v1141: ServerV1141Fixture,
) -> None:
    client = ServiceClient(
        base_url=server_v1141.base_url,
        api_key="sk-contract",
        organization_id=ORG_ID,
    )
    client.connect(ensure_session=False)
    try:
        balance = client.get_quota_balance()
        listed = client.list_quota_requests(limit=10, offset=20)
        created = client.request_quota("0.000000001", reason="one nano")
    finally:
        client.close()

    assert balance["available_nanos"] == 3_500_000_000
    assert listed["pagination"] == {"total_count": 0, "limit": 10, "offset": 20}
    assert created["amount_nanos"] == 1
    assert server_v1141.requests[-3]["query"] == {"org_id": [ORG_ID]}
    assert server_v1141.requests[-2]["query"] == {
        "org_id": [ORG_ID],
        "limit": ["10"],
        "offset": ["20"],
    }
    assert server_v1141.requests[-1]["query"] == {"org_id": [ORG_ID]}
    assert server_v1141.requests[-1]["json"] == {
        "amount_usd": "0.000000001",
        "reason": "one nano",
    }


@pytest.mark.parametrize(
    ("status", "payload", "headers", "expected"),
    [
        (
            402,
            {
                "error": "quota_exceeded",
                "message": "insufficient quota",
                "retryable": False,
                "details": {
                    "required_nanos": 2_500_000_000,
                    "available_nanos": 1_000_000_000,
                    "required_usd": "2.5",
                    "available_usd": "1",
                },
            },
            {},
            {
                "code": "quota_exceeded",
                "retryable": False,
                "required_nanos": 2_500_000_000,
                "available_nanos": 1_000_000_000,
                "required_usd": "2.5",
                "available_usd": "1",
                "retry_after": None,
            },
        ),
        (
            429,
            {
                "error": "rate_limited",
                "message": "rate limit exceeded",
                "retryable": True,
                "details": {
                    "retry_after_seconds": 2,
                    "retry_after_milliseconds": 1_500,
                },
            },
            {"Retry-After": "2"},
            {
                "code": "rate_limited",
                "retryable": True,
                "required_nanos": None,
                "available_nanos": None,
                "required_usd": None,
                "available_usd": None,
                "retry_after": "2",
            },
        ),
        (
            503,
            {
                "error": "quota_pricing_unavailable",
                "message": "quota pricing unavailable",
                "retryable": True,
            },
            {},
            {
                "code": "quota_pricing_unavailable",
                "retryable": True,
                "required_nanos": None,
                "available_nanos": None,
                "required_usd": None,
                "available_usd": None,
                "retry_after": None,
            },
        ),
    ],
)
def test_capacity_errors_match_v1141_wire_contract(
    server_v1141: ServerV1141Fixture,
    status: int,
    payload: dict[str, Any],
    headers: dict[str, str],
    expected: dict[str, Any],
) -> None:
    server_v1141.error_response = (status, payload, headers)
    client = APIClient(
        WeaverConfig(base_url=server_v1141.base_url, api_key="sk-contract"),
        max_retries=1,
    )
    try:
        with pytest.raises(WeaverAPIError) as caught:
            client.get("/api/v1/quota/balance")
    finally:
        client.close()

    error = caught.value
    assert error.status_code == status
    for field, value in expected.items():
        assert getattr(error, field) == value
    assert error.details == payload.get("details", {})
    # v1.14.1 has no request-id response middleware yet; SDK leaves it absent.
    assert error.request_id is None
