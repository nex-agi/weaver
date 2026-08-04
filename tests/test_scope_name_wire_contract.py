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

"""End-to-end HTTP contract for UUID/slug/name scope resolution."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

import pytest

from weaver._http import WeaverAPIError
from weaver.service_client import ServiceClient

ORG_ID = "11111111-1111-4111-8111-111111111111"
PROJECT_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture()
def scope_server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def reply(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            requests.append({"method": "GET", "path": parsed.path, "query": query})
            if query.get("organization") == ["Shared"]:
                self.reply(
                    409,
                    {
                        "error": "ambiguous_scope_reference",
                        "message": "scope reference is ambiguous",
                        "retryable": False,
                    },
                )
                return
            self.reply(
                200,
                {
                    "organization": {"id": ORG_ID, "slug": "research"},
                    "project": {"id": PROJECT_ID, "slug": "training"},
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            requests.append({"method": "POST", "path": parsed.path, "json": body})
            self.reply(201, {"id": "session-1", **body})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_name_scope_resolves_over_http_before_session_creation(
    monkeypatch: pytest.MonkeyPatch,
    scope_server: tuple[str, list[dict[str, Any]]],
) -> None:
    monkeypatch.delenv("WEAVER_ORGANIZATION_ID", raising=False)
    monkeypatch.delenv("WEAVER_PROJECT_ID", raising=False)
    monkeypatch.delenv("WEAVER_ORGANIZATION", raising=False)
    monkeypatch.delenv("WEAVER_PROJECT", raising=False)
    base_url, requests = scope_server
    client = ServiceClient(
        base_url=base_url,
        api_key="sk-contract",
        organization="research",
        project="Training Run",
    )
    client.connect(ensure_session=False)
    try:
        client.ensure_session()
    finally:
        client.close()

    assert requests[0] == {
        "method": "GET",
        "path": "/api/v1/scope/resolve",
        "query": {"organization": ["research"], "project": ["Training Run"]},
    }
    assert requests[1]["json"]["organization_id"] == ORG_ID
    assert requests[1]["json"]["project_id"] == PROJECT_ID


def test_ambiguous_name_surfaces_structured_409(
    scope_server: tuple[str, list[dict[str, Any]]],
) -> None:
    base_url, _requests = scope_server
    client = ServiceClient(base_url=base_url, api_key="sk-contract")
    client.connect(ensure_session=False)
    try:
        with pytest.raises(WeaverAPIError) as caught:
            client.resolve_scope("Shared")
    finally:
        client.close()

    assert caught.value.status_code == 409
    assert caught.value.code == "ambiguous_scope_reference"
    assert caught.value.retryable is False
