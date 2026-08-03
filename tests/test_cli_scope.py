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

"""Tests for organization/project discovery CLI commands."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from weaver.cli import cli


def test_list_organizations_is_sessionless(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)
    client = MagicMock()
    client.list_organizations.return_value = [{"id": "org-1", "name": "Research"}]
    with patch("weaver.cli.ServiceClient", return_value=client):
        result = CliRunner().invoke(cli, ["list", "organizations", "--format", "json"])

    assert result.exit_code == 0
    client.connect.assert_called_once_with(ensure_session=False)
    client.list_organizations.assert_called_once_with()
    client.close.assert_called_once_with()


def test_list_projects_accepts_org_id_and_is_sessionless(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)
    client = MagicMock()
    client.list_projects.return_value = [{"id": "project-1", "name": "Default Project"}]
    with patch("weaver.cli.ServiceClient", return_value=client) as constructor:
        result = CliRunner().invoke(
            cli,
            ["list", "projects", "--organization-id", "org-1", "--format", "json"],
        )

    assert result.exit_code == 0
    constructor.assert_called_once_with(base_url=None, api_key=None, organization_id="org-1")
    client.connect.assert_called_once_with(ensure_session=False)
    client.list_projects.assert_called_once_with("org-1")
    client.close.assert_called_once_with()
