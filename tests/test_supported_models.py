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
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from click.testing import CliRunner

from weaver.async_service_client import AsyncServiceClient
from weaver.cli import cli
from weaver.service_client import ServiceClient
from weaver.types import SupportedModel


def catalog_model(name: str = "Qwen/Qwen3.5-0.8B:262144") -> dict:
    return {
        "id": "model-1",
        "name": name,
        "status": "available",
        "visibility": "public",
        "training_modes": ["lora", "full_ft"],
        "metadata": {"context_length": 262144},
        "prices": {
            "training_tokens:lora": {
                "unit": "million_tokens",
                "unit_price_micros": 335_000,
                "version": "north-ledger-sku-2867",
            },
            "training_tokens:full_ft": {
                "unit": "million_tokens",
                "unit_price_micros": 3_350_000,
                "version": "north-ledger-sku-2868",
            },
            "sampling_prefill_tokens:lora": {
                "unit": "million_tokens",
                "unit_price_micros": 110_000,
                "version": "lora-prefill",
            },
            "sampling_cached_prefill_tokens:lora": {
                "unit": "million_tokens",
                "unit_price_micros": 22_000,
                "version": "lora-cached",
            },
            "sampling_output_tokens:lora": {
                "unit": "million_tokens",
                "unit_price_micros": 440_000,
                "version": "lora-output",
            },
            "sampling_prefill_tokens:full_ft": {
                "unit": "million_tokens",
                "unit_price_micros": 1_100_000,
                "version": "full-prefill",
            },
            "sampling_cached_prefill_tokens:full_ft": {
                "unit": "million_tokens",
                "unit_price_micros": 220_000,
                "version": "full-cached",
            },
            "sampling_output_tokens:full_ft": {
                "unit": "million_tokens",
                "unit_price_micros": 4_400_000,
                "version": "full-output",
            },
        },
    }


def test_detailed_supported_models_have_two_independent_prices() -> None:
    client = ServiceClient(organization_id="org-1")
    client._http = MagicMock()
    client._http.get.return_value = {
        "items": [catalog_model()],
        "pagination": {"total_count": 1},
    }

    models = client.list_supported_models(detailed=True)

    assert len(models) == 1
    assert isinstance(models[0], SupportedModel)
    assert models[0].training_mode("lora").price.unit_price_usd == Decimal("0.335")
    assert models[0].training_mode("full-ft").price.unit_price_usd == Decimal("3.35")
    assert models[0].training_mode("lora").price_for(
        "sampling_cached_prefill_tokens"
    ).unit_price_usd == Decimal("0.022")
    assert models[0].training_mode("full-ft").price_for(
        "sampling_output_tokens"
    ).unit_price_usd == Decimal("4.4")
    client._http.get.assert_called_once_with(
        "/api/v1/model-catalog",
        params={"limit": 100, "offset": 0, "organization_id": "org-1"},
    )


def test_legacy_sampling_fills_only_the_explicitly_quoted_training_mode() -> None:
    payload = catalog_model()
    payload["prices"] = {
        "training_tokens:lora": {
            "unit": "million_tokens",
            "unit_price_micros": 200_000,
            "version": "legacy",
        },
        "sampling_prefill_tokens": {
            "unit": "million_tokens",
            "unit_price_micros": 90_000,
            "version": "legacy-prefill",
        },
        "sampling_cached_prefill_tokens": {
            "unit": "million_tokens",
            "unit_price_micros": 18_000,
            "version": "legacy-cached",
        },
        "sampling_output_tokens": {
            "unit": "million_tokens",
            "unit_price_micros": 300_000,
            "version": "legacy-output",
        },
    }

    payload["training_modes"] = ["lora"]
    model = SupportedModel.from_payload(payload)
    assert [mode.mode for mode in model.training_modes] == ["lora"]
    assert model.training_mode("lora").price.unit_price_micros == 200_000
    assert (
        model.training_mode("lora").price_for("sampling_prefill_tokens").unit_price_micros == 90_000
    )
    assert model.training_mode("full_ft") is None


def test_missing_full_ft_quote_means_full_ft_is_not_supported() -> None:
    payload = catalog_model()
    payload["training_modes"] = ["lora", "full_ft"]
    del payload["prices"]["training_tokens:full_ft"]

    model = SupportedModel.from_payload(payload)

    assert [mode.mode for mode in model.training_modes] == ["lora"]
    assert model.training_mode("full_ft") is None


def test_async_detailed_supported_models_match_sync_shape() -> None:
    async def run() -> None:
        client = AsyncServiceClient(organization_id="org-async")
        client._http = MagicMock()
        client._http.get = AsyncMock(
            return_value={"items": [catalog_model()], "pagination": {"total_count": 1}}
        )

        models = await client.list_supported_models(detailed=True)

        assert models[0].training_mode("lora").display_name == "LoRA"
        assert client._http.get.await_args_list == [
            call(
                "/api/v1/model-catalog",
                params={"limit": 100, "offset": 0, "organization_id": "org-async"},
            )
        ]

    asyncio.run(run())


def test_cli_renders_one_complete_price_row_per_mode(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)
    client = MagicMock()
    client.list_supported_model_details.return_value = [
        SupportedModel.from_payload(catalog_model())
    ]
    with patch("weaver.cli.ServiceClient", return_value=client):
        result = CliRunner().invoke(cli, ["list", "supported-models"])

    assert result.exit_code == 0
    assert result.output.count("Qwen/Qwen3.5-0.8B:262144") == 1
    assert "LoRA" in result.output
    assert "Full-FT" in result.output
    for price in ("$0.335", "$0.11", "$0.022", "$0.44", "$3.35", "$1.1", "$0.22", "$4.4"):
        assert price in result.output
    assert "Cached input" in result.output
    client.connect.assert_called_once_with(ensure_session=False)
    client.close.assert_called_once_with()


def test_cli_json_can_select_one_training_mode(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)
    client = MagicMock()
    client.list_supported_model_details.return_value = [
        SupportedModel.from_payload(catalog_model())
    ]
    with patch("weaver.cli.ServiceClient", return_value=client):
        result = CliRunner().invoke(
            cli,
            ["list", "supported-models", "--mode", "full-ft", "--format", "json"],
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [item["mode"] for item in payload[0]["training_modes"]] == ["full_ft"]
    prices = payload[0]["training_modes"][0]["prices"]
    assert prices["training_tokens"]["unit_price_micros"] == 3_350_000
    assert prices["sampling_output_tokens"]["unit_price_micros"] == 4_400_000


@pytest.mark.parametrize("output_format", ["table", "json", "names"])
def test_cli_mode_filter_hides_models_that_do_not_support_the_mode(
    monkeypatch, output_format: str
) -> None:
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)
    payload = catalog_model("Qwen/LoRAOnly")
    payload["training_modes"] = ["lora"]
    client = MagicMock()
    client.list_supported_model_details.return_value = [SupportedModel.from_payload(payload)]
    with patch("weaver.cli.ServiceClient", return_value=client):
        result = CliRunner().invoke(
            cli,
            [
                "list",
                "supported-models",
                "--mode",
                "full-ft",
                "--format",
                output_format,
            ],
        )

    assert result.exit_code == 0
    assert "Qwen/LoRAOnly" not in result.output
    assert "Not supported" not in result.output
    if output_format == "json":
        assert json.loads(result.output) == []
    elif output_format == "table":
        assert "0 supported models" in result.output
