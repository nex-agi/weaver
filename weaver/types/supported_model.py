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

"""User-facing supported-model catalog types."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

MODE_PRICE_KINDS = (
    "training_tokens",
    "sampling_prefill_tokens",
    "sampling_cached_prefill_tokens",
    "sampling_output_tokens",
)


@dataclass(frozen=True)
class SupportedModelPrice:
    """One effective catalog price, represented without floating-point loss."""

    unit: str
    unit_price_micros: int
    version: str | None = None

    @property
    def unit_price_usd(self) -> Decimal:
        """Return the USD price for one catalog unit."""

        return Decimal(self.unit_price_micros) / Decimal(1_000_000)

    @classmethod
    def from_payload(cls, payload: Any) -> SupportedModelPrice | None:
        if not isinstance(payload, dict):
            return None
        try:
            unit_price_micros = int(payload["unit_price_micros"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            unit=str(payload.get("unit") or ""),
            unit_price_micros=unit_price_micros,
            version=str(payload["version"]) if payload.get("version") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "unit_price_micros": self.unit_price_micros,
            "unit_price_usd": str(self.unit_price_usd),
            "version": self.version,
        }


@dataclass(frozen=True)
class SupportedTrainingMode:
    """A model mode and its four independently configurable effective prices."""

    mode: str
    display_name: str
    prices: dict[str, SupportedModelPrice]

    @property
    def price(self) -> SupportedModelPrice | None:
        """Compatibility alias for the training-token price."""

        return self.prices.get("training_tokens")

    def price_for(self, usage_kind: str) -> SupportedModelPrice | None:
        """Return this mode's price for one catalog usage kind."""

        return self.prices.get(usage_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "display_name": self.display_name,
            "prices": {key: price.to_dict() for key, price in self.prices.items()},
        }


@dataclass(frozen=True)
class SupportedModel:
    """A usable model with only its explicitly supported, fully priced modes."""

    id: str
    name: str
    status: str
    visibility: str | None
    training_modes: tuple[SupportedTrainingMode, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SupportedModel:
        prices = payload.get("prices")
        prices = prices if isinstance(prices, dict) else {}

        def prices_for_mode(mode: str) -> dict[str, SupportedModelPrice]:
            result: dict[str, SupportedModelPrice] = {}
            for kind in MODE_PRICE_KINDS:
                price = SupportedModelPrice.from_payload(
                    prices.get(f"{kind}:{mode}")
                    or (prices.get(kind) if kind != "training_tokens" else None)
                )
                if price is not None:
                    result[kind] = price
            return result

        raw_modes = payload.get("training_modes")
        if isinstance(raw_modes, list):
            declared_modes = {str(mode).strip().lower().replace("-", "_") for mode in raw_modes}
        else:
            # Compatibility with servers that predate the capability field:
            # infer only from an exact mode-qualified training quote.
            declared_modes = {
                mode for mode in ("lora", "full_ft") if f"training_tokens:{mode}" in prices
            }
        training_modes_list: list[SupportedTrainingMode] = []
        for mode, display_name in (("lora", "LoRA"), ("full_ft", "Full-FT")):
            mode_prices = prices_for_mode(mode)
            if mode in declared_modes and len(mode_prices) == len(MODE_PRICE_KINDS):
                training_modes_list.append(
                    SupportedTrainingMode(
                        mode=mode,
                        display_name=display_name,
                        prices=mode_prices,
                    )
                )
        training_modes = tuple(training_modes_list)
        metadata = payload.get("metadata")
        return cls(
            id=str(payload.get("id") or ""),
            name=str(payload.get("name") or ""),
            status=str(payload.get("status") or "available"),
            visibility=(
                str(payload["visibility"]) if payload.get("visibility") is not None else None
            ),
            training_modes=training_modes,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    @property
    def sampling_prices(self) -> dict[str, SupportedModelPrice]:
        """Compatibility view of LoRA sampling prices.

        New code should use ``training_mode(...).prices`` so Full-FT sampling
        is never mistaken for LoRA pricing.
        """

        lora = self.training_mode("lora")
        if lora is None:
            return {}
        return {key: price for key, price in lora.prices.items() if key != "training_tokens"}

    def training_mode(self, mode: str) -> SupportedTrainingMode | None:
        """Return one normalized training-mode entry."""

        normalized = mode.strip().lower().replace("-", "_")
        return next((item for item in self.training_modes if item.mode == normalized), None)

    def to_dict(self, *, mode: str | None = None) -> dict[str, Any]:
        normalized = mode.strip().lower().replace("-", "_") if mode else None
        training_modes = [
            item.to_dict()
            for item in self.training_modes
            if normalized is None or item.mode == normalized
        ]
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "visibility": self.visibility,
            "training_modes": training_modes,
            "metadata": self.metadata,
        }
