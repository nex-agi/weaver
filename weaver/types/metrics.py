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

"""Local metric storage configuration and read-only Trainer observations."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_STATUSES = {"ok", "disabled", "not_sampled", "not_applicable", "unsupported", "invalid"}


@dataclass(frozen=True)
class MetricsStoreConfig:
    """Configure SDK-owned local metric storage, independently of W&B.

    Args:
        enabled: Write metric JSONL files. Disabling storage does not change
            collection or returned observations.
        path: Parent directory; None selects ``./weaver/.logs`` relative to the
            client construction directory. Ignored when storage is disabled.
    """

    enabled: bool = True
    path: str | os.PathLike[str] | None = None


@dataclass(frozen=True)
class MetricObservation:
    """One scalar/category or unavailable observation, optionally layer-labelled.

    A missing value is never a zero. Inspect ``status`` before consuming ``value``.
    Numeric vectors are represented as multiple points with layer/expert labels.
    """

    name: str
    status: str
    value: float | int | bool | str | None
    labels: Mapping[str, str]
    reason: str | None = None


@dataclass(frozen=True)
class MetricObservations:
    """Observations for one operation and process-lifetime optimizer attempt.

    ``attempt`` includes skipped updates; it is not a durable model step. Pair it
    with ``epoch`` and ``model_id``. Forward-only operations have no attempt.
    """

    model_id: str
    attempt: int | None
    epoch: str | None
    counter_scope: str
    points: tuple[MetricObservation, ...]

    @classmethod
    def from_response(cls, response: Any) -> MetricObservations | None:
        """Read an existing response without polling or changing legacy metrics.

        Returns None for older trainers or an absent response. Raises ValueError
        for an explicitly present malformed/unsupported envelope; raw responses
        remain accessible even if a newer version is not understood by this SDK.
        """
        if not isinstance(response, Mapping):
            return None
        payload = response.get("metric_observations")
        if payload is None and isinstance(response.get("result"), Mapping):
            payload = response["result"].get("metric_observations")
        if payload is None:
            return None
        if (
            not isinstance(payload, Mapping)
            or type(payload.get("version")) is not int
            or payload["version"] != 1
        ):
            raise ValueError("Unsupported metric_observations envelope version")
        model_id, attempt = payload.get("model_id"), payload.get("attempt")
        epoch, scope = payload.get("epoch"), payload.get("counter_scope")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Metric observations require a model_id")
        if attempt is not None and (type(attempt) is not int or attempt < 1):
            raise ValueError("Metric attempt must be a positive integer or None")
        if epoch is not None and not isinstance(epoch, str):
            raise ValueError("Metric epoch must be a string or None")
        if scope != "trainer_lifetime":
            raise ValueError("Unsupported metric counter scope")
        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("Metric observations points must be a list")
        points = []
        for point in raw_points:
            if not isinstance(point, Mapping):
                raise ValueError("Metric observation must be an object")
            name, status, value = point.get("name"), point.get("status"), point.get("value")
            labels, reason = point.get("labels", {}), point.get("reason")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(status, str)
                or status not in _STATUSES
            ):
                raise ValueError("Invalid metric name or status")
            if not isinstance(labels, Mapping) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items()
            ):
                raise ValueError("Metric labels must map strings to strings")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("Metric reason must be a string")
            if status == "ok":
                if type(value) not in (float, int, bool, str) or (
                    type(value) is float and not math.isfinite(value)
                ):
                    raise ValueError("Available metric value must be a finite JSON scalar")
            elif value is not None:
                raise ValueError("Unavailable metric must not carry a value")
            points.append(
                MetricObservation(name, status, value, MappingProxyType(dict(labels)), reason)
            )
        return cls(model_id, attempt, epoch, scope, tuple(points))
