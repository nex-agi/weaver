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

"""Readable W&B views of observations; never changes the raw/local records."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from urllib.parse import quote

from .types.metrics import MetricObservation, MetricObservations


def _chart_key(point: MetricObservation) -> str:
    key = point.name
    # Keep actual series dimensions, not descriptive/provenance labels. Default
    # group/rank values are implicit; other groups retain distinct short names.
    if point.name.startswith(("time/", "memory/")) and "operation" in point.labels:
        key += "/" + quote(point.labels["operation"], safe="-_")
    for label, default in (("loss_group", "1"), ("param_group", "0"), ("rank", "0")):
        if label in point.labels and point.labels[label] != default:
            key += f"/{label}_{quote(point.labels[label], safe='-_')}"
    if "layer" in point.labels:
        key += "/" + quote(point.labels["layer"], safe="/-_")
    return key


def _expert_histogram(points: list[MetricObservation]) -> Any | None:
    """Use expert IDs as bins, as Marin does; never re-bin assignment counts."""
    import wandb

    # W&B 0.19.8 allows at most 512 bins. Keep oversized/incomplete/ambiguous
    # observations in the existing table path rather than dropping or merging them.
    if len(points) > wandb.Histogram.MAX_LENGTH:
        return None
    try:
        ordered = sorted(points, key=lambda p: int(p.labels["expert"]))
        experts = [int(p.labels["expert"]) for p in ordered]
    except (KeyError, ValueError):
        return None
    if experts != list(range(len(points))):
        return None
    counts = [p.value for p in ordered]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in counts):
        return None
    return wandb.Histogram(np_histogram=(counts, list(range(len(points) + 1))))


class WandbMetricView:
    """Native scalar learning curves and per-layer expert histograms.

    Run config holds UUIDs/label definitions. No whole-training-history buffer,
    no per-expert metric keys, and no strings/bools masquerading as scalar charts.
    A fresh run is recommended when migrating from the old UUID-prefixed keys.
    """

    def __init__(self, run: Any) -> None:
        self.run = run
        self.defined: set[str] = set()
        self.contexts = list(run.config.get("weaver_metric_contexts", []))
        self.definitions = dict(run.config.get("weaver_metric_definitions", {}))
        self.forward_index = 0

    def _prefix(self, observations: MetricObservations) -> str:
        self.contexts = list(self.run.config.get("weaver_metric_contexts", []))
        for context in self.contexts:
            if (context["model_id"], context["trainer_run_id"]) == (
                observations.model_id,
                observations.epoch,
            ):
                return str(context["prefix"])
        models = list(dict.fromkeys(c["model_id"] for c in self.contexts))
        if observations.model_id not in models:
            models.append(observations.model_id)
        model_index = models.index(observations.model_id) + 1
        lifetime = 1 + sum(c["model_id"] == observations.model_id for c in self.contexts)
        prefix = f"model_{model_index}/" if model_index > 1 else ""
        if lifetime > 1:
            prefix += f"restart_{lifetime}/"
        self.contexts.append(
            {
                "model_id": observations.model_id,
                "trainer_run_id": observations.epoch,
                "prefix": prefix,
            }
        )
        self.run.config.update({"weaver_metric_contexts": self.contexts}, allow_val_change=True)
        return prefix

    def log(self, operation_id: str, observations: MetricObservations, *, commit: bool) -> None:
        """Publish a read-only display projection, retaining all statuses in a table."""
        import wandb

        prefix = self._prefix(observations)
        if observations.attempt is None:
            prefix += "forward/"
            self.forward_index += 1
            axis, step = prefix + "operation_index", self.forward_index
        else:
            axis, step = prefix + "step", observations.attempt
        if axis not in self.defined:
            self.run.define_metric(axis, hidden=True)
            self.defined.add(axis)
        values: dict[str, Any] = {axis: step}
        grouped: dict[str, list[MetricObservation]] = defaultdict(list)
        unavailable: set[str] = set()
        statuses = []
        self.definitions = dict(self.run.config.get("weaver_metric_definitions", {}))
        changed = False
        for point in observations.points:
            key = prefix + _chart_key(point)
            if key not in self.definitions:
                self.definitions[key] = {
                    "name": point.name,
                    "labels": {
                        k: v for k, v in point.labels.items() if k not in {"batch_id", "expert"}
                    },
                    "axis": axis,
                    "axis_meaning": (
                        "Optimizer attempts including skipped updates; resets on trainer restart"
                        if observations.attempt is not None
                        else "Forward-only operation index"
                    ),
                    "boolean_display": "0=false, 1=true" if isinstance(point.value, bool) else None,
                }
                changed = True
            if point.status != "ok" or isinstance(point.value, str):
                unavailable.add(key)
                statuses.append(
                    [
                        step,
                        point.name,
                        point.status,
                        point.value,
                        point.reason,
                        json.dumps(dict(point.labels), sort_keys=True),
                        operation_id,
                    ]
                )
            else:
                grouped[key].append(point)
        if changed:
            self.run.config.update(
                {"weaver_metric_definitions": self.definitions}, allow_val_change=True
            )
        for key, points in grouped.items():
            expert_counts = points[0].name == "router/load/counts/per_layer"
            if expert_counts and key not in unavailable:
                histogram = _expert_histogram(points)
                if histogram is not None:
                    hist_key = key + "/histogram"
                    if hist_key not in self.defined:
                        self.run.define_metric(hist_key, step_metric=axis, step_sync=False)
                        self.defined.add(hist_key)
                    values[hist_key] = histogram
                    continue
            if len(points) == 1 and not expert_counts:
                if key not in self.defined:
                    self.run.define_metric(key, step_metric=axis, step_sync=False)
                    self.defined.add(key)
                value = points[0].value
                values[key] = int(value) if isinstance(value, bool) else value
            else:
                # Unknown new dimensions must not silently overwrite each other.
                values[key + "/observations"] = wandb.Table(
                    columns=["step", "value", "labels"],
                    data=[
                        [step, p.value, json.dumps(dict(p.labels), sort_keys=True)] for p in points
                    ],
                )
        if statuses:
            values[prefix + "metrics/status"] = wandb.Table(
                columns=[
                    "step",
                    "metric",
                    "status",
                    "value",
                    "reason",
                    "labels",
                    "operation_id",
                ],
                data=statuses,
            )
        self.run.log(values, commit=commit)
