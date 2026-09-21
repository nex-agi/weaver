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

"""Pure CPU presentation tests; never constructs model tensors."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from weaver._wandb_metrics import WandbMetricView, _expert_histogram
from weaver.types.metrics import MetricObservation, MetricObservations


class Config(dict):
    def update(self, values, **_kwargs):
        super().update(deepcopy(values))


class Run:
    def __init__(self):
        self.config = Config()
        self.rows = []
        self.definitions = {}

    def define_metric(self, name, **kwargs):
        self.definitions[name] = kwargs

    def log(self, values, commit):
        self.rows.append((values, commit))


class Histogram:
    MAX_LENGTH = 512

    def __init__(self, *, np_histogram):
        self.histogram, self.bins = np_histogram


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Table=lambda **kwargs: {"_type": "table", **kwargs},
            Histogram=Histogram,
        ),
    )
    return Run()


def point(name, value, **labels):
    return MetricObservation(name, "ok", value, labels)


def envelope(*points, model="model-uuid", epoch="trainer-uuid", attempt=5):
    return MetricObservations(model, attempt, epoch, "trainer_lifetime", tuple(points))


def test_plain_names_numeric_booleans_and_explicit_text_table(run):
    view = WandbMetricView(run)
    raw = envelope(
        point("grad/norm/total", 2.5, phase="finalized_unscaled_pre_clip"),
        point("optim/update_successful", True),
        point("optim/skipped_step", False),
        point("optim/skip_reason", "none"),
        MetricObservation("router/z_loss/mean", "not_applicable", None, {}, "objective_inactive"),
    )
    view.log("operation-uuid", raw, commit=True)
    values, commit = run.rows[-1]
    assert values["grad/norm/total"] == 2.5
    assert type(values["optim/update_successful"]) is int
    assert values["optim/skipped_step"] == 0
    assert "optim/skip_reason" not in values
    assert "operation_id" not in values
    assert "router/z_loss/mean" not in values
    assert all("uuid" not in key and "phase=" not in key for key in values)
    assert run.definitions["grad/norm/total"]["step_metric"] == "step"
    assert run.definitions["step"]["hidden"] is True
    assert values["metrics/status"]["data"][0][1:4] == ["optim/skip_reason", "ok", "none"]
    assert raw.points[1].value is True  # projection cannot change local/raw types
    assert commit is True


def test_models_restarts_and_new_sink_instances_do_not_mix(run):
    p = point("grad/norm/total", 1.0)
    a, b = WandbMetricView(run), WandbMetricView(run)
    a.log("op1", envelope(p), commit=True)
    b.log("op2", envelope(p, model="model-b"), commit=False)
    a.log("op3", envelope(p, epoch="restart-uuid", attempt=1), commit=True)
    resumed = WandbMetricView(run)
    resumed.log("op4", envelope(p, model="model-b", attempt=7), commit=False)
    assert "grad/norm/total" in run.rows[0][0]
    assert "model_2/grad/norm/total" in run.rows[1][0]
    assert "restart_2/grad/norm/total" in run.rows[2][0]
    assert run.rows[3][0]["model_2/step"] == 7
    assert len(run.config["weaver_metric_contexts"]) == 3


def test_operation_and_parameter_groups_keep_separate_scalar_series(run):
    view = WandbMetricView(run)
    view.log(
        "op",
        envelope(
            point("time/operation_ms", 6000, operation="forward_backward"),
            point("time/operation_ms", 40, operation="optim_step"),
            point("optim/learning_rate", 1e-5, param_group="0"),
            point("optim/learning_rate", 2e-5, param_group="1"),
            point("router/aux_loss/load_balancing/mean", 0.1, loss_group="1"),
            point("router/aux_loss/load_balancing/mean", 0.2, loss_group="2"),
        ),
        commit=True,
    )
    values = run.rows[-1][0]
    assert values["time/operation_ms/forward_backward"] == 6000
    assert values["time/operation_ms/optim_step"] == 40
    assert values["optim/learning_rate/param_group_1"] == 2e-5
    assert values["router/aux_loss/load_balancing/mean/loss_group_2"] == 0.2


def test_forward_only_does_not_invent_optimizer_step(run):
    view = WandbMetricView(run)
    view.log("op", envelope(point("loss/task", 1), attempt=None), commit=True)
    values = run.rows[-1][0]
    assert values["forward/operation_index"] == 1
    assert "step" not in values


def test_per_layer_values_are_scalar_time_series_with_sparse_steps(run):
    view = WandbMetricView(run)
    for step in (5, 11):
        points = [point("grad/norm/per_layer", n + step, layer=f"decoder/{n}") for n in (10, 2, 1)]
        view.log(f"op-{step}", envelope(*points, attempt=step), commit=True)
        values = run.rows[-1][0]
        assert values["step"] == step
        for n in (10, 2, 1):
            key = f"grad/norm/per_layer/decoder/{n}"
            assert values[key] == n + step
            assert run.definitions[key]["step_metric"] == "step"
    assert len(run.rows) == 2  # No invented samples in the interval gaps.


def test_expert_histograms_preserve_expert_ids_and_counts_per_layer(run):
    p = [
        point(
            "router/load/counts/per_layer",
            100 * layer + expert,
            layer=f"decoder/{layer}",
            expert=str(expert),
        )
        for layer, expert in ((1, 1), (1, 0), (2, 0), (2, 1))
    ]
    WandbMetricView(run).log("op", envelope(*p), commit=True)
    values = run.rows[-1][0]
    assert len(values) == 3  # axis + ONE native histogram per layer
    for layer in (1, 2):
        key = f"router/load/counts/per_layer/decoder/{layer}/histogram"
        assert values[key].histogram == [100 * layer, 100 * layer + 1]
        assert values[key].bins == [0, 1, 2]
        assert run.definitions[key]["step_metric"] == "step"


def test_duplicate_layer_values_are_retained_in_table(run):
    p = point("grad/norm/per_layer", 1, layer="decoder/1")
    WandbMetricView(run).log("op", envelope(p, p), commit=True)
    rows = run.rows[-1][0]["grad/norm/per_layer/decoder/1/observations"]["data"]
    assert len(rows) == 2


@pytest.mark.parametrize("experts", [[0, 2], [0, 0], list(range(513))])
def test_incomplete_duplicate_or_oversized_experts_use_lossless_table(run, experts):
    p = [
        point("router/load/counts/per_layer", e + 1, layer="decoder/0", expert=str(e))
        for e in experts
    ]
    WandbMetricView(run).log("op", envelope(*p), commit=True)
    values = run.rows[-1][0]
    key = "router/load/counts/per_layer/decoder/0"
    assert key + "/histogram" not in values
    assert len(values[key + "/observations"]["data"]) == len(experts)


def test_unavailable_expert_does_not_become_a_zero_or_truncated_histogram(run):
    p = point("router/load/counts/per_layer", 3, layer="decoder/0", expert="0")
    missing = MetricObservation(p.name, "invalid", None, {"layer": "decoder/0", "expert": "1"})
    WandbMetricView(run).log("op", envelope(p, missing), commit=True)
    values = run.rows[-1][0]
    key = "router/load/counts/per_layer/decoder/0"
    assert key + "/histogram" not in values
    assert values[key + "/observations"]["data"][0][1] == 3
    assert values["metrics/status"]["data"][0][2] == "invalid"


def test_real_wandb_histogram_accepts_precomputed_expert_counts():
    wandb = pytest.importorskip("wandb")
    p = [point("router/load/counts/per_layer", n, expert=str(i)) for i, n in enumerate([3, 0, 8])]
    histogram = _expert_histogram(p)
    assert isinstance(histogram, wandb.Histogram)
    assert histogram.to_json() == {"_type": "histogram", "values": [3, 0, 8], "bins": [0, 1, 2, 3]}


def test_unknown_scalar_dimensions_are_preserved_in_table(run):
    p = [
        point("grad/clipping/coefficient", 0.5, optimizer="one"),
        point("grad/clipping/coefficient", 0.8, optimizer="two"),
    ]
    WandbMetricView(run).log("op", envelope(*p), commit=True)
    rows = run.rows[-1][0]["grad/clipping/coefficient/observations"]["data"]
    assert len(rows) == 2
    assert [row[1] for row in rows] == [0.5, 0.8]
