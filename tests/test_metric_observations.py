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

"""The same pure decoder is used by sync and async operation handles."""

from copy import deepcopy

import pytest

from weaver import AsyncOperationHandle, MetricObservations, OperationHandle


def response():
    return {
        "metrics": {"loss": 1.5},
        "metric_observations": {
            "version": 1,
            "model_id": "m",
            "attempt": 10,
            "counter_scope": "trainer_lifetime",
            "epoch": "worker-epoch",
            "points": [
                {
                    "name": "grad/norm/per_layer",
                    "status": "ok",
                    "value": 2.5,
                    "labels": {"layer": "decoder/1"},
                },
                {"name": "optim/skipped_step", "status": "ok", "value": False, "labels": {}},
                {"name": "optim/skip_reason", "status": "ok", "value": "none", "labels": {}},
                {
                    "name": "params/norm/total",
                    "status": "unsupported",
                    "reason": "storage_path",
                    "labels": {},
                },
            ],
        },
    }


@pytest.mark.parametrize("handle_cls", [OperationHandle, AsyncOperationHandle])
def test_shared_decoder_is_io_free_and_keeps_raw_response(handle_cls):
    raw = response()
    original = deepcopy(raw)
    handle = handle_cls(client=None, operation_id="op", _cached={"status": "done", "response": raw})
    observations = handle.metric_observations
    assert observations.attempt == 10
    assert observations.epoch == "worker-epoch"
    assert observations.points[0].labels == {"layer": "decoder/1"}
    assert observations.points[1].value is False
    assert observations.points[2].value == "none"
    assert observations.points[3].value is None
    with pytest.raises(TypeError):
        observations.points[0].labels["layer"] = "other"
    assert raw == original
    assert handle.response is raw


@pytest.mark.parametrize("raw", [None, {}, {"metrics": {"loss": 1.5}}])
def test_old_trainer_compatibility(raw):
    assert MetricObservations.from_response(raw) is None


def test_nested_envelope_and_forward_only():
    raw = response()
    raw["metric_observations"]["attempt"] = None
    assert MetricObservations.from_response({"result": raw}).attempt is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, [1.0, 2.0], {"x": 1}])
def test_malformed_available_scalar(value):
    raw = response()
    raw["metric_observations"]["points"][0]["value"] = value
    with pytest.raises(ValueError):
        MetricObservations.from_response(raw)


@pytest.mark.parametrize(
    "field,value",
    [("version", 2), ("version", True), ("attempt", True), ("attempt", 0), ("points", None)],
)
def test_invalid_envelope(field, value):
    raw = response()
    raw["metric_observations"][field] = value
    with pytest.raises(ValueError):
        MetricObservations.from_response(raw)


def test_unavailable_not_zero():
    raw = response()
    raw["metric_observations"]["points"][3]["value"] = 0
    with pytest.raises(ValueError):
        MetricObservations.from_response(raw)
