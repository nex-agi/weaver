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

import asyncio
import json
import sys
from types import SimpleNamespace

from weaver import AsyncOperationHandle, OperationHandle
from weaver.metric_persistence import MetricPersistence


def response():
    return {
        "metric_observations": {
            "version": 1,
            "model_id": "model-a",
            "attempt": 4,
            "epoch": "epoch-a",
            "counter_scope": "trainer_lifetime",
            "points": [
                {
                    "name": "loss/task",
                    "status": "ok",
                    "value": 1.25,
                    "labels": {"scope": "operation"},
                },
                {
                    "name": "grad/norm/per_layer",
                    "status": "ok",
                    "value": 2.5,
                    "labels": {"layer": "decoder/1"},
                },
                {
                    "name": "params/norm/total",
                    "status": "unsupported",
                    "reason": "storage_path",
                    "labels": {},
                },
            ],
        }
    }


def test_local_metrics_are_partitioned_by_model_and_metric(tmp_path):
    sink = MetricPersistence(local_path=tmp_path)
    sink.persist("operation-1", response())

    loss_file = tmp_path / "metrics-model-a" / "loss" / "task" / "observations.jsonl"
    layer_file = tmp_path / "metrics-model-a" / "grad" / "norm" / "per_layer" / "observations.jsonl"
    unsupported_file = (
        tmp_path / "metrics-model-a" / "params" / "norm" / "total" / "observations.jsonl"
    )
    assert json.loads(loss_file.read_text()) == {
        "operation_id": "operation-1",
        "model_id": "model-a",
        "attempt": 4,
        "epoch": "epoch-a",
        "counter_scope": "trainer_lifetime",
        "name": "loss/task",
        "status": "ok",
        "labels": {"scope": "operation"},
        "reason": None,
        "value": 1.25,
    }
    assert json.loads(layer_file.read_text())["labels"] == {"layer": "decoder/1"}
    assert json.loads(unsupported_file.read_text())["status"] == "unsupported"


def test_wandb_link_logs_scalar_points(monkeypatch, tmp_path):
    calls = []

    class Run:
        id = "run-1"
        entity = "entity"
        project = "project"
        settings = SimpleNamespace(base_url="https://api.wandb.ai", mode="online")

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values, commit=True):
            calls.append((values, commit))

    fake_wandb = SimpleNamespace(
        run=None,
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        login=lambda **kwargs: calls.append((kwargs, True)),
        init=lambda **kwargs: (calls.append((kwargs, True)) or Run()),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    sink = MetricPersistence(
        local_path=tmp_path, wandb_link="https://wandb.ai/entity/project/runs/run-1"
    )
    sink.persist("operation-1", response())

    init_call = next(values for values, _ in calls if "entity" in values)
    assert init_call["entity"] == "entity"
    assert init_call["project"] == "project"
    assert init_call["reinit"] is False  # W&B 0.19.8 accepts booleans, not string modes.
    logged = calls[-1][0]
    assert any(key.endswith("/loss/task/scope=operation") for key in logged)
    assert any(key.endswith("/grad/norm/per_layer/layer=decoder%2F1") for key in logged)
    assert not any("params/norm/total" in key for key in logged)
    assert calls[-1][1] is True


def test_operation_handles_persist_completed_results(tmp_path):
    sink = MetricPersistence(local_path=tmp_path)
    client = SimpleNamespace(metric_sink=sink)
    handle = OperationHandle(
        client=client,
        operation_id="operation-2",
        _cached={"status": "done", "response": response()},
    )
    assert handle.result() == response()
    assert handle.result() == response()
    loss_file = tmp_path / "metrics-model-a" / "loss" / "task" / "observations.jsonl"
    assert len(loss_file.read_text().splitlines()) == 1


def test_operation_handles_persist_server_callback_result_envelope(tmp_path):
    sink = MetricPersistence(local_path=tmp_path)
    client = SimpleNamespace(metric_sink=sink)
    callback_payload = {"status": "succeeded", "result": response()}
    handle = OperationHandle(
        client=client,
        operation_id="operation-nested",
        _cached={"status": "done", "response": callback_payload},
    )
    assert handle.result() == callback_payload
    loss_file = tmp_path / "metrics-model-a" / "loss" / "task" / "observations.jsonl"
    assert len(loss_file.read_text().splitlines()) == 1


def test_async_operation_handles_persist_off_event_loop(tmp_path):
    sink = MetricPersistence(local_path=tmp_path)
    client = SimpleNamespace(metric_sink=sink)
    handle = AsyncOperationHandle(
        client=client,
        operation_id="operation-3",
        _cached={"status": "done", "response": response()},
    )
    assert asyncio.run(handle.result()) == response()
    loss_file = tmp_path / "metrics-model-a" / "loss" / "task" / "observations.jsonl"
    assert len(loss_file.read_text().splitlines()) == 1


def test_default_path_is_weaver_logs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sink = MetricPersistence()
    assert sink.local_path == (tmp_path / "weaver/.logs").absolute()


def test_wandb_link_rejects_different_active_run(monkeypatch, tmp_path):
    active = SimpleNamespace(
        id="different",
        entity="entity",
        project="project",
        settings=SimpleNamespace(base_url="https://api.wandb.ai", mode="online"),
    )
    fake_wandb = SimpleNamespace(run=active)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    sink = MetricPersistence(
        local_path=tmp_path, wandb_link="https://wandb.ai/entity/project/runs/run-1"
    )
    sink.persist("operation-1", response())
    # The sink is best effort: local storage succeeds and the W&B mismatch is logged.
    assert sink.local_path.exists()


def test_wandb_link_uses_matching_active_run_without_finishing_it(monkeypatch, tmp_path):
    calls = []

    class Run:
        id = "run-1"
        entity = "entity"
        project = "project"
        settings = SimpleNamespace(base_url="https://api.wandb.ai", mode="online")

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values, commit=True):
            calls.append((values, commit))

        def finish(self):
            calls.append(("finish", True))

    active = Run()
    fake_wandb = SimpleNamespace(run=active)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    sink = MetricPersistence(
        local_path=tmp_path, wandb_link="https://wandb.ai/entity/project/runs/run-1"
    )
    sink.persist("operation-1", response())
    sink.close()

    assert calls[-1][1] is False
    assert not any(call[0] == "finish" for call in calls)
