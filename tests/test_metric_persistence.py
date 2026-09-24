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
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from weaver import (
    AsyncOperationHandle,
    AsyncServiceClient,
    MetricsStoreConfig,
    OperationHandle,
    ServiceClient,
)
from weaver.metric_persistence import MetricPersistence


class FakeConfig(dict):
    def update(self, values, **_kwargs):
        super().update(values)


@pytest.fixture
def wandb_mock(monkeypatch):
    run = Mock(
        id="run-1",
        entity="entity",
        project="project",
        settings=SimpleNamespace(base_url="https://api.wandb.ai", mode="online"),
        config=FakeConfig(),
    )
    module = SimpleNamespace(
        run=None,
        Settings=SimpleNamespace,
        login=Mock(),
        init=Mock(return_value=run),
        Table=lambda **kwargs: {"_type": "table", **kwargs},
    )
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def response():
    return {
        "metrics": {"loss": 1.25},
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
        },
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
        "step": 4,
        "trainer_run_id": "epoch-a",
        "counter_scope": "trainer_lifetime",
        "name": "loss/task",
        "status": "ok",
        "labels": {"scope": "operation"},
        "reason": None,
        "value": 1.25,
    }
    assert json.loads(layer_file.read_text())["labels"] == {"layer": "decoder/1"}
    assert json.loads(unsupported_file.read_text())["status"] == "unsupported"


@pytest.mark.parametrize("store_enabled", [True, False])
def test_wandb_link_logs_scalar_points(wandb_mock, tmp_path, store_enabled):
    sink = MetricPersistence(
        store=MetricsStoreConfig(enabled=store_enabled, path=tmp_path),
        wandb_link="https://wandb.ai/entity/project/runs/run-1",
    )
    sink.persist("operation-1", response())
    sink.close()

    wandb_mock.init.assert_called_once()
    init_call = wandb_mock.init.call_args.kwargs
    assert init_call["entity"] == "entity"
    assert init_call["project"] == "project"
    assert init_call["reinit"] is False  # W&B 0.19.8 accepts booleans, not string modes.
    assert init_call["dir"] == (str(tmp_path) if store_enabled else None)
    assert (tmp_path / "metrics-model-a").exists() is store_enabled
    run = wandb_mock.init.return_value
    run.log.assert_called_once()
    logged = run.log.call_args.args[0]
    assert logged["loss/task"] == 1.25
    assert logged["grad/norm/per_layer/decoder/1"] == 2.5
    assert "metrics/status" in logged
    assert not any("params/norm/total" in key for key in logged)
    assert run.log.call_args.kwargs == {"commit": True}
    run.finish.assert_called_once()


def test_wandb_link_rejects_different_active_run(wandb_mock, tmp_path, caplog):
    active = wandb_mock.init.return_value
    active.id = "different"
    wandb_mock.run = active
    sink = MetricPersistence(
        local_path=tmp_path, wandb_link="https://wandb.ai/entity/project/runs/run-1"
    )
    sink.persist("operation-1", response())
    # The sink is best effort: local storage succeeds and the W&B mismatch is logged.
    assert (tmp_path / "metrics-model-a/loss/task/observations.jsonl").exists()
    assert "wandb_link differs from the active run" in caplog.text
    wandb_mock.init.assert_not_called()
    active.log.assert_not_called()


@pytest.mark.parametrize("store_enabled", [True, False])
def test_wandb_link_uses_matching_active_run_without_finishing_it(
    wandb_mock, tmp_path, store_enabled
):
    active = wandb_mock.init.return_value
    wandb_mock.run = active

    sink = MetricPersistence(
        store=MetricsStoreConfig(enabled=store_enabled, path=tmp_path),
        wandb_link="https://wandb.ai/entity/project/runs/run-1",
    )
    sink.persist("operation-1", response())
    sink.close()

    active.log.assert_called_once()
    assert active.log.call_args.kwargs == {"commit": False}
    active.finish.assert_not_called()
    wandb_mock.login.assert_not_called()
    wandb_mock.init.assert_not_called()
    assert (tmp_path / "metrics-model-a").exists() is store_enabled


@pytest.mark.parametrize("client_type", [ServiceClient, AsyncServiceClient])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("custom_path", [True, False])
def test_metrics_store_config_controls_both_clients(
    monkeypatch, tmp_path, client_type, enabled, custom_path
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "custom" if custom_path else None
    client = client_type(metrics_store=MetricsStoreConfig(enabled=enabled, path=path))
    sink = client._metric_persistence
    sink.persist("operation-store", response())
    expected_parent = path or tmp_path / "weaver/.logs"
    expected_file = expected_parent / "metrics-model-a/loss/task/observations.jsonl"
    assert expected_file.exists() is enabled
    if enabled:
        assert json.loads(expected_file.read_text())["value"] == 1.25
    else:
        assert not list(tmp_path.iterdir())
        assert not sink._persisted_operations


@pytest.mark.parametrize("client_type", [ServiceClient, AsyncServiceClient])
def test_metrics_store_default_and_legacy_path(monkeypatch, tmp_path, client_type):
    monkeypatch.chdir(tmp_path)
    default = client_type()._metric_persistence
    assert default.store == MetricsStoreConfig()
    default.persist("operation-default", response())
    assert (tmp_path / "weaver/.logs/metrics-model-a/loss/task/observations.jsonl").exists()

    legacy = client_type(metrics_path=tmp_path / "legacy")._metric_persistence
    legacy.persist("operation-legacy", response())
    assert (tmp_path / "legacy/metrics-model-a/loss/task/observations.jsonl").exists()


@pytest.mark.parametrize("client_type", [ServiceClient, AsyncServiceClient])
def test_metrics_store_rejects_ambiguous_path(client_type, tmp_path):
    with pytest.raises(ValueError, match="metrics_store or metrics_path, not both"):
        client_type(metrics_store=MetricsStoreConfig(), metrics_path=tmp_path)


@pytest.mark.parametrize("handle_type", [OperationHandle, AsyncOperationHandle])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("nested", [True, False])
def test_handles_preserve_results_and_persist_once(
    monkeypatch, tmp_path, handle_type, enabled, nested
):
    monkeypatch.chdir(tmp_path)
    sink = MetricPersistence(store=MetricsStoreConfig(enabled=enabled, path=tmp_path))
    raw = response()
    if nested:
        raw = {"status": "succeeded", "result": raw}
    original = deepcopy(raw)
    handle = handle_type(
        client=SimpleNamespace(metric_sink=sink),
        operation_id="operation-1",
        _cached={"status": "done", "response": raw},
    )
    for _ in range(2):
        result = handle.result()
        if handle_type is AsyncOperationHandle:
            result = asyncio.run(result)
        assert result == original
    assert raw == original
    assert handle.metric_observations.points[0].value == 1.25
    assert handle.metric_observations.points[2].status == "unsupported"
    if enabled:
        loss_file = tmp_path / "metrics-model-a/loss/task/observations.jsonl"
        assert len(loss_file.read_text().splitlines()) == 1
    else:
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("store_enabled", [True, False])
def test_no_wandb_link_never_touches_caller_run(wandb_mock, tmp_path, store_enabled, caplog):
    caller_run = wandb_mock.init.return_value
    wandb_mock.run = caller_run
    sink = MetricPersistence(
        store=MetricsStoreConfig(enabled=store_enabled, path=tmp_path), wandb_link=None
    )
    sink.persist("operation-outer-owner", response())
    sink.close()
    wandb_mock.login.assert_not_called()
    wandb_mock.init.assert_not_called()
    caller_run.log.assert_not_called()
    caller_run.finish.assert_not_called()
    assert not caplog.records
