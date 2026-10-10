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

"""Pending diagnostics remain visible without changing operation completion."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from weaver import operations
from weaver.operations import AsyncOperationHandle, OperationHandle


def pending(code="waiting_for_scheduling", age=60):
    return {
        "id": "op-1",
        "status": "pending",
        "pending_info": {
            "code": code,
            "message": "Trainer is waiting for cluster scheduling; training has not started.",
            "wait_seconds": age,
            "resources": {
                "nodes": 4,
                "gpus_per_node": 8,
                "trainer_memory_gib_per_node": 560,
            },
        },
    }


@pytest.mark.parametrize("async_mode", [False, True])
def test_wait_reports_reason_changes_and_rate_limits_repeats(async_mode, monkeypatch, caplog):
    clock = [100.0]
    ticks = iter([1, 60, 1, 1])
    payloads = iter(
        [
            pending(age=61),
            pending(age=121),
            pending("trainer_starting", age=122),
            {"id": "op-1", "status": "done", "response": {"loss": 0.1}},
        ]
    )

    def sleep(_):
        clock[0] += next(ticks)

    async def async_sleep(delay):
        sleep(delay)

    def get(_):
        return next(payloads)

    async def async_get(path):
        return get(path)

    monkeypatch.setattr(
        operations, "time", SimpleNamespace(sleep=sleep, monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(operations.asyncio, "sleep", async_sleep)
    monkeypatch.setattr(operations, "_operation_poll_delays", lambda: iter([1] * 4))
    caplog.set_level(logging.WARNING, logger="weaver.operations")
    if async_mode:
        handle = AsyncOperationHandle(SimpleNamespace(get=async_get), "op-1", pending())
        assert asyncio.run(handle.result()) == {"loss": 0.1}
    else:
        handle = OperationHandle(SimpleNamespace(get=get), "op-1", pending())
        assert handle.result() == {"loss": 0.1}
    logs = [r.getMessage() for r in caplog.records]
    assert len(logs) == 3  # first reason, 60s repeat, immediate stage change
    assert "op-1 pending for 60s [waiting_for_scheduling]" in logs[0]
    assert "4 nodes, 8 GPUs/node, 560 GiB trainer memory/node" in logs[0]
    assert "121s" in logs[1]
    assert "[trainer_starting]" in logs[2]
    assert handle.pending_info is None


@pytest.mark.parametrize("async_mode", [False, True])
def test_old_server_still_completes_and_reports_long_wait(async_mode, monkeypatch, caplog):
    clock = [0.0]
    payloads = iter(
        [{"id": "op-1", "status": "pending"}, {"id": "op-1", "status": "done", "response": 42}]
    )

    def sleep(_):
        clock[0] += 31

    async def async_sleep(delay):
        sleep(delay)

    async def async_get(_):
        return next(payloads)

    monkeypatch.setattr(
        operations, "time", SimpleNamespace(sleep=sleep, monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(operations.asyncio, "sleep", async_sleep)
    monkeypatch.setattr(operations, "_operation_poll_delays", lambda: iter([1, 1]))
    initial = {"id": "op-1", "status": "pending"}
    caplog.set_level(logging.WARNING, logger="weaver.operations")
    if async_mode:
        handle = AsyncOperationHandle(SimpleNamespace(get=async_get), "op-1", initial)
        assert asyncio.run(handle.result()) == 42
    else:
        handle = OperationHandle(SimpleNamespace(get=lambda _: next(payloads)), "op-1", initial)
        assert handle.result() == 42
    assert len(caplog.records) == 1
    assert "31s [queued]" in caplog.records[0].getMessage()


def test_short_pending_is_quiet_and_accessor_is_optional(caplog):
    handle = OperationHandle(None, "op-1", {"status": "pending"})
    assert handle.pending_info is None
    handle._report_pending(operations.time.monotonic())
    assert not caplog.records
    handle._cached = {"status": "pending", "pending_info": "invalid"}
    assert handle.pending_info is None


def test_reported_scheduling_wait_updates_immediately_during_short_startup(caplog):
    handle = OperationHandle(None, "op-1", pending(age=0))
    caplog.set_level(logging.WARNING, logger="weaver.operations")
    started = operations.time.monotonic()
    handle._report_pending(started)
    handle._cached = pending("trainer_starting", age=1)
    handle._report_pending(started)
    assert len(caplog.records) == 2
    assert "[trainer_starting]" in caplog.records[-1].getMessage()
