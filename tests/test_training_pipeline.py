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

import pytest

import weaver.training_pipeline as training_pipeline
from weaver import types
from weaver.training_pipeline import (
    SubmitAheadQueue,
    TokenBudgetBatcher,
    predict_packed_batch_shape,
)


def test_predict_packed_batch_shape_balances_dp_and_microbatches():
    shape = predict_packed_batch_shape(
        [8, 7, 6, 5],
        dp_size=2,
        max_tokens_per_gpu=10,
    )

    assert shape.samples == 4
    assert shape.tokens == 26
    assert shape.tokens_per_dp == (13, 13)
    assert shape.samples_per_dp == (2, 2)
    assert shape.microbatches_per_dp == 2
    assert shape.max_microbatch_tokens == 8
    assert shape.global_microbatches == 4
    assert shape.dp_safe


def test_predict_packed_batch_shape_detects_second_stage_overflow():
    shape = predict_packed_batch_shape(
        [6, 6, 6],
        dp_size=1,
        max_tokens_per_gpu=10,
    )

    assert shape.microbatches_per_dp == 2
    assert shape.max_microbatch_tokens == 12


def test_token_budget_batcher_carries_second_stage_overflow():
    batches = list(
        TokenBudgetBatcher(
            range(4),
            length_fn=lambda _: 6,
            global_batch_size=2,
            dp_size=1,
            max_tokens_per_gpu=10,
        )
    )

    assert [batch.items for batch in batches] == [(0, 1), (2, 3)]
    assert all(batch.shape.max_microbatch_tokens <= 10 for batch in batches)


def test_token_budget_batcher_reads_only_through_next_boundary():
    reads: list[int] = []

    def source():
        index = 0
        while True:
            reads.append(index)
            yield index
            index += 1

    batches = iter(
        TokenBudgetBatcher(
            source(),
            length_fn=lambda _: 5,
            global_batch_size=4,
            dp_size=2,
            max_tokens_per_gpu=10,
        )
    )

    first = next(batches)

    assert first.items == tuple(range(8))
    assert first.shape.global_microbatches == 4
    assert reads == list(range(9))


def test_token_budget_batcher_carries_overflow_into_next_request():
    batches = iter(
        TokenBudgetBatcher(
            range(17),
            length_fn=lambda _: 5,
            global_batch_size=4,
            dp_size=2,
            max_tokens_per_gpu=10,
        )
    )

    assert next(batches).items == tuple(range(8))
    assert next(batches).items == tuple(range(8, 16))
    with pytest.raises(StopIteration):
        next(batches)


def test_token_budget_batcher_avoids_repartitioning_every_short_sample(monkeypatch):
    calls = 0
    original = training_pipeline.predict_packed_batch_shape

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(training_pipeline, "predict_packed_batch_shape", counted)
    batch = next(
        iter(
            TokenBudgetBatcher(
                range(1281),
                length_fn=lambda _: 1,
                global_batch_size=128,
                dp_size=8,
                max_tokens_per_gpu=10,
            )
        )
    )

    assert batch.shape.global_microbatches == 128
    assert batch.shape.tokens >= 1280 * 0.98
    assert calls <= 2


def test_token_budget_batcher_rejects_sample_over_budget():
    batches = iter(
        TokenBudgetBatcher(
            [11],
            length_fn=lambda value: value,
            global_batch_size=2,
            dp_size=1,
            max_tokens_per_gpu=10,
        )
    )

    with pytest.raises(ValueError, match="exceeds max_tokens_per_gpu"):
        next(batches)


class FakeHandle:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    async def result(self):
        self._events.append(f"wait_{self._name}")
        await asyncio.sleep(0)
        return {"result": {"metrics": {self._name: 1.0}}}


class FakeTrainingClient:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._forward_step = 0
        self._optimizer_step = 0

    async def forward_backward(self, data, loss_fn, *, wait):
        assert data and loss_fn == "cross_entropy" and wait is False
        name = f"fb{self._forward_step}"
        self._forward_step += 1
        self._events.append(f"submit_{name}")
        return FakeHandle(self._events, name)

    async def optim_step(self, optimizer, *, wait):
        assert optimizer is not None and wait is False
        name = f"opt{self._optimizer_step}"
        self._optimizer_step += 1
        self._events.append(f"submit_{name}")
        return FakeHandle(self._events, name)


def test_submit_ahead_queue_bounds_and_drains_in_step_order():
    async def exercise():
        events: list[str] = []
        queue = SubmitAheadQueue(FakeTrainingClient(events), submit_ahead=1)
        optimizer = types.AdamParams()
        datum = types.Datum(model_input=types.ModelInput.from_ints([1]))

        await queue.submit(0, [datum], optimizer)
        assert await queue.wait_for_room() is None
        await queue.submit(1, [datum], optimizer)
        first = await queue.wait_for_room()
        remaining = await queue.drain()
        return events, first, remaining

    events, first, remaining = asyncio.run(exercise())

    assert first is not None and first.step == 0
    assert [result.step for result in remaining] == [1]
    assert events == [
        "submit_fb0",
        "submit_opt0",
        "submit_fb1",
        "submit_opt1",
        "wait_fb0",
        "wait_opt0",
        "wait_fb1",
        "wait_opt1",
    ]


def test_submit_ahead_queue_requires_wait_for_room():
    async def exercise():
        queue = SubmitAheadQueue(FakeTrainingClient([]), submit_ahead=0)
        optimizer = types.AdamParams()
        datum = types.Datum(model_input=types.ModelInput.from_ints([1]))
        await queue.submit(0, [datum], optimizer)
        with pytest.raises(RuntimeError, match="wait_for_room"):
            await queue.submit(1, [datum], optimizer)
        await queue.drain()

    asyncio.run(exercise())
