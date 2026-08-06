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

"""Streaming batching and bounded submit-ahead for packed training."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    from .operations import AsyncOperationHandle
    from .types import AdamParams, Datum

T = TypeVar("T")


class _AsyncTrainingClient(Protocol):
    async def forward_backward(
        self, data: Sequence["Datum"], loss_fn: str, *, wait: Literal[False]
    ) -> "AsyncOperationHandle": ...

    async def optim_step(
        self, params: "AdamParams", *, wait: Literal[False]
    ) -> "AsyncOperationHandle": ...


@dataclass(frozen=True, slots=True)
class TokenBudgetBatch(Generic[T]):
    """One token-budgeted request selected from a streaming iterator."""

    items: tuple[T, ...]
    tokens: int


class TokenBudgetBatcher(Generic[T]):
    """Build best-effort token-budgeted requests with bounded memory.

    The target request size is ``global_batch_size * max_tokens_per_gpu``.
    The trainer remains responsible for actual packing. An item that would
    cross the target starts the next request, so the corpus is never planned or
    tokenized up front.

    Args:
        source: Stream of already rendered or tokenized source samples.
        length_fn: Returns the valid-token length of one sample.
        global_batch_size: Multiplier for the target request token budget.
        max_tokens_per_gpu: Token-budget unit and maximum source-sample length.
        drop_last: Drop an incomplete final request when the source ends.
    """

    def __init__(
        self,
        source: Iterable[T],
        *,
        length_fn: Callable[[T], int],
        global_batch_size: int,
        max_tokens_per_gpu: int,
        drop_last: bool = True,
    ) -> None:
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if max_tokens_per_gpu <= 0:
            raise ValueError("max_tokens_per_gpu must be positive")
        self._source = source
        self._length_fn = length_fn
        self._global_batch_size = global_batch_size
        self._max_tokens_per_gpu = max_tokens_per_gpu
        self._drop_last = drop_last

    def __iter__(self) -> Iterator[TokenBudgetBatch[T]]:
        target_tokens = self._global_batch_size * self._max_tokens_per_gpu
        items: list[T] = []
        total_tokens = 0

        for item in self._source:
            length = int(self._length_fn(item))
            if length <= 0:
                raise ValueError("sample token lengths must be positive")
            if length > self._max_tokens_per_gpu:
                raise ValueError(
                    f"sample length {length} exceeds max_tokens_per_gpu="
                    f"{self._max_tokens_per_gpu}"
                )

            if items and total_tokens + length > target_tokens:
                yield TokenBudgetBatch(tuple(items), total_tokens)
                items = []
                total_tokens = 0

            items.append(item)
            total_tokens += length
            if total_tokens == target_tokens:
                yield TokenBudgetBatch(tuple(items), total_tokens)
                items = []
                total_tokens = 0

        if items and not self._drop_last:
            yield TokenBudgetBatch(tuple(items), total_tokens)


@dataclass(frozen=True, slots=True)
class CompletedTrainingStep:
    """Results for one ordered forward/backward and optimizer pair."""

    step: int
    forward_backward: Any
    optimizer: Any


@dataclass(slots=True)
class _PendingTrainingStep:
    step: int
    forward_backward: Any
    optimizer: Any


class SubmitAheadQueue:
    """Bound a pipeline of ordered async Weaver training operations.

    Call :meth:`submit`, start preparing the next source batch, then call
    :meth:`wait_for_room`. This ordering overlaps preparation with remote
    training while retaining only operation handles in the queue.

    Args:
        training_client: Async Weaver training client for one model.
        submit_ahead: Number of future steps allowed behind the running step.
        loss_fn: Weaver loss function used by every submitted step.
    """

    def __init__(
        self,
        training_client: _AsyncTrainingClient,
        *,
        submit_ahead: int = 1,
        loss_fn: str = "cross_entropy",
    ) -> None:
        if submit_ahead < 0:
            raise ValueError("submit_ahead must be non-negative")
        self._client = training_client
        self._submit_ahead = submit_ahead
        self._loss_fn = loss_fn
        self._pending: deque[_PendingTrainingStep] = deque()

    @property
    def pending_count(self) -> int:
        """Return the number of submitted steps not yet resolved."""

        return len(self._pending)

    async def submit(self, step: int, data: Sequence["Datum"], optimizer: "AdamParams") -> None:
        """Submit one forward/backward and optimizer pair in sequence order.

        Args:
            step: Caller-defined optimizer-step index.
            data: Source samples for this step.
            optimizer: Optimizer parameters for this step.
        """

        if len(self._pending) > self._submit_ahead:
            raise RuntimeError("call wait_for_room() before submitting another step")
        forward_backward = await self._client.forward_backward(data, self._loss_fn, wait=False)
        try:
            optim = await self._client.optim_step(optimizer, wait=False)
        except Exception:
            try:
                await forward_backward.result()
            except Exception:
                pass
            raise
        self._pending.append(_PendingTrainingStep(step, forward_backward, optim))

    async def wait_for_room(self) -> CompletedTrainingStep | None:
        """Resolve the oldest step when the configured lookahead is full."""

        if len(self._pending) <= self._submit_ahead:
            return None
        return await self._finish_oldest()

    async def drain(self) -> list[CompletedTrainingStep]:
        """Resolve every pending step before a checkpoint, eval, or shutdown."""

        completed: list[CompletedTrainingStep] = []
        first_error: Exception | None = None
        while self._pending:
            try:
                completed.append(await self._finish_oldest())
            except Exception as exc:  # keep draining already-submitted work
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return completed

    async def _finish_oldest(self) -> CompletedTrainingStep:
        pending = self._pending.popleft()
        results: tuple[Any, Any] = await asyncio.gather(
            pending.forward_backward.result(),
            pending.optimizer.result(),
            return_exceptions=True,
        )
        forward_result, optimizer_result = results
        if isinstance(forward_result, BaseException):
            raise forward_result
        if isinstance(optimizer_result, BaseException):
            raise optimizer_result
        return CompletedTrainingStep(pending.step, forward_result, optimizer_result)
