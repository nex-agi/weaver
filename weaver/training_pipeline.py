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
import heapq
import math
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
_MAX_BOUNDARY_PROBES = 32
_MIN_SAFE_FILL_PERCENT = 98


class _AsyncTrainingClient(Protocol):
    async def forward_backward(
        self, data: Sequence["Datum"], loss_fn: str, *, wait: Literal[False]
    ) -> "AsyncOperationHandle": ...

    async def optim_step(
        self, params: "AdamParams", *, wait: Literal[False]
    ) -> "AsyncOperationHandle": ...


@dataclass(frozen=True, slots=True)
class PackedBatchShape:
    """Predicted shape after Weaver's balanced DP partition.

    Attributes:
        samples: Number of source samples in the request.
        tokens: Total source tokens in the request.
        microbatches_per_dp: Synchronized microbatch count on each DP rank.
        samples_per_dp: Source-sample counts assigned to each DP rank.
        tokens_per_dp: Token counts assigned to each DP rank.
    """

    samples: int
    tokens: int
    microbatches_per_dp: int
    samples_per_dp: tuple[int, ...]
    tokens_per_dp: tuple[int, ...]

    @property
    def global_microbatches(self) -> int:
        """Return the predicted global packed microbatch count."""

        return self.microbatches_per_dp * len(self.tokens_per_dp)

    @property
    def dp_safe(self) -> bool:
        """Return whether every DP rank has enough samples to form the shape."""

        return bool(self.microbatches_per_dp) and self.microbatches_per_dp <= min(
            self.samples_per_dp, default=0
        )


@dataclass(frozen=True, slots=True)
class TokenBudgetBatch(Generic[T]):
    """One source request selected from a streaming iterator."""

    items: tuple[T, ...]
    shape: PackedBatchShape


class _Partition:
    __slots__ = ("items", "total")

    def __init__(self) -> None:
        self.items: list[tuple[int, int]] = []
        self.total = 0

    def add(self, index: int, value: int) -> None:
        self.items.append((index, value))
        self.total += value

    def merge(self, other: "_Partition") -> None:
        self.items.extend(other.items)
        self.total += other.total

    def __lt__(self, other: "_Partition") -> bool:
        return (self.total, len(self.items), self.items) < (
            other.total,
            len(other.items),
            other.items,
        )


class _PartitionState:
    __slots__ = ("partitions", "width")

    def __init__(self, items: Sequence[tuple[int, int]], width: int) -> None:
        self.width = width
        self.partitions = [_Partition() for _ in range(width)]
        for partition, (index, value) in zip(self.partitions, items, strict=False):
            partition.add(index, value)
        self.partitions.sort(reverse=True)

    @property
    def spread(self) -> int:
        return self.partitions[0].total - self.partitions[-1].total

    def merge(self, other: "_PartitionState") -> None:
        for index in range(self.width):
            self.partitions[index].merge(other.partitions[self.width - 1 - index])
        self.partitions.sort(reverse=True)

    def __lt__(self, other: "_PartitionState") -> bool:
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.partitions[0] > other.partitions[0]


def _balanced_partitions(lengths: Sequence[int], count: int) -> list[list[int]]:
    """Match Weaver trainer's balanced DP partition for length-only planning."""

    if len(lengths) < count:
        return [list(range(rank, len(lengths), count)) for rank in range(count)]

    states: list[_PartitionState] = []
    for length, index in sorted((length, index) for index, length in enumerate(lengths)):
        heapq.heappush(states, _PartitionState([(index, length)], count))
    while len(states) > 1:
        first = heapq.heappop(states)
        second = heapq.heappop(states)
        first.merge(second)
        heapq.heappush(states, first)
    return [sorted(index for index, _ in part.items) for part in states[0].partitions]


def predict_packed_batch_shape(
    lengths: Sequence[int], *, dp_size: int, max_tokens_per_gpu: int
) -> PackedBatchShape:
    """Predict the current balanced trainer's request shape from token lengths.

    Args:
        lengths: Valid-token length of each source sample.
        dp_size: Data-parallel world size.
        max_tokens_per_gpu: Trainer token budget for one packed microbatch.

    Returns:
        The predicted DP assignment and synchronized microbatch count.

    Raises:
        ValueError: If a size is invalid or a source sample exceeds the budget.
    """

    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if max_tokens_per_gpu <= 0:
        raise ValueError("max_tokens_per_gpu must be positive")
    if any(length <= 0 for length in lengths):
        raise ValueError("sample token lengths must be positive")
    if lengths and max(lengths) > max_tokens_per_gpu:
        raise ValueError(
            f"sample length {max(lengths)} exceeds max_tokens_per_gpu=" f"{max_tokens_per_gpu}"
        )
    if not lengths:
        return PackedBatchShape(0, 0, 0, (0,) * dp_size, (0,) * dp_size)

    partitions = _balanced_partitions(lengths, dp_size)
    tokens_per_dp = tuple(sum(lengths[index] for index in part) for part in partitions)
    samples_per_dp = tuple(len(part) for part in partitions)
    microbatches = max(math.ceil(tokens / max_tokens_per_gpu) for tokens in tokens_per_dp)
    return PackedBatchShape(
        samples=len(lengths),
        tokens=sum(lengths),
        microbatches_per_dp=microbatches,
        samples_per_dp=samples_per_dp,
        tokens_per_dp=tokens_per_dp,
    )


class TokenBudgetBatcher(Generic[T]):
    """Build full packed requests from a source stream with bounded memory.

    The batcher reads only until one DP-safe full request boundary is known.
    Any item read past that boundary is carried into the next request; the
    corpus is never planned or tokenized up front.

    Args:
        source: Stream of already rendered or tokenized source samples.
        length_fn: Returns the valid-token length of one sample.
        global_batch_size: Desired global number of packed microbatches.
        dp_size: Data-parallel world size used by the registered model.
        max_tokens_per_gpu: Trainer token budget for one packed microbatch.
        drop_last: Drop an incomplete final request when the source ends.
    """

    def __init__(
        self,
        source: Iterable[T],
        *,
        length_fn: Callable[[T], int],
        global_batch_size: int,
        dp_size: int,
        max_tokens_per_gpu: int,
        drop_last: bool = True,
    ) -> None:
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if dp_size <= 0 or global_batch_size % dp_size:
            raise ValueError("global_batch_size must be divisible by dp_size")
        if max_tokens_per_gpu <= 0:
            raise ValueError("max_tokens_per_gpu must be positive")
        self._source = source
        self._length_fn = length_fn
        self._global_batch_size = global_batch_size
        self._dp_size = dp_size
        self._max_tokens_per_gpu = max_tokens_per_gpu
        self._drop_last = drop_last

    def __iter__(self) -> Iterator[TokenBudgetBatch[T]]:
        target = self._global_batch_size // self._dp_size
        items: list[T] = []
        lengths: list[int] = []
        shape: PackedBatchShape | None = None
        total_tokens = 0
        max_length = 0
        probing_boundary = False
        boundary_probes = 0

        for item in self._source:
            length = int(self._length_fn(item))
            if length <= 0:
                raise ValueError("sample token lengths must be positive")
            if length > self._max_tokens_per_gpu:
                raise ValueError(
                    f"sample length {length} exceeds max_tokens_per_gpu="
                    f"{self._max_tokens_per_gpu}"
                )
            items.append(item)
            lengths.append(length)
            total_tokens += length
            max_length = max(max_length, length)

            if not probing_boundary:
                # Karmarkar-Karp's final partition spread is no larger than the
                # longest sample. Until this bound crosses the per-DP capacity,
                # the exact O(n log n) partition cannot overflow the target.
                average_ceiling = math.ceil(total_tokens / self._dp_size)
                partition_upper_bound = average_ceiling + max_length
                if partition_upper_bound <= target * self._max_tokens_per_gpu:
                    continue
                probing_boundary = True
                if len(items) > 1:
                    previous = predict_packed_batch_shape(
                        lengths[:-1],
                        dp_size=self._dp_size,
                        max_tokens_per_gpu=self._max_tokens_per_gpu,
                    )
                    if previous.microbatches_per_dp == target and previous.dp_safe:
                        global_capacity = self._global_batch_size * self._max_tokens_per_gpu
                        if previous.tokens * 100 >= global_capacity * _MIN_SAFE_FILL_PERCENT:
                            yield TokenBudgetBatch(tuple(items[:-1]), previous)
                            items = [item]
                            lengths = [length]
                            shape = None
                            total_tokens = length
                            max_length = length
                            probing_boundary = False
                            boundary_probes = 0
                            continue
                        shape = previous

            candidate = predict_packed_batch_shape(
                lengths,
                dp_size=self._dp_size,
                max_tokens_per_gpu=self._max_tokens_per_gpu,
            )
            boundary_probes += 1
            if candidate.microbatches_per_dp <= target:
                if candidate.microbatches_per_dp == target and candidate.dp_safe:
                    shape = candidate
                    if boundary_probes >= _MAX_BOUNDARY_PROBES:
                        yield TokenBudgetBatch(tuple(items), shape)
                        items = []
                        lengths = []
                        shape = None
                        total_tokens = 0
                        max_length = 0
                        probing_boundary = False
                        boundary_probes = 0
                continue

            overflow_item = items.pop()
            overflow_length = lengths.pop()
            if shape is None:
                raise ValueError(
                    "source stream crossed the packed batch boundary before a "
                    "DP-safe full request was formed"
                )
            yield TokenBudgetBatch(tuple(items), shape)
            items = [overflow_item]
            lengths = [overflow_length]
            shape = None
            total_tokens = overflow_length
            max_length = overflow_length
            probing_boundary = False
            boundary_probes = 0

        final_shape = (
            predict_packed_batch_shape(
                lengths,
                dp_size=self._dp_size,
                max_tokens_per_gpu=self._max_tokens_per_gpu,
            )
            if items
            else None
        )
        if (
            items
            and final_shape is not None
            and final_shape.microbatches_per_dp == target
            and final_shape.dp_safe
        ):
            yield TokenBudgetBatch(tuple(items), final_shape)
        elif items and not self._drop_last:
            assert final_shape is not None
            if not final_shape.dp_safe:
                raise ValueError("incomplete final request is not DP-safe")
            yield TokenBudgetBatch(tuple(items), final_shape)


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
