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

"""Streaming, packed SFT with bounded preparation and submission lookahead.

Input is JSONL with ``{"prompt": "...", "response": "..."}`` records. The
source is read and tokenized only until one full trainer request is known; the
corpus is never preprocessed in memory. While the trainer runs that request, a
worker thread prepares the next one.

The batcher's DP size and token budget must match the registered trainer.

Example:
    python examples/streaming_sft.py data.jsonl \
        --base-model Qwen/Qwen3-8B --global-batch-size 128 \
        --dp-size 8 --max-tokens-per-gpu 262144
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from weaver import AsyncServiceClient, types
from weaver.training_pipeline import (
    CompletedTrainingStep,
    SubmitAheadQueue,
    TokenBudgetBatch,
    TokenBudgetBatcher,
)


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    """One tokenized causal-LM training example."""

    input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]

    def to_datum(self) -> types.Datum:
        """Convert the example to a Weaver datum."""

        return types.Datum(
            model_input=types.ModelInput.from_ints(self.input_tokens),
            loss_fn_inputs={
                "target_tokens": torch.tensor(self.target_tokens, dtype=torch.int64),
                "weights": torch.tensor(self.weights, dtype=torch.float32),
            },
        )


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """A materialized request plus preparation telemetry."""

    data: list[types.Datum]
    samples: int
    tokens: int
    packed_microbatches: int
    prepare_seconds: float


def iter_tokenized_examples(path: Path, tokenizer: Any) -> Iterator[TokenizedExample]:
    """Stream and repeat a JSONL corpus without retaining an epoch in memory."""

    while True:
        seen = 0
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("prompt")
                response = record.get("response")
                if not isinstance(prompt, str) or not isinstance(response, str):
                    raise ValueError(f"{path}:{line_number}: prompt and response must be strings")
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                tokens = prompt_tokens + response_tokens
                if len(tokens) < 2:
                    continue
                seen += 1
                yield TokenizedExample(
                    input_tokens=tokens[:-1],
                    target_tokens=tokens[1:],
                    weights=([0.0] * len(prompt_tokens) + [1.0] * len(response_tokens))[1:],
                )
        if not seen:
            raise RuntimeError(f"{path} produced no trainable examples")


def prepare_next(
    batches: Iterator[TokenBudgetBatch[TokenizedExample]],
) -> PreparedRequest | None:
    """Prepare one request; intended to run through ``asyncio.to_thread``."""

    started = time.perf_counter()
    batch = next(batches, None)
    if batch is None:
        return None
    return PreparedRequest(
        data=[example.to_datum() for example in batch.items],
        samples=batch.shape.samples,
        tokens=batch.shape.tokens,
        packed_microbatches=batch.shape.global_microbatches,
        prepare_seconds=time.perf_counter() - started,
    )


def result_metrics(result: Any) -> dict[str, Any]:
    """Extract metrics from an operation result."""

    if not isinstance(result, dict):
        return {}
    nested = result.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("metrics"), dict):
        return nested["metrics"]
    metrics = result.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def report_step(completed: CompletedTrainingStep) -> None:
    """Print one completed optimizer step."""

    metrics = result_metrics(completed.forward_backward)
    metrics.update(result_metrics(completed.optimizer))
    print(f"step={completed.step} metrics={metrics}", flush=True)


async def cancel_preparation(task: asyncio.Task[PreparedRequest | None]) -> None:
    """Cancel and consume a pending preparation task."""

    if task.done():
        if not task.cancelled():
            task.exception()
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def run(args: argparse.Namespace) -> None:
    """Run packed SFT."""

    async with AsyncServiceClient() as service:
        training = await service.create_model(
            base_model=args.base_model,
            training_mode=args.training_mode,
        )
        tokenizer = await asyncio.to_thread(training.get_tokenizer)
        batches = iter(
            TokenBudgetBatcher(
                # Requires Megatron Bridge balanced packing: TRAINER_MICRO_BATCH_SIZE=0,
                # matching DP/token budget, and router replay disabled.
                iter_tokenized_examples(args.data, tokenizer),
                length_fn=lambda example: len(example.input_tokens),
                global_batch_size=args.global_batch_size,
                dp_size=args.dp_size,
                max_tokens_per_gpu=args.max_tokens_per_gpu,
            )
        )
        prepared = asyncio.create_task(asyncio.to_thread(prepare_next, batches))
        queue = SubmitAheadQueue(training, submit_ahead=args.submit_ahead)
        optimizer = types.AdamParams(learning_rate=args.learning_rate)

        try:
            for step in range(args.steps):
                request = await prepared
                if request is None:
                    raise RuntimeError("source ended before a full request was formed")

                await queue.submit(step, request.data, optimizer)
                print(
                    f"submitted step={step} samples={request.samples} "
                    f"tokens={request.tokens} packed={request.packed_microbatches} "
                    f"prepare_s={request.prepare_seconds:.3f}",
                    flush=True,
                )

                if step + 1 < args.steps:
                    prepared = asyncio.create_task(asyncio.to_thread(prepare_next, batches))
                completed = await queue.wait_for_room()
                if completed is not None:
                    report_step(completed)

            for completed in await queue.drain():
                report_step(completed)
        finally:
            await cancel_preparation(prepared)
            try:
                if queue.pending_count:
                    await queue.drain()
            finally:
                await training.terminate()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--training-mode", choices=("full_ft", "lora"), default="full_ft")
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--max-tokens-per-gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--submit-ahead", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
