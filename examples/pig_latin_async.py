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
"""Pig Latin fine-tuning walkthrough using the **async** Weaver SDK.

This mirrors ``pig_latin.py`` but uses the asyncio-native client stack so the
event loop stays free while the server works:

* Training is **pipelined** — every ``forward_backward`` / ``optim_step`` for
  all steps is submitted with ``wait=False`` (the submit returns an
  ``AsyncOperationHandle`` immediately, each carrying a monotonic ``seq_id``),
  then the handles are awaited together with ``AsyncOperationHandle.wait_all``.
  The submits never block, and the trainer applies the ops in ``seq_id`` order,
  so the loss curve matches a blocking (``wait=True``) loop.
* Sampling is **concurrent** — several prompts are sampled at once via
  ``asyncio.gather``; each ``await`` yields the loop instead of blocking it.
  Sampling requests are independent, so concurrency here is always correct.

This entry point owns the loop via ``asyncio.run(main())``. The client itself
creates no loop and runs on the caller's loop — when embedding in an existing
async app (FastAPI, Jupyter, ...) ``await`` the client directly instead of
calling ``asyncio.run``. See the "Event loop model" section in
``AsyncServiceClient`` for the full integration contract.

Run with:  ``python examples/pig_latin_async.py``  (needs ``WEAVER_API_KEY``).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

import torch

from weaver import AsyncOperationHandle, AsyncServiceClient, types

EXAMPLES: List[Dict[str, str]] = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]

TEST_PROMPTS: List[str] = [
    "coffee break",
    "mountain trail",
    "electric guitar",
    "midnight snack",
]

NUM_STEPS = 6


def process_example(example: Dict[str, str], tokenizer) -> types.Datum:
    prompt = f"English: {example['input']}\nPig Latin:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": torch.tensor(target_tokens, dtype=torch.int64),
            "weights": torch.tensor(weights, dtype=torch.float32),
        },
    )


def _extract_logprobs(output: Dict[str, Any]) -> torch.Tensor:
    value = output.get("logprobs") or output.get("Logprobs")
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        raise ValueError("Missing logprobs in forward/backward output")
    return torch.as_tensor(value, dtype=torch.float32)


def _loss_per_token(fwdbwd_result: Dict[str, Any], examples: List[types.Datum]) -> float:
    outputs = fwdbwd_result.get("result", {}).get("loss_fn_outputs") or []
    logprobs = torch.cat([_extract_logprobs(output) for output in outputs], dim=0)
    weights = torch.cat([ex.loss_fn_inputs["weights"] for ex in examples], dim=0)
    return float(-torch.dot(logprobs, weights) / weights.sum())


async def main() -> None:
    base_model = os.getenv("WEAVER_BASE_MODEL", "Qwen/Qwen3-8B")
    async with AsyncServiceClient(api_key=os.getenv("WEAVER_API_KEY")) as service_client:
        training_client = await service_client.create_model(base_model=base_model)
        print(f"Model ID: {training_client.model_id}")
        tokenizer = training_client.get_tokenizer()

        processed_examples = [process_example(example, tokenizer) for example in EXAMPLES]

        # --- Phase 1: submit ALL training ops without blocking -------------
        # Each `await ...(wait=False)` only awaits the (fast) submit POST and
        # returns a handle immediately. We fire every step's forward_backward
        # and optim_step up front; the server executes them in seq order.
        adam = types.AdamParams(learning_rate=1e-4)
        fb_handles: List[AsyncOperationHandle] = []
        optim_handles: List[AsyncOperationHandle] = []
        for _ in range(NUM_STEPS):
            fb_handles.append(
                await training_client.forward_backward(
                    processed_examples, "cross_entropy", wait=False
                )
            )
            optim_handles.append(await training_client.optim_step(adam, wait=False))
        print(
            f"Submitted {len(fb_handles)} forward_backward + {len(optim_handles)} optim_step "
            "operations (none awaited yet)."
        )

        # --- Phase 2: await the in-flight results -------------------------
        # wait_all awaits concurrently; the event loop is free the whole time.
        fb_results = await AsyncOperationHandle.wait_all(fb_handles)
        await AsyncOperationHandle.wait_all(optim_handles)
        for step, fb_result in enumerate(fb_results):
            print(f"Step {step}: loss/token={_loss_per_token(fb_result, processed_examples):.4f}")

        # --- Phase 3: export weights and sample concurrently --------------
        sampling_client = await training_client.save_weights_and_get_sampling_client(
            name="pig-latin-model"
        )
        params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"])

        async def sample_one(text: str) -> Dict[str, Any]:
            prompt_tokens = tokenizer.encode(
                f"English: {text}\nPig Latin:", add_special_tokens=True
            )
            return await sampling_client.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                sampling_params=params,
                num_samples=1,
            )

        # Fan out all prompts at once — the requests run concurrently.
        results = await asyncio.gather(*(sample_one(text) for text in TEST_PROMPTS))
        for text, result in zip(TEST_PROMPTS, results):
            sequences = result.get("sequences", [])
            decoded = tokenizer.decode(sequences[0].get("tokens", [])) if sequences else ""
            print(f"{text!r} -> {decoded!r}")


if __name__ == "__main__":
    asyncio.run(main())
