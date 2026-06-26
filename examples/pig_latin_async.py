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

* Training is a **sequential per-step loop** (mirrors tinker's
  ``forward_backward_async`` / ``optim_step_async`` + ``.result()`` pattern):
  each step submits ``forward_backward`` then ``optim_step`` with ``wait=False``
  (the submits overlap and never block the loop) and then awaits both before the
  next step. The per-step ``optim_step`` writes weights that the next step's
  ``forward_backward`` reads, so the loss is printed each step and the curve
  matches a blocking (``wait=True``) loop.
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

from weaver import AsyncServiceClient, types

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

        # --- Training: sequential per-step loop ---------------------------
        # Each step submits forward_backward + optim_step without blocking
        # (wait=False returns a handle right away, like tinker's *_async), then
        # awaits both before the next step. optim_step writes the weights the
        # next step's forward_backward reads, so steps stay strictly ordered.
        adam = types.AdamParams(learning_rate=1e-4)
        for step in range(NUM_STEPS):
            fb_handle = await training_client.forward_backward(
                processed_examples, "cross_entropy", wait=False
            )
            optim_handle = await training_client.optim_step(adam, wait=False)
            fb_result = await fb_handle  # loss for this step
            await optim_handle  # ensure the update lands before the next step
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
