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

"""Baseline: run LoRA (6 steps) and FullFT (6 steps) to record reference losses."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import torch

from weaver import ServiceClient, types

EXAMPLES: List[Dict[str, str]] = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]


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


def compute_loss(
    fwdbwd_result: Dict[str, Any],
    processed_examples: Sequence[types.Datum],
) -> float:
    outputs = fwdbwd_result.get("result", {}).get("loss_fn_outputs") or []
    logprobs = torch.cat([_extract_logprobs(o) for o in outputs], dim=0)
    weights = torch.cat([ex.loss_fn_inputs["weights"] for ex in processed_examples], dim=0)
    return float(-torch.dot(logprobs, weights) / weights.sum())


def run_baseline(training_mode: str | None, lr: float, steps: int = 6) -> List[float]:
    with ServiceClient(api_key=os.getenv("WEAVER_API_KEY")) as client:
        kwargs: Dict[str, Any] = {"base_model": "Qwen/Qwen3-8B"}
        if training_mode is not None:
            kwargs["training_mode"] = training_mode
        tc = client.create_model(**kwargs)
        tokenizer = tc.get_tokenizer()
        data = [process_example(ex, tokenizer) for ex in EXAMPLES]

        adam = types.AdamParams(learning_rate=lr)
        losses: List[float] = []
        for step in range(steps):
            result = tc.forward_backward(data, "cross_entropy", wait=True)
            _ = tc.optim_step(adam, wait=True)
            loss = compute_loss(result, data)
            losses.append(loss)
            print(f"  Step {step}: loss/token={loss:.6f}")
        return losses


def main() -> None:
    print("=" * 60)
    print("BASELINE: LoRA 6 steps (lr=1e-4)")
    print("=" * 60)
    lora_losses = run_baseline(None, lr=1e-4, steps=6)

    print()
    print("=" * 60)
    print("BASELINE: FullFT 6 steps (lr=1e-5)")
    print("=" * 60)
    fullft_losses = run_baseline("full_ft", lr=1e-5, steps=6)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("LoRA losses:", [f"{l:.6f}" for l in lora_losses])
    print("FullFT losses:", [f"{l:.6f}" for l in fullft_losses])


if __name__ == "__main__":
    main()
