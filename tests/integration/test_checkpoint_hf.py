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

"""Test 3: FullFT load_state with original HuggingFace model path.

Phase 1 - Train 3 steps (full_ft), record losses.
Phase 2 - load_state with HF base model path to reset weights, train 3 more steps.
Verify: Phase 2 losses ≈ Phase 1 losses (weights reset to original).
"""

from __future__ import annotations

import os
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

LR = 1e-5
HF_MODEL_PATH = "/gpfs/models/huggingface.co/models/Qwen/Qwen3-8B"


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


def main() -> None:
    phase1_losses: List[float] = []
    phase2_losses: List[float] = []

    with ServiceClient(api_key=os.getenv("WEAVER_API_KEY")) as client:
        tc = client.create_model(
            base_model="Qwen/Qwen3-8B",
            training_mode="full_ft",
        )
        tokenizer = tc.get_tokenizer()
        data = [process_example(ex, tokenizer) for ex in EXAMPLES]

        # ==============================================================
        # Phase 1: Train 3 steps
        # ==============================================================
        print("=" * 60)
        print("Phase 1: FullFT train 3 steps")
        print("=" * 60)

        adam = types.AdamParams(learning_rate=LR)
        for step in range(3):
            result = tc.forward_backward(data, "cross_entropy", wait=True)
            _ = tc.optim_step(adam, wait=True)
            loss = compute_loss(result, data)
            phase1_losses.append(loss)
            print(f"  Step {step}: loss/token={loss:.6f}")

        # ==============================================================
        # Phase 2: load_state with HuggingFace path, then train 3 steps
        # ==============================================================
        print()
        print("=" * 60)
        print(f"Phase 2: load_state('{HF_MODEL_PATH}') + train 3 steps")
        print("=" * 60)

        print(f"  Loading HF model weights: {HF_MODEL_PATH}")
        load_result = tc.load_state(HF_MODEL_PATH, wait=True)
        print(f"  Load result: {load_result}")

        for step in range(3):
            result = tc.forward_backward(data, "cross_entropy", wait=True)
            _ = tc.optim_step(adam, wait=True)
            loss = compute_loss(result, data)
            phase2_losses.append(loss)
            print(f"  Step {step + 3}: loss/token={loss:.6f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 60)
    print("TEST 3 RESULTS: FullFT load HuggingFace model")
    print("=" * 60)
    print("\nPhase 1 (initial training):")
    for i, loss in enumerate(phase1_losses):
        print(f"  Step {i}: loss/token={loss:.6f}")
    print("\nPhase 2 (after loading HF weights):")
    for i, loss in enumerate(phase2_losses):
        print(f"  Step {i + 3}: loss/token={loss:.6f}")

    print("\nComparison:")
    for i in range(3):
        diff = abs(phase2_losses[i] - phase1_losses[i])
        pct = diff / phase1_losses[i] * 100 if phase1_losses[i] else 0
        print(
            f"  Step {i} vs Step {i + 3}: "
            f"{phase1_losses[i]:.6f} vs {phase2_losses[i]:.6f} "
            f"(diff={diff:.6f}, {pct:.2f}%)"
        )

    print()
    print("Expected: Phase 2 losses should be close to Phase 1 losses")
    print("(weights reset to original HuggingFace model).")


if __name__ == "__main__":
    main()
