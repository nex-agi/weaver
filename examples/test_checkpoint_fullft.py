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

"""Test 2: FullFT save_state / load_state checkpoint round-trip.

Phase 1 - New session: train 3 steps (full_ft), save_state, terminate.
Phase 2 - Brand new session & model: load_state, train 3 more steps.
Verify: combined 6-step losses ≈ baseline FullFT 6-step losses.
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

    # ==================================================================
    # Phase 1: Fresh session — FullFT train 3 steps, save checkpoint
    # ==================================================================
    print("=" * 60)
    print("Phase 1: FullFT train 3 steps + save_state (new session)")
    print("=" * 60)

    with ServiceClient(api_key=os.getenv("WEAVER_API_KEY")) as client:
        tc = client.create_model(
            base_model="Qwen/Qwen3-8B",
            training_mode="full_ft",
        )
        tokenizer = tc.get_tokenizer()
        data = [process_example(ex, tokenizer) for ex in EXAMPLES]

        adam = types.AdamParams(learning_rate=LR)
        for step in range(3):
            result = tc.forward_backward(data, "cross_entropy", wait=True)
            _ = tc.optim_step(adam, wait=True)
            loss = compute_loss(result, data)
            phase1_losses.append(loss)
            print(f"  Step {step}: loss/token={loss:.6f}")

        # Save checkpoint
        checkpoint = tc.save_state(name="fullft-checkpoint-test2")
        print(f"\n  Checkpoint saved: id={checkpoint.id}, path={checkpoint.path}")
        print(f"  Checkpoint type={checkpoint.checkpoint_type}, status={checkpoint.status}")

        checkpoints = tc.list_checkpoints()
        print(f"  Total checkpoints for model: {len(checkpoints)}")

        saved_checkpoint_id = checkpoint.id

    print(f"\n  Phase 1 complete. Saved checkpoint_id: {saved_checkpoint_id}")

    # ==================================================================
    # Phase 2: Brand new session & model — load checkpoint, train 3 more
    # ==================================================================
    print()
    print("=" * 60)
    print("Phase 2: New session + new FullFT model + load_state + train 3 steps")
    print("=" * 60)

    with ServiceClient(api_key=os.getenv("WEAVER_API_KEY")) as client:
        tc2 = client.create_model(
            base_model="Qwen/Qwen3-8B",
            training_mode="full_ft",
        )
        tokenizer = tc2.get_tokenizer()
        data = [process_example(ex, tokenizer) for ex in EXAMPLES]

        # Trigger provisioning with a forward_backward
        print("  Triggering provisioning with initial forward_backward...")
        warmup_result = tc2.forward_backward(data, "cross_entropy", wait=True)
        warmup_loss = compute_loss(warmup_result, data)
        print(f"  Provisioning done (warmup loss={warmup_loss:.6f})")

        # Load the checkpoint from phase 1
        print(f"  Loading checkpoint: {saved_checkpoint_id}")
        load_result = tc2.load_state(saved_checkpoint_id, wait=True)
        print(f"  Load result: {load_result}")

        adam = types.AdamParams(learning_rate=LR)
        for step in range(3):
            result = tc2.forward_backward(data, "cross_entropy", wait=True)
            _ = tc2.optim_step(adam, wait=True)
            loss = compute_loss(result, data)
            phase2_losses.append(loss)
            print(f"  Step {step + 3}: loss/token={loss:.6f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 60)
    print("TEST 2 RESULTS: FullFT save_state / load_state")
    print("=" * 60)
    all_losses = phase1_losses + phase2_losses
    for i, loss in enumerate(all_losses):
        print(f"  Step {i}: loss/token={loss:.6f}")
    print()
    print("Compare with baseline FullFT 6-step losses.")
    print("Steps 0-2 should match baseline (same training run).")
    print("Step 3 loss should match baseline Step 3 (same weights loaded).")
    print("Steps 4-5 may differ slightly (optimizer state not restored).")


if __name__ == "__main__":
    main()
