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
"""Pig Latin fine-tuning walkthrough using the Weaver SDK.

Run with ``--tensor-transport http-binary`` to use binary tensor packs. Zstandard
compression is enabled by default for binary packs; select ``--tensor-compression raw``
to disable it.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

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


def visualize(datum: types.Datum, tokenizer) -> None:
    print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
    print("-" * 50)
    for inp, tgt, wgt in zip(
        datum.model_input.to_ints(),
        datum.loss_fn_inputs["target_tokens"].tolist(),
        datum.loss_fn_inputs["weights"].tolist(),
    ):
        print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")


def _extract_logprobs(output: Dict[str, Any]) -> torch.Tensor:
    value = output.get("logprobs") or output.get("Logprobs")
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        raise ValueError("Missing logprobs in forward/backward output")
    return torch.as_tensor(value, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    """Parse command-line tensor transport options."""

    parser = argparse.ArgumentParser(description="Run the Pig Latin fine-tuning example.")
    parser.add_argument(
        "--tensor-transport",
        choices=("default", "http-binary"),
        default=None,
        help="defaults to WEAVER_TENSOR_TRANSPORT or default",
    )
    parser.add_argument(
        "--tensor-compression",
        choices=("raw", "zstd"),
        default=None,
        help="compression for http-binary; defaults to WEAVER_TENSOR_COMPRESSION or zstd",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = "Qwen/Qwen3-8B"
    with ServiceClient(
        api_key=os.getenv("WEAVER_API_KEY"),
        tensor_transport=args.tensor_transport,
        tensor_compression=args.tensor_compression,
    ) as service_client:
        training_client = service_client.create_model(base_model=base_model)
        print(f"Model ID: {training_client.model_id}")
        tokenizer = training_client.get_tokenizer()

        processed_examples = [process_example(example, tokenizer) for example in EXAMPLES]
        visualize(processed_examples[0], tokenizer)

        adam = types.AdamParams(learning_rate=1e-4)
        for step in range(6):
            fwdbwd_result = training_client.forward_backward(
                processed_examples,
                "cross_entropy",
                wait=True,
            )
            _ = training_client.optim_step(adam, wait=True)

            outputs = fwdbwd_result.get("result", {}).get("loss_fn_outputs") or []
            logprobs = torch.cat([_extract_logprobs(output) for output in outputs], dim=0)
            weights = torch.cat(
                [example.loss_fn_inputs["weights"] for example in processed_examples],
                dim=0,
            )
            loss = -torch.dot(logprobs, weights) / weights.sum()
            print(f"Step {step}: loss/token={float(loss):.4f}")

        sampling_client = training_client.save_weights_and_get_sampling_client(
            name="pig-latin-model"
        )
        prompt_tokens = tokenizer.encode(
            "English: coffee break\nPig Latin:", add_special_tokens=True
        )
        prompt = types.ModelInput.from_ints(prompt_tokens)
        params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"])
        sample_result = sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=8,
        )
        for idx, sequence in enumerate(sample_result.get("sequences", [])):
            decoded = tokenizer.decode(sequence.get("tokens", []))
            print(f"{idx}: {decoded!r}")


if __name__ == "__main__":
    main()
