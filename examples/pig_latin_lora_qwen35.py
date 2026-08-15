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
"""Train and evaluate a Pig Latin LoRA on Qwen3.5-9B-Base."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from weaver import ServiceClient, types

BASE_MODEL = "Qwen/Qwen3.5-9B-Base:262144"
VOWELS = frozenset("aeiouy")
INSTRUCTION = (
    "Convert the lowercase English text to Pig Latin. For each word, move the leading "
    "consonants before the first a, e, i, o, u, or y to the end and add -ay. If a word "
    "starts with one of those letters, add -way. Keep word order. Answer with only the "
    "converted text."
)

DEMONSTRATION_INPUTS = (
    "apple bright",
    "string crypt",
    "under queen",
    "young flower",
    "year yard",
    "violin village",
    "window walnut",
    "cricket cradle",
    "plastic scheme",
    "oasis umpire",
    "scholar apricot",
    "cabin velvet",
)

# The held-out inputs below do not share words with these training examples.
SEED_TRAIN_INPUTS = [
    "banana split",
    "quantum physics",
    "donut shop",
    "pickle jar",
    "space exploration",
    "rubber duck",
    "coding wizard",
    "apple pie",
    "engine room",
    "island ferry",
    "olive branch",
    "underwater cave",
    "bright moon",
    "green forest",
    "black pepper",
    "fresh bread",
    "small river",
    "great mountain",
    "quick fox",
    "lazy dog",
    "smart robot",
    "friendly dragon",
    "blue flower",
    "white snow",
    "red cherry",
    "golden star",
    "quiet library",
    "modern city",
    "ancient temple",
    "open window",
    "early morning",
    "evening walk",
    "music player",
    "paper notebook",
    "wooden chair",
    "glass bottle",
    "strong bridge",
    "tiny mouse",
    "giant whale",
    "clever rabbit",
    "stormy night",
    "sunny beach",
    "frozen lake",
    "warm jacket",
    "purple candle",
    "silver spoon",
    "round table",
    "young artist",
    "string crypt",
    "under queen",
    "young flower",
]

# Real words matter here as well as synthetic coverage: the base model brings
# strong English priors (for example, treating initial "y" as a consonant) that
# pseudo-words alone do not reliably override. These examples cover the rule's
# difficult boundaries without reusing any held-out word.
CURATED_BOUNDARY_INPUTS = [
    "year yard",
    "yoga yogurt",
    "youth yummy",
    "yawn yelp",
    "yonder yarn",
    "yak yodel",
    "yes yesterday",
    "yeast yearbook",
    "violin village",
    "vivid vision",
    "victor vintage",
    "velvet vanilla",
    "vocal visitor",
    "vulture vacation",
    "vessel villain",
    "viper virus",
    "vinyl vigor",
    "plaza plenty",
    "plastic plum",
    "please pledge",
    "plover plank",
    "plush playground",
    "scheme scholar",
    "schooner schism",
    "schedule schwa",
    "schnitzel schnauzer",
    "cricket cradle",
    "crocus cream",
    "crown crook",
    "crazy creature",
    "crunch crater",
    "candle cocoa",
    "cabin cedar",
    "walnut willow",
    "window windy",
    "oasis orchard",
    "umpire urgent",
    "echo artist",
    "ivory object",
    "onion acorn",
    "urban eagle",
    "isotope avocado",
    "apricot oven",
    "elm avenue",
    "harvest velvet",
    "village harbor",
    "lecture banjo",
    "windy blizzard",
    "terminal railway",
    "bucket citrus",
    "mist scarlet",
    "pause cafe",
    "submersible yolk",
    "tortoise cheerful",
]

# Exercise every common consonant-prefix shape with many different word pieces.
# These deterministic synthetic words keep the held-out words genuinely unseen
# while giving the adapter enough character-level coverage to learn the rule.
SYNTHETIC_ONSETS = (
    "",
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "w",
    "z",
    "bl",
    "br",
    "ch",
    "cl",
    "cr",
    "dr",
    "fl",
    "fr",
    "gl",
    "gr",
    "ph",
    "pl",
    "pr",
    "sc",
    "sch",
    "sh",
    "sk",
    "sl",
    "sm",
    "sn",
    "sp",
    "spl",
    "spr",
    "st",
    "str",
    "sw",
    "th",
    "thr",
    "tr",
    "wh",
)
# Keep the suffix side deliberately varied.  A small, onset-grouped synthetic
# set lets the adapter memorize the six suffix strings and then drift toward
# whichever consonant cluster it saw most recently.  These vowel/y-leading
# pieces exercise different lengths and character shapes without leaking any
# held-out word.
SYNTHETIC_RIMES = (
    "amber",
    "echo",
    "ivory",
    "orbit",
    "umber",
    "yonder",
    "alder",
    "ember",
    "inlet",
    "otter",
    "upper",
    "yarn",
    "acorn",
    "easel",
    "igloo",
    "onion",
    "uncle",
    "arrow",
    "event",
    "item",
    "oval",
    "urban",
    "yodel",
    "apron",
)
SYNTHETIC_WORDS = [f"{onset}{rime}" for onset in SYNTHETIC_ONSETS for rime in SYNTHETIC_RIMES]
SYNTHETIC_INPUTS = [
    f"{SYNTHETIC_WORDS[index]} {SYNTHETIC_WORDS[index + 1]}"
    for index in range(0, len(SYNTHETIC_WORDS), 2)
]
TRAIN_INPUTS = SEED_TRAIN_INPUTS * 2 + CURATED_BOUNDARY_INPUTS * 4 + SYNTHETIC_INPUTS

TEST_INPUTS = [
    "coffee break",
    "crimson cloud",
    "orange basket",
    "train station",
    "yellow submarine",
    "crystal meadow",
    "guitar lesson",
    "winter breeze",
    "planet ocean",
    "happy turtle",
    "violet garden",
    "school umbrella",
]

assert not (
    {word for text in (*TRAIN_INPUTS, *DEMONSTRATION_INPUTS) for word in text.split()}
    & {word for text in TEST_INPUTS for word in text.split()}
), "held-out Pig Latin words must not appear in training or demonstrations"


def pig_latin_word(word: str) -> str:
    if word[0] in VOWELS:
        return f"{word}-way"
    first_vowel = next((index for index, char in enumerate(word) if char in VOWELS), len(word))
    return f"{word[first_vowel:]}-{word[:first_vowel]}ay"


def pig_latin(text: str) -> str:
    return " ".join(pig_latin_word(word) for word in text.split())


def prompt_for(text: str) -> str:
    demonstrations = "\n".join(
        f"English: {source}\nPig Latin: {pig_latin(source)}" for source in DEMONSTRATION_INPUTS
    )
    return f"{INSTRUCTION}\n\n{demonstrations}\nEnglish: {text}\nPig Latin:"


def process_example(text: str, tokenizer: Any) -> types.Datum:
    prompt_tokens = tokenizer.encode(prompt_for(text), add_special_tokens=True)
    completion_tokens = tokenizer.encode(f" {pig_latin(text)}\n", add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": torch.tensor(tokens[1:], dtype=torch.int64),
            "weights": torch.tensor(weights[1:], dtype=torch.float32),
        },
    )


def batches(items: Sequence[types.Datum], batch_size: int, step: int) -> list[types.Datum]:
    # Mix natural and synthetic examples, and mix consonant onsets inside every
    # batch. Walking TRAIN_INPUTS in declaration order feeds long runs of one
    # onset (b*, then c*, ...), which caused low training loss but catastrophic
    # held-out oscillation after each optimizer step. Each epoch gets its own
    # reproducible permutation and still visits every example exactly once.
    start = (step - 1) * batch_size
    batch: list[types.Datum] = []
    for absolute_index in range(start, start + batch_size):
        epoch, epoch_index = divmod(absolute_index, len(items))
        order = list(range(len(items)))
        random.Random(42 + epoch).shuffle(order)
        batch.append(items[order[epoch_index]])
    return batch


def learning_rate_for_step(base_learning_rate: float, step: int, max_steps: int) -> float:
    """Cosine-decay the LR to 10% so a learned rule does not drift late in training."""
    progress = (step - 1) / max(max_steps - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * (0.1 + 0.9 * cosine)


def _extract_logprobs(output: dict[str, Any]) -> torch.Tensor:
    value = output.get("logprobs") or output.get("Logprobs")
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        raise ValueError("Missing logprobs in forward/backward output")
    return torch.as_tensor(value, dtype=torch.float32)


def compute_loss(result: dict[str, Any], examples: Sequence[types.Datum]) -> float:
    outputs = result.get("result", {}).get("loss_fn_outputs") or []
    logprobs = torch.cat([_extract_logprobs(output) for output in outputs], dim=0)
    weights = torch.cat([example.loss_fn_inputs["weights"] for example in examples], dim=0)
    return float(-torch.dot(logprobs, weights) / weights.sum())


def normalize_prediction(text: str) -> str:
    return " ".join(text.strip().lower().split())


def evaluate(
    sampling_client: Any,
    tokenizer: Any,
    inputs: Iterable[str],
) -> tuple[float, float, list[tuple[str, str, str]]]:
    results: list[tuple[str, str, str]] = []
    params = types.SamplingParams(max_tokens=64, temperature=0.0, stop=["\n"], seed=42)
    for text in inputs:
        prompt = types.ModelInput.from_ints(
            tokenizer.encode(prompt_for(text), add_special_tokens=True)
        )
        response = sampling_client.sample(prompt=prompt, sampling_params=params)
        sequences = response.get("sequences", [])
        if not sequences:
            raise RuntimeError(f"Sampler returned no sequence for {text!r}")
        prediction = normalize_prediction(
            tokenizer.decode(sequences[0].get("tokens", []), skip_special_tokens=True)
        )
        expected = pig_latin(text)
        results.append((text, expected, prediction))

    phrase_correct = sum(expected == prediction for _, expected, prediction in results)
    word_correct = 0
    word_total = 0
    for _, expected, prediction in results:
        expected_words = expected.split()
        predicted_words = prediction.split()
        word_total += len(expected_words)
        word_correct += sum(
            expected_word == predicted_word
            for expected_word, predicted_word in zip(expected_words, predicted_words, strict=False)
        )
    return phrase_correct / len(results), word_correct / word_total, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-steps", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.5,
        help="Held-out phrase exact-match smoke threshold (default: 0.5).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Stop after save_state (useful for training/sampling smoke tests).",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    if not 0.0 <= args.target_accuracy <= 1.0:
        raise ValueError("--target-accuracy must be between 0 and 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.initial_steps <= 0 or args.max_steps <= 0:
        raise ValueError("--initial-steps and --max-steps must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.initial_steps > args.max_steps:
        raise ValueError("--initial-steps cannot exceed --max-steps")

    with ServiceClient(
        api_key=os.getenv("WEAVER_API_KEY"),
        name="pig-latin-qwen35-lora",
        labels={"task": "pig-latin", "base_model": BASE_MODEL, "training_mode": "lora"},
    ) as service:
        training = service.create_model(
            base_model=BASE_MODEL,
            training_mode="lora",
            lora_config=types.LoraConfig(rank=32, seed=42),
            user_metadata={"recipe": "examples/pig_latin_lora_qwen35.py"},
        )
        print(f"session_id={service.session_id}", flush=True)
        print(f"model_id={training.model_id}", flush=True)
        tokenizer = training.get_tokenizer()
        train_data = [process_example(text, tokenizer) for text in TRAIN_INPUTS]
        step = 0
        phrase_accuracy = 0.0
        word_accuracy = 0.0
        while step < args.max_steps and (
            step < args.initial_steps or phrase_accuracy < args.target_accuracy
        ):
            next_eval = (
                args.initial_steps
                if step < args.initial_steps
                else min(step + args.eval_every, args.max_steps)
            )
            while step < next_eval:
                step += 1
                batch = batches(train_data, args.batch_size, step)
                result = training.forward_backward(batch, "cross_entropy", wait=True)
                learning_rate = learning_rate_for_step(args.learning_rate, step, args.max_steps)
                optimizer = types.AdamParams(
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                )
                training.optim_step(optimizer, wait=True)
                loss = compute_loss(result, batch)
                print(
                    f"step={step} loss_per_token={loss:.6f} " f"learning_rate={learning_rate:.8g}",
                    flush=True,
                )

            sampler = training.save_weights_and_get_sampling_client(
                name=f"pig-latin-qwen35-lora-step-{step}",
                ttl_seconds=1800,
            )
            phrase_accuracy, word_accuracy, eval_results = evaluate(sampler, tokenizer, TEST_INPUTS)
            training.log_metrics(
                {
                    "eval/phrase_exact_match": phrase_accuracy,
                    "eval/word_exact_match": word_accuracy,
                },
                step=step,
                labels={"split": "held_out"},
            )
            print(
                f"eval step={step} phrase_exact_match={phrase_accuracy:.4f} "
                f"word_exact_match={word_accuracy:.4f}",
                flush=True,
            )
            for source, expected, prediction in eval_results:
                status = "PASS" if expected == prediction else "FAIL"
                print(
                    f"  {status} input={source!r} expected={expected!r} prediction={prediction!r}",
                    flush=True,
                )

        checkpoint = training.save_state(
            name=f"pig-latin-qwen35-lora-final-step-{step}",
            checkpoint_type="weight",
        )
        print(f"checkpoint_id={checkpoint.id}", flush=True)
        print(f"checkpoint_path={checkpoint.path}", flush=True)
        if not args.skip_export:
            artifact = training.export_weights(
                checkpoint=checkpoint,
                merge_adapter=False,
            )
            if artifact.status != "completed" or artifact.kind != "hf_adapter":
                raise RuntimeError(
                    "LoRA export did not produce a completed HuggingFace adapter: "
                    f"status={artifact.status!r} kind={artifact.kind!r} "
                    f"error={artifact.error!r}"
                )
            print(f"weights_artifact_id={artifact.id}", flush=True)
            print(f"weights_artifact_uri={artifact.uri}", flush=True)
            print(f"weights_artifact_size_bytes={artifact.size_bytes}", flush=True)
        print(f"final_step={step}", flush=True)
        print(f"final_phrase_exact_match={phrase_accuracy:.4f}", flush=True)
        print(f"final_word_exact_match={word_accuracy:.4f}", flush=True)
        if phrase_accuracy < args.target_accuracy:
            raise RuntimeError(
                f"Held-out phrase accuracy {phrase_accuracy:.4f} did not reach "
                f"target {args.target_accuracy:.4f}"
            )


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
