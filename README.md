# Weaver Python SDK

[![PyPI version](https://img.shields.io/pypi/v/nex-weaver)](https://pypi.org/project/nex-weaver/)
[![Python](https://img.shields.io/pypi/pyversions/nex-weaver)](https://pypi.org/project/nex-weaver/)
[![CI](https://github.com/nex-agi/weaver/actions/workflows/ci.yml/badge.svg)](https://github.com/nex-agi/weaver/actions/workflows/ci.yml)

English | [中文](README_zh.md)

Python client for the NexWeave Weaver server. The SDK mirrors the REST API exposed by
`weaver-server` and provides ergonomic helpers for training, sampling, telemetry, and
operations management.

## Installing locally

```bash
pip install nex-weaver
```

## Configuration

Configuration can be provided via keyword arguments or environment variables:
- `WEAVER_API_KEY`
- `WEAVER_ORGANIZATION_ID` / `WEAVER_PROJECT_ID` for canonical IDs
- `WEAVER_ORGANIZATION` / `WEAVER_PROJECT` for UUIDs, slugs, or display names

Canonical IDs take precedence. When no organization or project is configured,
the server keeps its stable personal-organization/default-project fallback.

## Quickstart

```python
from weaver import ServiceClient

def main():
    with ServiceClient() as client:
        session = client.ensure_session()
        print(session)

if __name__ == "__main__":
    main()
```

Give a new Session an optional experiment name and searchable string labels while
selecting its scope by human-readable references:

```python
with ServiceClient(
    organization="research",
    project="alignment",
    name="PPO baseline",
    labels={"dataset": "math", "environment": "staging"},
) as client:
    client.ensure_session()
```

Empty `name`/`labels` are omitted from the create request, preserving compatibility
with legacy servers and existing call sites.

```bash
weaver organizations list
weaver projects list --organization research
weaver scope resolve --organization research --project alignment
```

Organization slugs are globally unique. Project slugs and names are unique inside
their organization; an ambiguous display name is rejected instead of guessed.

## Usage

See [`examples/weaver_walkthrough.ipynb`](examples/weaver_walkthrough.ipynb) for an interactive
walkthrough of the full SDK workflow using a Pig Latin translation task — covering data
preparation, LoRA / full fine-tuning, sampling, and checkpoint management.

For a complete runnable script, see [`examples/pig_latin.py`](examples/pig_latin.py).
For large packed datasets, [`examples/streaming_sft.py`](examples/streaming_sft.py)
shows bounded token-budget batching and submit-ahead.

### Generation control (full fine-tuning only)

RL weight swaps need in-flight generation to stop before new weights land. A sampling
client can freeze its inference engine and resume it afterwards:

```python
with sampling_client.paused(mode=PauseMode.ABORT):
    path = training_client.save_weights_for_sampler(name="step-42")
    new_client = service.create_sampling_client(
        model_path=path, model_id=model_id, base_model=base_model
    )
```

Two things to know before using it:

- **The pause is engine-wide, not per sampling session.** It freezes *every* in-flight
  request on the engine serving that model, including ones issued through an earlier
  sampling session. That is what makes it usable for weight swaps — the requests you
  want to abort belong to the previous weight epoch — but it also means a pause is never
  scoped to "just my requests".
- **Full fine-tuning only.** Those models get a dedicated engine. LoRA adapters are
  served from one shared engine per base model, where a pause would abort generation for
  unrelated tenants, so the call is rejected before any request is sent.

Prefer `paused()` over calling `pause_generation()` / `continue_generation()` directly:
a pause that never reaches its resume leaves the engine frozen indefinitely, and there is
no server-side auto-resume. The async client mirrors this as `async with`.

### HuggingFace weights export

Checkpoints are stored in the trainer's native distributed format. `export_weights()`
converts one into a HuggingFace directory — a full model for full fine-tuning, a PEFT
adapter for LoRA — and `download_weights()` fetches it to disk:

```python
artifact = training_client.export_weights()          # save current weights + export
artifact = training_client.export_weights(checkpoint=ckpt)  # export an existing checkpoint

service.download_weights(artifact, "./hf-weights")
```

```bash
weaver checkpoint export weaver://<model>/checkpoints/step-42
weaver checkpoint download weaver://<model>/checkpoints/step-42 -o ./hf-weights
```

Three things to know:

- **Export is explicit, download never triggers one.** Converting a full model is minutes
  of compute and tens of GB of storage, so `download_weights()` fails with a "run
  export_weights first" error rather than silently starting a conversion.
- **Artifacts expire independently of their checkpoint** (7 days by default, `ttl_seconds`
  to change). Deleting the source checkpoint does not delete the artifact, and vice versa.
- **LoRA exports an adapter by default.** Pass `merge_adapter=True` to fold it into the
  base model and get a full HF model instead; it is rejected for full fine-tuning models.

Downloads run in parallel, resume interrupted files, refresh expired URLs, and verify each
file's sha256 before publishing it. Both methods have `Async*` twins.

## Ecosystem

[NexRL](https://github.com/nex-agi/NexRL) is the companion RL training framework.
In its **training-service** mode, NexRL orchestrates the full RL loop
(rollouts, trajectory collection, policy updates) while Weaver handles the underlying
training and inference services.

## Use Cases

**OpenClaw Autonomous Learning** —
[MetaClaw](https://github.com/aiming-lab/MetaClaw) has integrated Weaver as an RL
backend. By setting `rl.backend=weaver`, MetaClaw turns every live conversation into a
learning signal and uses Weaver for cloud-based LoRA training, enabling personal agents
to continuously evolve without a local GPU.

## Deep Dive

For more technical details, see [Deep Dive into Weaver](https://dawning-road.github.io/blog/deep-dive-weaver).
