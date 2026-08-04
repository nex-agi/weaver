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

Select a scope by human-readable references, or discover and resolve available scopes:

```python
with ServiceClient(organization="research", project="alignment") as client:
    client.ensure_session()
```

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
