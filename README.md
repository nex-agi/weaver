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

## Interactive Walkthrough

See [`examples/weaver_walkthrough.ipynb`](examples/weaver_walkthrough.ipynb) for a hands-on
tutorial that walks through the full Weaver SDK workflow using a Pig Latin translation task:

- Data preparation and token-level Datum construction
- LoRA training and full fine-tuning
- Sampling / inference
- Checkpoint save and restore

## RL Training with NexRL

For reinforcement learning workflows, we recommend using Weaver together with
[NexRL](https://github.com/nex-agi/NexRL) in its **training-service** mode.
NexRL orchestrates the full RL loop (rollouts, trajectory collection, policy updates)
while Weaver handles the underlying training and inference services, so you get
end-to-end RL training without managing GPU resources directly.

## Autonomous Learning with MetaClaw

[MetaClaw](https://github.com/aiming-lab/MetaClaw) integrates Weaver as an RL backend
for OpenClaw autonomous learning. MetaClaw turns every live conversation into a learning
signal — with `rl.backend=weaver`, it uses Weaver for cloud-based LoRA training so your
personal agent continuously evolves without a local GPU.

## Deep Dive

For more technical details, see [Deep Dive into Weaver](https://dawning-road.github.io/blog/deep-dive-weaver).

## Full Example

For a complete script example, see `weaver/examples/pig_latin.py`.
