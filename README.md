# Weaver Python SDK

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

## Deep Dive

For more technical details, see [Deep Dive into Weaver](https://dawning-road.github.io/blog/deep-dive-weaver).

## Full Example

For a complete script example, see `weaver/examples/pig_latin.py`.
