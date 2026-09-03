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

### Training tensor transport

Training tensors use inline JSON by default, preserving the behavior of existing
clients. To opt into compressed binary transport, configure the service client:

```python
from weaver import ServiceClient

with ServiceClient(
    tensor_transport="http-binary",
    tensor_compression="zstd",  # Optional: Zstandard is the binary-pack default.
) as client:
    ...
```

`AsyncServiceClient` accepts the same options. The runnable Pig Latin examples expose
the same settings as command-line options:

```bash
python examples/pig_latin.py \
  --base-model Qwen/Qwen3.5-0.8B:262144 \
  --lora-rank 16 \
  --tensor-transport http-binary \
  --tensor-compression zstd
```

All clients also honor environment variables, so the same example can be configured
without command-line options:

```bash
WEAVER_TENSOR_TRANSPORT=http-binary \
WEAVER_TENSOR_COMPRESSION=zstd \
python examples/pig_latin.py
```

| Transport | Compression | Behavior |
| --- | --- | --- |
| `default` | Ignored | Legacy inline JSON (the default) |
| `http-binary` | `raw` | Binary tensor packs without compression |
| `http-binary` | `zstd` | Zstandard-compressed binary tensor packs |

`tensor_compression` (or `WEAVER_TENSOR_COMPRESSION`) only takes effect with
`http-binary`. For `cross_entropy` requests, the SDK moves eligible dense input tensors
into the binary pack; control metadata and other values remain JSON. When an operation
returns output tensors, the SDK downloads and materializes them back into the legacy
public response shape automatically. `Datum` construction, training calls, and result
handling therefore do not need to change. Keep the `default` transport when connecting
to an older Weaver server/trainer deployment that does not support binary tensor packs.

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

### Supported models and training modes

The backwards-compatible call still returns model names. Request details when choosing
between the independently priced LoRA and Full-FT modes:

```python
models = client.list_supported_models(detailed=True)
for model in models:
    for mode in model.training_modes:
        print(model.name, mode.display_name, mode.prices)
```

`training_modes` contains only modes that the model explicitly supports and
that have all four effective prices. A missing Full-FT quote means Full-FT is
not supported; the SDK never fabricates a 10× display price.

The CLI uses one compact row per mode with `Train`, `Input`, `Cached input`, and
`Output` columns. Filter to one mode, emit names only, or request stable JSON when needed:

```bash
weaver list supported-models
weaver list supported-models --mode full-ft
weaver list supported-models --format names
weaver list supported-models --format json
```

### Managed datasets

Managed datasets expose authorized catalog metadata and stable sample references. Every
immutable dataset version declares `content_visibility`: `protected` data keeps source
messages and token identities private, while `public` data can return its real content and
be downloaded. Select an explicit dataset `name` and `version`, then ask the model-bound
training client for effective lengths before packing whole samples:

```python
from weaver import ServiceClient
from weaver.types import Datum, SampleRef

with ServiceClient() as service:
    service.ensure_session()
    for dataset in service.datasets.list(compatible_model="Qwen/Qwen3-8B"):
        print(
            dataset.name,
            dataset.version,
            dataset.sample_count,
            dataset.content_visibility,
        )

    ref = SampleRef(dataset="hq-math", version="2026-08", sample_idx=42)
    trainer = service.create_model(
        base_model="Qwen/Qwen3-8B",
        training_max_sequence_length=4096,
    )
    resolved = trainer.resolve_sample_ref_lengths([ref])[0]
    length = resolved.input_token_count
    # Opaque identity of the server-private model data profile. Persist it if
    # exact resume must fail closed when that profile changes.
    model_data_revision = resolved.model_data_revision
    print(length)

    datum = Datum.from_sample_ref(
        dataset=ref.dataset,
        version=ref.version,
        sample_idx=ref.sample_idx,
        datum_id="batch-7-item-3",
    )
    result = trainer.forward_backward([datum], "cross_entropy")

    public = service.datasets.get(name="open-math", version="2026-08")
    if public.content_visibility == "public":
        service.datasets.download(
            name=public.name,
            version=public.version,
            destination="open-math.jsonl",
        )
```

The model fixes the tokenizer, chat template, and complete pre-shift token limit; clients
cannot override them per reference. `input_token_count` is the resulting autoregressive
training length after shifting (and is smaller than the pinned full-token limit), which
is sufficient for client-side whole-sample packing.
Protected references support only built-in `cross_entropy` `forward_backward`, and their
`loss_fn_inputs` must be empty. Forward logprobs, per-token losses, custom/surrogate loss,
and sampling would expose label-dependent signals and are rejected by the server. A
protected response carries a server-resolved `content_visibility="protected"`; any
token-shaped identity field contains only the response-only `-8` sentinel at the true
length. Never feed `-8` back into `ModelInput` or `target_tokens`.

Public references use the same `SampleRef` request shape, but their server-resolved
`content_visibility="public"` response can contain real non-negative token IDs and the
ordinary loss-specific output fields. Forward and custom/surrogate training requests are
therefore available for public data. `datasets.download` always streams authenticated
canonical JSONL to a same-directory `.part`, verifies exact size and SHA-256, then
publishes atomically; it refuses to overwrite an existing path unless `overwrite=True`.

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

### Deploying a checkpoint

`deploy_checkpoint()` publishes a checkpoint as a public, OpenAI-compatible endpoint: the
server converts it to HuggingFace format, launches a dedicated inference workload, and
registers that workload on the NorthGate gateway under the name you choose.

```python
deployment = training_client.deploy_checkpoint(ckpt, name="my-chat-model")
print(deployment.endpoint)          # OpenAI-compatible URL

service.list_deployments()
service.get_deployment(deployment.id)
service.delete_deployment(deployment.id)
```

```bash
weaver deployment create weaver://<model>/checkpoints/step-42 --name my-chat-model
weaver deployment list
weaver deployment get <deployment-id>
weaver deployment delete <deployment-id>
```

Four things to know:

- **Publishing is permission-gated and off by default.** The `deployment.publish` capability
  is granted by principal origin, not by Weaver role: an SSO session always qualifies, an API
  key only when it was minted under an IAM `biz_code` on the server's allowlist, and a service
  credential never. A server with the feature switched off answers 503. Both cases raise a
  `WeaverAPIError` that names what has to change. Listing, reading and deleting your own
  deployments need no capability — whoever published an endpoint can always take it down.
- **A deployment is independent and long-lived.** It does not share the training inference
  instance, it outlives the training session, and it holds its GPUs until you delete it. It
  also pins the source checkpoint and the exported artifact against garbage collection.
- **The name is global and must be valid everywhere it lands.** It is the served model name,
  the gateway's `model_name`, and a Kubernetes label at once: at most 63 characters of
  letters, digits, `.`, `-` and `_`, starting and ending alphanumerically. `overwrite=True`
  replaces an existing *gateway* registration; it does not free a name Weaver already uses.
- **It takes tens of minutes**, dominated by the conversion. Pass `wait=False` to get an
  `OperationHandle` instead of blocking. Every method has an `Async*` twin.

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
