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
"""Export trained weights to HuggingFace format and download them locally.

What happens under the hood:

1. ``TrainingClient.export_weights()`` asks the server to convert the
   checkpoint's native training-format weights (Megatron DCP) into a
   HuggingFace directory. The conversion runs server-side on a dedicated CPU
   converter pool — no GPU and no live trainer are involved, so it also works
   long after the training run's compute has been reclaimed.
2. While the conversion is running the export operation reports per-shard
   progress; the artifact stays ``pending`` until every file (weights shards,
   config, tokenizer) has been written to object storage with a sha256
   manifest.
3. ``ServiceClient.download_weights()`` streams every file of the finished
   artifact to a local directory in parallel, resumes on transport errors,
   and verifies each file against the manifest. The result is a standard HF
   directory you can load with ``transformers`` / ``vllm`` / ``sglang``.

What gets exported depends on the training mode:

- full fine-tuning checkpoint -> kind ``hf_model`` (full HF model directory)
- LoRA checkpoint             -> kind ``hf_adapter`` (HF PEFT adapter);
  pass ``merge_adapter=True`` to bake the adapter into the base weights and
  export a full ``hf_model`` instead.

CLI equivalents::

    weaver checkpoint export <checkpoint-id-or-weaver-uri> [--merge-adapter]
    weaver checkpoint download <weaver-uri-or-artifact-id> -o ./hf-export

Run with::

    export WEAVER_API_KEY=sk-your-api-key-here
    python examples/export_and_download_weights.py
"""

import os
import time

from weaver import OperationHandle, ServiceClient, WeightsArtifact

BASE_MODEL = os.getenv("WEAVER_BASE_MODEL", "Qwen/Qwen3-8B")
DOWNLOAD_DIR = os.getenv("WEAVER_DOWNLOAD_DIR", "./hf-export")


def lookup(payload: dict, key: str):
    """Case-insensitive key lookup (server payloads use Go-style casing)."""
    for k, v in payload.items():
        if k.lower() == key:
            return v
    return None


def wait_with_progress(handle: OperationHandle) -> WeightsArtifact:
    """Poll an export operation, printing conversion progress as it runs.

    ``export_weights(wait=True)`` does the same wait without the progress
    printing — use this pattern when you want user-visible feedback.
    """
    while not handle.done():
        payload = handle.refresh()
        metadata = lookup(payload, "metadata") or {}
        progress = metadata.get("conversion_progress") or {}
        shards = progress.get("shards_done")
        gigabytes = (progress.get("bytes_done") or 0) / 1e9
        detail = f"{shards} shard(s), {gigabytes:.1f} GB uploaded" if shards else "starting"
        print(f"  converting... {detail}")
        time.sleep(15)
    # The finished operation's response is the artifact itself.
    return WeightsArtifact.from_payload(handle.result())


def main():
    with ServiceClient(api_key=os.getenv("WEAVER_API_KEY")) as service:
        # Any training run works here; a fresh one keeps the example
        # self-contained. In your own code, call export_weights() on the
        # TrainingClient you already train with.
        training = service.create_training_client(base_model=BASE_MODEL)
        print(f"Training run: {training.model_id}")

        # ... forward_backward / optim_step loop elided; see pig_latin.py ...

        # One-step export: saves the CURRENT weights as a checkpoint and
        # converts them in a single call. To export an earlier checkpoint
        # instead, pass checkpoint="<id>" or a weaver:// checkpoint path, e.g.
        #   training.export_weights(checkpoint="weaver://<model-id>/checkpoints/<name>")
        # For LoRA runs this produces an hf_adapter artifact; add
        # merge_adapter=True to export a merged full model instead.
        result = training.export_weights(wait=False)

        if isinstance(result, WeightsArtifact):
            # Idempotent hit: a completed artifact for these weights already
            # existed, no new conversion was started.
            artifact = result
        else:
            artifact = wait_with_progress(result)

        print(f"Artifact {artifact.id}: kind={artifact.kind} status={artifact.status}")
        if artifact.size_bytes:
            print(f"Total size: {artifact.size_bytes / 1e9:.1f} GB")

        # Download and verify. Accepts the artifact object, an artifact id, or
        # a weaver:// checkpoint URI (kind= selects between hf_model and
        # hf_adapter when a checkpoint has both).
        dest = service.download_weights(artifact, DOWNLOAD_DIR)
        print(f"Downloaded to {dest.resolve()} (sha256-verified)")


if __name__ == "__main__":
    main()
