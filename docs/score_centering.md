# Score-centering sampling (experimental)

Requires the coordinated server, trainer and NexRL score-centering changes.

```python
sample = sampling_client.sample(
    prompt=prompt,
    sampling_params=weaver.types.SamplingParams(temperature=1, top_p=1, top_k=-1),
    topk_output_logprobs=128,
)
```

The opt-in flag requests **generated-token** top-k statistics. It does not change
sampling top_k and does not request prompt top-k statistics. Default 0 preserves
existing behavior. K must be between 1 and 128.

Each sequence contains response-aligned sampler_logprobs [R],
sampler_topk_ids/logprobs/mask [R,K], and sampler_distribution metadata with
schema behavior/unfiltered/v1. These are the probabilities reported at rollout
time; keep them separate from logprobs recomputed later for importance sampling.

The initial mode requires temperature=1, top_p=1, top_k=-1, no other distribution
processors, router replay, sampling masks, or prompt logprob requests. Server
validation additionally rejects penalties, minimum generation lengths and
structured/constrained sampling. An unsupported server returning incomplete
fields fails explicitly, including OperationHandle.result() after wait=False.

Both sync and async clients preserve all fields. Dense target-aligned SC tensors
use the existing default JSON or http-binary training transport (raw or zstd).
Do not send a full vocabulary distribution: the trainer computes probabilities
only at the selected IDs, with exact vocabulary normalization and gradients.

See tests/test_score_centering.py. Live GPU rollout/training and payload throughput
remain deployment acceptance checks.
