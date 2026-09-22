# Score-centering sampling (experimental)

Request the sampled token's log probability and the top-k token log probabilities
at each generated position for use with score-centering objectives.

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

Use temperature=1, top_p=1, and top_k=-1. This option cannot be combined with
penalties, minimum generation lengths, structured/constrained sampling, other
distribution processors, router replay, sampling masks, or prompt logprob
requests. Incomplete response fields raise an error, including when retrieving
OperationHandle.result() after wait=False.

Both sync and async clients return these fields. When preparing training data,
align the generated-token statistics with the corresponding target-token
positions and exclude prompt positions from the loss mask. Training requests
support the default JSON and http-binary transports, with raw or zstd encoding.
