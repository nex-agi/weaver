# Score-centering sampling (experimental)

Request the sampled token's log probability and the top-k token log probabilities
at each generated position for use with score-centering objectives.

```python
sample = sampling_client.sample(
    prompt=prompt,
    sampling_params=weaver.types.SamplingParams(temperature=1, top_p=1, top_k=-1),
    topk_output_logprobs=128,
    sampler_distribution_transport="ref",
)
```

The opt-in flag requests **generated-token** top-k statistics. It does not change
sampling top_k and does not request prompt top-k statistics. Default 0 preserves
existing behavior. K must be between 1 and 128.

For large responses, use `sampler_distribution_transport="ref"` with a sampling
client bound to your full-finetuning model. Each sequence contains
`sampler_logprobs` [R], `sampler_distribution` metadata, and an opaque
`sampler_distribution_ref`. Keep the reference unchanged and pass it in the
training datum's metadata with `sampler_distribution`. The reference belongs to
that model, identifies the sampling weight version, and expires at `expires_at`.
Consume it before expiry; it is not a permanent dataset artifact.

Set the datum's `loss_mask` to one only at generated target positions. For a
retained subrange of the response, also provide
`sampler_distribution_slice={"start": start, "stop": stop}` in metadata, using
zero-based response indices and an exclusive stop. Its length must equal the
number of active loss positions. Ref mode does not require `sampler_topk_*` or
`sampler_logprobs` training inputs. Do not combine those inputs with a reference.

The default `sampler_distribution_transport="inline"` returns
`sampler_topk_ids`, `sampler_topk_logprobs`, and `sampler_topk_mask` [R,K] as JSON
arrays. Use it for small payloads or when you need to inspect the probabilities.
Both modes report rollout-time probabilities; keep them separate from logprobs
recomputed later for importance sampling.

Use temperature=1, top_p=1, and top_k=-1. This option cannot be combined with
penalties, minimum generation lengths, structured/constrained sampling, other
distribution processors, router replay, sampling masks, or prompt logprob
requests. Incomplete response fields raise an error, including when retrieving
OperationHandle.result() after wait=False.

Both sync and async clients return these fields. When preparing training data,
align the generated-token statistics with the corresponding target-token
positions and exclude prompt positions from the loss mask. Training requests
support the default JSON and http-binary transports, with raw or zstd encoding.

For large training inputs, enable compressed binary tensors when creating the
service client:

```python
service = weaver.ServiceClient(
    tensor_transport="http-binary",
    tensor_compression="zstd",
)
```

The equivalent environment variables are `WEAVER_TENSOR_TRANSPORT=http-binary`
and `WEAVER_TENSOR_COMPRESSION=zstd`. These settings apply to training tensors;
inline top-k sampling results return as JSON arrays. Ref mode keeps top-k arrays
out of the client response and subsequent training request.

Each training tensor pack is limited to 8 GiB both before and after compression.
Budget for token IDs, masks, and other inputs as well as log probabilities.
Compression reduces transfer size but does not reduce the decoded tensor size
or automatically split a large batch into smaller requests.

### Optional probability-ratio weighting

`loss_fn_config` accepts `importance_weighting="none"` (default), `"tis"`, or
`"mis"`. The ratio is the current sampled-token probability divided by its
rollout-time probability. TIS caps this ratio at `tis_cap` (default `2.0`);
MIS keeps it only inside the inclusive `[mis_min, mis_max]` interval
(defaults `0.5` and `5.0`), assigning zero weight outside it. The weight is
held constant during differentiation. Bounds must be finite and positive,
and `mis_min` must not exceed `mis_max`.
