# Sampling mask references

For long responses, request an opaque sampling mask reference instead of candidate ID lists:

```python
result = sampler.sample(
    prompt=prompt,
    sampling_params=SamplingParams(temperature=1.0, top_k=128),
    return_sampling_mask=True,
    sampling_mask_transport="ref",
)
sequence = result["sequences"][0]
ref = sequence["sampling_mask_ref"]
```

References require a sampling client bound to a full fine-tuning model. The response contains the generated tokens and log probabilities, plus `sampling_mask_ref`; it does not contain `sampling_masks`. The default transport remains `"inline"`.

Pass the reference in a training datum's metadata. `sampling_mask_positions` contains zero-based positions in `target_tokens`, one for each referenced response token. For example, with prompt `[1, 2]` and response `[3, 4]`, the datum has input `[1, 2, 3]`, targets `[2, 3, 4]`, and positions `[1, 2]`:

```python
metadata = {
    "sampling_mask_ref": ref,
    "sampling_mask_positions": [1, 2],
    "sampling_mask_slice": {"start": 0, "stop": 2},
}
```

Use the same metadata for `forward_logprob` and subsequent training with `importance_sampling`, `truncated_importance_sampling`, or `ppo_clip`. Do not also supply `loss_fn_inputs["sampling_mask"]`. Positions outside the reference use singleton target support, as with prompt positions in an inline mask. Empty candidate rows mean the complete vocabulary; singleton rows represent greedy sampling.

For truncation, keep the reference unchanged and shorten the half-open `sampling_mask_slice`. For multiple turns, use `metadata["sampling_mask_refs"]`, a list of the metadata objects above, with positions ordered across turns. Do not combine the single-reference and multiple-reference forms.

A reference belongs to the model that generated it and expires at its `expires_at` timestamp (currently 24 hours after generation). It cannot be reused with another model. Treat its contents as opaque; preserve the handle without modifying its fields. Expired, incomplete, or mismatched references fail explicitly. Sampling mask references cannot be combined with score centering.
