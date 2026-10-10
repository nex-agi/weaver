# Composite payload refs

Samplers return large per-token payloads as opaque leaf refs, for example
`sequence["moe_topk_indices_ref"]` (R3 router-replay indices) or
`sequence["sampling_mask_ref"]`. To merge a multi-turn trajectory into one
training datum, you have to slice and concatenate these refs the same way you
slice and concatenate token lists. `CompositeRef` handles that without
downloading the payloads:

```python
from weaver.types.composite_ref import CompositeRef

a = CompositeRef.from_leaf(ref_a)    # rows: ref["meta"]["num_rows"], else shape[0], else token_count
b = CompositeRef.from_leaf(ref_b, num_rows=12)   # pass num_rows when the ref has none of these
merged = a[:5] + b[5:12]             # step-1 slices, `+` concatenation, len() gives rows
payload = merged.to_payload()        # JSON-able; sent wherever a leaf ref is accepted
```

On the wire, a composite looks like this:

```json
{"kind": "composite", "schema": "weaver.payload_ref.composite.v1",
 "segments": [{"ref": {"...leaf ref, verbatim...": ""}, "start": 0, "count": 5},
              {"ref": {"...": ""}, "start": 5, "count": 7}]}
```

The rules:

- `start` and `count` are in the leaf's own row coordinates.
- Rows are concatenated in segment order.
- Composites are always flat: a segment's `ref` is never itself a composite.
- A composite has at most 1024 segments (`MAX_COMPOSITE_SEGMENTS`).
- Adjacent contiguous slices of the same leaf are coalesced into one segment.

Treat each leaf ref as opaque, and keep it unchanged.

## Merging R3 router-replay refs across turns

R3 refs are target-aligned. Row `i` holds the routing for input token `i`, so
a sequence of N tokens has N-1 rows. The last sampled token of a turn is never
fed forward in that turn. Its routing row first appears during the next turn's
prefill. This is why the merge boundary sits at `L1 - 1`:

Newer servers attach a `meta` object to each R3 leaf ref (older servers omit it):

```json
{"token_alignment": "target_aligned", "num_tokens": 12, "prompt_tokens": 9, "num_rows": 11}
```

`num_tokens` is prompt + response tokens of that sampling request, and
`num_rows == num_tokens - 1`. The SDK validates it on receipt: a malformed
meta makes `sample()` raise `ValueError` in both the sync and async clients.
`sequence["moe_topk_indices_ref"]` stays a raw dict. To get the typed form,
call `parse_moe_topk_indices_ref`:

```python
from weaver.types import parse_moe_topk_indices_ref

ref = parse_moe_topk_indices_ref(seq)   # PayloadRef, or None if the sequence has no ref
if ref is not None and ref.meta is not None:   # meta is None on older servers
    ref.meta.prompt_tokens       # P
    ref.meta.num_tokens          # P + R
    ref.meta.num_rows            # P + R - 1
    ref.meta.response_tokens     # R
    ref.meta.response_row_start  # P - 1: the first row whose target is a response token
    ref.meta.extra               # meta keys this SDK does not know yet (kept on round trip)
ref.to_payload() == seq["moe_topk_indices_ref"]   # lossless
```

`PayloadRefMeta.from_payload(d)` parses a bare meta object, and
`ref_meta(ref)` returns the meta as a plain dict. `CompositeRef.from_leaf` and
`merge_router_replay_prefix` accept either `PayloadRef` objects or raw dicts.
With meta, `merge_router_replay_prefix` derives each side's token count itself,
so you can omit the counts. A merged composite over L tokens has L-1 rows, so
its count is `len(ref) + 1`:

```python
from weaver.types.composite_ref import merge_router_replay_prefix

# Turn k+1's full sequence starts with turn k's full sequence (NexRL trace_prefix_merge).
ref = turns[0].moe_topk_indices_ref
for turn in turns[1:]:
    ref = merge_router_replay_prefix(ref, None, turn.moe_topk_indices_ref, None)
    # or, as keywords: merge_router_replay_prefix(ref, nxt=turn.moe_topk_indices_ref)
# ref == leaf1[0:L1-1] + leaf2[L1-1:L2-1] + ... ; len(ref) == L_last - 1
value_ref = ref.to_payload()
```

You still have to check that each turn's tokens extend the previous turn's
tokens (`turn.tokens[:len(prev_tokens)] == prev_tokens`). If you pass the
counts explicitly, for example `merge_router_replay_prefix(ref, len(tokens),
next_ref, len(next_tokens))`, and a leaf's `meta.num_tokens` differs from the
count you passed, the call raises `ValueError`. That mismatch means the tokens
you built are not the tokens the sampler saw, for example because of
retokenization drift. A leaf with no meta, no `shape`, and no `token_count`
needs explicit counts. A leaf whose `meta.num_rows` and `shape[0]` differ is
rejected as inconsistent.

For debugging, `weaver.types.payload_ref.materialize_payload_ref(composite)`
loads each leaf from the shared filesystem (set `WEAVER_PAYLOAD_REF_ROOT`),
slices it, and concatenates the slices into a single tensor. Reading
`safetensors` leaves requires the `safetensors` package.
