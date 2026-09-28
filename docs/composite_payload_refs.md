# Composite payload refs

Samplers return large per-token payloads as opaque leaf refs, for example
`sequence["moe_topk_indices_ref"]` (R3 router-replay indices) or
`sequence["sampling_mask_ref"]`. To merge a multi-turn trajectory into one
training datum, you have to slice and concatenate these refs the same way you
slice and concatenate token lists. `CompositeRef` handles that without
downloading the payloads:

```python
from weaver.types.composite_ref import CompositeRef

a = CompositeRef.from_leaf(ref_a)    # row count from ref["shape"][0] or ref["token_count"]
b = CompositeRef.from_leaf(ref_b, num_rows=12)   # pass num_rows when the ref has neither
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

```python
from weaver.types.composite_ref import merge_router_replay_prefix

# Turn k+1's full sequence starts with turn k's full sequence (NexRL trace_prefix_merge).
ref, tokens = turns[0].moe_topk_indices_ref, turns[0].tokens
for turn in turns[1:]:
    assert turn.tokens[: len(tokens)] == tokens  # the merge precondition
    ref = merge_router_replay_prefix(ref, len(tokens), turn.moe_topk_indices_ref, len(turn.tokens))
    tokens = turn.tokens
num_tokens = len(tokens)
# ref == leaf1[0:L1-1] + leaf2[L1-1:L2-1] + ... ; len(ref) == num_tokens - 1
value_ref = ref.to_payload()
```

For debugging, `weaver.types.payload_ref.materialize_payload_ref(composite)`
loads each leaf from the shared filesystem (set `WEAVER_PAYLOAD_REF_ROOT`),
slices it, and concatenates the slices into a single tensor. Reading
`safetensors` leaves requires the `safetensors` package.
