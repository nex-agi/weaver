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

"""Tests for composite payload refs (slice/concat algebra, R3 merge, materialize)."""

import copy
import json

import pytest

from weaver.types import (
    COMPOSITE_REF_SCHEMA,
    MAX_COMPOSITE_SEGMENTS,
    CompositeRef,
    is_composite_ref,
    merge_router_replay_prefix,
)
from weaver.types.payload_ref import PayloadRefMaterializationError, materialize_payload_ref

R3_SCHEMA = "weaver.router_replay.r2_recorded_indices.v1"


def r3_leaf(rid: str, rows: int | None, *, layers: int = 2, topk: int = 2) -> dict:
    leaf = {
        "storage": "gpfs",
        "relative_path": f"sglang_router_replay/{rid}/seq-0.safetensors",
        "format": "safetensors",
        "schema": R3_SCHEMA,
        "dtype": "int16",
        "size_bytes": 123,
        "uri": f"weaver://sglang_router_replay/{rid}/seq-0.safetensors",
        "created_at": "2026-09-28T00:00:00Z",
        "expires_at": "2026-09-29T00:00:00Z",
    }
    if rows is not None:
        leaf["shape"] = [rows, layers, topk]
    return leaf


def mask_leaf(rid: str, token_count: int) -> dict:
    return {
        "storage": "gpfs",
        "relative_path": f"sampling_masks/{rid}.bin",
        "format": "weaver.sampling_mask.v1",
        "schema": "weaver.sampling_mask.v1",
        "token_count": token_count,
        "empty_row": "full-vocabulary",
    }


def spans(ref: CompositeRef) -> list[tuple[str, int, int]]:
    return [(seg.ref["uri"], seg.start, seg.stop) for seg in ref.segments]


# --------------------------------------------------------------------- from_leaf


def test_from_leaf_row_count_sources():
    assert len(CompositeRef.from_leaf(r3_leaf("a", 10))) == 10
    assert len(CompositeRef.from_leaf(mask_leaf("m", 7))) == 7
    assert len(CompositeRef.from_leaf(r3_leaf("a", None), num_rows=5)) == 5
    # explicit num_rows may reference a prefix of a known-size leaf
    assert len(CompositeRef.from_leaf(r3_leaf("a", 10), num_rows=4)) == 4


def test_from_leaf_requires_num_rows_when_unknown():
    with pytest.raises(ValueError, match="num_rows is required"):
        CompositeRef.from_leaf(r3_leaf("a", None))


def test_from_leaf_rejects_num_rows_beyond_shape():
    with pytest.raises(ValueError, match="exceed"):
        CompositeRef.from_leaf(r3_leaf("a", 3), num_rows=4)


def test_from_leaf_deep_copies_input():
    leaf = r3_leaf("a", 4)
    ref = CompositeRef.from_leaf(leaf)
    leaf["shape"][0] = 999
    leaf["uri"] = "mutated"
    assert ref.to_payload()["segments"][0]["ref"]["shape"] == [4, 2, 2]
    assert ref.to_payload()["segments"][0]["ref"]["uri"].startswith("weaver://")


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda leaf: leaf.pop("storage"), "storage"),
        (lambda leaf: leaf.update(format=""), "format"),
        (lambda leaf: leaf.pop("schema"), "schema"),
        (lambda leaf: (leaf.pop("uri"), leaf.pop("relative_path")), "relative_path"),
        (lambda leaf: leaf.update(kind="composite"), "nested"),
        (lambda leaf: leaf.update(segments=[]), "nested"),
    ],
)
def test_leaf_validation_errors(mutate, match):
    leaf = r3_leaf("a", 4)
    mutate(leaf)
    with pytest.raises(ValueError, match=match):
        CompositeRef.from_leaf(leaf, num_rows=4)


def test_leaf_only_uri_or_only_relative_path_is_enough():
    leaf = r3_leaf("a", 4)
    leaf.pop("uri")
    assert len(CompositeRef.from_leaf(leaf)) == 4
    leaf = r3_leaf("a", 4)
    leaf.pop("relative_path")
    assert len(CompositeRef.from_leaf(leaf)) == 4


def test_leaf_must_be_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        CompositeRef([("not-a-dict", 0, 1)])


def test_constructor_validates_segments():
    leaf = r3_leaf("a", 4)
    with pytest.raises(ValueError, match="start"):
        CompositeRef([(leaf, -1, 1)])
    with pytest.raises(ValueError, match="count"):
        CompositeRef([(leaf, 0, True)])
    with pytest.raises(ValueError, match="exceed"):
        CompositeRef([(leaf, 3, 2)])
    with pytest.raises(ValueError, match="nested"):
        CompositeRef([(CompositeRef.from_leaf(leaf), 0, 1)])
    with pytest.raises(TypeError):
        CompositeRef([{"ref": leaf, "start": 0, "count": 1}])


def test_from_payload_strict_validation():
    leaf = r3_leaf("a", 4)
    good = {
        "kind": "composite",
        "schema": COMPOSITE_REF_SCHEMA,
        "segments": [{"ref": leaf, "start": 0, "count": 2}],
    }
    assert len(CompositeRef.from_payload(good)) == 2

    def bad(**changes):
        payload = copy.deepcopy(good)
        payload.update(changes)
        return payload

    cases = [
        (bad(schema="weaver.payload_ref.composite.v2"), "schema"),
        (bad(segments=[]), "non-empty"),
        (bad(segments="nope"), "non-empty"),
        (bad(extra=1), "unknown keys"),
        (bad(segments=[{"ref": leaf, "start": 0, "count": 0}]), "positive"),
        (bad(segments=[{"ref": leaf, "start": -1, "count": 1}]), "start"),
        (bad(segments=[{"ref": leaf, "start": 3, "count": 2}]), "exceed"),
        (bad(segments=[{"ref": leaf, "start": 0}]), "count"),
        (bad(segments=[{"start": 0, "count": 1}]), "missing 'ref'"),
        (bad(segments=[{"kind": "fill", "value": 0, "count": 1}]), "unsupported keys"),
        (bad(segments=[{"ref": good, "start": 0, "count": 1}]), "nested"),
        (bad(segments=["x"]), "must be an object"),
        (
            bad(segments=[{"ref": leaf, "start": 0, "count": 1}] * (MAX_COMPOSITE_SEGMENTS + 1)),
            "maximum",
        ),
    ]
    for payload, match in cases:
        with pytest.raises(ValueError, match=match):
            CompositeRef.from_payload(payload)

    with pytest.raises(TypeError):
        CompositeRef.from_payload(["not", "a", "mapping"])
    with pytest.raises(ValueError, match="does not match"):
        CompositeRef.from_payload(good, num_rows=3)


def test_is_composite_ref():
    leaf = r3_leaf("a", 4)
    ref = CompositeRef.from_leaf(leaf)
    assert is_composite_ref(ref)
    assert is_composite_ref(ref.to_payload())
    assert not is_composite_ref(leaf)
    assert not is_composite_ref(None)
    assert not is_composite_ref([1, 2])


# --------------------------------------------------------------------- algebra


def test_slicing_matches_list_semantics():
    a = CompositeRef.from_leaf(r3_leaf("a", 5))
    b = CompositeRef.from_leaf(r3_leaf("b", 4))
    ref = a + b
    rows = [("a", i) for i in range(5)] + [("b", i) for i in range(4)]

    def expand(r: CompositeRef) -> list[tuple[str, int]]:
        out = []
        for seg in r.segments:
            rid = seg.ref["uri"].split("/")[-2]
            out.extend((rid, i) for i in range(seg.start, seg.stop))
        return out

    bounds = [None, -20, -9, -5, -1, 0, 1, 3, 5, 6, 9, 20]
    for start in bounds:
        for stop in bounds:
            sliced = ref[start:stop]
            assert expand(sliced) == rows[start:stop], (start, stop)
            assert len(sliced) == len(rows[start:stop])
    assert expand(ref[2:8:1]) == rows[2:8]


def test_slicing_rejects_step_and_int():
    ref = CompositeRef.from_leaf(r3_leaf("a", 5))
    with pytest.raises(ValueError, match="step"):
        ref[::2]
    with pytest.raises(ValueError, match="step"):
        ref[::-1]
    with pytest.raises(TypeError, match="slices"):
        ref[0]


def test_concat_and_coalesce():
    leaf = r3_leaf("a", 10)
    ref = CompositeRef.from_leaf(leaf)
    rejoined = ref[:3] + ref[3:7] + ref[7:]
    assert len(rejoined.segments) == 1
    assert rejoined == ref

    # non-contiguous slices of the same leaf do not coalesce
    gap = ref[:3] + ref[4:6]
    assert spans(gap) == [(leaf["uri"], 0, 3), (leaf["uri"], 4, 6)]

    # same uri but non-identical dict (e.g. refreshed expires_at) still coalesces
    other = dict(leaf, expires_at="2099-01-01T00:00:00Z")
    joined = ref[:3] + CompositeRef([(other, 3, 2)])
    assert spans(joined) == [(leaf["uri"], 0, 5)]

    # different leaves never coalesce
    b = CompositeRef.from_leaf(r3_leaf("b", 2))
    assert len((ref[:2] + b).segments) == 2


def test_empty_segments_dropped_and_empty_ref():
    leaf = r3_leaf("a", 4)
    ref = CompositeRef([(leaf, 0, 0), (leaf, 0, 2), (leaf, 2, 0)])
    assert len(ref.segments) == 1
    empty = ref[3:1]
    assert len(empty) == 0
    assert empty + ref == ref
    assert ref + empty == ref
    with pytest.raises(ValueError, match="empty"):
        empty.to_payload()


def test_add_requires_composite_and_sum_works():
    leaf = r3_leaf("a", 4)
    ref = CompositeRef.from_leaf(leaf)
    with pytest.raises(TypeError):
        ref + leaf  # pylint: disable=pointless-statement
    with pytest.raises(TypeError):
        leaf + ref  # pylint: disable=pointless-statement
    parts = [ref[0:1], ref[1:2], ref[2:4]]
    assert sum(parts) == ref


def test_immutable_and_unhashable():
    ref = CompositeRef.from_leaf(r3_leaf("a", 4))
    with pytest.raises(AttributeError):
        ref._segments = ()  # pylint: disable=protected-access
    with pytest.raises(TypeError):
        hash(ref)
    # segments hands out copies
    ref.segments[0].ref["uri"] = "mutated"
    assert ref.segments[0].ref["uri"].startswith("weaver://")


def test_payload_round_trip_is_json_and_verbatim():
    a_leaf, b_leaf = r3_leaf("a", 6), r3_leaf("b", 8)
    ref = CompositeRef.from_leaf(a_leaf)[:5] + CompositeRef.from_leaf(b_leaf)[5:7]
    payload = ref.to_payload()
    assert payload == {
        "kind": "composite",
        "schema": COMPOSITE_REF_SCHEMA,
        "segments": [
            {"ref": a_leaf, "start": 0, "count": 5},
            {"ref": b_leaf, "start": 5, "count": 2},
        ],
    }
    assert json.loads(json.dumps(payload)) == payload
    assert CompositeRef.from_payload(payload) == ref
    assert CompositeRef.from_payload(ref) is ref
    payload["segments"][0]["ref"]["uri"] = "mutated"
    assert ref.to_payload()["segments"][0]["ref"]["uri"] == a_leaf["uri"]


def test_to_payload_enforces_segment_cap():
    leaf = r3_leaf("a", 2 * MAX_COMPOSITE_SEGMENTS + 2)
    base = CompositeRef.from_leaf(leaf)
    ok = sum((base[2 * i : 2 * i + 1] for i in range(MAX_COMPOSITE_SEGMENTS)), CompositeRef())
    assert len(ok.to_payload()["segments"]) == MAX_COMPOSITE_SEGMENTS
    too_many = ok + base[2 * MAX_COMPOSITE_SEGMENTS : 2 * MAX_COMPOSITE_SEGMENTS + 1]
    with pytest.raises(ValueError, match="maximum"):
        too_many.to_payload()


# --------------------------------------------------------------- R3 prefix merge


def test_merge_two_turns_boundary():
    l1, l2 = 10, 16
    leaf1, leaf2 = r3_leaf("t1", l1 - 1), r3_leaf("t2", l2 - 1)
    merged = merge_router_replay_prefix(leaf1, l1, leaf2, l2)
    assert len(merged) == l2 - 1
    assert spans(merged) == [(leaf1["uri"], 0, l1 - 1), (leaf2["uri"], l1 - 1, l2 - 1)]


def test_merge_three_turn_chain():
    l1, l2, l3 = 5, 9, 14
    leaves = [r3_leaf("t1", l1 - 1), r3_leaf("t2", l2 - 1), r3_leaf("t3", l3 - 1)]
    merged = merge_router_replay_prefix(leaves[0], l1, leaves[1], l2)
    merged = merge_router_replay_prefix(merged, l2, leaves[2], l3)
    assert len(merged) == l3 - 1
    assert spans(merged) == [
        (leaves[0]["uri"], 0, l1 - 1),
        (leaves[1]["uri"], l1 - 1, l2 - 1),
        (leaves[2]["uri"], l2 - 1, l3 - 1),
    ]
    # composite payload dicts are accepted as prev too
    again = merge_router_replay_prefix(
        merge_router_replay_prefix(leaves[0], l1, leaves[1], l2).to_payload(),
        l2,
        leaves[2],
        l3,
    )
    assert again == merged


def test_merge_without_shape_assumes_num_tokens_minus_one():
    merged = merge_router_replay_prefix(r3_leaf("t1", None), 4, r3_leaf("t2", None), 7)
    assert [(s, e) for _, s, e in spans(merged)] == [(0, 3), (3, 6)]


def test_merge_equal_lengths_takes_prev_only():
    leaf1 = r3_leaf("t1", 5)
    merged = merge_router_replay_prefix(leaf1, 6, r3_leaf("t2", 5), 6)
    assert spans(merged) == [(leaf1["uri"], 0, 5)]


def test_merge_errors():
    with pytest.raises(ValueError, match="must be <="):
        merge_router_replay_prefix(r3_leaf("a", 9), 10, r3_leaf("b", 5), 6)
    with pytest.raises(ValueError, match=">= 1"):
        merge_router_replay_prefix(r3_leaf("a", 0), 0, r3_leaf("b", 5), 6)
    with pytest.raises(ValueError, match="prev has 3 rows"):
        merge_router_replay_prefix(r3_leaf("a", 3), 10, r3_leaf("b", 15), 16)
    with pytest.raises(ValueError, match="nxt has 10 rows"):
        merge_router_replay_prefix(r3_leaf("a", 9), 10, r3_leaf("b", 10), 16)
    with pytest.raises(ValueError, match="int"):
        merge_router_replay_prefix(r3_leaf("a", 9), 10.0, r3_leaf("b", 15), 16)
    with pytest.raises(TypeError):
        merge_router_replay_prefix("nope", 10, r3_leaf("b", 15), 16)


# ------------------------------------------------------------------ materialize


def _write_safetensors(root, rid: str, tensor) -> dict:
    from safetensors.torch import save_file

    leaf = r3_leaf(rid, int(tensor.shape[0]), layers=tensor.shape[1], topk=tensor.shape[2])
    path = root / leaf["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"indices": tensor}, str(path))
    return leaf


def test_materialize_composite_equals_cat_of_leaf_slices(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    monkeypatch.setenv("WEAVER_PAYLOAD_REF_ROOT", str(tmp_path))
    t1 = torch.arange(9 * 3 * 2, dtype=torch.int16).reshape(9, 3, 2)
    t2 = (torch.arange(15 * 3 * 2, dtype=torch.int16) + 1000).reshape(15, 3, 2)
    leaf1, leaf2 = _write_safetensors(tmp_path, "t1", t1), _write_safetensors(tmp_path, "t2", t2)

    # plain leaf materialization of safetensors returns the tensor dict
    assert torch.equal(materialize_payload_ref(leaf1, field="indices"), t1)
    assert torch.equal(materialize_payload_ref(leaf1)["indices"], t1)

    merged = merge_router_replay_prefix(leaf1, 10, leaf2, 16)
    expected = torch.cat([t1[0:9], t2[9:15]], dim=0)
    for ref in (merged, merged.to_payload()):
        value = materialize_payload_ref(ref)
        assert value.dtype == torch.int16
        assert torch.equal(value, expected)
    assert torch.equal(materialize_payload_ref(merged, field="indices"), expected)


def test_materialize_composite_mixed_leaf_formats_upcasts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    monkeypatch.setenv("WEAVER_PAYLOAD_REF_ROOT", str(tmp_path))
    t1 = torch.arange(4 * 2 * 2, dtype=torch.int16).reshape(4, 2, 2)
    leaf1 = _write_safetensors(tmp_path, "t1", t1)
    t2 = torch.arange(3 * 2 * 2, dtype=torch.int32).reshape(3, 2, 2)
    torch.save({"indices": t2}, tmp_path / "legacy.pt")
    leaf2 = {
        "storage": "gpfs",
        "format": "torch.save",
        "schema": R3_SCHEMA,
        "relative_path": "legacy.pt",
    }
    ref = CompositeRef.from_leaf(leaf1)[1:3] + CompositeRef.from_leaf(leaf2, num_rows=3)[0:2]
    value = materialize_payload_ref(ref)
    assert value.dtype == torch.int32
    assert torch.equal(value, torch.cat([t1[1:3].to(torch.int32), t2[0:2]]))


def test_materialize_composite_errors(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    monkeypatch.setenv("WEAVER_PAYLOAD_REF_ROOT", str(tmp_path))
    leaf1 = _write_safetensors(tmp_path, "t1", torch.zeros(4, 2, 2, dtype=torch.int16))
    leaf_topk3 = _write_safetensors(tmp_path, "t3", torch.zeros(4, 2, 3, dtype=torch.int16))

    mismatched = CompositeRef.from_leaf(leaf1)[:2] + CompositeRef.from_leaf(leaf_topk3)[2:]
    with pytest.raises(PayloadRefMaterializationError, match="trailing dims"):
        materialize_payload_ref(mismatched)

    # the leaf on disk is shorter than the ref claims (shape metadata absent)
    short = dict(leaf1)
    short.pop("shape")
    with pytest.raises(PayloadRefMaterializationError, match="exceed"):
        materialize_payload_ref(CompositeRef.from_leaf(short, num_rows=6))

    with pytest.raises(PayloadRefMaterializationError, match="does not contain field"):
        materialize_payload_ref(CompositeRef.from_leaf(leaf1), field="missing")

    floats = {
        "storage": "gpfs",
        "format": "json",
        "schema": "test",
        "relative_path": "floats.json",
    }
    (tmp_path / "floats.json").write_text(json.dumps([[[0.5, 0.5], [0.5, 0.5]]]))
    mixed = CompositeRef.from_leaf(leaf1)[:1] + CompositeRef.from_leaf(floats, num_rows=1)
    with pytest.raises(PayloadRefMaterializationError, match="incompatible dtypes"):
        materialize_payload_ref(mixed)


def test_materialize_existing_leaf_behaviour_unchanged(tmp_path):
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[[1, 2]]])
    path = tmp_path / "shard.pt"
    torch.save({"indices": tensor}, path)
    value = materialize_payload_ref(
        {"storage": "gpfs", "format": "torch.save", "path": str(path)}, field="indices"
    )
    assert torch.equal(value, tensor)
