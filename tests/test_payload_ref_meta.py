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

"""Typed parsing of server-attached R3 leaf ref ``meta``."""

from __future__ import annotations

import asyncio
import copy
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from weaver._sampling_utils import normalize_sample_result, validate_sequence_payload_refs
from weaver.async_sampling_client import AsyncSamplingClient
from weaver.sampling_client import SamplingClient
from weaver.types import (
    CompositeRef,
    ModelInput,
    PayloadRef,
    PayloadRefMeta,
    merge_router_replay_prefix,
    parse_moe_topk_indices_ref,
    ref_meta,
)

R3_SCHEMA = "weaver.router_replay.r2_recorded_indices.v1"


def meta_payload(prompt: int = 4, response: int = 3, **overrides) -> dict:
    num_tokens = prompt + response
    return {
        "token_alignment": "target_aligned",
        "num_tokens": num_tokens,
        "prompt_tokens": prompt,
        "num_rows": num_tokens - 1,
        **overrides,
    }


def r3_ref(rid: str = "t1", prompt: int = 4, response: int = 3, *, meta: bool = True) -> dict:
    rows = prompt + response - 1
    ref = {
        "storage": "gpfs",
        "relative_path": f"sglang_router_replay/{rid}/seq-0.safetensors",
        "format": "safetensors",
        "schema": R3_SCHEMA,
        "dtype": "int16",
        "size_bytes": 123,
        "uri": f"weaver://sglang_router_replay/{rid}/seq-0.safetensors",
        "shape": [rows, 2, 2],
        "expires_at": "2026-09-30T00:00:00Z",
    }
    if meta:
        ref["meta"] = meta_payload(prompt, response)
    return ref


# ---------------------------------------------------------------- PayloadRefMeta


def test_meta_typed_parse_and_properties():
    meta = PayloadRefMeta.from_payload(meta_payload(9, 3))
    assert (meta.token_alignment, meta.num_tokens, meta.prompt_tokens, meta.num_rows) == (
        "target_aligned",
        12,
        9,
        11,
    )
    assert meta.is_target_aligned
    assert meta.response_tokens == 3
    # rows [8, 11) are the forwards of tokens 8..10 whose targets are response tokens 9..11
    assert meta.response_row_start == 8
    assert meta.extra == {}
    assert meta.to_payload() == meta_payload(9, 3)


def test_meta_response_row_start_edges():
    assert PayloadRefMeta.from_payload(meta_payload(1, 5)).response_row_start == 0
    assert PayloadRefMeta.from_payload(meta_payload(0, 5)).response_row_start == 0
    other = PayloadRefMeta.from_payload(meta_payload(4, 3, token_alignment="input_aligned"))
    assert not other.is_target_aligned
    assert other.response_row_start is None


def test_meta_unknown_keys_preserved_in_extra():
    payload = meta_payload(extra_field={"nested": [1, 2]}, version=2)
    meta = PayloadRefMeta.from_payload(payload)
    assert meta.extra == {"extra_field": {"nested": [1, 2]}, "version": 2}
    assert meta.to_payload() == payload
    # extra is deep-copied both ways
    payload["extra_field"]["nested"].append(3)
    assert meta.extra["extra_field"] == {"nested": [1, 2]}
    out = meta.to_payload()
    out["extra_field"]["nested"].append(4)
    assert meta.extra["extra_field"] == {"nested": [1, 2]}


@pytest.mark.parametrize(
    "payload, match",
    [
        (None, "meta must be an object"),
        ([1], "meta must be an object"),
        ({"num_tokens": 7}, r"missing required keys: \['token_alignment', 'prompt_tokens'"),
        (meta_payload(token_alignment=3), "token_alignment must be a non-empty string"),
        (meta_payload(num_tokens="7"), "meta.num_tokens must be an int"),
        (meta_payload(prompt_tokens=False), "meta.prompt_tokens must be an int"),
        (meta_payload(num_rows=-1), "meta.num_rows must be >= 0"),
        (meta_payload(prompt_tokens=8), "meta.prompt_tokens=8 exceeds meta.num_tokens=7"),
        (meta_payload(num_rows=7), "requires num_rows == num_tokens - 1, got num_rows=7"),
    ],
)
def test_meta_validation_errors(payload, match):
    with pytest.raises(ValueError, match=match):
        PayloadRefMeta.from_payload(payload)


def test_meta_direct_construction_is_validated_and_frozen():
    with pytest.raises(ValueError, match="requires num_rows"):
        PayloadRefMeta("target_aligned", 7, 4, 5)
    with pytest.raises(ValueError, match="must not repeat typed keys"):
        PayloadRefMeta("target_aligned", 7, 4, 6, extra={"num_rows": 6})
    meta = PayloadRefMeta("target_aligned", 7, 4, 6)
    with pytest.raises(AttributeError):
        meta.num_rows = 5  # type: ignore[misc]
    assert PayloadRefMeta.from_payload(meta) is meta


# -------------------------------------------------------------------- PayloadRef


def test_payload_ref_parses_meta_and_round_trips_losslessly():
    raw = r3_ref(meta=True)
    raw["meta"]["future_key"] = "x"
    ref = PayloadRef.from_payload(raw)
    assert isinstance(ref.meta, PayloadRefMeta)
    assert ref.meta.num_rows == 6 and ref.meta.extra == {"future_key": "x"}
    assert "meta" not in ref.metadata
    assert ref.metadata["shape"] == [6, 2, 2]
    out = ref.to_payload()
    assert out == raw
    assert json.loads(json.dumps(out)) == raw


def test_payload_ref_without_meta_unchanged():
    raw = r3_ref(meta=False)
    ref = PayloadRef.from_payload(raw)
    assert ref.meta is None
    assert ref.to_payload() == raw
    # an explicit null meta stays an untyped key and round-trips too
    raw["meta"] = None
    ref = PayloadRef.from_payload(raw)
    assert ref.meta is None and ref.to_payload() == raw


def test_payload_ref_rejects_bad_meta_and_shape_mismatch():
    raw = r3_ref()
    raw["meta"]["num_rows"] = 3
    with pytest.raises(ValueError, match="requires num_rows"):
        PayloadRef.from_payload(raw)
    raw = r3_ref()
    raw["shape"][0] = 5
    with pytest.raises(
        ValueError, match=r"inconsistent leaf ref: meta.num_rows=6 but shape\[0\]=5"
    ):
        PayloadRef.from_payload(raw)


# -------------------------------------------------------- sequence-level helper


def test_parse_moe_topk_indices_ref_with_and_without_meta():
    seq = {"tokens": [5, 6, 7], "moe_topk_indices_ref": r3_ref(prompt=4, response=3)}
    ref = parse_moe_topk_indices_ref(seq)
    assert isinstance(ref, PayloadRef) and ref.meta is not None
    assert (ref.meta.prompt_tokens, ref.meta.num_rows, ref.meta.response_tokens) == (4, 6, 3)
    assert ref.meta.response_tokens == len(seq["tokens"])
    assert ref.to_payload() == seq["moe_topk_indices_ref"]

    legacy = parse_moe_topk_indices_ref({"moe_topk_indices_ref": r3_ref(meta=False)})
    assert legacy is not None and legacy.meta is None
    assert parse_moe_topk_indices_ref({"tokens": [1]}) is None
    assert parse_moe_topk_indices_ref({"moe_topk_indices_ref": None}) is None
    assert parse_moe_topk_indices_ref({"moe_topk_indices_ref": ref}) is ref
    with pytest.raises(ValueError, match="must be an object"):
        parse_moe_topk_indices_ref({"moe_topk_indices_ref": "shard-1"})


# ----------------------------------------------- CompositeRef from typed refs


def test_composite_from_payload_ref_uses_typed_meta():
    raw = r3_ref(prompt=4, response=3)
    del raw["shape"]  # rows come from meta alone
    typed = PayloadRef.from_payload(raw)
    assert len(CompositeRef.from_leaf(typed)) == 6
    assert CompositeRef.from_leaf(typed) == CompositeRef.from_leaf(raw)
    assert CompositeRef.from_payload(typed) == CompositeRef.from_leaf(raw)
    assert CompositeRef([(typed, 0, 2)]) == CompositeRef([(raw, 0, 2)])
    assert ref_meta(typed) == raw["meta"]
    # the typed leaf is written back verbatim (meta included) into the composite
    assert CompositeRef.from_leaf(typed).to_payload()["segments"][0]["ref"] == raw


def test_merge_accepts_payload_refs_using_meta_counts():
    t1, t2, t3 = r3_ref("t1", 4, 3), r3_ref("t2", 9, 3), r3_ref("t3", 13, 4)
    typed = [PayloadRef.from_payload(ref) for ref in (t1, t2, t3)]
    merged = merge_router_replay_prefix(typed[0], None, typed[1], None)
    merged = merge_router_replay_prefix(merged, nxt=typed[2])
    expected = merge_router_replay_prefix(merge_router_replay_prefix(t1, 7, t2, 12), 12, t3, 17)
    assert merged == expected
    assert len(merged) == 16
    with pytest.raises(ValueError, match="prev_num_tokens=8 disagrees"):
        merge_router_replay_prefix(typed[0], 8, typed[1], 12)


def test_payload_ref_with_path_is_rejected_as_composite_leaf():
    typed = PayloadRef.from_payload({**r3_ref(), "path": "/gpfs/abs.safetensors"})
    with pytest.raises(ValueError, match="must not carry 'path'"):
        CompositeRef.from_leaf(typed)


# -------------------------------------------------- sample result normalization


def _sample_payload(ref: dict | None, *, wrapped: bool = True) -> dict:
    seq = {"tokens": [5, 6, 7], "text": "abc", "stop_reason": "stop"}
    if ref is not None:
        seq["moe_topk_indices_ref"] = ref
    return {"result": {"sequences": [seq]}} if wrapped else {"sequences": [seq]}


@pytest.mark.parametrize("wrapped", [True, False])
def test_normalize_keeps_raw_ref_dict_and_validates_meta(wrapped):
    ref = r3_ref()
    out = normalize_sample_result(
        _sample_payload(copy.deepcopy(ref), wrapped=wrapped), lambda: None
    )
    seq = out["sequences"][0]
    assert seq["moe_topk_indices_ref"] == ref
    assert isinstance(seq["moe_topk_indices_ref"], dict)
    assert parse_moe_topk_indices_ref(seq).meta.num_rows == 6


@pytest.mark.parametrize("wrapped", [True, False])
def test_normalize_rejects_malformed_meta_on_receipt(wrapped):
    ref = r3_ref()
    ref["meta"]["num_rows"] = 99
    with pytest.raises(ValueError, match=r"sequence 0: invalid moe_topk_indices_ref: .*num_rows"):
        normalize_sample_result(_sample_payload(ref, wrapped=wrapped), lambda: None)


def test_normalize_without_meta_or_with_opaque_ref_still_works():
    ref = r3_ref(meta=False)
    out = normalize_sample_result(_sample_payload(copy.deepcopy(ref)), lambda: None)
    assert out["sequences"][0]["moe_topk_indices_ref"] == ref
    # older/minimal refs that are not full leaves are not validated beyond meta
    minimal = {"uri": "weaver://x", "format": "safetensors"}
    out = normalize_sample_result(_sample_payload(dict(minimal)), lambda: None)
    assert out["sequences"][0]["moe_topk_indices_ref"] == minimal
    validate_sequence_payload_refs(None)
    validate_sequence_payload_refs(["not-a-dict", {"moe_topk_indices_ref": "opaque"}])


def _run_sample(async_mode: bool, payload: dict):
    service, handle = MagicMock(), MagicMock()
    kwargs = dict(prompt=ModelInput.from_ints([1, 2, 3, 4]), return_moe_topk_indices=True)
    if async_mode:
        service.enqueue_operation = AsyncMock(return_value=handle)
        handle.result = AsyncMock(return_value=payload)
        client = AsyncSamplingClient(service=service, sampling_session_id="s", base_model="m")
        client._ensure_tokenizer_source = AsyncMock()  # type: ignore[method-assign]
        return asyncio.run(client.sample(**kwargs))
    service.enqueue_operation.return_value = handle
    handle.result.return_value = payload
    client = SamplingClient(service=service, sampling_session_id="s", base_model="m")
    return client.sample(**kwargs)


@pytest.mark.parametrize("async_mode", [False, True])
def test_sync_and_async_sample_carry_meta(async_mode):
    ref = r3_ref()
    result = _run_sample(async_mode, _sample_payload(copy.deepcopy(ref)))
    seq = result["sequences"][0]
    assert seq["moe_topk_indices_ref"] == ref
    typed = parse_moe_topk_indices_ref(seq)
    assert typed is not None and typed.meta is not None
    assert typed.meta.prompt_tokens == 4 and typed.meta.response_tokens == 3


@pytest.mark.parametrize("async_mode", [False, True])
def test_sync_and_async_sample_reject_malformed_meta(async_mode):
    ref = r3_ref()
    ref["meta"]["prompt_tokens"] = 100
    with pytest.raises(ValueError, match="invalid moe_topk_indices_ref"):
        _run_sample(async_mode, _sample_payload(ref))


@pytest.mark.parametrize("async_mode", [False, True])
def test_sync_and_async_sample_without_meta(async_mode):
    ref = r3_ref(meta=False)
    seq = _run_sample(async_mode, _sample_payload(copy.deepcopy(ref)))["sequences"][0]
    assert seq["moe_topk_indices_ref"] == ref
    assert parse_moe_topk_indices_ref(seq).meta is None


def test_normalize_rejects_meta_response_count_mismatch():
    ref = r3_ref()  # meta describes 3 response tokens
    payload = _sample_payload(ref)
    seqs = payload["result"]["sequences"] if "result" in payload else payload["sequences"]
    seqs[0]["tokens"] = seqs[0]["tokens"][:-1]
    with pytest.raises(ValueError, match=r"sequence 0: .*describes 3 response tokens .* has 2"):
        normalize_sample_result(payload, lambda: None)
