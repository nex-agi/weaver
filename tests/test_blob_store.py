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

"""Tests for the offloaded-blob store (packing, offsets, read cache)."""

import numpy as np
import pytest
import torch

from weaver.blob_store import BlobStore


def _store(monkeypatch, tmp_path, *, enabled=True):
    monkeypatch.setenv("WEAVER_BLOB_OFFLOAD", "1" if enabled else "0")
    monkeypatch.setenv("WEAVER_BLOB_ROOT", str(tmp_path))
    return BlobStore()


def test_offload_disabled_is_inert(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path, enabled=False)
    assert store.enabled is False


def test_put_tensor_round_trips(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    ref = store.put_tensor(
        torch.tensor([1, 2, 3], dtype=torch.int64), model_id="m", seq_id=0, field="x"
    )
    assert "offset" not in ref["$blob"]  # standalone file, not a packed slice
    out = store.get_array(ref)
    assert out.dtype == np.int64
    assert out.tolist() == [1, 2, 3]


def test_pack_writes_one_file_with_offsets(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    pack = store.open_pack(model_id="m", seq_id=7)
    r_tokens = pack.put_tensor(torch.tensor([1, 2, 3, 4], dtype=torch.int64), field="d0_tokens")
    r_target = pack.put_tensor(torch.tensor([2, 3, 4], dtype=torch.int64), field="d0_target")
    r_weights = pack.put_tensor(torch.tensor([0.5, 0.25], dtype=torch.float32), field="d0_w")
    pack.commit()

    # All three references point at one file, distinguished by byte offset.
    keys = {r["$blob"]["relative_path"] for r in (r_tokens, r_target, r_weights)}
    assert len(keys) == 1
    seq_dir = tmp_path / store.run / "m" / "seq-7"
    assert [p.name for p in seq_dir.iterdir()] == ["pack.bin"]
    assert r_tokens["$blob"]["offset"] == 0
    assert r_target["$blob"]["offset"] == 4 * 8  # after four int64s
    assert r_weights["$blob"]["offset"] == 4 * 8 + 3 * 8

    # Mixed dtypes resolve back exactly from their slices.
    assert store.get_array(r_tokens).tolist() == [1, 2, 3, 4]
    assert store.get_array(r_target).tolist() == [2, 3, 4]
    w = store.get_array(r_weights)
    assert w.dtype == np.float32
    assert np.array_equal(w, np.float32([0.5, 0.25]))


def test_pack_read_uses_single_cached_read(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    pack = store.open_pack(model_id="m", seq_id=1)
    refs = [
        pack.put_tensor(torch.tensor([i, i], dtype=torch.int64), field=f"d{i}") for i in range(5)
    ]
    pack.commit()

    for i, ref in enumerate(refs):
        assert store.get_array(ref).tolist() == [i, i]
    # One packed file read once, then five slices served from the cache.
    assert len(store._read_cache) == 1  # pylint: disable=protected-access


def test_empty_pack_commit_writes_nothing(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    pack = store.open_pack(model_id="m", seq_id=2)
    pack.commit()  # must not raise or create a file
    assert not (tmp_path / store.run / "m" / "seq-2").exists()


def test_put_packed_round_trips_in_order(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    arrays = [[-0.1, -0.2, -0.3], [-1.5, -2.5], [-9.9]]
    ref = store.put_packed(arrays, model_id="m", field="lfo")
    assert ref["$blob"]["splits"] == [3, 2, 1]  # order/lengths preserved
    out = store.get_packed(ref)
    assert len(out) == len(arrays)
    for got, want in zip(out, arrays):
        assert np.array_equal(got, np.float32(want))


def test_get_packed_rejects_mismatched_splits(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    ref = store.put_packed([[1.0, 2.0, 3.0], [4.0, 5.0]], model_id="m", field="f")
    ref["$blob"]["splits"] = [3, 3]  # sums to 6, but the blob holds 5 floats
    with pytest.raises(ValueError, match="splits sum"):
        store.get_packed(ref)


@pytest.mark.parametrize(
    "bad_path",
    ["../escape.bin", "/etc/hostname", "a/../../escape.bin", "a\\b.bin", "./x.bin", ""],
)
def test_get_array_rejects_path_traversal(monkeypatch, tmp_path, bad_path):
    """A crafted $blob relative_path must not resolve outside the store root."""
    store = _store(monkeypatch, tmp_path)
    ref = {
        "$blob": {
            "storage": "filesystem",
            "format": "raw-tensor",
            "relative_path": bad_path,
            "dtype": "int32",
            "shape": [4],
            "codec": "raw",
            "size_bytes": 16,
        }
    }
    with pytest.raises(ValueError, match="unsafe blob relative_path"):
        store.get_array(ref)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda p: {"loss_fn_outputs_packed": p},  # top level
        lambda p: {"result": {"loss_fn_outputs_packed": p}},  # under result
        lambda p: {  # nested per task under forward_task_results
            "result": {"forward_task_results": {"t1": {"loss_fn_outputs_packed": p}}}
        },
    ],
)
def test_resolve_result_blobs_handles_all_shapes(monkeypatch, tmp_path, wrap):
    """The leg-2 SDK resolver must restore loss_fn_outputs in every response
    shape the consumer accepts (top level, under ``result``, or nested under
    ``forward_task_results``)."""
    import weaver.blob_store as blob_store
    from weaver.operations import _resolve_result_blobs

    monkeypatch.setenv("WEAVER_BLOB_OFFLOAD", "1")
    monkeypatch.setenv("WEAVER_BLOB_ROOT", str(tmp_path))
    blob_store.get_blob_store.cache_clear()
    try:
        store = blob_store.get_blob_store()
        ref = store.put_packed([[-1.0, -2.0]], model_id="m", field="f")
        packed = {
            "ref": ref,
            "field": "logprobs",
            "siblings": [{"logprobs": {"dtype": "float32", "shape": [2]}}],
            "count": 1,
        }
        resolved = _resolve_result_blobs(wrap(packed))

        found = []

        def walk(node):
            if isinstance(node, dict):
                if "loss_fn_outputs" in node and "loss_fn_outputs_packed" not in node:
                    found.append(node["loss_fn_outputs"])
                for value in node.values():
                    walk(value)

        walk(resolved)
        assert found, "packed descriptor was not resolved"
        data = found[0][0]["logprobs"]["data"]
        assert np.array_equal(np.float32(data), np.float32([-1.0, -2.0]))
    finally:
        blob_store.get_blob_store.cache_clear()
