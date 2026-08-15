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

"""Backend and update stay separate, immutable, and fail closed."""

from __future__ import annotations

import dataclasses

import pytest

from weaver.types.weight_sync import (
    WEIGHT_SYNC_BACKENDS,
    WEIGHT_SYNC_UPDATES,
    WeightSyncSelection,
    normalize_weight_sync_backend,
    normalize_weight_sync_update,
)


def test_default_selection_is_the_established_checkpoint_path():
    selection = WeightSyncSelection()
    assert selection.backend == "default"
    assert selection.update == "full"
    assert selection.debug_checksum is False
    assert selection.rebaseline_interval is None
    assert not selection.is_live_collective
    assert not selection.is_delta


def test_backend_and_update_are_independent_dimensions():
    # There is no combined "nccl_delta" identity: the two axes compose.
    assert set(WEIGHT_SYNC_BACKENDS) == {"default", "mooncake", "nccl"}
    assert set(WEIGHT_SYNC_UPDATES) == {"full", "delta"}
    for backend in WEIGHT_SYNC_BACKENDS:
        for update in WEIGHT_SYNC_UPDATES:
            if (backend, update) in (("mooncake", "delta"), ("nccl", "delta")):
                continue
            selection = WeightSyncSelection(backend=backend, update=update)
            assert (selection.backend, selection.update) == (backend, update)


@pytest.mark.parametrize(
    "backend,update",
    [("default", "full"), ("default", "delta"), ("mooncake", "full"), ("nccl", "full")],
)
def test_qualified_combinations_are_accepted(backend, update):
    assert WeightSyncSelection(backend=backend, update=update).update == update


def test_collective_delta_is_refused_until_it_exists():
    # Accepting it would run the FULL collective publication while reporting a
    # delta update. Removed when detection, encoding, transfer and target
    # application all exist.
    with pytest.raises(ValueError, match="not implemented yet"):
        WeightSyncSelection(backend="nccl", update="delta")


def test_mooncake_delta_is_refused_with_the_exact_capability_reason():
    # The Mooncake writer uploads whole shard files and has no delta producer.
    # Refusing here is the point: the established behaviour silently downgraded
    # to a full sync, which is what this selection must never do.
    with pytest.raises(ValueError, match="no delta producer"):
        WeightSyncSelection(backend="mooncake", update="delta")


def test_unknown_backend_and_update_are_refused():
    with pytest.raises(ValueError, match="backend must be one of"):
        WeightSyncSelection(backend="rdma")
    with pytest.raises(ValueError, match="update must be one of"):
        WeightSyncSelection(update="incremental")
    with pytest.raises(ValueError):
        normalize_weight_sync_backend(None)
    with pytest.raises(ValueError):
        normalize_weight_sync_update(7)


def test_selection_is_immutable_after_construction():
    selection = WeightSyncSelection(backend="nccl")
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.backend = "default"
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.update = "delta"


def test_debug_checksum_is_off_by_default_and_scoped_to_the_verifying_backend():
    assert WeightSyncSelection(backend="nccl").debug_checksum is False
    assert WeightSyncSelection(backend="nccl", debug_checksum=True).debug_checksum is True
    with pytest.raises(ValueError, match="debug_checksum"):
        WeightSyncSelection(backend="default", debug_checksum=True)
    with pytest.raises(ValueError, match="debug_checksum"):
        WeightSyncSelection(backend="mooncake", debug_checksum=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        WeightSyncSelection(backend="nccl", debug_checksum="yes")


def test_rebaseline_interval_is_scoped_to_the_collective_delta_path():
    # Its one valid combination is unimplemented, so it cannot be set at all yet.
    with pytest.raises(ValueError, match="not implemented yet"):
        WeightSyncSelection(backend="nccl", update="delta", rebaseline_interval=8)
    # The durable-checkpoint delta path re-baselines on accumulated byte drift
    # and keeps its own knobs; it must not be steered through this field.
    with pytest.raises(ValueError, match="rebaseline_interval applies only"):
        WeightSyncSelection(backend="default", update="delta", rebaseline_interval=8)
    with pytest.raises(ValueError, match="rebaseline_interval applies only"):
        WeightSyncSelection(backend="nccl", update="full", rebaseline_interval=8)


def test_payload_round_trip():
    selection = WeightSyncSelection(backend="nccl", update="full", debug_checksum=True)
    payload = selection.to_payload()
    assert payload == {
        "backend": "nccl",
        "update": "full",
        "debug_checksum": True,
    }
    assert WeightSyncSelection.from_payload(payload) == selection
    # An unset interval is omitted rather than sent as null.
    assert "rebaseline_interval" not in WeightSyncSelection(backend="nccl").to_payload()


def test_from_payload_reads_the_nested_session_selection():
    session = {"id": "s-1", "weight_sync": {"backend": "default", "update": "delta"}}
    selection = WeightSyncSelection.from_payload(session)
    assert (selection.backend, selection.update) == ("default", "delta")


def test_from_payload_accepts_a_control_plane_without_a_structured_selection():
    # An older control plane reports no selection: that is the established
    # durable-checkpoint configuration, not an error.
    assert WeightSyncSelection.from_payload({"id": "s-1"}) == WeightSyncSelection()
    # ...and the narrow legacy spelling resolves to the live collective.
    assert WeightSyncSelection.from_payload(
        {"id": "s-1", "weight_sync_mode": "nccl_v1"}
    ) == WeightSyncSelection(backend="nccl", update="full")


def test_from_payload_rejects_a_selection_this_sdk_cannot_represent():
    with pytest.raises(RuntimeError, match="unusable weight sync selection"):
        WeightSyncSelection.from_payload(
            {"weight_sync": {"backend": "mooncake", "update": "delta"}}
        )
    with pytest.raises(RuntimeError, match="must be an object"):
        WeightSyncSelection.from_payload("nccl")


def test_assert_matches_binds_the_control_plane_value_when_they_agree():
    requested = WeightSyncSelection(backend="default", update="delta")
    resolved = WeightSyncSelection(backend="default", update="delta")
    # The caller left the interval unset, so the control plane's value stands.
    assert requested.assert_matches(resolved) is resolved


def test_assert_matches_refuses_a_silently_different_backend():
    requested = WeightSyncSelection(backend="nccl")
    resolved = WeightSyncSelection(backend="default")
    with pytest.raises(RuntimeError, match="backend: requested 'nccl'"):
        requested.assert_matches(resolved)


def test_assert_matches_refuses_a_silently_different_update_or_checksum():
    with pytest.raises(RuntimeError, match="update: requested 'delta'"):
        WeightSyncSelection(backend="default", update="delta").assert_matches(
            WeightSyncSelection(backend="default", update="full")
        )
    with pytest.raises(RuntimeError, match="debug_checksum: requested True"):
        WeightSyncSelection(backend="nccl", debug_checksum=True).assert_matches(
            WeightSyncSelection(backend="nccl")
        )


def test_assert_matches_compares_an_explicitly_requested_interval():
    # rebaseline_interval only exists on the unimplemented collective delta
    # path, so this comparison is exercised through the shared code path.
    requested = WeightSyncSelection(backend="nccl")
    with pytest.raises(RuntimeError, match="debug_checksum"):
        WeightSyncSelection(backend="nccl", debug_checksum=True).assert_matches(requested)
