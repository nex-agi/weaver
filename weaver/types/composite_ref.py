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

"""Composite payload refs: lazy slice/concat algebra over leaf payload refs.

Samplers return large per-token payloads (R3 router-replay indices, sampling
masks) as opaque *leaf refs* pointing at files the producer wrote. Clients that
merge multi-turn trajectories need to slice and concatenate those payloads the
same way they slice and concatenate token lists, without downloading them. A
:class:`CompositeRef` describes such a result lazily; the trainer resolves it by
loading each leaf, slicing rows, and concatenating along the row axis.

Wire format (``weaver.payload_ref.composite.v1``)::

    {
        "kind": "composite",
        "schema": "weaver.payload_ref.composite.v1",
        "segments": [
            {"ref": {<leaf ref, verbatim>}, "start": 0, "count": 99},
            {"ref": {<leaf ref, verbatim>}, "start": 99, "count": 60},
        ],
    }

Rules:

* ``start``/``count`` are in the leaf's native row coordinates.
* The destination is implicit: rows are concatenated in segment order.
* Composites are flat: a segment's ``ref`` is never itself composite.
* v1 supports only ``ref`` segments (no ``fill``/``inline``).

Example::

    >>> leaf = {"storage": "gpfs", "format": "safetensors", "schema": "s",
    ...         "relative_path": "a.safetensors", "shape": [10, 2, 2]}
    >>> ref = CompositeRef.from_leaf(leaf)
    >>> len(ref), len(ref[2:-3]), len(ref[:4] + ref[6:])
    (10, 5, 8)
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

COMPOSITE_REF_KIND = "composite"
COMPOSITE_REF_SCHEMA = "weaver.payload_ref.composite.v1"
MAX_COMPOSITE_SEGMENTS = 1024

_TOP_LEVEL_KEYS = frozenset({"kind", "schema", "segments"})
_SEGMENT_KEYS = frozenset({"ref", "start", "count"})


@dataclass(frozen=True)
class CompositeSegment:
    """One ``ref`` segment: rows ``[start, start + count)`` of a leaf ref."""

    ref: Mapping[str, Any]
    start: int
    count: int

    @property
    def stop(self) -> int:
        return self.start + self.count


def is_composite_ref(obj: Any) -> bool:
    """Return True if ``obj`` looks like a composite ref (no validation)."""

    if isinstance(obj, CompositeRef):
        return True
    return isinstance(obj, Mapping) and obj.get("kind") == COMPOSITE_REF_KIND


class CompositeRef:
    """Immutable, flat, lazily-described concatenation of leaf-ref row ranges.

    Supports ``len()``, slicing (step 1 only, list-like clamping and negative
    bounds) and ``+`` with another :class:`CompositeRef`. Adjacent contiguous
    segments on the same leaf are coalesced and empty segments dropped, so the
    representation stays minimal across repeated slice/concat.

    Leaf dicts are never guessed into composites: wrap them explicitly with
    :meth:`from_leaf`, which needs a row count (explicit, ``shape[0]`` or
    ``token_count``).
    """

    __slots__ = ("_segments", "_len")
    _segments: tuple[CompositeSegment, ...]
    _len: int

    def __init__(self, segments: Iterable[CompositeSegment | tuple[Any, int, int]] = ()) -> None:
        validated: list[CompositeSegment] = []
        for index, segment in enumerate(segments):
            if isinstance(segment, Mapping):
                raise TypeError(
                    "CompositeRef segments must be (ref, start, count) tuples; "
                    "use CompositeRef.from_payload for wire dicts."
                )
            if isinstance(segment, CompositeSegment):
                segment = (segment.ref, segment.start, segment.count)
            try:
                ref, start, count = segment
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"segment {index} must be a (ref, start, count) tuple, got {segment!r}"
                ) from exc
            leaf = _validated_leaf(ref, where=f"segment {index}")
            start = _non_negative_int(start, f"segment {index} start")
            count = _non_negative_int(count, f"segment {index} count")
            _check_leaf_bounds(leaf, start, count, where=f"segment {index}")
            validated.append(CompositeSegment(leaf, start, count))
        self._init_normalized(validated)

    # ------------------------------------------------------------------ builders

    @classmethod
    def _from_trusted(cls, segments: Iterable[CompositeSegment]) -> CompositeRef:
        """Build from already-validated segments (leaves shared, not copied)."""

        obj = cls.__new__(cls)
        obj._init_normalized(segments)
        return obj

    def _init_normalized(self, segments: Iterable[CompositeSegment]) -> None:
        normalized: list[CompositeSegment] = []
        for segment in segments:
            if segment.count == 0:
                continue
            if normalized:
                last = normalized[-1]
                if last.stop == segment.start and _same_leaf(last.ref, segment.ref):
                    normalized[-1] = CompositeSegment(
                        last.ref, last.start, last.count + segment.count
                    )
                    continue
            normalized.append(segment)
        object.__setattr__(self, "_segments", tuple(normalized))
        object.__setattr__(self, "_len", sum(seg.count for seg in normalized))

    @classmethod
    def from_leaf(cls, ref: Mapping[str, Any], num_rows: int | None = None) -> CompositeRef:
        """Wrap a leaf ref covering rows ``[0, num_rows)``.

        ``num_rows`` defaults to ``ref["shape"][0]`` (R3 indices refs), then
        ``ref["token_count"]`` (sampling-mask refs). If neither is present the
        row count cannot be inferred and ``num_rows`` must be passed.
        """

        leaf = _validated_leaf(ref, where="leaf")
        if num_rows is None:
            num_rows = _leaf_num_rows(leaf)
            if num_rows is None:
                raise ValueError(
                    "num_rows is required: leaf ref has neither 'shape' nor 'token_count' "
                    "to infer its row count from."
                )
        num_rows = _non_negative_int(num_rows, "num_rows")
        _check_leaf_bounds(leaf, 0, num_rows, where="leaf")
        return cls._from_trusted([CompositeSegment(leaf, 0, num_rows)])

    @classmethod
    def from_payload(cls, obj: Any, num_rows: int | None = None) -> CompositeRef:
        """Parse a composite wire payload (strictly) or wrap a leaf ref.

        ``num_rows`` only applies to leaf refs (see :meth:`from_leaf`); for a
        composite payload it must be omitted or equal to the composite length.
        """

        if isinstance(obj, CompositeRef):
            result = obj
        elif is_composite_ref(obj):
            result = cls._parse_composite(obj)
        elif isinstance(obj, Mapping):
            return cls.from_leaf(obj, num_rows)
        else:
            raise TypeError(f"expected a payload ref mapping or CompositeRef, got {type(obj)!r}")
        if num_rows is not None and num_rows != len(result):
            raise ValueError(
                f"num_rows={num_rows} does not match composite ref length {len(result)}"
            )
        return result

    @classmethod
    def _parse_composite(cls, payload: Mapping[str, Any]) -> CompositeRef:
        unknown = set(payload) - _TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"composite ref has unknown keys: {sorted(unknown)}")
        if payload.get("schema") != COMPOSITE_REF_SCHEMA:
            raise ValueError(
                f"composite ref schema must be {COMPOSITE_REF_SCHEMA!r}, "
                f"got {payload.get('schema')!r}"
            )
        segments = payload.get("segments")
        if not isinstance(segments, list | tuple) or not segments:
            raise ValueError("composite ref 'segments' must be a non-empty list")
        if len(segments) > MAX_COMPOSITE_SEGMENTS:
            raise ValueError(
                f"composite ref has {len(segments)} segments; "
                f"maximum is {MAX_COMPOSITE_SEGMENTS}"
            )
        parsed: list[CompositeSegment] = []
        for index, segment in enumerate(segments):
            where = f"segment {index}"
            if not isinstance(segment, Mapping):
                raise ValueError(f"{where} must be an object, got {type(segment).__name__}")
            unknown = set(segment) - _SEGMENT_KEYS
            if unknown:
                raise ValueError(
                    f"{where} has unsupported keys {sorted(unknown)}; "
                    "v1 supports only {'ref', 'start', 'count'} segments"
                )
            if "ref" not in segment:
                raise ValueError(f"{where} is missing 'ref'")
            leaf = _validated_leaf(segment["ref"], where=where)
            start = _non_negative_int(segment.get("start"), f"{where} start")
            count = _non_negative_int(segment.get("count"), f"{where} count")
            if count == 0:
                raise ValueError(f"{where} count must be positive")
            _check_leaf_bounds(leaf, start, count, where=where)
            parsed.append(CompositeSegment(leaf, start, count))
        return cls._from_trusted(parsed)

    # ------------------------------------------------------------------ export

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-able wire payload (leaf dicts are deep-copied)."""

        if not self._segments:
            raise ValueError("cannot serialize an empty CompositeRef")
        if len(self._segments) > MAX_COMPOSITE_SEGMENTS:
            raise ValueError(
                f"CompositeRef has {len(self._segments)} segments; "
                f"maximum is {MAX_COMPOSITE_SEGMENTS}"
            )
        return {
            "kind": COMPOSITE_REF_KIND,
            "schema": COMPOSITE_REF_SCHEMA,
            "segments": [
                {"ref": copy.deepcopy(dict(seg.ref)), "start": seg.start, "count": seg.count}
                for seg in self._segments
            ],
        }

    @property
    def segments(self) -> tuple[CompositeSegment, ...]:
        """Normalized segments (leaf dicts are copies; mutating them is harmless)."""

        return tuple(
            CompositeSegment(copy.deepcopy(dict(seg.ref)), seg.start, seg.count)
            for seg in self._segments
        )

    # ------------------------------------------------------------------ algebra

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("CompositeRef is not iterable; use .segments")

    def __getitem__(self, key: slice) -> CompositeRef:
        if not isinstance(key, slice):
            raise TypeError(
                f"CompositeRef indices must be slices, not {type(key).__name__}; "
                "use ref[i:i + 1] for a single row"
            )
        if key.step not in (None, 1):
            raise ValueError("CompositeRef slicing supports only step None or 1")
        start, stop, _ = key.indices(self._len)
        if stop <= start:
            return CompositeRef._from_trusted(())
        out: list[CompositeSegment] = []
        offset = 0
        for seg in self._segments:
            seg_lo, seg_hi = offset, offset + seg.count
            offset = seg_hi
            if seg_hi <= start:
                continue
            if seg_lo >= stop:
                break
            lo = max(start, seg_lo) - seg_lo
            hi = min(stop, seg_hi) - seg_lo
            out.append(CompositeSegment(seg.ref, seg.start + lo, hi - lo))
        return CompositeRef._from_trusted(out)

    def __add__(self, other: Any) -> CompositeRef:
        if not isinstance(other, CompositeRef):
            return NotImplemented
        return CompositeRef._from_trusted(self._segments + other._segments)

    def __radd__(self, other: Any) -> CompositeRef:
        # ``sum(refs)`` starts from int 0; treat it as the empty composite.
        if isinstance(other, int) and not isinstance(other, bool) and other == 0:
            return self
        if isinstance(other, CompositeRef):
            return other.__add__(self)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeRef):
            return NotImplemented
        return self._segments == other._segments

    __hash__ = None  # type: ignore[assignment]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CompositeRef is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CompositeRef is immutable")

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{_leaf_label(seg.ref)}[{seg.start}:{seg.stop}]" for seg in self._segments
        )
        return f"CompositeRef(len={self._len}, segments=[{parts}])"


def merge_router_replay_prefix(
    prev: Mapping[str, Any] | CompositeRef,
    prev_num_tokens: int,
    nxt: Mapping[str, Any] | CompositeRef,
    next_num_tokens: int,
) -> CompositeRef:
    """Merge R3 router-replay refs of a turn and its prefix-extending next turn.

    Use when turn 2's full token sequence (length ``next_num_tokens``) starts
    with turn 1's full sequence (length ``prev_num_tokens``), i.e. the two
    turns are merged into one training datum over turn 2's tokens.

    R3 refs are *target-aligned*: row ``i`` holds the routing of input token
    ``i``, and a leaf for an N-token sequence has N-1 rows (prompt +
    output[:-1]), because the last sampled token is never fed forward. The
    merged ref is therefore::

        prev[0 : prev_num_tokens - 1] + nxt[prev_num_tokens - 1 : next_num_tokens - 1]

    Why the boundary is ``prev_num_tokens - 1``: turn 1 never forwarded its
    last sampled token, so turn 1's ref has no routing row for it. That token
    is forwarded for the first time during turn 2's prefill, so its row (and
    everything after it) must come from turn 2's ref. Rows before it were
    produced by turn 1's own forward passes and are what turn 1 was actually
    sampled with.

    ``prev``/``nxt`` may be leaf dicts (row count from ``shape[0]``, else
    assumed ``num_tokens - 1``), :class:`CompositeRef` objects, or composite
    payload dicts, so chained merges over K turns work::

        merged = merge_router_replay_prefix(ref1, L1, ref2, L2)
        merged = merge_router_replay_prefix(merged, L2, ref3, L3)

    Raises:
        ValueError: if ``prev_num_tokens > next_num_tokens``, a token count is
            not positive, or either side has too few rows.
    """

    prev_num_tokens = _non_negative_int(prev_num_tokens, "prev_num_tokens")
    next_num_tokens = _non_negative_int(next_num_tokens, "next_num_tokens")
    if prev_num_tokens < 1:
        raise ValueError("prev_num_tokens must be >= 1")
    if prev_num_tokens > next_num_tokens:
        raise ValueError(
            f"prev_num_tokens ({prev_num_tokens}) must be <= next_num_tokens "
            f"({next_num_tokens}); the next turn must extend the previous sequence"
        )
    boundary = prev_num_tokens - 1
    end = next_num_tokens - 1
    prev_ref = _as_router_replay_composite(prev, prev_num_tokens, "prev")
    next_ref = _as_router_replay_composite(nxt, next_num_tokens, "nxt")
    if len(prev_ref) < boundary:
        raise ValueError(
            f"prev has {len(prev_ref)} rows but prev_num_tokens={prev_num_tokens} "
            f"requires at least {boundary}"
        )
    if len(next_ref) < end:
        raise ValueError(
            f"nxt has {len(next_ref)} rows but next_num_tokens={next_num_tokens} "
            f"requires at least {end}"
        )
    return prev_ref[:boundary] + next_ref[boundary:end]


# ---------------------------------------------------------------------- helpers


def _as_router_replay_composite(obj: Any, num_tokens: int, name: str) -> CompositeRef:
    if is_composite_ref(obj):
        return CompositeRef.from_payload(obj)
    if not isinstance(obj, Mapping):
        raise TypeError(f"{name} must be a leaf ref dict or CompositeRef, got {type(obj)!r}")
    num_rows = _leaf_num_rows(obj)
    return CompositeRef.from_leaf(obj, num_tokens - 1 if num_rows is None else num_rows)


def _validated_leaf(ref: Any, *, where: str) -> Mapping[str, Any]:
    if isinstance(ref, CompositeRef) or (
        isinstance(ref, Mapping)
        and (
            ref.get("kind") == COMPOSITE_REF_KIND
            or ref.get("schema") == COMPOSITE_REF_SCHEMA
            or "segments" in ref
        )
    ):
        raise ValueError(f"{where}: nested composite refs are not allowed (composites are flat)")
    if not isinstance(ref, Mapping):
        raise ValueError(f"{where}: leaf ref must be a dict, got {type(ref).__name__}")
    for key in ("storage", "format", "schema"):
        value = ref.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}: leaf ref requires a non-empty string {key!r}")
    if not any(isinstance(ref.get(key), str) and ref.get(key) for key in ("relative_path", "uri")):
        raise ValueError(f"{where}: leaf ref requires a non-empty 'relative_path' or 'uri'")
    return copy.deepcopy(dict(ref))


def _leaf_num_rows(ref: Mapping[str, Any]) -> int | None:
    shape = ref.get("shape")
    if isinstance(shape, list | tuple) and shape and _is_int(shape[0]) and shape[0] >= 0:
        return int(shape[0])
    token_count = ref.get("token_count")
    if isinstance(token_count, int) and not isinstance(token_count, bool) and token_count >= 0:
        return token_count
    return None


def _check_leaf_bounds(leaf: Mapping[str, Any], start: int, count: int, *, where: str) -> None:
    rows = _leaf_num_rows(leaf)
    if rows is not None and start + count > rows:
        raise ValueError(f"{where}: rows [{start}, {start + count}) exceed the leaf's {rows} rows")


def _same_leaf(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a is b or a == b:
        return True
    if a.get("storage") != b.get("storage"):
        return False
    for key in ("uri", "relative_path"):
        value = a.get(key)
        if value and value == b.get(key):
            return True
    return False


def _leaf_label(ref: Mapping[str, Any]) -> str:
    return str(ref.get("uri") or ref.get("relative_path"))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_int(value: Any, name: str) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)
