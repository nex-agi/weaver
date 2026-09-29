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

Leaf meta: newer servers attach a ``meta`` object to each R3 leaf ref::

    {"token_alignment": "target_aligned", "num_tokens": P + R,
     "prompt_tokens": P, "num_rows": P + R - 1}

It is parsed by :class:`~weaver.types.payload_ref.PayloadRefMeta` and, when
present, is the authoritative row/token count (see :func:`ref_meta`); older
servers omit it and everything falls back to ``shape``/``token_count`` or
explicit counts. Leaves may be raw dicts or typed
:class:`~weaver.types.payload_ref.PayloadRef` objects; they (meta included) are
carried verbatim into composites.

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

from .payload_ref import TOKEN_ALIGNMENT_TARGET_ALIGNED, PayloadRef, PayloadRefMeta

COMPOSITE_REF_KIND = "composite"
COMPOSITE_REF_SCHEMA = "weaver.payload_ref.composite.v1"
MAX_COMPOSITE_SEGMENTS = 1024

LeafRef = Mapping[str, Any] | PayloadRef

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
    :meth:`from_leaf`, which needs a row count (explicit, ``meta.num_rows``,
    ``shape[0]`` or ``token_count``).
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
    def from_leaf(cls, ref: LeafRef, num_rows: int | None = None) -> CompositeRef:
        """Wrap a leaf ref covering rows ``[0, num_rows)``.

        Row count precedence: explicit ``num_rows`` > ``ref["meta"]["num_rows"]``
        (server-attached, see :func:`ref_meta`) > ``ref["shape"][0]`` (R3
        indices refs) > ``ref["token_count"]`` (sampling-mask refs). If none is
        available ``num_rows`` must be passed. ``ref`` may be a leaf dict or a
        typed :class:`PayloadRef` (its ``meta`` is used the same way).

        Raises:
            ValueError: if ``meta.num_rows`` and ``shape[0]`` both exist and
                differ (inconsistent ref), or ``meta`` is malformed.
        """

        leaf = _validated_leaf(ref, where="leaf")
        if num_rows is None:
            num_rows = _leaf_num_rows(leaf)
            if num_rows is None:
                raise ValueError(
                    "num_rows is required: leaf ref has no 'meta.num_rows', 'shape' or "
                    "'token_count' to infer its row count from."
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
        elif isinstance(obj, Mapping | PayloadRef):
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


def ref_meta(ref: Any) -> dict[str, Any] | None:
    """Return a copy of a leaf ref's server-attached ``meta``, or None.

    Newer servers attach, to each R3 leaf ref::

        {"token_alignment": "target_aligned", "num_tokens": P + R,
         "prompt_tokens": P, "num_rows": P + R - 1}

    where ``P``/``R`` are the prompt/response token counts of that sampling
    request. Older servers omit it; composites and non-refs return None. For
    the typed form use :class:`~weaver.types.payload_ref.PayloadRef` ``.meta``.

    Raises:
        ValueError: if ``meta`` is present but malformed or self-inconsistent.
    """

    if isinstance(ref, PayloadRef):
        return None if ref.meta is None else ref.meta.to_payload()
    if isinstance(ref, CompositeRef) or is_composite_ref(ref) or not isinstance(ref, Mapping):
        return None
    meta = _leaf_meta(ref, where="ref")
    return None if meta is None else meta.to_payload()


def merge_router_replay_prefix(
    prev: LeafRef | CompositeRef,
    prev_num_tokens: int | None = None,
    nxt: LeafRef | CompositeRef | None = None,
    next_num_tokens: int | None = None,
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

    ``prev``/``nxt`` may be leaf dicts, typed
    :class:`~weaver.types.payload_ref.PayloadRef` leaves, :class:`CompositeRef`
    objects, or composite payload dicts. Token counts are optional; when ``None`` they are
    derived from the ref:

    * leaf: ``meta.num_tokens`` (see :func:`ref_meta`), else row count + 1
      (row count from ``meta.num_rows``/``shape[0]``/``token_count``);
    * composite: ``len(ref) + 1`` (a merged R3 composite over L tokens has
      L-1 rows).

    When a count is given explicitly and the leaf's ``meta.num_tokens``
    disagrees, the tokens being merged are not the tokens the sampler saw
    (e.g. retokenization drift) and a ValueError is raised. For leaves without
    row information an explicit count is required (rows assumed ``count - 1``).
    Chained merges over K turns work by feeding the result back as ``prev``::

        merged = merge_router_replay_prefix(ref1, None, ref2, None)  # meta-only
        merged = merge_router_replay_prefix(merged, nxt=ref3)
        merged = merge_router_replay_prefix(ref1, L1, ref2, L2)      # explicit counts

    Raises:
        ValueError: if ``prev_num_tokens > next_num_tokens``, a token count is
            not positive, cannot be derived, disagrees with ``meta``, or either
            side has too few rows.
    """

    if nxt is None:
        raise TypeError("merge_router_replay_prefix() missing required argument 'nxt'")
    prev, nxt = _leaf_mapping(prev), _leaf_mapping(nxt)
    prev_num_tokens = _router_replay_num_tokens(prev, prev_num_tokens, "prev", "prev_num_tokens")
    next_num_tokens = _router_replay_num_tokens(nxt, next_num_tokens, "nxt", "next_num_tokens")
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


def _router_replay_num_tokens(obj: Any, num_tokens: Any, name: str, count_name: str) -> int:
    """Resolve (or cross-check) the token count of one side of an R3 merge."""

    if num_tokens is not None:
        num_tokens = _non_negative_int(num_tokens, count_name)
    if is_composite_ref(obj):
        if num_tokens is not None:
            return num_tokens
        return len(CompositeRef.from_payload(obj)) + 1
    if not isinstance(obj, Mapping):
        raise TypeError(f"{name} must be a leaf ref dict or CompositeRef, got {type(obj)!r}")
    meta = _leaf_meta(obj, where=name)
    if meta is not None and not meta.is_target_aligned:
        raise ValueError(
            f"{name}: router replay refs must be {TOKEN_ALIGNMENT_TARGET_ALIGNED!r}, "
            f"got meta.token_alignment={meta.token_alignment!r}"
        )
    meta_tokens = None if meta is None else meta.num_tokens
    if num_tokens is not None:
        if meta_tokens is not None and meta_tokens != num_tokens:
            raise ValueError(
                f"{count_name}={num_tokens} disagrees with {name} meta.num_tokens="
                f"{meta_tokens}: the tokens being merged are not the tokens the sampler "
                "saw (retokenization drift?)"
            )
        return num_tokens
    if meta_tokens is not None:
        return meta_tokens
    rows = _leaf_num_rows(obj, where=name)
    if rows is None:
        raise ValueError(
            f"{count_name} is required: {name} has no 'meta.num_tokens', 'meta.num_rows', "
            "'shape' or 'token_count' to derive it from"
        )
    return rows + 1


def _as_router_replay_composite(obj: Any, num_tokens: int, name: str) -> CompositeRef:
    if is_composite_ref(obj):
        return CompositeRef.from_payload(obj)
    num_rows = _leaf_num_rows(obj, where=name)
    return CompositeRef.from_leaf(obj, num_tokens - 1 if num_rows is None else num_rows)


def _validated_leaf(ref: Any, *, where: str) -> Mapping[str, Any]:
    ref = _leaf_mapping(ref)
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
    # relative_path is the canonical locator (storage + relative_path); uri is
    # only a convenience handle, so it may not stand alone or disagree.
    relative_path = ref.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{where}: leaf ref requires a non-empty 'relative_path'")
    uri = ref.get("uri")
    if uri is not None and uri != f"weaver://{relative_path}":
        raise ValueError(
            f"{where}: leaf ref uri {uri!r} does not match relative_path {relative_path!r}"
        )
    if "path" in ref:
        # The server and trainer reject client-supplied absolute paths on
        # composite leaves; fail here instead of at submission time.
        raise ValueError(f"{where}: leaf ref must not carry 'path'; use 'relative_path'")
    _leaf_num_rows(ref, where=where)  # validates meta and meta/shape consistency
    return copy.deepcopy(dict(ref))


def _leaf_mapping(obj: Any) -> Any:
    """Turn a typed :class:`PayloadRef` into its wire dict; pass anything else through."""

    return obj.to_payload() if isinstance(obj, PayloadRef) else obj


def _leaf_meta(ref: Mapping[str, Any], *, where: str) -> PayloadRefMeta | None:
    """Parse ``ref["meta"]`` with :class:`PayloadRefMeta`, or None when absent."""

    meta = ref.get("meta")
    if meta is None:
        return None
    try:
        return PayloadRefMeta.from_payload(meta)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid leaf ref {exc}") from exc


def _leaf_num_rows(ref: Mapping[str, Any], *, where: str = "leaf") -> int | None:
    """Row count: ``meta.num_rows`` > ``shape[0]`` > ``token_count`` > None."""

    meta = _leaf_meta(ref, where=where)
    meta_rows = None if meta is None else meta.num_rows
    shape_rows = None
    shape = ref.get("shape")
    if isinstance(shape, list | tuple) and shape and _is_int(shape[0]) and shape[0] >= 0:
        shape_rows = int(shape[0])
    if meta_rows is not None:
        if shape_rows is not None and shape_rows != meta_rows:
            raise ValueError(
                f"{where}: inconsistent leaf ref: meta.num_rows={meta_rows} but "
                f"shape[0]={shape_rows}"
            )
        return meta_rows
    if shape_rows is not None:
        return shape_rows
    token_count = ref.get("token_count")
    if isinstance(token_count, int) and not isinstance(token_count, bool) and token_count >= 0:
        return token_count
    return None


def _check_leaf_bounds(leaf: Mapping[str, Any], start: int, count: int, *, where: str) -> None:
    rows = _leaf_num_rows(leaf, where=where)
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
