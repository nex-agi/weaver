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

"""HF weights export/download helpers shared by the sync and async clients.

Everything in this module is pure (no IO) so both client stacks build
identical requests and interpret identical responses; see
``.claude/rules/async-compatibility.md``.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional

from ._utils import lookup_case_insensitive

# Artifact kinds the server can produce. The server derives the kind
# (full_ft -> hf_model; lora -> hf_adapter, merge_adapter=true -> hf_model);
# clients only ever use it to *select* among existing artifacts.
ARTIFACT_KINDS = ("hf_model", "hf_adapter")

# Export artifacts default to a bounded 7-day TTL: they are regenerable from
# the source checkpoint, and full HF exports are tens of GB of object storage.
DEFAULT_EXPORT_TTL_SECONDS = 604800  # 7 days

# Per-file retry budgets for artifact downloads (shared by both stacks).
# Descriptor re-fetches are idempotent and cheap, but bounded so a URL the
# server keeps signing wrong fails fast instead of looping.
DOWNLOAD_MAX_URL_REFRESHES = 3
DOWNLOAD_MAX_TRANSPORT_RETRIES = 3

# ``weaver://{model_id}/checkpoints/{name}`` optionally followed by
# ``/artifacts/{kind}`` (the artifact URI shape).
# Windows device names are reserved in EVERY directory and with any extension
# ("CON.txt" still names the console device). Checked per path component.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# URL path segments derived from URIs must never contain dot-segments or
# separator characters; server ids are hyphenated-hex UUIDs, well within this.
_SAFE_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")

_ARTIFACT_URI_RE = re.compile(
    r"^weaver://(?P<model_id>[^/]+)/checkpoints/(?P<name>[^/]+)"
    r"(?:/artifacts/(?P<kind>[^/]+))?/?$"
)


def validate_resource_id(value: str, *, kind: str) -> str:
    """Validate a caller-supplied resource id before it becomes a URL segment.

    Raw ids are interpolated into API paths; without this check a value like
    ``../checkpoints/<uuid>`` is dot-segment-normalized by the HTTP stack into
    a DIFFERENT route (e.g. deleting a checkpoint through the deployments
    surface). Server resource ids are UUIDs, so anything else is rejected
    before any request is built.

    Returns:
        The canonical lowercase hyphenated UUID text, safe to interpolate.

    Raises:
        ValueError: If *value* is not a canonical UUID.
    """
    candidate = (value or "").strip()
    try:
        parsed = uuid.UUID(candidate)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{kind} id must be a UUID, got {value!r}") from exc
    canonical = str(parsed)
    if candidate.lower() != canonical:
        # Reject aliases (unhyphenated/braced/urn forms) so the text that
        # reaches the URL is exactly the text that was validated.
        raise ValueError(f"{kind} id must be a canonical hyphenated UUID, got {value!r}")
    return canonical


@dataclass(frozen=True)
class ArtifactTarget:
    """A parsed ``download_weights`` target.

    Exactly one of ``artifact_id`` or (``model_id`` + ``checkpoint_path``)
    is populated. ``kind`` is set only when the target was an artifact URI
    that names it explicitly.
    """

    artifact_id: Optional[str] = None
    model_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    kind: Optional[str] = None


def parse_download_target(target: str) -> ArtifactTarget:
    """Classify a string download target.

    Args:
        target: An artifact ``weaver://.../artifacts/{kind}`` URI, a
            checkpoint ``weaver://...`` URI, or a raw artifact id.

    Returns:
        An :class:`ArtifactTarget` describing how to resolve the artifact.

    Raises:
        ValueError: If *target* is empty, or is a ``weaver://`` URI that does
            not match the checkpoint/artifact shape, or names an unknown
            artifact kind.
    """
    normalized = (target or "").strip()
    if not normalized:
        raise ValueError("download target must not be empty")
    if not normalized.startswith("weaver://"):
        # Anything that is not a weaver URI is treated as an artifact id, and
        # must therefore be a canonical UUID (it becomes a URL path segment).
        return ArtifactTarget(artifact_id=validate_resource_id(normalized, kind="artifact"))
    match = _ARTIFACT_URI_RE.match(normalized)
    if not match:
        raise ValueError(
            f"Unrecognized weaver URI {normalized!r}: expected "
            "weaver://{model_id}/checkpoints/{name} or "
            "weaver://{model_id}/checkpoints/{name}/artifacts/{kind}"
        )
    kind = match.group("kind")
    if kind is not None and kind not in ARTIFACT_KINDS:
        raise ValueError(
            f"Unknown artifact kind {kind!r} in {normalized!r}; expected one of {ARTIFACT_KINDS}"
        )
    model_id = match.group("model_id")
    if not _SAFE_SEGMENT_RE.fullmatch(model_id):
        # The model id is interpolated into /api/v1/models/{id}/... routes;
        # '..', encoded dots, '\\' etc. would let a crafted URI reroute the
        # request. Real model ids are UUIDs, so a conservative charset is safe.
        raise ValueError(f"unsafe model id in weaver URI: {model_id!r}")
    name = match.group("name")
    return ArtifactTarget(
        model_id=model_id,
        checkpoint_path=f"weaver://{model_id}/checkpoints/{name}",
        kind=kind,
    )


def is_artifact_payload(payload: Any) -> bool:
    """Return True when *payload* is an artifact JSON, not an operation envelope.

    The HTTP layer flattens status codes, so an idempotent completed hit
    (HTTP 200 artifact body) and a freshly enqueued export (HTTP 202
    operation body) both arrive as plain dicts. Only artifacts carry a
    ``kind`` of ``hf_model``/``hf_adapter``; operation envelopes never do.
    """
    if not isinstance(payload, dict):
        return False
    kind = lookup_case_insensitive(payload, "kind")
    return isinstance(kind, str) and kind in ARTIFACT_KINDS


def resolve_checkpoint_id_from_listing(
    items: List[Dict[str, Any]], checkpoint_path: str
) -> Optional[str]:
    """Find the checkpoint id whose storage ``path`` equals *checkpoint_path*."""
    for item in items:
        if not isinstance(item, dict):
            continue
        path = lookup_case_insensitive(item, "path")
        if path is not None and str(path) == checkpoint_path:
            identifier = lookup_case_insensitive(item, "id")
            if identifier is not None:
                return str(identifier)
    return None


def select_artifact_payload(
    items: List[Dict[str, Any]],
    kind: Optional[str],
    *,
    context: str,
) -> Dict[str, Any]:
    """Pick the completed artifact to download from a listing.

    Args:
        items: Raw artifact dicts from ``GET /checkpoints/{id}/artifacts``.
        kind: Required artifact kind, or ``None`` to accept the single
            completed artifact.
        context: Human-readable target description for error messages.

    Returns:
        The selected artifact payload.

    Raises:
        RuntimeError: If no completed artifact matches — downloads never
            trigger exports implicitly, so the caller is told to run
            ``export_weights`` first.
        ValueError: If *kind* is omitted while several completed kinds exist.
    """
    completed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = lookup_case_insensitive(item, "status")
        if status is not None and str(status).lower() != "completed":
            continue
        item_kind = lookup_case_insensitive(item, "kind")
        if kind is not None and str(item_kind) != kind:
            continue
        completed.append(item)
    if not completed:
        wanted = f" of kind {kind!r}" if kind else ""
        raise RuntimeError(
            f"No completed HF weights artifact{wanted} exists for {context}. "
            "Downloads never trigger a conversion implicitly; call "
            "export_weights() (or `weaver checkpoint export`) first, then retry."
        )
    if len(completed) > 1:
        kinds = sorted(str(lookup_case_insensitive(item, "kind")) for item in completed)
        raise ValueError(
            f"Multiple completed artifacts exist for {context} (kinds: {kinds}); "
            "disambiguate with kind='hf_model' or kind='hf_adapter'."
        )
    return completed[0]


@dataclass(frozen=True)
class ArtifactFile:
    """One downloadable file from an artifact download descriptor."""

    name: str
    url: str
    size: Optional[int] = None
    sha256: Optional[str] = None
    url_expires_at: Optional[str] = None


def descriptor_files(descriptor: Any) -> List[ArtifactFile]:
    """Validate and normalize ``GET /artifacts/{id}/download`` file entries.

    File names are server-controlled input used to build local paths, so the
    validation is strict: absolute names, ``..`` traversal segments, names
    that normalize to nothing (``.``/``./``, which would alias the destination
    directory itself), non-canonical spellings (``a//b``, ``a/./b``, trailing
    slashes — anything whose literal text differs from its normalized form),
    and duplicate normalized destinations are all rejected before any file is
    created.

    Raises:
        ValueError: On a malformed descriptor or an unsafe file name.
    """
    payload = descriptor if isinstance(descriptor, dict) else {}
    raw_files = lookup_case_insensitive(payload, "files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("artifact download descriptor contains no files")
    files: List[ArtifactFile] = []
    seen_names: set = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError(f"malformed descriptor file entry: {raw!r}")
        name = str(lookup_case_insensitive(raw, "name") or "")
        url = str(lookup_case_insensitive(raw, "url") or "")
        if not name or not url:
            raise ValueError(f"descriptor file entry missing name or url: {raw!r}")
        if "\\" in name:
            # PurePosixPath treats backslashes as ordinary characters, but the
            # final ``dest_dir / name`` uses HOST semantics — on Windows a
            # backslash is a separator and ``..\\owned.bin`` escapes dest_dir.
            raise ValueError(f"unsafe file name in download descriptor: {name!r}")
        windows_view = PureWindowsPath(name)
        if windows_view.drive or windows_view.root or ".." in windows_view.parts:
            # Rejects drive-absolute (C:\\...), drive-relative (C:foo) and
            # rooted (\\Windows\\...) spellings that are traversal on Windows.
            raise ValueError(f"unsafe file name in download descriptor: {name!r}")
        pure = PurePosixPath(name)
        if not pure.parts or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe file name in download descriptor: {name!r}")
        for component in pure.parts:
            stripped = component.rstrip(" .")
            stem = component.split(".", 1)[0].upper()
            if (
                ":" in component  # ADS syntax / drive remnants
                or stripped != component  # trailing dot/space alias on Windows
                or stem in _WINDOWS_RESERVED_NAMES  # device names, with or without extension
            ):
                raise ValueError(f"unsafe file name in download descriptor: {name!r}")
        if name != str(pure):
            # A name whose literal text differs from its normalized form
            # (``a//b``, ``a/./b``, ``foo/``) can alias another entry's
            # destination or smuggle an empty segment; require canonical form.
            raise ValueError(f"non-canonical file name in download descriptor: {name!r}")
        if name in seen_names:
            raise ValueError(f"duplicate file name in download descriptor: {name!r}")
        seen_names.add(name)
        size = lookup_case_insensitive(raw, "size")
        sha256 = lookup_case_insensitive(raw, "sha256")
        expires = lookup_case_insensitive(raw, "url_expires_at")
        files.append(
            ArtifactFile(
                name=name,
                url=url,
                size=int(size) if size is not None else None,
                sha256=str(sha256) if sha256 else None,
                url_expires_at=str(expires) if expires else None,
            )
        )
    return files


def file_sha256(path: Path) -> str:
    """Compute the sha256 hex digest of *path* by streaming it from disk.

    Hashing what actually landed on disk (rather than the bytes seen on the
    wire) is deliberate: it also catches short writes and torn resumes.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_downloaded_file(part_path: Path, entry: ArtifactFile, *, verify: bool) -> None:
    """Validate a fully downloaded ``.part`` file against its manifest entry.

    Deletes the corrupt file before raising so a later retry cannot resume
    from poisoned bytes.

    Raises:
        RuntimeError: On size or sha256 mismatch.
    """
    actual_size = part_path.stat().st_size
    if entry.size is not None and actual_size != entry.size:
        part_path.unlink()
        raise RuntimeError(
            f"Downloaded size mismatch for {entry.name!r}: "
            f"expected {entry.size} bytes, got {actual_size}"
        )
    if verify and entry.sha256:
        actual = file_sha256(part_path)
        if actual != entry.sha256:
            part_path.unlink()
            raise RuntimeError(
                f"sha256 mismatch for {entry.name!r}: " f"expected {entry.sha256}, got {actual}"
            )


def is_file_already_complete(final_path: Path, entry: ArtifactFile, *, verify: bool) -> bool:
    """Return True when *final_path* already matches its manifest entry.

    Lets a re-run of ``download_weights`` skip files that finished earlier.
    Requires a known size (and matching sha256 when *verify*); otherwise the
    file is re-downloaded.
    """
    if not final_path.is_file() or entry.size is None:
        return False
    if final_path.stat().st_size != entry.size:
        return False
    if verify and entry.sha256:
        return file_sha256(final_path) == entry.sha256
    return True


def ensure_within_directory(dest_dir: Path, candidate_parent: Path) -> None:
    """Require *candidate_parent* (resolved) to stay under *dest_dir* (resolved).

    Lexical name validation cannot see the filesystem: a pre-existing symlink
    inside the destination (``dest/link -> /tmp/outside``) turns the validated
    name ``link/owned.bin`` into a write outside the tree. Resolving both
    sides closes that hole; combined with O_NOFOLLOW on the final open this
    leaves only a narrow, accepted check-to-use window.

    Raises:
        ValueError: When the resolved parent escapes the destination.
    """
    resolved_dest = dest_dir.resolve()
    resolved_parent = candidate_parent.resolve()
    if not (resolved_parent == resolved_dest or resolved_parent.is_relative_to(resolved_dest)):
        raise ValueError(
            f"descriptor file resolves outside the destination directory: "
            f"{candidate_parent} -> {resolved_parent}"
        )
