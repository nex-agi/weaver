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

"""Public failure information for resumable direct tensor uploads."""

from pathlib import Path


def get_tensor_upload_recovery_path(error: BaseException) -> Path | None:
    """Find a tensor upload journal across exception propagation wrappers.

    Python 3.10 may replace a task's cancellation exception and retain the
    original as its context. This helper supports that behavior without
    changing cancellation semantics.

    Args:
        error: Upload failure or cancellation caught by the caller.

    Returns:
        The retained journal path, or None when no upload journal was recorded.
    """
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        recovery = getattr(current, "recovery_path", None)
        if isinstance(recovery, Path):
            return recovery
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return None


class TensorUploadInterrupted(RuntimeError):
    """A remote submission failed while its original source remains recoverable.

    Attributes:
        recovery_path: Private local journal accepted by ``resume_tensor_upload``.
        source_path: Original tensor pack; keep it until recovery or explicit abort.
        upload_id: Stable caller nonce used by the original server publication.
    """

    def __init__(self, recovery_path: Path, source_path: Path, upload_id: str) -> None:
        super().__init__(f"Tensor upload interrupted; resume from {recovery_path}")
        self.recovery_path = recovery_path
        self.source_path = source_path
        self.upload_id = upload_id
