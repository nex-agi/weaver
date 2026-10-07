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
