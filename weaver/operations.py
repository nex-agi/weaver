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

"""Helpers for Weaver long-running operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ._http import APIClient, WeaverAPIError, backoff_delays
from ._utils import extract_id, lookup_case_insensitive


class WeaverOperationError(RuntimeError):
    def __init__(self, payload: Dict[str, Any]):
        message = lookup_case_insensitive(payload, "error") or "operation_failed"
        super().__init__(f"Operation failed: {message}")
        self.payload = payload


@dataclass
class OperationHandle:
    client: APIClient
    operation_id: str
    _cached: Dict[str, Any]

    @classmethod
    def from_payload(cls, client: APIClient, payload: Dict[str, Any]) -> "OperationHandle":
        op_id = extract_id(payload)
        return cls(client=client, operation_id=str(op_id), _cached=payload)

    @property
    def status(self) -> Optional[str]:
        status = lookup_case_insensitive(self._cached, "status")
        return str(status).lower() if status is not None else None

    @property
    def response(self) -> Any:
        return lookup_case_insensitive(self._cached, "response")

    @property
    def error(self) -> Optional[str]:
        err = lookup_case_insensitive(self._cached, "error")
        return str(err) if err else None

    def done(self) -> bool:
        status = self.status
        return status in {"done", "error"}

    def refresh(self) -> Dict[str, Any]:
        path = f"/api/v1/operations/{self.operation_id}"
        self._cached = self.client.get(path)
        return self._cached

    def wait(self) -> Dict[str, Any]:
        if self.done():
            return self._cached
        for delay in backoff_delays():
            time.sleep(delay)
            self.refresh()
            if self.done():
                break
        if not self.done():
            raise WeaverAPIError(504, "timeout", "Operation polling timed out", True)
        if self.status == "error":
            raise WeaverOperationError(self._cached)
        return self._cached

    def result(self) -> Any:
        payload = self.wait()
        return lookup_case_insensitive(payload, "response")


def build_operation_handle(client: APIClient, payload: Dict[str, Any]) -> OperationHandle:
    return OperationHandle.from_payload(client, payload)
