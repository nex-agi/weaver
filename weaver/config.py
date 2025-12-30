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

"""Configuration helpers for the Weaver SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_DEFAULT_BASE_URL = "https://weaver-console.nex-agi.com"


@dataclass(slots=True)
class WeaverConfig:
    """Holds connection + auth settings for the Weaver server."""

    base_url: str = _DEFAULT_BASE_URL
    api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "WeaverConfig":
        """Load configuration from kwargs with env fallbacks.

        The api_key should be the complete API key starting with 'sk-'
        obtained from the API Keys page.
        """

        return cls(
            base_url=base_url or os.getenv("WEAVER_BASE_URL") or _DEFAULT_BASE_URL,
            api_key=api_key or os.getenv("WEAVER_API_KEY"),
        )

    def require_auth(self) -> None:
        """Raise if auth credentials are missing."""

        if not self.api_key:
            raise RuntimeError(
                "Weaver credentials missing. Provide api_key or set "
                f"WEAVER_API_KEY environment variable. Get your API key "
                f"from the Weaver at {self.base_url}/api-keys"
            )
