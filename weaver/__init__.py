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

"""Weaver SDK public exports."""

# Version management - single source of truth
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("weaver")
except Exception:
    # Fallback when package is not installed (e.g., development mode without pip install -e)
    __version__ = "0.3.0"

from . import types  # noqa: F401
from .service_client import ServiceClient  # noqa: F401

__all__ = ["ServiceClient", "types", "__version__"]
