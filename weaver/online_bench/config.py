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

"""Resolve the fixed per-run benchmark configuration before training starts."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OnlineBenchSuite(BaseModel):
    """Named, image-owned Harbor preset; not a path on the SDK host."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    harbor_config: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\.yaml$")


class OnlineBenchConfig(BaseModel):
    """Immutable run policy, separate from metrics and GPU model registration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    enabled: bool = False
    every_n_steps: int = Field(default=100, ge=1)
    timeout_seconds: int = Field(default=3600, ge=1, le=86400)
    results_path: str | None = None
    suites: tuple[OnlineBenchSuite, ...] = Field(default=(), max_length=16)

    @model_validator(mode="before")
    @classmethod
    def normalize_suites(cls, value: Any) -> Any:
        """Accept YAML/JSON arrays while keeping the resolved policy immutable."""
        if isinstance(value, Mapping):
            value = dict(value)
            if isinstance(value.get("suites"), list):
                value["suites"] = tuple(value["suites"])
        return value

    @model_validator(mode="after")
    def validate_suites(self) -> OnlineBenchConfig:
        """Reject ambiguous suite identity and enabled runs without any tasks."""
        names = [suite.name for suite in self.suites]
        if len(set(names)) != len(names):
            raise ValueError("online-bench suite names must be unique")
        if self.enabled and not self.suites:
            raise ValueError("enabled online-bench requires at least one suite")
        return self

    def server_payload(self) -> dict[str, Any]:
        """Return public policy only; the driver-local output path stays local."""
        return self.model_dump(mode="json", exclude={"results_path"})


ConfigSource = OnlineBenchConfig | Mapping[str, Any] | str | Path | None


def resolve_config(source: ConfigSource = None) -> OnlineBenchConfig:
    """Load defaults plus exactly one explicit or environment-selected override.

    A YAML file or mapping contains an ``online_bench`` section. Explicit input
    takes precedence over WEAVER_ONLINE_BENCH_CONFIG_PATH. Lists replace defaults.
    The result path is resolved once against the current working directory.
    """
    defaults = yaml.safe_load(files(__package__).joinpath("defaults.yaml").read_text())
    if source is None:
        source = os.environ.get("WEAVER_ONLINE_BENCH_CONFIG_PATH") or None
    if isinstance(source, OnlineBenchConfig):
        config = source
    else:
        if isinstance(source, (str, Path)):
            source = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        if source is not None:
            if not isinstance(source, Mapping) or set(source) != {"online_bench"}:
                raise ValueError("expected a mapping containing only online_bench")
            override = source["online_bench"]
            if not isinstance(override, Mapping):
                raise ValueError("online_bench must be a mapping")
            defaults["online_bench"].update(override)
        config = OnlineBenchConfig.model_validate(defaults["online_bench"])
    path = Path(config.results_path or "./weaver/.logs").expanduser().resolve()
    return config.model_copy(update={"results_path": str(path)})
