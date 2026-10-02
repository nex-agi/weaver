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

import copy
import json
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

# Separate bounded sync/reporting work from each suite's execution budget.
SYNC_TIMEOUT_SECONDS = 1800
REPORT_TIMEOUT_SECONDS = 180


class _UniqueLoader(yaml.SafeLoader):
    """Reject accidental duplicate options instead of silently taking the last one."""


def _mapping(loader: _UniqueLoader, node: Any) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("configuration requires unique string keys")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path: Path) -> Any:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)


def _check_credentials(value: Any) -> None:
    """Keep deployment secrets out of persisted configuration and control payloads."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Harbor configuration keys must be strings")
            if key == "env" and not isinstance(child, dict):
                raise ValueError("use mapping-form env in Harbor configuration")
            sensitive = re.search(
                r"(^|_)(api_key|token|password|secret|credential|authorization|access_key_id|secret_access_key)$",
                key,
                re.I,
            )
            if (
                sensitive
                and child
                and not (isinstance(child, str) and re.fullmatch(r"\$\{[A-Z_][A-Z0-9_]*\}", child))
            ):
                raise ValueError(
                    "use deployment environment references instead of inline credentials"
                )
            _check_credentials(child)
    elif isinstance(value, str) and re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        if urlsplit(value).username is not None or urlsplit(value).password is not None:
            raise ValueError("credential-bearing URLs are not allowed in Harbor configuration")
    elif isinstance(value, list):
        for child in value:
            _check_credentials(child)


class OnlineBenchSuite(BaseModel):
    """A native Harbor child YAML beside the main config, or an inline mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    harbor_config: str | dict[str, Any]
    timeout_seconds: int = Field(default=3600, ge=1, le=86400)
    _harbor_json: str = PrivateAttr(default="")

    @field_validator("harbor_config")
    @classmethod
    def validate_child(cls, value: str | dict[str, Any]) -> str | dict[str, Any]:
        """Child references are sibling files, never paths inside the worker image."""
        if isinstance(value, str) and not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\.ya?ml", value
        ):
            raise ValueError("harbor_config must name a sibling YAML file")
        return value

    def execution_config(self) -> dict[str, Any]:
        """Return a copy of the startup snapshot; later caller/file edits cannot alter it."""
        if not self._harbor_json:
            raise ValueError("Harbor configuration has not been resolved")
        return json.loads(self._harbor_json)


class OnlineBenchConfig(BaseModel):
    """Frozen run policy; one results root for summaries and optional artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    enabled: bool = False
    every_n_steps: int = Field(default=100, ge=1)
    results_path: str | None = None
    save_artifacts: bool = False
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

    @property
    def round_timeout_seconds(self) -> int:
        """Internal watchdog, not a competing user-configured suite deadline."""
        return (
            SYNC_TIMEOUT_SECONDS
            + REPORT_TIMEOUT_SECONDS
            + sum(s.timeout_seconds for s in self.suites)
        )

    def server_payload(self) -> dict[str, Any]:
        """Pass native config contents, not client-local file references, to the worker."""
        return dict(
            enabled=self.enabled,
            every_n_steps=self.every_n_steps,
            results_path=self.results_path,
            save_artifacts=self.save_artifacts,
            suites=[
                dict(
                    name=s.name,
                    timeout_seconds=s.timeout_seconds,
                    harbor_config=s.execution_config(),
                )
                for s in self.suites
            ],
        )


ConfigSource = OnlineBenchConfig | Mapping[str, Any] | str | Path | None


def resolve_config(source: ConfigSource = None) -> OnlineBenchConfig:
    """Load defaults and one override, then freeze sibling native Harbor YAMLs.

    Explicit input wins over WEAVER_ONLINE_BENCH_CONFIG_PATH. File references
    resolve beside the main YAML (cwd for mappings). Results resolve against cwd,
    matching SDK metrics storage. Native Harbor includes are not reimplemented.
    """
    defaults = yaml.safe_load(files(__package__).joinpath("defaults.yaml").read_text())
    base = Path.cwd()
    if source is None:
        source = os.environ.get("WEAVER_ONLINE_BENCH_CONFIG_PATH") or None
    if isinstance(source, OnlineBenchConfig):
        config = source.model_copy(deep=True)
    else:
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser().resolve()
            base, source = path.parent, _read(path)
        if source is not None:
            if not isinstance(source, Mapping) or set(source) != {"online_bench"}:
                raise ValueError("expected a mapping containing only online_bench")
            override = source["online_bench"]
            if not isinstance(override, Mapping):
                raise ValueError("online_bench must be a mapping")
            defaults["online_bench"].update(copy.deepcopy(override))
        config = OnlineBenchConfig.model_validate(defaults["online_bench"])
    if config.enabled:
        for suite in config.suites:
            if suite._harbor_json:
                continue
            data = (
                _read(base / suite.harbor_config)
                if isinstance(suite.harbor_config, str)
                else suite.harbor_config
            )
            if not isinstance(data, dict) or not data:
                raise ValueError("Harbor child configuration must be a nonempty mapping")
            if "__base__" in data:
                raise ValueError(
                    "supply a complete native Harbor child YAML; resolve kit __base__ layers with its launcher first"
                )
            _check_credentials(data)
            suite._harbor_json = json.dumps(data, allow_nan=False, sort_keys=True)
    path = Path(config.results_path or "./weaver/.logs").expanduser().resolve()
    return config.model_copy(update={"results_path": str(path)})
