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

"""Shared protocol validation and state transitions for both client stacks."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .._http import WeaverAPIError
from .config import OnlineBenchConfig


class EvaluationStatus(BaseModel):
    """Small status response; large case results are fetched only at completion."""

    model_config = ConfigDict(extra="ignore", strict=True)
    evaluation_id: str
    model_id: str
    completed_step: int = Field(ge=1)
    config_digest: str
    status: Literal["preparing", "running", "completed", "failed", "cancelled"]
    sync_ready: bool = False
    target_id: str = ""
    weight_version: str = ""
    error: str = ""


class BenchState:
    """No HTTP or event-loop ownership; one logical step producer per instance."""

    def __init__(self, model_id: str, config: OnlineBenchConfig) -> None:
        # IDs are also used in paths and URLs, never trust a server-supplied path.
        UUID(model_id)
        self.model_id = model_id
        self.config = config
        self.path = f"/api/v1/models/{model_id}/online-bench"
        self.directory = Path(config.results_path or ".") / f"online-bench-{model_id}"
        self.digest = ""
        self.target_id = ""
        self.active: EvaluationStatus | None = None
        self.deadline = 0.0
        self.last_step = 0
        self.finished = False
        self.failure: BaseException | None = None
        self.observed: set[str] = set()

    def configure(self, response: Mapping[str, Any]) -> None:
        """Require an allocation/preflight acknowledgement, not mere acceptance."""
        if response.get("model_id") != self.model_id or response.get("status") != "ready":
            raise ValueError("online-bench configuration is not ready for this training run")
        self.digest = str(response.get("config_digest", ""))
        self.target_id = str(response.get("target_id", ""))
        UUID(self.target_id)
        if not self.digest:
            raise ValueError("online-bench configuration digest is missing")

    def check_step(self, step: int) -> bool:
        """Check monotonicity and prevent skipping a required trigger boundary."""
        self.check_live()
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step < self.last_step
            or step <= 0
        ):
            raise ValueError("completed_step must be a nondecreasing positive integer")
        if step == self.last_step:
            return False
        interval = self.config.every_n_steps
        next_boundary = (self.last_step // interval + 1) * interval
        if step > next_boundary:
            raise ValueError(f"missed online-bench boundary at completed step {next_boundary}")
        self.last_step = step
        return step % interval == 0

    def check_live(self) -> None:
        """A failed/cancelled hook must not silently reopen the training gate."""
        if self.failure is not None:
            raise RuntimeError(
                "online-bench hook previously failed; training must not advance"
            ) from self.failure
        if self.finished:
            raise RuntimeError("online-bench hook is already finished")

    def status(self, payload: Any, *, submitted_step: int | None = None) -> EvaluationStatus:
        """Bind every observation to the same source step, target and configuration."""
        result = EvaluationStatus.model_validate(payload)
        UUID(result.evaluation_id)
        if result.model_id != self.model_id or result.config_digest != self.digest:
            raise ValueError("online-bench response belongs to another run/configuration")
        expected_step = (
            submitted_step
            if submitted_step is not None
            else self.active.completed_step if self.active else None
        )
        if result.completed_step != expected_step:
            raise ValueError("online-bench response has the wrong source step")
        if self.active:
            if result.evaluation_id != self.active.evaluation_id:
                raise ValueError("online-bench polling returned another evaluation")
            if self.active.sync_ready and (
                not result.sync_ready or result.weight_version != self.active.weight_version
            ):
                raise ValueError("online-bench weight version changed after sync-ready")
        if result.sync_ready:
            if result.target_id != self.target_id or not result.weight_version:
                raise ValueError(
                    "sync-ready response lacks the configured target/exact weight version"
                )
        if result.status in {"running", "completed"} and not result.sync_ready:
            raise ValueError("online-bench execution preceded weight readiness")
        self.active = result
        return result

    def check_status(self) -> None:
        """Execution failures and expired deadlines are errors, never quality scores."""
        if self.active is None:
            return
        if self.active.status in {"failed", "cancelled"}:
            raise WeaverAPIError(
                500,
                "online_bench_failed",
                self.active.error or self.active.status,
                False,
                details={
                    "evaluation_id": self.active.evaluation_id,
                    "completed_step": self.active.completed_step,
                },
            )
        if self.active.status != "completed" and time.monotonic() >= self.deadline:
            raise TimeoutError("online-bench whole-evaluation deadline exceeded")

    def result(self, payload: Any) -> dict[str, Any]:
        """Validate terminal attribution and complete suite results before saving."""
        if not isinstance(payload, dict) or self.active is None:
            raise ValueError("missing online-bench result")
        for key in (
            "evaluation_id",
            "model_id",
            "completed_step",
            "config_digest",
            "target_id",
            "weight_version",
        ):
            if payload.get(key) != getattr(self.active, key):
                raise ValueError(f"online-bench result attribution mismatch: {key}")
        if payload.get("status") != "completed":
            raise ValueError("online-bench result is not completed")
        suites = payload.get("suites")
        if not isinstance(suites, list) or len(suites) != len(self.config.suites):
            raise ValueError("online-bench result has missing/extra suites")
        names = [suite.get("name") for suite in suites if isinstance(suite, dict)]
        if len(set(names)) != len(suites) or set(names) != {s.name for s in self.config.suites}:
            raise ValueError("online-bench result has duplicate/unknown suites")
        for suite in suites:
            cases = suite.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError("online-bench suite has no case results")
            if type(suite.get("expected_cases")) is not int or suite["expected_cases"] != len(
                cases
            ):
                raise ValueError("online-bench suite case count is incomplete")
            identities = set()
            for case in cases:
                if not isinstance(case, dict) or case.get("status") != "completed":
                    raise ValueError("online-bench case execution did not complete")
                name, ref, score = case.get("name"), case.get("ref"), case.get("score")
                if not isinstance(name, str) or not name or not isinstance(ref, str) or not ref:
                    raise ValueError("online-bench case lacks task identity")
                attempt = case.get("attempt")
                if (
                    type(attempt) is not int
                    or attempt < 1
                    or (name, attempt) in identities
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not math.isfinite(score)
                ):
                    raise ValueError("online-bench case has duplicate identity or invalid score")
                identities.add((name, attempt))
        return payload

    def save_config(self) -> None:
        """Create the ordinary caller-owned output directory at configuration time."""
        policy = self.config.server_payload()
        snapshots = {}
        for suite in policy["suites"]:
            filename = "suite-" + suite["name"] + ".yaml"
            snapshots[filename] = yaml.safe_dump(suite["harbor_config"], sort_keys=False)
            suite["harbor_config"] = filename
        snapshots["config.yaml"] = yaml.safe_dump({"online_bench": policy}, sort_keys=False)
        # Reserve the run directory before publishing anything. A second SDK
        # instance must never overwrite an active run's frozen config, even when
        # its subsequent server configure request would be rejected.
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.directory.mkdir(mode=0o700, exist_ok=False)
        for filename, contents in snapshots.items():
            with (self.directory / filename).open("x", encoding="utf-8") as stream:
                stream.write(contents)

    def save_timing(self, step: int, phase: str, timings: dict[str, float]) -> None:
        """Record driver blocking time separately from asynchronous benchmark duration."""
        record = dict(completed_step=step, phase=phase, timings=timings)
        with (self.directory / "timings.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def save_result(self, result: dict[str, Any]) -> None:
        """Save normal JSON and JSONL with in-process repeated-observation dedup."""
        evaluation_id = str(result["evaluation_id"])
        UUID(evaluation_id)
        if evaluation_id in self.observed:
            return
        step = result["completed_step"]
        directory = self.directory / f"step-{step:06d}-{evaluation_id}"
        directory.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        (directory / "result.json").write_text(encoded + "\n", encoding="utf-8")
        with (self.directory / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        self.observed.add(evaluation_id)
