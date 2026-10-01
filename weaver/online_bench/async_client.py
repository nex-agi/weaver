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

"""Asyncio-native boundary hook; benchmark execution stays server-side."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

from ._state import BenchState
from .config import ConfigSource, resolve_config

if TYPE_CHECKING:
    from ..async_training_client import AsyncTrainingClient


class AsyncOnlineBench:
    """One hook per training run; call after every completed logical optimizer step.

    Do not prequeue training operations beyond a trigger boundary. This object
    creates no background loop/task and must be used by one step producer.
    Call ``finish`` before closing the parent service/session.
    """

    def __init__(self, training: AsyncTrainingClient, state: BenchState) -> None:
        self._training = training
        self._http = training._service.http
        self._state = state
        self._poll_interval = 2.0
        self._guard = threading.Lock()
        self._owner_thread = threading.get_ident()
        self._configuring = True
        self._owner_task = asyncio.current_task()

    def _assert_producer(self) -> None:
        if (
            threading.get_ident() != self._owner_thread
            or asyncio.current_task() is not self._owner_task
        ):
            raise RuntimeError("online-bench supports only its configuring logical step producer")

    def _assert_mutation_allowed(self) -> None:
        self._assert_producer()
        self._state.check_live()
        if self._configuring or self._guard.locked():
            raise RuntimeError("training mutation rejected while online-bench holds the boundary")

    @classmethod
    async def configure(
        cls, training: AsyncTrainingClient, source: ConfigSource = None
    ) -> AsyncOnlineBench:
        """Resolve policy once and wait for server-owned resource preparation."""
        config = await asyncio.to_thread(resolve_config, source)
        self = cls(training, BenchState(training.model_id, config))
        if config.enabled:
            guard = self._assert_mutation_allowed
            if (
                training._service._online_bench_guards.setdefault(training.model_id, guard)
                is not guard
            ):
                raise RuntimeError("online-bench is already configured for this training run")
            try:
                response = await self._http.post(
                    self._state.path, json=config.server_payload(), max_retries=1
                )
                # Provisioning may queue much longer than one HTTP request. The
                # server returns a small preparing record, not a held connection.
                while response.get("status") == "preparing":
                    await asyncio.sleep(self._poll_interval)
                    response = await self._http.get(self._state.path)
                self._state.configure(response)
                await asyncio.to_thread(self._state.save_config)
                self._configuring = False
            except BaseException as exc:
                self._state.failure = exc
                raise
        return self

    async def after_step(self, *, completed_step: int) -> list[dict[str, Any]]:
        """Observe progress; at cadence boundaries drain, export and await sync-ready.

        Returns newly collected results, attributed to their source steps. Valid
        low scores do not close the gate. Execution/transport errors propagate;
        subsequent calls cannot silently advance a failed hook.
        """
        if self._state.config.enabled:
            self._assert_producer()
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("online-bench supports only one logical step producer")
        try:
            if not self._state.config.enabled:
                return []
            trigger = self._state.check_step(completed_step)
            if trigger:
                await self._http.post(
                    self._state.path + "/boundary",
                    json={"completed_step": completed_step},
                    max_retries=1,
                )
            results = await self._observe(wait=trigger)
            if trigger:
                self._state.deadline = time.monotonic() + self._state.config.timeout_seconds
                response = await self._http.post(
                    self._state.path + "/evaluations",
                    json={
                        "completed_step": completed_step,
                        "seq_id": self._training._service.next_operation_seq(self._state.model_id),
                    },
                    max_retries=1,
                )
                self._state.status(response, submitted_step=completed_step)
                while True:
                    await self._check_status()
                    if self._state.active is not None and self._state.active.sync_ready:
                        break
                    await asyncio.sleep(self._poll_interval)
                    await self._poll()
            return results
        except BaseException as exc:
            self._state.failure = exc
            raise
        finally:
            self._guard.release()

    async def finish(self) -> list[dict[str, Any]]:
        """Collect the outstanding evaluation, with no final off-cadence trigger."""
        if self._state.config.enabled:
            self._assert_producer()
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("online-bench supports only one logical step producer")
        try:
            if self._state.finished or not self._state.config.enabled:
                return []
            self._state.check_live()
            results = await self._observe(wait=True)
            await self._http.post(self._state.path + "/finish", json={}, max_retries=1)
            self._state.finished = True
            return results
        except BaseException as exc:
            self._state.failure = exc
            raise
        finally:
            self._guard.release()

    async def _check_status(self) -> None:
        try:
            self._state.check_status()
        except (RuntimeError, TimeoutError):
            if self._state.active is not None:
                result = self._state.active.model_dump(mode="json")
                if result["status"] not in {"failed", "cancelled"}:
                    result.update(status="failed", error="online-bench deadline exceeded")
                await asyncio.to_thread(self._state.save_result, result)
            raise

    async def _poll(self) -> None:
        if self._state.active is not None:
            response = await self._http.get(
                self._state.path + "/evaluations/" + self._state.active.evaluation_id
            )
            self._state.status(response)

    async def _observe(self, *, wait: bool) -> list[dict[str, Any]]:
        if self._state.active is None:
            return []
        await self._poll()
        while True:
            active = self._state.active
            await self._check_status()
            if active is not None and active.status == "completed":
                response = await self._http.get(
                    self._state.path + "/evaluations/" + active.evaluation_id + "/result"
                )
                result = self._state.result(response)
                await asyncio.to_thread(self._state.save_result, result)
                self._state.active = None
                return [result]
            if not wait:
                return []
            await asyncio.sleep(self._poll_interval)
            await self._poll()
