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
        """Resolve policy once; operational failures are loud monitoring failures."""
        config = await asyncio.to_thread(resolve_config, source)
        self = cls(training, BenchState(training.model_id, config))
        if not config.enabled:
            return self
        guard = self._assert_mutation_allowed
        if training._service._online_bench_guards.setdefault(training.model_id, guard) is not guard:
            raise RuntimeError("online-bench is already configured for this training run")
        try:
            await asyncio.to_thread(self._state.save_config)
        except FileExistsError:
            raise
        except Exception as exc:
            await asyncio.to_thread(self._state.report_failure, 0, "save_config", exc)
        try:
            response = await self._http.post(
                self._state.path, json=config.server_payload(), max_retries=1
            )
            while response.get("status") == "preparing":
                await asyncio.sleep(self._poll_interval)
                response = await self._http.get(self._state.path)
            self._state.configure(response)
        except Exception as exc:
            await self._abandon(0)
            await asyncio.to_thread(self._state.report_failure, 0, "configure", exc)
        finally:
            self._configuring = False
        return self

    async def after_step(self, *, completed_step: int) -> list[dict[str, Any]]:
        """Collect progress and trigger scheduled rounds without failing training.

        A failed round is returned/stored as a failure, never a zero score. Later
        scheduled rounds remain enabled. Producer/cadence misuse still raises.
        """
        if self._state.config.enabled:
            self._assert_producer()
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("online-bench supports only one logical step producer")
        try:
            if not self._state.config.enabled:
                return []
            trigger = self._state.check_step(completed_step)
            return await self._after_step(completed_step, trigger)
        finally:
            self._guard.release()

    async def _after_step(self, completed_step: int, trigger: bool) -> list[dict[str, Any]]:
        started = time.monotonic()
        results: list[dict[str, Any]] = []
        try:
            results.extend(await self._observe(wait=trigger))
            observed = time.monotonic()
            if trigger:
                if not self._state.digest:
                    self._state.configure(await self._http.get(self._state.path))
                self._state.deadline = time.monotonic() + self._state.config.round_timeout_seconds
                response = await self._http.post(
                    self._state.path + "/evaluations",
                    json={"completed_step": completed_step},
                    max_retries=1,
                )
                self._state.status(response, submitted_step=completed_step)
                while self._state.active is not None:
                    terminal = await self._collect_terminal()
                    if terminal is not None:
                        results.append(terminal)
                        break
                    if self._state.active.sync_ready:
                        break
                    await asyncio.sleep(self._poll_interval)
                    await self._poll()
            timing = dict(
                observe_or_boundary_wait_seconds=observed - started,
                sync_wait_seconds=time.monotonic() - observed if trigger else 0.0,
                hook_seconds=time.monotonic() - started,
            )
            await asyncio.to_thread(
                self._state.persist_timing, completed_step, "after_step", timing
            )
        except Exception as exc:
            await self._abandon(completed_step)
            results.append(
                await asyncio.to_thread(
                    self._state.report_failure,
                    completed_step,
                    "after_step",
                    exc,
                    self._state.active,
                )
            )
            self._state.active = None
        return results

    async def finish(self) -> list[dict[str, Any]]:
        """Drain and clean up; monitoring/cleanup errors do not fail training."""
        if self._state.config.enabled:
            self._assert_producer()
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("online-bench supports only one logical step producer")
        results: list[dict[str, Any]] = []
        try:
            if self._state.finished or not self._state.config.enabled:
                return results
            started = time.monotonic()
            try:
                results.extend(await self._observe(wait=True))
            except Exception as exc:
                await self._abandon(self._state.last_step)
                results.append(
                    await asyncio.to_thread(
                        self._state.report_failure,
                        self._state.last_step,
                        "collect",
                        exc,
                        self._state.active,
                    )
                )
            try:
                await self._http.post(self._state.path + "/finish", json={}, max_retries=1)
                self._state.finished = True
                timing = dict(final_drain_seconds=time.monotonic() - started)
                await asyncio.to_thread(
                    self._state.persist_timing, self._state.last_step, "finish", timing
                )
            except Exception as exc:
                results.append(
                    await asyncio.to_thread(
                        self._state.report_failure, self._state.last_step, "finish", exc
                    )
                )
            return results
        finally:
            self._guard.release()

    async def _poll(self) -> None:
        if self._state.active is not None:
            response = await self._http.get(
                self._state.path + "/evaluations/" + self._state.active.evaluation_id
            )
            self._state.status(response)

    async def _collect_terminal(self) -> dict[str, Any] | None:
        active = self._state.active
        if active is None:
            return None
        terminal = active.status in {"completed", "failed", "cancelled"}
        if not terminal and time.monotonic() < self._state.deadline:
            return None
        if terminal and active.status != "completed" and not active.sync_ready:
            await self._abandon(active.completed_step)
        result = active.model_dump(mode="json")
        if terminal:
            try:
                response = await self._http.get(
                    self._state.path + "/evaluations/" + active.evaluation_id + "/result"
                )
                result = self._state.result(response)
            except Exception as exc:
                result.update(status="failed", collection_error=type(exc).__name__)
                result["error"] = result.get("error") or "online-bench result collection failed"
        else:
            await self._abandon(active.completed_step)
            result.update(status="failed", error="online-bench deadline exceeded")
        await asyncio.to_thread(self._state.persist_result, result)
        # The server retains the slot until old consumers are drained/fenced.
        # An SDK timeout alone is never permission to load the next weights.
        self._state.active = None
        return result

    async def _observe(self, *, wait: bool) -> list[dict[str, Any]]:
        if self._state.active is None:
            return []
        await self._poll()
        while True:
            result = await self._collect_terminal()
            if result is not None:
                return [result]
            if not wait:
                return []
            await asyncio.sleep(self._poll_interval)
            await self._poll()

    async def _abandon(self, step: int) -> None:
        # A monitoring HTTP/parsing error must also release the server-side gate.
        # The slot remains fenced until cancellation/drain, so this is not replay.
        if self._state.active is not None:
            step = self._state.active.completed_step
        try:
            await self._http.post(
                self._state.path + "/abandon", json={"completed_step": step}, max_retries=1
            )
        except Exception as exc:
            await asyncio.to_thread(self._state.report_failure, step, "release_training_gate", exc)
