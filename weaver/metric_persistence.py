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

"""Client-owned local JSONL storage and optional W&B publishing of v1 metrics."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from ._wandb_metrics import WandbMetricView
from .types.metrics import MetricObservations

logger = logging.getLogger(__name__)
DEFAULT_METRICS_PATH = Path("weaver/.logs")


def _safe_component(value: str) -> str:
    # Percent encoding is reversible: decoder/1 and decoder_1 must not collide.
    if value in {"", ".", ".."}:
        return {"": "%00", ".": "%2E", "..": "%2E%2E"}[value]
    return quote(value, safe="-_")


def _parse_wandb_link(link: str) -> tuple[str, str, str, str]:
    """Return API host, entity, project and run ID, including self-hosted URLs."""
    parsed = urlparse(link.strip())
    if parsed.scheme in {"http", "https"}:
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("wandb_link must be a run URL without credentials")
        host = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.hostname in {"wandb.ai", "www.wandb.ai"}:
            host = "https://api.wandb.ai"
        parts = parsed.path.strip("/").split("/")
    elif parsed.scheme == "wandb":
        host = os.getenv("WANDB_BASE_URL") or os.getenv("WANDB_HOST") or "https://api.wandb.ai"
        parts = [parsed.netloc, *parsed.path.strip("/").split("/")]
    elif not parsed.scheme:
        host = os.getenv("WANDB_BASE_URL") or os.getenv("WANDB_HOST") or "https://api.wandb.ai"
        parts = link.strip().strip("/").split("/")
    else:
        raise ValueError("wandb_link must be an HTTP(S) run URL or entity/project/run-id")
    if len(parts) == 4 and parts[2] == "runs":
        parts.pop(2)
    if len(parts) != 3 or any(not p or p in {".", ".."} for p in parts):
        raise ValueError("wandb_link must identify <entity>/<project>/runs/<run-id>")
    return host.rstrip("/"), parts[0], parts[1], parts[2]


class MetricPersistence:
    """Save every received v1 observation locally; optionally also publish to W&B.

    ``local_path=None`` selects ``./weaver/.logs`` relative to the working
    directory at construction, not the installed Python package. Files are
    created lazily at ``<parent>/metrics-<model-id>/<metric>/observations.jsonl``.
    Local files retain all values, labels and unavailable statuses.

    A matching active W&B run is borrowed, never finished or auto-advanced:
    its owner commits each training step. Repeated keys in that step take the
    latest value in W&B; JSONL retains every observation. A standalone SDK-owned
    run commits each operation and is finished by ``close``. Concurrent writers
    to the same run in different processes require application coordination.
    """

    def __init__(
        self,
        *,
        local_path: str | os.PathLike[str] | None = None,
        wandb_link: str | None = None,
    ) -> None:
        self.local_path = Path(local_path if local_path is not None else DEFAULT_METRICS_PATH)
        self.local_path = self.local_path.expanduser().absolute()
        self.wandb_link = wandb_link
        self._wandb_target = _parse_wandb_link(wandb_link) if wandb_link is not None else None
        self._lock = threading.Lock()
        self._persisted_operations: set[str] = set()
        self._wandb_run: Any = None
        self._owns_wandb_run = False
        self._wandb_view: WandbMetricView | None = None

    def persist(self, operation_id: str, response: Any) -> None:
        """Persist one completion; failures are visible without changing its result."""
        observations = MetricObservations.from_response(response)
        if observations is None:
            return
        with self._lock:
            if operation_id in self._persisted_operations:
                return
            # Independent sinks: a disk failure must not prevent W&B publishing,
            # nor a W&B failure prevent the local record from being written.
            try:
                self._persist_local(operation_id, observations)
            except Exception:
                logger.exception(
                    "Local metrics write failed for %s in %s", operation_id, self.local_path
                )
            if self._wandb_target is not None:
                try:
                    self._persist_wandb(operation_id, observations)
                except Exception:
                    logger.exception("W&B metrics publish failed for operation %s", operation_id)
            # At most one attempt per completion in this client. Do not duplicate
            # append-only rows following a partial sink failure.
            self._persisted_operations.add(operation_id)

    def _persist_local(self, operation_id: str, observations: MetricObservations) -> None:
        root = self.local_path / f"metrics-{_safe_component(observations.model_id)}"
        records: dict[Path, list[str]] = defaultdict(list)
        for point in observations.points:
            metric_dir = root.joinpath(*(_safe_component(part) for part in point.name.split("/")))
            record: dict[str, Any] = {
                "operation_id": operation_id,
                "model_id": observations.model_id,
                "step": observations.attempt,
                "trainer_run_id": observations.epoch,
                "counter_scope": observations.counter_scope,
                "name": point.name,
                "status": point.status,
                "labels": dict(point.labels),
                "reason": point.reason,
            }
            if point.status == "ok":
                record["value"] = point.value
            records[metric_dir].append(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
        # Open once per metric, not once per layer/expert scalar (important on GPFS).
        for metric_dir, lines in records.items():
            metric_dir.mkdir(parents=True, exist_ok=True)
            with (metric_dir / "observations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")

    def _ensure_wandb_run(self) -> Any:
        if self._wandb_run is not None:
            return self._wandb_run
        import wandb  # Optional dependency; never imported with wandb_link=None.

        assert self._wandb_target is not None
        host, entity, project, run_id = self._wandb_target
        active = wandb.run
        if active is not None:
            active_host = active.settings.base_url.rstrip("/")
            if (active_host, active.entity, active.project, active.id) != self._wandb_target:
                raise ValueError(
                    "wandb_link differs from the active run; refusing to log to another run"
                )
            self._wandb_run = active
        else:
            # Match NexRL's WANDB_HOST/WANDB_KEY while also accepting standard
            # W&B credentials. A URL locates the run; it is not a credential.
            if os.getenv("WANDB_MODE") not in {"offline", "disabled", "dryrun"}:
                wandb.login(
                    host=host,
                    key=os.getenv("WANDB_API_KEY") or os.getenv("WANDB_KEY"),
                    timeout=30,
                )
            self._wandb_run = wandb.init(
                entity=entity,
                project=project,
                id=run_id,
                resume="allow",
                # Only initialize when no run is active. The boolean form also
                # supports W&B 0.19.8; newer string modes are unnecessary here.
                reinit=False,
                settings=wandb.Settings(base_url=host, init_timeout=30),
                dir=str(self.local_path),
            )
            self._owns_wandb_run = True
        if getattr(self._wandb_run.settings, "mode", "online") != "online":
            logger.warning("W&B is not in online mode; metrics will not update live in the UI")
        return self._wandb_run

    def _persist_wandb(self, operation_id: str, observations: MetricObservations) -> None:
        if not observations.points:
            return
        run = self._ensure_wandb_run()
        if self._wandb_view is None:
            self._wandb_view = WandbMetricView(run)
        # NexRL supplies explicit step=N itself. Advancing its implicit W&B
        # history here would cause subsequent NexRL log(step=N) calls to drop.
        self._wandb_view.log(operation_id, observations, commit=self._owns_wandb_run)

    def close(self) -> None:
        """Flush only an SDK-owned W&B run; never finish a caller/NexRL run."""
        with self._lock:
            if self._owns_wandb_run and self._wandb_run is not None:
                try:
                    self._wandb_run.finish()
                except Exception:
                    logger.exception("Could not finish the SDK-owned W&B run")
                finally:
                    self._wandb_run = None
                    self._owns_wandb_run = False
