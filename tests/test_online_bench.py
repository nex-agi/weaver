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

"""The online-bench boundary is independent of training math and GPU execution."""

import asyncio
import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest

from weaver._async_http import AsyncAPIClient
from weaver.async_training_client import AsyncTrainingClient
from weaver.config import WeaverConfig
from weaver.online_bench.async_client import AsyncOnlineBench
from weaver.online_bench.client import OnlineBench
from weaver.online_bench.config import resolve_config
from weaver.training_client import TrainingClient


class Backend:
    def __init__(self):
        self.model_id = str(uuid4())
        self.target_id = str(uuid4())
        self.status = None
        self.submissions = []
        self.polls = 0
        self.terminal = False
        self.ready = True
        self.posts = []
        self.boundaries = []
        self.requests = []

    def post(self, path, *, json, max_retries=1):
        assert max_retries == 1
        self.posts.append(path)
        self.requests.append(("POST", path, copy.deepcopy(json)))
        if path.endswith("/boundary"):
            self.boundaries.append(json["completed_step"])
            return {}
        if path.endswith("/finish"):
            return {}
        if path.endswith("/evaluations"):
            self.submissions.append(json)
            self.terminal = False
            self.status = dict(
                evaluation_id=str(uuid4()),
                model_id=self.model_id,
                completed_step=json["completed_step"],
                config_digest="digest",
                status="running" if self.ready else "preparing",
                sync_ready=self.ready,
                target_id=self.target_id,
                weight_version=f"v{json['completed_step']}",
            )
            return copy.deepcopy(self.status)
        return dict(
            model_id=self.model_id, target_id=self.target_id, status="ready", config_digest="digest"
        )

    def get(self, path):
        self.requests.append(("GET", path, None))
        self.polls += 1
        result = copy.deepcopy(self.status)
        if self.terminal:
            result["status"] = "completed"
        if path.endswith("/result"):
            result["suites"] = [
                dict(
                    name="probe",
                    expected_cases=1,
                    cases=[
                        dict(name="task", ref="sha256:abc", score=0, status="completed", attempt=1)
                    ],
                )
            ]
        return result


def config(tmp_path, **overrides):
    return {
        "online_bench": dict(
            enabled=True,
            results_path=str(tmp_path),
            every_n_steps=100,
            suites=[
                dict(name="probe", harbor_config={"tasks": [{"name": "task", "ref": "sha256:abc"}]})
            ],
            **overrides,
        )
    }


def training(backend, asynchronous=False):
    cls = AsyncTrainingClient if asynchronous else TrainingClient
    service = SimpleNamespace(
        http=backend,
        next_operation_seq=Mock(side_effect=range(1, 100)),
        _online_bench_guards={},
    )
    return cls(
        service=service, model_id=backend.model_id, base_model="test", session_id=str(uuid4())
    )


def test_config_precedence_replacement_and_frozen_path(tmp_path, monkeypatch):
    file = tmp_path / "config.yaml"
    file.write_text("online_bench:\n  enabled: false\n  every_n_steps: 7\n")
    monkeypatch.setenv("WEAVER_ONLINE_BENCH_CONFIG_PATH", str(file))
    assert resolve_config().every_n_steps == 7
    explicit = resolve_config(config(tmp_path))
    assert explicit.every_n_steps == 100
    assert explicit.results_path == str(tmp_path)
    with pytest.raises(ValueError):
        resolve_config({"online_bench": {"unknown": 1}})
    with pytest.raises(ValueError):
        resolve_config({"online_bench": {"enabled": True}})
    with pytest.raises(ValueError):
        resolve_config({"online_bench": {"every_n_steps": True}})
    with pytest.raises(ValueError):
        resolve_config(
            {"online_bench": {"suites": [{"name": "bad", "harbor_config": "../secret.yaml"}]}}
        )


def test_disabled_is_noop(tmp_path):
    backend = Backend()
    hook = training(backend).configure_online_bench(
        {"online_bench": {"results_path": str(tmp_path)}}
    )
    hook.after_step(completed_step=100)
    hook.finish()
    assert not backend.posts
    assert not list(tmp_path.iterdir())


def test_early_collection_low_score_and_finish_without_extra_trigger(tmp_path):
    backend = Backend()
    hook = training(backend).configure_online_bench(config(tmp_path))
    assert hook.after_step(completed_step=100) == []
    assert len(backend.submissions) == 1
    backend.terminal = True
    result = hook.after_step(completed_step=101)
    assert result[0]["completed_step"] == 100
    assert result[0]["suites"][0]["cases"][0]["score"] == 0
    hook.after_step(completed_step=150)
    hook.finish()
    hook.finish()
    assert len(backend.submissions) == 1
    lines = list(tmp_path.rglob("evaluations.jsonl"))[0].read_text().splitlines()
    assert len(lines) == 1


def test_next_boundary_waits_before_second_export(tmp_path):
    backend = Backend()
    at100 = threading.Event()
    completed = threading.Event()
    finish = threading.Event()
    errors = []

    def producer():
        try:
            hook = OnlineBench.configure(training(backend), config(tmp_path))
            hook._poll_interval = 0.005
            hook.after_step(completed_step=100)
            at100.set()
            hook.after_step(completed_step=200)
            completed.set()
            assert finish.wait(2)
            hook.finish()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    try:
        assert at100.wait(2)
        assert not completed.wait(0.05)
        assert len(backend.submissions) == 1
        assert backend.boundaries == [100, 200]
        backend.terminal = True
        assert completed.wait(2)
        assert len(backend.submissions) == 2
    finally:
        backend.terminal = True
        finish.set()
        thread.join(timeout=2)
    assert not errors


def test_sync_ready_is_a_separate_barrier(tmp_path):
    backend = Backend()
    backend.ready = False
    hook = OnlineBench.configure(training(backend), config(tmp_path))
    hook._poll_interval = 0.001
    original = backend.get

    def get(path):
        if backend.polls == 2:
            backend.status.update(sync_ready=True, status="running")
        return original(path)

    backend.get = get
    hook.after_step(completed_step=100)
    assert backend.polls == 3
    assert not backend.terminal


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "unknown"},
        {"status": "failed", "error": "sandbox failed"},
        {"target_id": str(uuid4())},
        {"completed_step": 99},
        {"weight_version": "other"},
        {"status": "completed", "sync_ready": False},
    ],
)
def test_fail_closed_on_bad_status(tmp_path, patch):
    backend = Backend()
    hook = OnlineBench.configure(training(backend), config(tmp_path))
    hook.after_step(completed_step=100)
    backend.status.update(patch)
    with pytest.raises((ValueError, RuntimeError)):
        hook.after_step(completed_step=101)
    with pytest.raises(RuntimeError, match="previously failed"):
        hook.after_step(completed_step=200)
    assert len(backend.submissions) == 1


def test_missing_result_and_no_submission_replay(tmp_path):
    backend = Backend()
    hook = OnlineBench.configure(training(backend), config(tmp_path))
    hook.after_step(completed_step=100)
    backend.terminal = True
    original = backend.get
    backend.get = lambda path: (
        dict(original(path), suites=[]) if path.endswith("/result") else original(path)
    )
    with pytest.raises(ValueError, match="missing/extra suites"):
        hook.after_step(completed_step=200)
    assert len(backend.submissions) == 1


def test_missed_boundary_and_deadline(tmp_path):
    backend = Backend()
    hook = OnlineBench.configure(training(backend), config(tmp_path))
    with pytest.raises(ValueError, match="missed"):
        hook.after_step(completed_step=101)
    backend = Backend()  # Failed runs are not resumed over their existing snapshots.
    hook = OnlineBench.configure(training(backend), config(tmp_path))
    hook.after_step(completed_step=100)
    hook._state.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        hook.after_step(completed_step=101)


@pytest.mark.timeout(15)
def test_async_real_socket_boundary_and_ticker(tmp_path):
    backend = Backend()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.reply(backend.post(self.path, json=body))

        def do_GET(self):
            self.reply(backend.get(self.path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def scenario():
        async with AsyncAPIClient(
            WeaverConfig(base_url=f"http://127.0.0.1:{server.server_port}", api_key="test")
        ) as http:
            train = training(backend, asynchronous=True)
            train._service.http = http
            hook = await train.configure_online_bench(config(tmp_path))
            assert isinstance(hook, AsyncOnlineBench)
            hook._poll_interval = 0.01
            await hook.after_step(completed_step=100)

            async def ticker():
                ticks = 0
                for _ in range(15):
                    await asyncio.sleep(0.005)
                    ticks += 1
                assert ticks == 15
                assert len(backend.submissions) == 1
                assert backend.boundaries == [100, 200]
                backend.terminal = True

            ticking = asyncio.create_task(ticker())
            await hook.after_step(completed_step=200)
            await ticking
            assert len(backend.submissions) == 2
            backend.terminal = True
            await hook.finish()

    try:
        asyncio.run(asyncio.wait_for(scenario(), timeout=8))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_repeat_step_and_sibling_client_guard(tmp_path):
    backend = Backend()
    train = training(backend)
    hook = train.configure_online_bench(config(tmp_path))
    hook.after_step(completed_step=100)
    hook.after_step(completed_step=100)
    assert len(backend.submissions) == 1
    original_config = (hook._state.directory / "config.yaml").read_text()
    sibling = TrainingClient(
        service=train._service,
        model_id=train.model_id,
        base_model="test",
        session_id=train.session_id,
    )
    with pytest.raises(RuntimeError, match="already configured"):
        sibling.configure_online_bench(config(tmp_path))
    assert (hook._state.directory / "config.yaml").read_text() == original_config
    with hook._guard:
        with pytest.raises(RuntimeError, match="mutation rejected"):
            sibling._next_seq()
        with pytest.raises(RuntimeError, match="mutation rejected"):
            sibling.load_state("weaver://test/checkpoint", wait=False)
    backend.status.update(status="failed", error="verifier infrastructure failure")
    with pytest.raises(RuntimeError):
        hook.after_step(completed_step=101)
    record = json.loads((hook._state.directory / "evaluations.jsonl").read_text())
    assert record["completed_step"] == 100 and record["status"] == "failed"
    assert record["evaluation_id"] == backend.status["evaluation_id"]
    with pytest.raises(RuntimeError, match="previously failed"):
        sibling._next_seq()


def test_async_failure_and_producer_guard(tmp_path):
    async def scenario():
        backend = Backend()
        train = training(backend, asynchronous=True)
        train._service.http = SimpleNamespace(
            post=AsyncMock(side_effect=backend.post), get=AsyncMock(side_effect=backend.get)
        )
        hook = await train.configure_online_bench(config(tmp_path))
        await hook.after_step(completed_step=100)
        await hook.after_step(completed_step=100)

        async def foreign_producer():
            with pytest.raises(RuntimeError, match="configuring logical"):
                await train.load_state("weaver://test/checkpoint", wait=False)

        await asyncio.create_task(foreign_producer())
        backend.status.update(status="failed", error="sandbox failed")
        with pytest.raises(RuntimeError):
            await hook.after_step(completed_step=101)
        record = json.loads((hook._state.directory / "evaluations.jsonl").read_text())
        assert record["status"] == "failed" and record["completed_step"] == 100
        with pytest.raises(RuntimeError, match="previously failed"):
            await train.load_state("weaver://test/checkpoint", wait=False)

    asyncio.run(scenario())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_boundary_precedes_drain_and_only_occurs_at_new_cadence(tmp_path, asynchronous):
    async def scenario():
        backend = Backend()
        train = training(backend, asynchronous=asynchronous)
        if asynchronous:
            train._service.http = SimpleNamespace(
                post=AsyncMock(side_effect=backend.post), get=AsyncMock(side_effect=backend.get)
            )

        async def call(method, **kwargs):
            result = method(**kwargs)
            return await result if asynchronous else result

        hook = await call(train.configure_online_bench, config=config(tmp_path))
        for step in (1, 99):
            assert await call(hook.after_step, completed_step=step) == []
        assert backend.boundaries == []
        assert await call(hook.after_step, completed_step=100) == []
        for step in (100, 101, 199):
            assert await call(hook.after_step, completed_step=step) == []
        assert backend.boundaries == [100]

        previous_id = backend.status["evaluation_id"]
        backend.terminal = True
        backend.requests.clear()
        results = await call(hook.after_step, completed_step=200)
        assert [result["completed_step"] for result in results] == [100]
        path = hook._state.path
        assert backend.requests == [
            ("POST", path + "/boundary", {"completed_step": 200}),
            ("GET", path + "/evaluations/" + previous_id, None),
            ("GET", path + "/evaluations/" + previous_id + "/result", None),
            ("POST", path + "/evaluations", {"completed_step": 200, "seq_id": 2}),
        ]
        assert backend.boundaries == [100, 200]
        assert await call(hook.after_step, completed_step=200) == []
        backend.terminal = True
        results = await call(hook.finish)
        assert [result["completed_step"] for result in results] == [200]
        assert backend.boundaries == [100, 200]

    asyncio.run(scenario())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failed_step", [100, 200])
def test_boundary_transport_failure_is_sticky(tmp_path, asynchronous, failed_step):
    async def scenario():
        backend = Backend()
        original_post = backend.post
        failure = httpx.ConnectError("boundary transport unavailable")

        def post(path, *, json, max_retries=1):
            response = original_post(path, json=json, max_retries=max_retries)
            if path.endswith("/boundary") and json["completed_step"] == failed_step:
                raise failure
            return response

        backend.post = post
        train = training(backend, asynchronous=asynchronous)
        if asynchronous:
            train._service.http = SimpleNamespace(
                post=AsyncMock(side_effect=backend.post), get=AsyncMock(side_effect=backend.get)
            )

        async def call(method, **kwargs):
            result = method(**kwargs)
            return await result if asynchronous else result

        hook = await call(train.configure_online_bench, config=config(tmp_path))
        if failed_step == 200:
            await call(hook.after_step, completed_step=100)
        backend.requests.clear()
        with pytest.raises(httpx.ConnectError, match="boundary transport unavailable"):
            await call(hook.after_step, completed_step=failed_step)
        expected = [("POST", hook._state.path + "/boundary", {"completed_step": failed_step})]
        assert backend.requests == expected
        assert len(backend.submissions) == failed_step // 100 - 1
        for step in (failed_step, failed_step + 1):
            with pytest.raises(RuntimeError, match="previously failed"):
                await call(hook.after_step, completed_step=step)
        with pytest.raises(RuntimeError, match="previously failed"):
            await call(hook.finish)
        with pytest.raises(RuntimeError, match="previously failed"):
            train._next_seq()
        assert backend.requests == expected

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "override",
    [
        {"suites": [{"name": "probe", "harbor_config": {}, "timeout_seconds": 86401}]},
        {"timeout_seconds": 3600},  # Removed whole-round setting must not shadow suite budgets.
        {
            "suites": [
                {"name": f"probe_{index}", "harbor_config": "swe-verified-smoke-v1.yaml"}
                for index in range(17)
            ]
        },
    ],
)
def test_config_limits_match_server(override):
    with pytest.raises(ValueError):
        resolve_config({"online_bench": override})
