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

"""Sibling Harbor configuration snapshots and multi-suite SDK control tests."""

import asyncio
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from tests.test_online_bench import Backend, config, training
from weaver.online_bench.config import (
    REPORT_TIMEOUT_SECONDS,
    SYNC_TIMEOUT_SECONDS,
    resolve_config,
)


def native_config(prefix="swe", count=5):
    """Small native-shaped fixture; no evaluator, sandbox or model is launched."""
    return {
        "n_concurrent_trials": 3,
        "n_attempts": 1,
        "tasks": [
            {"name": f"{prefix}/task-{index}", "ref": f"sha256:{index:064x}"}
            for index in range(count)
        ],
        "environment": {"type": "e2b", "env": {}},
        "agents": [{"env": {"JUDGE_API_KEY": "${JUDGE_API_KEY}"}}],
    }


def write_config(directory, *, names=("swe", "terminal"), save_artifacts=True):
    directory.mkdir(parents=True, exist_ok=True)
    suites = []
    for index, name in enumerate(names):
        filename = f"{name}-input.yaml"
        (directory / filename).write_text(yaml.safe_dump(native_config(name)))
        suites.append(dict(name=name, harbor_config=filename, timeout_seconds=3600 - index * 3000))
    main = directory / "online-bench.yaml"
    main.write_text(
        yaml.safe_dump(
            {
                "online_bench": dict(
                    enabled=True,
                    every_n_steps=5,
                    results_path="output",
                    save_artifacts=save_artifacts,
                    suites=suites,
                )
            }
        )
    )
    return main


def test_children_resolve_beside_main_but_output_resolves_against_sdk_cwd(tmp_path, monkeypatch):
    main = write_config(tmp_path / "recipe")
    sdk_cwd = tmp_path / "driver"
    sdk_cwd.mkdir()
    monkeypatch.chdir(sdk_cwd)
    resolved = resolve_config(main)
    payload = resolved.server_payload()
    assert resolved.results_path == str(sdk_cwd / "output")
    assert payload["results_path"] == resolved.results_path
    assert payload["save_artifacts"] is True
    assert [suite["name"] for suite in payload["suites"]] == ["swe", "terminal"]
    assert [suite["timeout_seconds"] for suite in payload["suites"]] == [3600, 600]
    assert payload["suites"][0]["harbor_config"] == native_config("swe")
    assert payload["suites"][1]["harbor_config"] == native_config("terminal")
    assert "timeout_seconds" not in payload
    assert resolved.round_timeout_seconds == 4200 + SYNC_TIMEOUT_SECONDS + REPORT_TIMEOUT_SECONDS


def test_native_file_content_and_returned_payload_are_frozen_at_startup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main = write_config(tmp_path / "recipe")
    resolved = resolve_config(main)
    expected = resolved.server_payload()
    (main.parent / "swe-input.yaml").write_text("tasks: []\n")
    main.write_text("online_bench:\n  enabled: false\n")
    copied_payload = resolved.server_payload()
    copied_payload["suites"][0]["harbor_config"]["tasks"].clear()
    assert resolved.server_payload() == expected
    assert resolve_config(resolved).server_payload() == expected
    assert resolve_config(main).enabled is False


def test_mapping_inputs_and_exposed_nested_dicts_cannot_mutate_execution_snapshot(tmp_path):
    native = native_config()
    source = {
        "online_bench": dict(
            enabled=True,
            results_path=str(tmp_path),
            suites=[dict(name="swe", harbor_config=native)],
        )
    }
    resolved = resolve_config(source)
    expected = copy.deepcopy(resolved.server_payload())
    native["tasks"].clear()
    source["online_bench"]["suites"].clear()
    resolved.suites[0].harbor_config["tasks"].clear()
    assert resolved.server_payload() == expected
    assert resolve_config(resolved).server_payload() == expected


@pytest.mark.parametrize("save_artifacts", [False, True])
def test_saved_config_and_native_children_are_reloadable_sanitized_snapshots(
    tmp_path, monkeypatch, save_artifacts
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JUDGE_API_KEY", "test-runtime-secret-do-not-persist")
    main = write_config(
        tmp_path / "recipe", names=("config", "terminal"), save_artifacts=save_artifacts
    )
    backend = Backend()
    hook = training(backend).configure_online_bench(main)
    directory = hook._state.directory
    saved_main = directory / "config.yaml"
    snapshot = yaml.safe_load(saved_main.read_text())["online_bench"]
    assert snapshot["save_artifacts"] is save_artifacts
    assert snapshot["results_path"] == str(tmp_path / "output")
    for suite in snapshot["suites"]:
        assert suite["harbor_config"] != "config.yaml"
        assert (directory / suite["harbor_config"]).is_file()
        child = yaml.safe_load((directory / suite["harbor_config"]).read_text())
        assert child == native_config(suite["name"])
    assert resolve_config(saved_main).server_payload() == hook._state.config.server_payload()
    assert "test-runtime-secret-do-not-persist" not in "".join(
        file.read_text() for file in directory.glob("*.yaml")
    )
    assert "test-runtime-secret-do-not-persist" not in json.dumps(backend.requests)


@pytest.mark.parametrize("location", ["main", "child"])
def test_duplicate_yaml_keys_fail_before_submission(tmp_path, monkeypatch, location):
    monkeypatch.chdir(tmp_path)
    main = write_config(tmp_path / "recipe")
    target = main if location == "main" else main.parent / "swe-input.yaml"
    target.write_text(
        "online_bench:\n  enabled: true\n  enabled: false\n"
        if location == "main"
        else "n_attempts: 1\nn_attempts: 2\n"
    )
    backend = Backend()
    with pytest.raises(ValueError, match="unique string keys"):
        training(backend).configure_online_bench(main)
    assert backend.requests == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "ANTHROPIC_AUTH_TOKEN",
        "password",
        "credential",
        "GITHUB_TOKEN",
        "Authorization",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
    ],
)
def test_inline_credentials_are_rejected_without_echoing_the_value(tmp_path, key):
    source = {
        "online_bench": dict(
            enabled=True,
            results_path=str(tmp_path),
            suites=[
                dict(name="probe", harbor_config={"agents": [{"env": {key: "private-value"}}]})
            ],
        )
    }
    backend = Backend()
    with pytest.raises(ValueError, match="inline credentials") as error:
        training(backend).configure_online_bench(source)
    assert "private-value" not in str(error.value)
    assert backend.requests == []
    assert list(tmp_path.iterdir()) == []


def test_shipped_twenty_step_recipe_selects_two_suites_with_five_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main = Path(__file__).resolve().parents[1] / "examples/online-bench/online-bench.yaml"
    resolved = resolve_config(main)
    assert resolved.every_n_steps == 5
    assert resolved.save_artifacts
    assert len(resolved.suites) == 2
    assert all(len(suite.execution_config()["tasks"]) == 5 for suite in resolved.suites)
    assert all(suite.execution_config()["n_attempts"] == 1 for suite in resolved.suites)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.timeout(10)
def test_twenty_training_steps_four_rounds_two_suites_five_tasks(
    tmp_path, monkeypatch, asynchronous
):
    """Control-only simulation: old work drains before publication; no real training."""
    monkeypatch.chdir(tmp_path)
    main = write_config(tmp_path / "recipe")

    class MultiSuiteBackend(Backend):
        def __init__(self):
            super().__init__()
            self.policy = None
            self.drained = set()
            self.wait_polls = 0
            self.finishing = False

        def post(self, path, *, json, max_retries=1):
            if path.endswith("/online-bench"):
                self.policy = copy.deepcopy(json)
            if path.endswith("/evaluations"):
                if self.status is not None:
                    assert self.status["evaluation_id"] in self.drained
                self.wait_polls = 0
            return super().post(path, json=json, max_retries=max_retries)

        def get(self, path):
            if self.finishing or self.boundaries[-1] > self.status["completed_step"]:
                self.wait_polls += 1
                if self.wait_polls >= 2:
                    self.terminal = True
            result = super().get(path)
            if path.endswith("/result"):
                assert result["status"] == "completed"
                result["suites"] = [
                    dict(
                        name=suite["name"],
                        expected_cases=len(suite["harbor_config"]["tasks"]),
                        metrics={"mean": 0.0},
                        timings={"wall_seconds": 10.0},
                        cases=[
                            dict(
                                task,
                                score=0,
                                status="completed",
                                attempt=1,
                                timings={"wall_seconds": 1.0},
                            )
                            for task in suite["harbor_config"]["tasks"]
                        ],
                    )
                    for suite in self.policy["suites"]
                ]
                self.drained.add(result["evaluation_id"])
            return result

    async def scenario():
        backend = MultiSuiteBackend()
        train = training(backend, asynchronous=asynchronous)
        if asynchronous:
            train._service.http = SimpleNamespace(
                post=AsyncMock(side_effect=backend.post), get=AsyncMock(side_effect=backend.get)
            )

        async def call(method, **kwargs):
            result = method(**kwargs)
            return await result if asynchronous else result

        hook = await call(train.configure_online_bench, config=main)
        hook._poll_interval = 0.001
        completed_steps, collected = [], []
        for step in range(1, 21):
            # The ordinary training mutation guard remains open between hooks.
            train._next_seq()
            completed_steps.append(step)
            before = time.monotonic()
            collected.extend(await call(hook.after_step, completed_step=step))
            assert len(backend.submissions) == step // 5
            if step % 5 == 0:
                assert hook._state.active.completed_step == step
                assert hook._state.deadline >= before + hook._state.config.round_timeout_seconds
                assert not backend.terminal  # New round remains async after sync-ready.
            elif step > 5:
                assert hook._state.active is not None
        backend.finishing = True
        collected.extend(await call(hook.finish))
        assert completed_steps == list(range(1, 21))
        assert backend.boundaries == [5, 10, 15, 20]
        assert [item["completed_step"] for item in backend.submissions] == [5, 10, 15, 20]
        assert [item["completed_step"] for item in collected] == [5, 10, 15, 20]
        assert [item["weight_version"] for item in collected] == ["v5", "v10", "v15", "v20"]
        assert sum(len(suite["cases"]) for item in collected for suite in item["suites"]) == 40
        assert all(
            suite["expected_cases"] == len(suite["cases"]) == 5
            and suite["metrics"] == {"mean": 0.0}
            and suite["timings"] == {"wall_seconds": 10.0}
            for item in collected
            for suite in item["suites"]
        )
        assert len(list(hook._state.directory.glob("step-*/result.json"))) == 4
        assert len((hook._state.directory / "evaluations.jsonl").read_text().splitlines()) == 4
        timings = [
            json.loads(line)
            for line in (hook._state.directory / "timings.jsonl").read_text().splitlines()
        ]
        assert [item["completed_step"] for item in timings[:-1]] == list(range(1, 21))
        assert timings[-1]["phase"] == "finish"
        assert all(value >= 0 for item in timings for value in item["timings"].values())
        assert backend.posts[-1].endswith("/finish")

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


@pytest.mark.parametrize(
    "url",
    [
        "https://user:private-value@example.invalid/eval",
        "https://private-value@example.invalid/eval",
        "https://user:private%2Dvalue@example.invalid/eval",
    ],
)
def test_credential_bearing_urls_rejected_before_saving_or_transmission(tmp_path, url):
    source = config(tmp_path)
    source["online_bench"]["suites"][0]["harbor_config"]["registry"] = {"url": url}
    backend = Backend()
    with pytest.raises(ValueError, match="credential-bearing URLs") as error:
        training(backend).configure_online_bench(source)
    assert url not in str(error.value)
    assert "private-value" not in str(error.value)
    assert backend.requests == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_independent_sdk_cannot_overwrite_existing_run_snapshots(tmp_path, asynchronous):
    async def scenario():
        backend = Backend()
        first, second = training(backend, asynchronous), training(backend, asynchronous)
        if asynchronous:
            for client in (first, second):
                client._service.http = SimpleNamespace(
                    post=AsyncMock(side_effect=backend.post), get=AsyncMock(side_effect=backend.get)
                )

        async def configure(client, policy):
            result = client.configure_online_bench(policy)
            return await result if asynchronous else result

        hook = await configure(first, config(tmp_path))
        original = {file.name: file.read_bytes() for file in hook._state.directory.iterdir()}
        original_requests = copy.deepcopy(backend.requests)
        changed = config(tmp_path)
        changed["online_bench"]["every_n_steps"] = 5
        changed["online_bench"]["suites"][0]["harbor_config"]["tasks"][0]["name"] = "changed"
        with pytest.raises(FileExistsError):
            await configure(second, changed)
        assert {
            file.name: file.read_bytes() for file in hook._state.directory.iterdir()
        } == original
        assert backend.requests == original_requests  # No replacement configure request was sent.
        first._next_seq()  # The established producer remains usable.
        with pytest.raises(RuntimeError, match="previously failed"):
            second._next_seq()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_cases", None),
        ("expected_cases", 0),
        ("expected_cases", -1),
        ("expected_cases", 2),
        ("expected_cases", True),
        ("expected_cases", 1.0),
        ("attempt", None),
        ("attempt", 0),
        ("attempt", -1),
        ("attempt", True),
        ("attempt", 1.0),
        ("attempt", "1"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_incomplete_counts_and_invalid_attempts_fail_closed(tmp_path, field, value, asynchronous):
    async def scenario():
        backend = Backend()
        original_get = backend.get

        def get(path):
            result = original_get(path)
            if path.endswith("/result"):
                suite = result["suites"][0]
                target = suite if field == "expected_cases" else suite["cases"][0]
                if value is None:
                    target.pop(field)
                else:
                    target[field] = value
            return result

        backend.get = get
        client = training(backend, asynchronous)
        if asynchronous:
            client._service.http = SimpleNamespace(
                post=AsyncMock(side_effect=backend.post), get=AsyncMock(side_effect=backend.get)
            )

        async def call(method, **kwargs):
            result = method(**kwargs)
            return await result if asynchronous else result

        hook = await call(client.configure_online_bench, config=config(tmp_path))
        await call(hook.after_step, completed_step=100)
        backend.terminal = True
        with pytest.raises(ValueError):
            await call(hook.after_step, completed_step=101)
        with pytest.raises(RuntimeError, match="previously failed"):
            await call(hook.after_step, completed_step=200)
        assert len(backend.submissions) == 1
        assert not (hook._state.directory / "evaluations.jsonl").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("duplicate", [False, True], ids=["distinct-attempts", "duplicate-attempt"])
def test_case_identity_includes_attempt(tmp_path, duplicate):
    backend = Backend()
    original_get = backend.get

    def get(path):
        result = original_get(path)
        if path.endswith("/result"):
            suite = result["suites"][0]
            suite["expected_cases"] = 2
            extra = dict(suite["cases"][0], attempt=1 if duplicate else 2)
            suite["cases"].append(extra)
        return result

    backend.get = get
    hook = training(backend).configure_online_bench(config(tmp_path))
    hook.after_step(completed_step=100)
    backend.terminal = True
    if duplicate:
        with pytest.raises(ValueError, match="duplicate identity"):
            hook.after_step(completed_step=101)
    else:
        result = hook.after_step(completed_step=101)
        assert [case["attempt"] for case in result[0]["suites"][0]["cases"]] == [1, 2]


def test_legacy_environment_list_cannot_bypass_secret_guard(tmp_path):
    source = config(tmp_path)
    source["online_bench"]["suites"][0]["harbor_config"] = {
        "environment": {"env": ["GITHUB_TOKEN=private-value"]}
    }
    with pytest.raises(ValueError, match="mapping-form env") as error:
        resolve_config(source)
    assert "private-value" not in str(error.value)


def test_non_secret_token_budget_and_deployment_reference_are_allowed(tmp_path):
    source = config(tmp_path)
    source["online_bench"]["suites"][0]["harbor_config"] = {
        "max_tokens": 8192,
        "agents": [{"env": {"GITHUB_TOKEN": "${BENCH_AUTH}"}}],
    }
    resolved = resolve_config(source)
    assert resolved.suites[0].execution_config()["max_tokens"] == 8192
