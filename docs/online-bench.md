# Online benchmarks (experimental)

Online-bench is independent of RL validation. The same hook can be used by an
SFT or RL training driver. It is disabled by default and requires a server with
the enabled online-bench runtime and matching CPU worker image. Runtime wiring
is implemented; real deployment/GPU/E2B qualification is a separate E2E step.
A disabled or older server returns 503 for enabled requests.

```python
bench = training.configure_online_bench("online-bench.yaml")
for step in range(1, num_steps + 1):
    train_one_step(step)  # await/complete all operations belonging to this step
    results = bench.after_step(completed_step=step)
bench.finish()           # collect outstanding work; no extra final evaluation
```

The asynchronous counterparts are `await training.configure_online_bench(...)`,
`await bench.after_step(...)`, and `await bench.finish()`. They own no event loop
or background task. HTTP/poll sleeps yield the caller's loop; filesystem work is
sent to a thread.

```yaml
online_bench:
  enabled: true
  every_n_steps: 100
  timeout_seconds: 3600
  results_path: null
  suites:
    - name: swe_probe
      harbor_config: swe-verified-smoke-v1.yaml
```

Defaults are packaged at `weaver/online_bench/defaults.yaml`. An explicit config
path/mapping takes precedence over `WEAVER_ONLINE_BENCH_CONFIG_PATH`; exactly one
override is loaded. Lists replace defaults; unknown keys are rejected. A mapping
has the same `online_bench` root as YAML. GPU resources remain in the exact
registered model's separate `resource.online_bench_inference` recipe. Preset
names address files in the worker image, not files on the SDK host. The whole
evaluation timeout is capped at 86,400 seconds, with at most 16 named suites; the
initial catalog contains only the pinned three-task SWE Verified smoke preset.
The initial runtime supports dedicated full-FT, single-cluster SGLang deployments
with the existing file/Mooncake weight transport, not LoRA/time-sharing/NCCL.

At step 100 the hook closes the server-side boundary gate, reserves an evaluation,
submits an ordered export and waits
for **sync-ready**, not merely export completion. Training may then continue
while Harbor runs. At step 200 it waits for evaluation 100 before publishing
step 200; it must not submit step 201 first. Valid low scores are returned, not
turned into automatic training stops. Execution, synchronization, protocol and
storage errors raise; a failed hook cannot silently reopen the training gate.

Call from one configuring thread/task. Mutation guards apply to sibling training
clients sharing the same ServiceClient; concurrent optimizer/export/load calls
are rejected while the hook holds a boundary or after failure. The server also enforces source-operation admission across separate clients and
V1/V2 operation submission, and rejects preparation/boundaries with pending
trainer work. These guards do not cancel already-enqueued work or turn multiple
producers into a supported training driver. Never prequeue future steps. Repeated calls for the current known
step observe existing work without resubmission. Backwards counters, missed
boundaries and automatic resume/reconstruction are not supported.

Results are normal caller-owned JSON plus JSONL under:

```
<results_path or ./weaver/.logs>/online-bench-<model_id>/
  config.yaml
  evaluations.jsonl
  step-000100-<evaluation_id>/result.json
```

Paths resolve once at configuration time and are never sent to the server.
Results keep the source step, exact version, target and config digest. Terminal
execution failures are also saved with attribution. Repeated observations are
deduplicated in memory; crash-safe exactly-once storage is not promised. Raw
Harbor working directories may contain credentials and are not downloaded.

`finish()` drains the outstanding evaluation before asking the server to release
its task-owned resources. Call it before normal session teardown. On operational
failure, use normal task error/teardown handling rather than submitting another
benchmark. No special lost-response replay or pod-loss recovery is implemented.
