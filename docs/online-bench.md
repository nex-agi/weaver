# Online benchmarks (experimental)

Online-bench is independent of RL validation. The same hook can be used by an
SFT or RL driver. It is disabled by default and requires the matching server and
CPU worker image. A disabled/older server returns 503 for enabled requests.
The initial three-task SWE SFT E2E qualified the distributed execution path;
the expanded two-suite recipe still requires a new real GPU/E2B qualification.

```python
bench = training.configure_online_bench("examples/online-bench/online-bench.yaml")
for step in range(1, 21):
    train_one_step(step)  # complete all operations belonging to this logical step
    results = bench.after_step(completed_step=step)
bench.finish()           # collect outstanding work; no extra final evaluation
```

The async equivalents are `await training.configure_online_bench(...)`,
`await bench.after_step(...)`, and `await bench.finish()`. They own no event loop
or background task. HTTP/poll sleeps yield the caller's loop; filesystem work is
sent to a thread. These hooks do not change the training data, optimizer, loss,
learning-rate schedule or checkpoint semantics.

## SDK-owned configuration

Keep the editable main and native Harbor child configurations together, as in
[`examples/online-bench/`](../examples/online-bench/):

```text
online-bench/
  online-bench.yaml
  swe-probe.yaml
  terminal-probe.yaml
```

```yaml
online_bench:
  enabled: true
  every_n_steps: 5
  results_path: ./weaver/.logs
  save_artifacts: true
  suites:
    - name: swe_probe
      harbor_config: swe-probe.yaml
      timeout_seconds: 5400
    - name: terminal_probe
      harbor_config: terminal-probe.yaml
      timeout_seconds: 3600
```

`harbor_config` names a sibling `.yaml`/`.yml` file. The SDK resolves children
relative to the **main YAML**, not the worker's filesystem or the driver's cwd.
The child is a complete native Harbor configuration: task names/revisions,
harness, concurrency, attempts, retry and verifier options use their existing
Harbor fields. A mapping instead of a filename is also accepted for programmatic
callers. This does not introduce a second task-selection language.

Defaults are packaged at `weaver/online_bench/defaults.yaml`. An explicit path or
mapping takes precedence over `WEAVER_ONLINE_BENCH_CONFIG_PATH`; exactly one
override is loaded. Lists replace defaults; unknown wrapper keys, duplicate YAML
keys and duplicate suite names are rejected. A mapping has the same
`online_bench` root as YAML; its child filenames resolve relative to cwd.
There may be at most 16 suites; each suite timeout is 1–86,400 seconds.
The old top-level `timeout_seconds` setting and image-owned preset names are not
supported. GPU topology remains in the exact registered model's separate
`resource.online_bench_inference` recipe, not this file.

All selected child contents are loaded and snapshotted at startup. Later edits
to the source files or caller dictionaries do not change the running evaluation.
The SDK sends resolved native configurations to the server/worker, not local
filenames. Do not put credentials in YAML. Use deployment environment variables;
`${NAME}` references are not expanded in the SDK's saved configuration. Inline
credential fields (including `*_TOKEN`/`Authorization`) and URL userinfo credentials
are rejected before persistence or control-plane submission.

The CPU worker remains a thin adapter to the packaged **code-harbor** revision,
not a generic benchmark/plugin system. Tasks and grading belong to code-harbor;
the server has no SWE task allowlist. The current model endpoint binding uses
`nexau_agent.claude_code_direct:ClaudeCodeDirect`; this does not claim arbitrary
harness support. Child configurations use explicit native Harbor `tasks` lists
with exact revision references and `datasets: []`. Arbitrary available task
selections are accepted, not only the original three SWE tasks. Nonempty dataset
selections are rejected in this version rather than silently changing dataset
metric grouping or resolving a mutable selection again every round. Harbor's
native configuration/task resolution validates these lists; there is no new
Weaver task-selection language.
Child configs must be complete: code-harbor launcher `__base__` composition is
not reimplemented by the SDK. Runtime model endpoint, identity and output
location are supplied by Weaver so copied offline settings cannot select the
wrong model or output directory. The worker image must contain the selected
tasks and their dependencies; changing to an unavailable upstream task revision
still requires updating that packaged kit.

## Scheduling and timeouts

One **round** contains all configured suites and task attempts at one exact
weight version. Only one aggregate round may be in flight. The worker executes
suites sequentially; task concurrency remains a Harbor option inside each suite.
Each suite's deadline begins when that suite starts, not while it waits behind
another suite. It covers setup, execution, verification and result finalization;
per-agent/verifier limits remain in the native child configuration. An internal
round watchdog accounts for the sum of suite budgets plus bounded weight-sync
and reporting overhead, rather than allowing one suite to consume another's
budget.

At step 5 the hook closes the server-side boundary gate, reserves a round,
submits an ordered export and waits for **sync-ready**, not merely export
completion. Training may then continue while Harbor runs. At step 10 it waits
for all step-5 suites before publishing step-10 weights; do not submit step 11
first. The same applies at steps 15 and 20. `finish()` drains the final round
before normal resource release; it never creates an off-cadence evaluation.

Valid low scores are monitoring results, not automatic training stops.
Execution, synchronization, protocol and storage errors raise; a failed hook
cannot silently reopen the training gate. A timeout is not a score of zero and
must not permit loading new weights while old execution is still active.

Call from one configuring thread/task. Mutation guards apply to sibling training
clients sharing the same ServiceClient; concurrent optimizer/export/load calls
are rejected while the hook holds a boundary or after failure. The server also
enforces source-operation admission across separate clients and V1/V2 operation
submission, and rejects preparation/boundaries with pending trainer work. These
guards do not cancel prequeued work or support multiple producers. Never prequeue
future steps. Repeated calls for the current known step observe existing work
without resubmission. Backwards counters, missed boundaries and automatic
resume/reconstruction are not supported. NexRL integration remains a separate
qualification, especially its logical-step and async task ownership conventions.

The runtime supports dedicated full-FT, single-cluster SGLang deployments with
the existing file/Mooncake weight transport, not LoRA/time-sharing/NCCL. Benchmark
execution overlaps training, but weight sync, busy-boundary waits and final drain
are intentional blocking points—not zero-overhead evaluation.

## One results root

`results_path` defaults to `./weaver/.logs`. The SDK resolves it once against its
cwd and sends the absolute location. It must be an approved shared filesystem
mount writable by the worker; the worker must not resolve a client-relative path
against its own cwd. A private SDK-only filesystem cannot receive direct worker
artifacts.

```text
<results_path>/online-bench-<model_id>/
  config.yaml                       # frozen policy snapshot, not editable input
  suite-swe_probe.yaml              # frozen native child snapshot
  suite-terminal_probe.yaml
  evaluations.jsonl
  timings.jsonl
  step-000005-<evaluation_id>/
    result.json                     # compact score/status/timing report
    swe_probe/
      artifacts/                    # worker writes retained large outputs directly
    terminal_probe/
      artifacts/
```

Scores, status, timings and artifact references return worker → server → SDK.
The SDK reserves each model's run directory exclusively before saving snapshots;
a second SDK instance cannot overwrite an existing run's configuration. Existing
run directories are not resumed automatically. A failed/new launch needs a new
model/run identity or results root. The SDK writes compact JSON/JSONL. Large
trajectories/logs are written directly
by the worker under this same root, never downloaded and re-uploaded through the
control plane. Native credential-bearing configs/logs must not be exposed as
public artifacts; only the worker's selected safe outputs are retained.

`save_artifacts: false` disables optional long-artifact retention, not score or
timing output. Harbor may still produce temporary working files needed to run
and grade tasks; this option does not promise to eliminate all internal IO.

Results retain source step, exact version, target and configuration digest.
Terminal execution failures are saved with attribution. Repeated observations
are deduplicated in memory; crash-safe exactly-once storage is not promised.

Compact suite reports include metrics, the expected case count and individual
case identities/attempt numbers and scores. Task/suite timings belong to evaluator
execution. `timings.jsonl` separately
records SDK hook observation/boundary wait, sync wait, total hook time and final
drain. Parallel task durations must not be summed and described as suite wall
time; asynchronous evaluation duration is not equal to blocked training time.

Call `finish()` before normal session teardown. On operational failure, use
normal task error/teardown handling rather than submitting another benchmark.
No special lost-response replay or pod-loss recovery is implemented.
