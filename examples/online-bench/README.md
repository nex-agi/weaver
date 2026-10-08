# Two-suite online-bench probe

Keep these three YAML files together:

- `online-bench.yaml`: Weaver cadence, output, artifact policy and suite deadlines.
- `swe-probe.yaml`: native Harbor configuration selecting five SWE Verified tasks.
- `terminal-probe.yaml`: native Harbor configuration selecting five Terminal-Bench tasks.

The main config selects both children by filename. Child paths resolve beside the
main YAML; `results_path` resolves against the SDK driver's cwd. Copy this directory
into your training recipe and edit the native task references and budgets there.
The SDK freezes the selected contents at startup and saves non-secret snapshots
with the run's results; editing input files afterward does not change an active run.

Use this configuration with a **20-step SFT driver**, calling the hook after each
completed logical training step and `finish()` before teardown:

```python
bench = training.configure_online_bench("online-bench/online-bench.yaml")
for step in range(1, 21):
    train_one_step(step)
    results = bench.after_step(completed_step=step)
final_results = bench.finish()
```

Training steps are controlled by the training driver, not by this benchmark YAML.
Rounds trigger at steps 5, 10, 15 and 20. Each contains two suites × five tasks ×
one attempt: **40 task executions total**. Training resumes once weight sync is
ready, and waits at the next trigger boundary if the prior aggregate round is
still running. Suites run sequentially with independent execution budgets;
Harbor controls task concurrency within each suite.

Use explicit native Harbor `tasks` lists with immutable revision references and
`datasets: []`. Nonempty dataset selections and kit `__base__` composition are not
supported in this version; the server does not restrict task names to this probe.

The examples enable large-artifact retention for E2E verification. Set
`save_artifacts: false` to retain only compact results/timings; evaluator temporary
working files may still be necessary. The configured output root must be shared
with and writable by the worker. Inference GPU resources and worker shared mounts
remain model-registration/deployment configuration.

These configurations target the packaged code-harbor/Claude Code adapter. They
are not arbitrary harness/plugin examples, and task revisions must be available
in the deployed worker kit. Do not put API keys in configuration files. Deployment
credentials are supplied separately, and Weaver binds the runtime model endpoint
and output directory.

The expanded 20-step run is a planned real GPU/E2B acceptance test; mocked SDK
control tests do not establish benchmark correctness or GPU weight equality.
See [online-bench documentation](../../docs/online-bench.md) for full semantics.
