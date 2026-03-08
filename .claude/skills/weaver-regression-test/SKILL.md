---
name: weaver-regression-test
description: Run the full Weaver provisioner regression test suite against the production service. Covers full_ft, LoRA, debug modes, and edge cases.
---

# Weaver Provisioner Regression Testing Skill

## Overview

Run end-to-end regression tests for the Weaver provisioner following the test plan in `.claude/skills/weaver-regression-test/regression_test_plan.md`. Tests cover full fine-tuning, LoRA, debug modes (auto/manual), and edge cases.

## Prerequisites

```bash
conda activate weaver
export WEAVER_API_KEY="<your-key>"
```

## Instructions

1. **Follow `regression_test_plan.md` strictly** (co-located in this skill directory). Execute every test case as documented.

2. **Use the production service** at `https://weaver-console.nex-agi.cn` — do not build or run locally.

3. **IAM errors can be retried** without recording. If an IAM authentication error occurs, simply retry the request.

4. **Use shared knowledge skills for log analysis.** For each test, use the skills from `china-qijizhifeng/nex-taas-shared-knowledge` to:
   - Query Infrawaves for pod status (by `model_id`)
   - Query Volcengine TLS for provisioner logs
   - Cross-reference with `china-qijizhifeng/weaver-server` source code for root cause analysis

5. **Run 3 rounds of testing.** Create a GitHub issue in `china-qijizhifeng/weaver-server` and post each round's results as a comment:
   - **Pass**: Record as passed
   - **Fail**: Analyze root cause using log skills, record the analysis, and create a sub-issue in `china-qijizhifeng/weaver-server` linked to the main test issue
   - **Flaky (passes on retry)**: Still record the failure, including log analysis from the failed attempt
   - **Blocking failure**: Skip the test, move on to the next one, but do not continue multi-round iteration for that test
   - **Between rounds**: Verify all tasks have stopped, then wait 10 minutes before starting the next round

6. **Post a final summary report** to the issue after all rounds complete.

7. **Use example scripts** from the `examples/` directory:
   - `examples/pig_latin_fullft.py` — Full fine-tuning (F1-F4)
   - `examples/pig_latin_lora.py` — LoRA training (L1-L4)
   - `examples/pig_latin_fullft_debug_auto.py` — Debug auto mode (DA1-DA4)
   - `examples/pig_latin_fullft_debug_manual.py` — Debug manual mode (DM1-DM3)
   - `examples/pig_latin_lora_alt.py` — LoRA with alternate base model (L4)

## Test Groups

| Group | Tests | What it validates |
|-------|-------|-------------------|
| Full FT Basic | F1, F2 | Single full_ft provision, independent pods, auto-termination |
| Full FT Concurrent | F3, F4 | 2-3 concurrent full_ft with full isolation |
| LoRA | L1-L4 | LoRA provision, shared trainer dedup, different base_model independence |
| Debug Auto | DA1-DA4 | Debug auto provision, pod reuse, crash recovery, concurrent dedup |
| Debug Manual | DM1-DM3 | Lazy provision on forward_backward, sleep infinity, manual torchrun |
| Edge Cases | E1-E4 | Pod crash recovery, invalid model errors, mixed mode concurrency |

## Log Analysis

When a test fails, follow this troubleshooting order:

1. Check script logs in `/tmp/test_*.log`
2. Query pod status via Infrawaves skill (by `model_id`)
3. Query provisioner logs via Volcengine TLS skill with keywords: `provisioning`, `terminate`, `skip`, `debug`, `stale`, `lora dedup`
4. Cross-reference with weaver-server code:
   - Auto-provision: `internal/services/instance_orchestrator.go`
   - LoRA dedup: `checkExistingLoRATrainer()`
   - Debug mode: `extractDebugMode()`, `provisionNewTrainer()`
   - Terminate: `HandleTerminate()`

## GitHub Issue Format

```markdown
# Weaver Provisioner Integration Test — Round N

| Test | Result | Notes |
|------|--------|-------|
| F1   | PASS   |       |
| F2   | PASS   |       |
| ...  | ...    | ...   |

## Details
[Per-test details for any failures or notable observations]
```
