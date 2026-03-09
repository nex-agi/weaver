## Prerequisites

```bash
# Verify service status before each test
conda activate weaver
# Record the current weaver-server version
curl -s http://<WEAVER_SERVER>/health
```

**Pod query conventions** (used throughout):
- Use the **infrawaves** skill from `nex-taas-shared-knowledge` to query pods by `model_id`
- Use the **volc-tls** skill to query provisioner logs

**Naming convention reference**:
| Type | Naming format | Example |
|------|---------------|---------|
| full_ft trainer | `{USER}-trainer-full_ft-{model_id}` | `sunpeng-trainer-full_ft-{uuid}` |
| full_ft inference | `fullft-{model_id}` | `fullft-{uuid}` |
| LoRA trainer | `trainer-lora-weaver-{base_model}` | `trainer-lora-weaver-qwen-qwen3-8b` |
| LoRA inference | `weaver-{base_model}` | `weaver-qwen-qwen3-8b` |

---

## GROUP 1: Full FT Basic Scenarios (F1 - F2)

### F1: First full_ft launch

**Goal**: Verify full_ft auto-provision works on first launch, and pods auto-terminate after training completes.

**Steps**:
1. Start the training script:
   ```bash
   conda activate weaver
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f1.log
   ```
2. Wait for `Model ID: <uuid>` to appear in logs and record the `model_id` (referred to as `MODEL_F1` below).

**Expected behavior**:
- [ ] Logs show `Model ID: MODEL_F1`
- [ ] Cluster shows both `sunpeng-trainer-full_ft-MODEL_F1` and `fullft-MODEL_F1` pods running normally
- [ ] Loss starts around 4.x and decreases steadily to ~0.1
- [ ] After training completes, both pods auto-terminate (disappear within ~2 minutes)

**Verification commands**:
```bash
# Query pods (using infrawaves skill)
# Search by model_id = MODEL_F1

# After training completes, confirm pods are cleaned up
# If pods remain, use volc-tls skill to check provisioner logs
```

**If issues arise**: Use the volc-tls skill to check weaver provisioner logs. Cross-reference with the terminate logic in `internal/services/instance_orchestrator.go`.

---

### F2: Second launch (new model_id, same base_model)

**Goal**: Verify that each full_ft creates independent trainer/inference pods and does not reuse the previous ones.

**Steps**:
1. Confirm all F1 pods have terminated.
2. Launch again:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f2.log
   ```
3. Record the new `model_id` (referred to as `MODEL_F2`).

**Expected behavior**:
- [ ] `MODEL_F2 != MODEL_F1` (new UUID)
- [ ] New pods `sunpeng-trainer-full_ft-MODEL_F2` and `fullft-MODEL_F2` are created, with completely different names from F1
- [ ] Training completes normally, pods auto-terminate

---

> Wait 2 minutes before starting the next group

---

## GROUP 2: Full FT Concurrent Scenarios (F3 - F4)

### F3: 2 concurrent full_ft jobs (core scenario)

**Goal**: Verify two concurrent full_ft jobs are fully isolated and do not interfere with each other.

**Steps**:
1. In Terminal A, start the first job:
   ```bash
   conda activate weaver
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f3a.log
   ```
2. Wait ~10 seconds (first job is running), then start the second in Terminal B:
   ```bash
   conda activate weaver
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f3b.log
   ```
3. Record both `model_id` values (referred to as `MODEL_F3A` and `MODEL_F3B`).

**Expected behavior**:
- [ ] Both scripts output their own `Model ID` with different UUIDs
- [ ] Cluster shows **4 pods** simultaneously:
  - `sunpeng-trainer-full_ft-MODEL_F3A`
  - `fullft-MODEL_F3A`
  - `sunpeng-trainer-full_ft-MODEL_F3B`
  - `fullft-MODEL_F3B`
- [ ] Both loss curves decrease independently from ~4.x without interfering with each other
- [ ] After completion, all 4 pods auto-terminate

---

### F4: 3 concurrent full_ft jobs (stress test)

**Goal**: Verify the provisioner has no race conditions or lost requests under high concurrency.

**Steps**:
1. Terminal A:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f4a.log
   ```
2. ~5 seconds later, Terminal B:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f4b.log
   ```
3. ~5 seconds later, Terminal C:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f4c.log
   ```

**Expected behavior**:
- [ ] All 3 scripts obtain different `Model ID` values
- [ ] Cluster shows **6 pods** total (3 sets of trainer + inference)
- [ ] All 3 loss curves decrease independently
- [ ] After completion, all 6 pods are automatically cleaned up

---

> Wait 2 minutes before starting the next group

---

## GROUP 3: LoRA Basic & Sharing Scenarios (L1 - L4)

### L1: First LoRA launch

**Goal**: Verify LoRA auto-provision works, with pods named as `weaver-{base_model}`.

**Steps**:
```bash
conda activate weaver
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l1.log
```

**Expected behavior**:
- [ ] Logs show `Model ID: MODEL_L1`
- [ ] Trainer pod is named `trainer-lora-weaver-<base_model>` (**not** model_id)
- [ ] Inference pod is named `weaver-<base_model>`
- [ ] Loss decreases normally
- [ ] Pods auto-terminate after training completes (same terminate behavior as full_ft)

---

### L2: Second LoRA launch (shared trainer verification, core test)

**Goal**: Verify that a second LoRA with the same base_model does not create new pods but reuses the existing trainer/inference.

**Steps**:
1. **While L1 pods are still running**, or after L1 ends and pods restart:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l2.log
   ```
2. Record the new `MODEL_L2`.

**Expected behavior**:
- [ ] `MODEL_L2 != MODEL_L1` (new UUID)
- [ ] **No new pods created** — still uses the same `trainer-lora-weaver-<base_model>` pod
- [ ] Pod count on the cluster remains unchanged (no additions)
- [ ] Training completes normally

> Note: If L1 pods have already terminated, L2 will trigger a new provision — this is expected (rebuild after idle cleanup). The key scenario to test is launching L2 **before** L1 finishes.

---

### L3: 2 concurrent LoRA jobs (same base_model, dedup verification)

**Goal**: Verify the `checkExistingLoRATrainer` dedup logic — only one set of pods should be created under concurrent launches.

**Steps**:
1. Terminal A:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l3a.log
   ```
2. **Immediately** (at the same time) in Terminal B:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l3b.log
   ```

**Expected behavior**:
- [ ] Both scripts obtain their own `Model ID`
- [ ] Cluster shows **only 1 set of pods** (trainer + inference), named `weaver-{base_model}`
- [ ] No duplicate provisioned pods appear
- [ ] Both training jobs run normally (sharing the same trainer)

---

### L4: 2 concurrent LoRA jobs (different base_model, independent provision)

**Goal**: Verify that LoRA jobs with different base_models provision independently.

**Steps**:
1. Terminal A (base_model A, e.g., Qwen3-8B):
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l4a.log
   ```
2. Terminal B (base_model B, e.g., Qwen3-14B, if available):
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora_14b.py 2>&1 | tee /tmp/test_l4b.log
   ```

**Expected behavior**:
- [ ] Cluster shows **2 differently named pod sets** (`weaver-qwen3-8b` series + `weaver-qwen3-14b` series)
- [ ] Both run without interfering with each other, each with decreasing loss

---

> Wait 2 minutes before starting the next group

---

## GROUP 4: Debug Mode Auto (DA1 - DA4)

> **Prerequisite**: Register a supported model with `debug_mode: "auto"` (see PR #28 for example)

```bash
curl -X PUT http://<WEAVER_SERVER>/api/v1/supported-models \
  -H "X-WEAVER-API-KEY: <key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "debug-auto/pig-latin",
    "config": {
      "debug_mode": "auto",
      "base_model_path": "/gpfs/models/...",
      "resource": { "train": { "world_size": 8, "gpus_per_pod": 8, "memory_per_gpu": 220 }, ... }
    }
  }'
```

---

### DA1: First debug auto launch

**Steps**:
```bash
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da1.log
```

**Expected behavior**:
- [ ] Provisions normally (torchrun starts automatically)
- [ ] After training completes, **pods do NOT terminate** (debug mode disables auto-termination)
- [ ] Cluster shows `sunpeng-trainer-full_ft-<model_id>` and `fullft-<model_id>` still alive

---

### DA2: Second launch (pods still running)

**Steps** (execute immediately after DA1 finishes, while pods are still alive):
```bash
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da2.log
```

**Expected behavior**:
- [ ] **Skips provision** — no new pods created (logs should show debug skip output)
- [ ] `seq_id` is reset, training starts from the beginning
- [ ] Reuses model weights already in GPU memory (no checkpoint loading)
- [ ] Pod count remains unchanged

---

### DA3: Second launch (pods have crashed)

**Steps**:
1. Manually delete the pods created by DA1:
   ```bash
   kubectl delete job sunpeng-trainer-full_ft-<da1_model_id> -n <namespace>
   ```
2. Launch the script:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da3.log
   ```

**Expected behavior**:
- [ ] Detects that pods no longer exist, **triggers re-provision**
- [ ] New pods are created and run normally
- [ ] Training completes successfully

---

### DA4: 2 concurrent debug auto launches (race condition verification)

**Goal**: Verify no duplicate provision occurs under concurrent launches.

**Steps**: Run in two terminals simultaneously (< 1 second apart)
```bash
# Terminal A & B run at the same time
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da4a.log
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da4b.log
```

**Expected behavior**:
- [ ] **Only one set of pods** is created (no duplicate provision)
- [ ] Both scripts run successfully (one provisions, the other skips)
- [ ] No duplicate pods appear

---

> Wait 2 minutes before starting the next group

---

## GROUP 5: Debug Mode Manual (DM1 - DM3)

> **Prerequisite**: Register a supported model with `debug_mode: "manual"`

```bash
curl -X PUT http://<WEAVER_SERVER>/api/v1/supported-models \
  -H "X-WEAVER-API-KEY: <key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "debug-manual/pig-latin",
    "config": {
      "debug_mode": "manual",
      "base_model_path": "/gpfs/models/...",
      "resource": { "train": { "world_size": 8, "gpus_per_pod": 8, "memory_per_gpu": 220 } }
    }
  }'
```

---

### DM1: Verify debug_info from create_model (no provision triggered)

**Goal**: Verify that `create_model` only creates a model record and returns `debug_info`, **without triggering provision**. Provision is triggered by subsequent `forward_backward` or `forward` calls.

**Steps**:
```bash
curl -X POST http://<WEAVER_SERVER>/api/v1/sessions/<session_id>/models \
  -H "X-WEAVER-API-KEY: <key>" \
  -H "Content-Type: application/json" \
  -d '{"base_model": "debug-manual/pig-latin", "training_mode": "full_ft", "model_seq_id": 1}'
```

**Expected behavior**:
- [ ] Response contains a `debug_info` field with:
  - `debug_mode: "manual"`
  - `job_name: "sunpeng-trainer-full_ft-<model_id>"`
  - `namespace`
  - `kubectl_exec: "kubectl exec -it sunpeng-trainer-full_ft-<model_id>-master-0 -n <ns> -- /bin/bash"`
  - `config_file: "/tmp/trainer.env"`
- [ ] **No pods are created on the cluster at this point** (`create_model` does not trigger provision)

---

### DM2: forward_backward triggers provision, manually run torchrun

**Goal**: Verify that provision is triggered only when `forward_backward` (or `forward`) is called, and in manual mode the pod runs `sleep infinity`.

**Steps**:
1. Use the SDK to call `forward_backward` (or `forward`) to trigger provision:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_manual.py 2>&1 | tee /tmp/test_dm2.log
   ```
2. Wait for pod creation, then confirm the pod is running `sleep infinity` (not torchrun):
   ```bash
   kubectl get pod -n <namespace> | grep sunpeng-trainer-full_ft-<model_id>
   ```
3. Exec into the pod using the `kubectl_exec` command from DM1:
   ```bash
   kubectl exec -it sunpeng-trainer-full_ft-<model_id>-master-0 -n <namespace> -- /bin/bash
   ```
4. Verify the config file:
   ```bash
   cat /tmp/trainer.env
   ```
5. Manually start torchrun:
   ```bash
   torchrun --nnodes=$WORLD_SIZE --nproc_per_node=8 --node_rank=$RANK \
     --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
     -m weaver-trainer.worker_process --env-file /tmp/trainer.env
   ```

**Expected behavior**:
- [ ] Provision is triggered only after `forward_backward`/`forward` is called — pods start creating
- [ ] Pod starts running `sleep infinity` (not torchrun)
- [ ] `kubectl get pod` confirms pod is in Running state
- [ ] `/tmp/trainer.env` contains correct content (model config, server_url, api_key, etc.)
- [ ] Manual torchrun starts successfully and loss begins decreasing

---

### DM3: Second forward_backward (pods still running, skip provision)

**Steps** (while DM2 pods are still alive):
1. Call `create_model` again:
   ```bash
   curl -X POST http://<WEAVER_SERVER>/api/v1/sessions/<session_id>/models \
     -H "X-WEAVER-API-KEY: <key>" \
     -H "Content-Type: application/json" \
     -d '{"base_model": "debug-manual/pig-latin", "training_mode": "full_ft", "model_seq_id": 2}'
   ```
2. Call `forward_backward` (or `forward`) again to trigger training.

**Expected behavior**:
- [ ] `create_model` returns `debug_info` (reusing existing pod information)
- [ ] `forward_backward`/`forward` **skips provision** — no new pods created
- [ ] `kubectl exec` command still works (pod has not changed)

---

> Wait 2 minutes before starting the next group

---

## GROUP 6: Edge & Error Scenarios (E1 - E4)

### E1: full_ft pod mid-training crash, verify stale recovery

**Steps**:
1. Start a full_ft job and wait for pods to be created:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py &
   ```
2. Once pods are created, kill the trainer pod:
   ```bash
   kubectl delete pod sunpeng-trainer-full_ft-<model_id>-master-0 -n <namespace>
   ```

**Expected behavior**:
- [ ] Provisioner detects the stale instance (abnormal instance state)
- [ ] Use the volc-tls skill to check provisioner logs and confirm stale detection and re-provision log entries

---

### E2: LoRA pod disappears, verify restart

**Steps**:
1. While a LoRA job is running, delete the trainer pod.
2. Start a new LoRA job with the same base_model.

**Expected behavior**:
- [ ] `checkExistingLoRATrainer` detects the pod is unhealthy
- [ ] Re-triggers provision (does not reuse the dead pod)
- [ ] New pod is created normally

---

### E3: Launch with non-existent base_model

**Steps**:
```bash
curl -X POST http://<WEAVER_SERVER>/api/v1/sessions/<session_id>/models \
  -H "X-WEAVER-API-KEY: <key>" \
  -d '{"base_model": "nonexistent/model-xyz", "training_mode": "full_ft"}'
```

**Expected behavior**:
- [ ] Returns a 4xx error (not 500)
- [ ] **No orphaned instance records** are left in the database

---

### E4: Concurrent full_ft + LoRA with the same base_model

**Steps**:
1. Terminal A — start full_ft:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_e4a.log
   ```
2. Terminal B — simultaneously start LoRA:
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_e4b.log
   ```

**Expected behavior**:
- [ ] full_ft creates `sunpeng-trainer-full_ft-<model_id>` and `fullft-<model_id>`
- [ ] LoRA creates `trainer-lora-weaver-<base_model>` and `weaver-<base_model>`
- [ ] Both pod sets run without interfering with each other
- [ ] Naming is completely different with no confusion

---

## Troubleshooting Guide (Global)

When any test case encounters issues, follow this troubleshooting order:

1. **Check script logs**: `/tmp/test_*.log`
2. **Check pod status** (infrawaves skill, search by model_id)
3. **Check provisioner logs** (volc-tls skill):
   - Keywords: `provisioning`, `terminate`, `skip`, `debug`, `stale`, `lora dedup`
4. **Check weaver-server code**:
   - Auto-provision logic: `internal/services/instance_orchestrator.go`
   - LoRA dedup: `checkExistingLoRATrainer()`
   - Debug mode: `extractDebugMode()`, `provisionNewTrainer()`
   - Terminate: `HandleTerminate()` — skipped when `getTrainingMode() == "lora"`
5. **After identifying the issue**: Create a bug issue in `china-qijizhifeng/weaver-server` and link it to the feedback repo

---

## Completion Criteria

| Group | Test Cases | Pass Criteria |
|-------|------------|---------------|
| Full FT Basic | F1, F2 | Pods correctly named, loss decreasing, auto-terminated |
| Full FT Concurrent | F3, F4 | All pods exist independently, no interference |
| LoRA | L1-L4 | Shared trainer dedup works correctly, naming follows convention |
| Debug Auto | DA1-DA4 | First launch normal, second skips provision, concurrent no duplicates |
| Debug Manual | DM1-DM3 | create_model does not trigger provision, forward_backward triggers provision, sleep infinity, skip and reuse |
| Edge Cases | E1-E4 | Correct error handling, no data pollution |
