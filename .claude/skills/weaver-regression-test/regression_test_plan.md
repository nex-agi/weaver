## 前置准备

```bash
# 每次测试前确认服务状态
conda activate weaver
# 记录 weaver-server 当前版本
curl -s http://<WEAVER_SERVER>/health
```

**Pod 查询约定**（贯穿全文）：
- 使用 `nex-taas-shared-knowledge` 里的 **infrawaves** skill，按 `model_id` 查询 pod
- 使用 **volc-tls** skill 查询 provisioner 日志

**命名规则速查**：
| 类型 | 命名格式 | 示例 |
|------|----------|------|
| full_ft trainer | `{USER}-trainer-full_ft-{model_id}` | `sunpeng-trainer-full_ft-{uuid}` |
| full_ft inference | `fullft-{model_id}` | `fullft-{uuid}` |
| LoRA trainer | `trainer-lora-weaver-{base_model}` | `trainer-lora-weaver-qwen-qwen3-8b` |
| LoRA inference | `weaver-{base_model}` | `weaver-qwen-qwen3-8b` |

---

## GROUP 1：Full FT 基础场景（F1 ~ F2）

### F1：首次启动单个 full_ft 任务

**目的**：验证 full_ft 首次 auto-provision 正常，任务结束后 pod 自动终止。

**步骤**：
1. 启动训练脚本：
   ```bash
   conda activate weaver
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f1.log
   ```
2. 等待日志中出现 `Model ID: <uuid>`，记录该 `model_id`（以下称 `MODEL_F1`）。

**预期行为**：
- [ ] 日志中出现 `Model ID: MODEL_F1`
- [ ] 集群上能看到 `sunpeng-trainer-full_ft-MODEL_F1` 和 `fullft-MODEL_F1` 两个 pod 均正常运行
- [ ] loss 从约 4.x 开始持续下降至 0.1 附近
- [ ] 训练结束后，两个 pod 均自动终止（约 2min 内消失）

**验证命令**：
```bash
# 查询 pod（使用 infrawaves skill）
# 按 model_id = MODEL_F1 查

# 任务结束后确认 pod 已清理
# 如 pod 残留，使用 volc-tls skill 查 provisioner 日志
```

**如有问题**：使用 volc-tls skill 检查 weaver provisioner 日志，结合 `internal/services/instance_orchestrator.go` 的 terminate 逻辑分析原因。

---

### F2：第二次启动（新 model_id，同 base_model）

**目的**：验证 full_ft 每次新建独立 trainer/inference，不复用前一次。

**步骤**：
1. 确认 F1 的 pod 已全部终止。
2. 再次启动：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f2.log
   ```
3. 记录新的 `model_id`（称 `MODEL_F2`）。

**预期行为**：
- [ ] `MODEL_F2 ≠ MODEL_F1`（新 UUID）
- [ ] 新创建 `sunpeng-trainer-full_ft-MODEL_F2` 和 `fullft-MODEL_F2`，与 F1 命名完全不同
- [ ] 训练正常完成，pod 自动终止

---

> ⏳ **等待 2 分钟后开始下一组**

---

## GROUP 2：Full FT 并发场景（F3 ~ F4）

### F3：并发 2 个 full_ft（核心场景）

**目的**：验证两个 full_ft 并发完全隔离、互不影响。

**步骤**：
1. 在 Terminal A 启动第一个：
   ```bash
   conda activate weaver
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f3a.log
   ```
2. 等待约 10 秒（第一个正在运行），在 Terminal B 启动第二个：
   ```bash
   conda activate weaver
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f3b.log
   ```
3. 分别记录两个 `model_id`（称 `MODEL_F3A`、`MODEL_F3B`）。

**预期行为**：
- [ ] 两个脚本都输出各自的 `Model ID`，两个 UUID 不同
- [ ] 集群上能同时看到 **4 个 pod**：
  - `sunpeng-trainer-full_ft-MODEL_F3A`
  - `fullft-MODEL_F3A`
  - `sunpeng-trainer-full_ft-MODEL_F3B`
  - `fullft-MODEL_F3B`
- [ ] 两边 loss 各自独立从 4.x 下降，不互相干扰
- [ ] 任务结束后，4 个 pod 全部自动终止

---

### F4：并发 3 个 full_ft（压力场景）

**目的**：验证 provisioner 在高并发下无竞态、无丢失。

**步骤**：
1. Terminal A 启动：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f4a.log
   ```
2. 约 5 秒后 Terminal B：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f4b.log
   ```
3. 约 5 秒后 Terminal C：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_f4c.log
   ```

**预期行为**：
- [ ] 3 个脚本都成功获得不同的 `Model ID`
- [ ] 集群上最终存在 **6 个 pod**（3 组 trainer + inference）
- [ ] 3 个 loss 曲线各自独立下降
- [ ] 结束后 6 个 pod 均自动清理

---

> ⏳ **等待 2 分钟后开始下一组**

---

## GROUP 3：LoRA 基础 & 共享场景（L1 ~ L4）

### L1：首次启动单个 LoRA 任务

**目的**：验证 LoRA 首次 auto-provision，命名为 `weaver-{base_model}`。

**步骤**：
```bash
conda activate weaver
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l1.log
```

**预期行为**：
- [ ] 日志中出现 `Model ID: MODEL_L1`
- [ ] 集群上 trainer pod 名为 `trainer-lora-weaver-<base_model>`（**非** model_id）
- [ ] inference pod 名为 `weaver-<base_model>`
- [ ] loss 正常下降
- [ ] 任务结束后 pod 自动终止（LoRA 任务结束触发 terminate，与 full_ft 相同）

---

### L2：第二次 LoRA 启动（共享验证，核心）

**目的**：验证相同 base_model 的第二个 LoRA 不新建 pod，复用已有 trainer/inference。

**步骤**：
1. **L1 的 pod 仍在运行时**，或 L1 结束后重新启动：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l2.log
   ```
2. 记录新的 `MODEL_L2`。

**预期行为**：
- [ ] `MODEL_L2 ≠ MODEL_L1`（新 UUID）
- [ ] **不产生新的 pod**，仍是 `trainer-lora-weaver-<base_model>` 那一个
- [ ] 集群上 pod 数量不变（没有新增）
- [ ] 训练正常完成

> ⚠️ 若 L1 结束后 pod 已终止，L2 会触发新的 provision——这是正常行为（idle cleanup 后重建）。此场景重点测 L1 **结束前** 立即启动 L2。

---

### L3：并发 2 个 LoRA（相同 base_model，去重验证）

**目的**：验证 `checkExistingLoRATrainer` 去重逻辑，并发时只创建一组 pod。

**步骤**：
1. Terminal A：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l3a.log
   ```
2. **立即**（同一时刻）Terminal B：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l3b.log
   ```

**预期行为**：
- [ ] 两个脚本都获得各自的 `Model ID`
- [ ] 集群上**只有 1 组 pod**（trainer + inference），命名为 `weaver-{base_model}` 系列
- [ ] 没有出现重复 provision 的 pod
- [ ] 两个训练任务都正常运行（共享同一 trainer）

---

### L4：并发 2 个 LoRA（不同 base_model，独立 provision）

**目的**：验证不同 base_model 的 LoRA 各自独立 provision。

**步骤**：
1. Terminal A（base_model A，如 Qwen3-8B）：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_l4a.log
   ```
2. Terminal B（base_model B，如 Qwen3-14B，如有）：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora_14b.py 2>&1 | tee /tmp/test_l4b.log
   ```

**预期行为**：
- [ ] 集群上出现 **2 组不同命名的 pod**（`weaver-qwen3-8b` 系列 + `weaver-qwen3-14b` 系列）
- [ ] 两者互不干扰，各自 loss 下降

---

> ⏳ **等待 2 分钟后开始下一组**

---

## GROUP 4：Debug Mode Auto（DA1 ~ DA4）

> **前置**：注册一个 `debug_mode: "auto"` 的 supported model（见 PR #28 示例）

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

### DA1：首次启动 debug auto

**步骤**：
```bash
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da1.log
```

**预期行为**：
- [ ] 正常 provision（torchrun 自动运行）
- [ ] 训练完成后 **pod 不终止**（debug 模式禁用 auto-termination）
- [ ] 集群上 `sunpeng-trainer-full_ft-<model_id>` 和 `fullft-<model_id>` 仍存活

---

### DA2：第二次启动（pod 仍在）

**步骤**（DA1 结束后，pod 仍存活时立即执行）：
```bash
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da2.log
```

**预期行为**：
- [ ] **跳过 provision**，不新建 pod（日志中应有 debug skip 相关输出）
- [ ] `seq_id` 被重置，训练从头开始
- [ ] 复用 GPU 内存中的模型权重（无 checkpoint 加载）
- [ ] pod 数量不变

---

### DA3：第二次启动（pod 已崩溃）

**步骤**：
1. 手动删除 DA1 创建的 pod：
   ```bash
   kubectl delete job sunpeng-trainer-full_ft-<da1_model_id> -n <namespace>
   ```
2. 启动脚本：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da3.log
   ```

**预期行为**：
- [ ] 检测到 pod 不存在，**重新触发 provision**
- [ ] 新的 pod 创建并正常运行
- [ ] 训练正常完成

---

### DA4：并发 2 次 debug auto（竞态验证）

**目的**：验证并发情况下不产生重复 provision。

**步骤**：两个 Terminal 同时运行（间隔 < 1s）
```bash
# Terminal A & B 同时执行
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da4a.log
python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_auto.py 2>&1 | tee /tmp/test_da4b.log
```

**预期行为**：
- [ ] **只创建一组 pod**（不重复 provision）
- [ ] 两个脚本都能正常运行（一个 provision、一个跳过）
- [ ] 无重复 pod 出现

---

> ⏳ **等待 2 分钟后开始下一组**

---

## GROUP 5：Debug Mode Manual（DM1 ~ DM3）

> **前置**：注册 `debug_mode: "manual"` 的 supported model

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

### DM1：create_model 验证 debug_info（不触发 provision）

**目的**：验证 `create_model` 只创建模型记录并返回 `debug_info`，**不会触发 provision**。Provision 由后续的 `forward_backward` 或 `forward` 调用触发。

**步骤**：
```bash
curl -X POST http://<WEAVER_SERVER>/api/v1/sessions/<session_id>/models \
  -H "X-WEAVER-API-KEY: <key>" \
  -H "Content-Type: application/json" \
  -d '{"base_model": "debug-manual/pig-latin", "training_mode": "full_ft", "model_seq_id": 1}'
```

**预期行为**：
- [ ] Response 中包含 `debug_info` 字段，内容包括：
  - `debug_mode: "manual"`
  - `job_name: "sunpeng-trainer-full_ft-<model_id>"`
  - `namespace`
  - `kubectl_exec: "kubectl exec -it sunpeng-trainer-full_ft-<model_id>-master-0 -n <ns> -- /bin/bash"`
  - `config_file: "/tmp/trainer.env"`
- [ ] **此时集群上没有 pod 创建**（`create_model` 不触发 provision）

---

### DM2：forward_backward 触发 provision，手动运行 torchrun

**目的**：验证调用 `forward_backward`（或 `forward`）时才触发 provision，manual 模式下 pod 运行 `sleep infinity`。

**步骤**：
1. 使用 SDK 调用 `forward_backward`（或 `forward`），触发 provision：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft_debug_manual.py 2>&1 | tee /tmp/test_dm2.log
   ```
2. 等待 pod 创建完成，确认 pod 运行 `sleep infinity`（不是 torchrun）：
   ```bash
   kubectl get pod -n <namespace> | grep sunpeng-trainer-full_ft-<model_id>
   ```
3. 根据 DM1 返回的 `kubectl_exec` 命令进入 pod：
   ```bash
   kubectl exec -it sunpeng-trainer-full_ft-<model_id>-master-0 -n <namespace> -- /bin/bash
   ```
4. 验证配置文件：
   ```bash
   cat /tmp/trainer.env
   ```
5. 手动运行 torchrun：
   ```bash
   torchrun --nnodes=$WORLD_SIZE --nproc_per_node=8 --node_rank=$RANK \
     --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
     -m weaver-trainer.worker_process --env-file /tmp/trainer.env
   ```

**预期行为**：
- [ ] `forward_backward`/`forward` 调用后才触发 provision，pod 开始创建
- [ ] Pod 启动后运行 `sleep infinity`（不是 torchrun）
- [ ] `kubectl get pod` 确认 pod 处于 Running 状态
- [ ] `/tmp/trainer.env` 内容正确（包含 model 配置、server_url、api_key 等）
- [ ] 手动 torchrun 正常启动，loss 开始下降

---

### DM3：第二次 forward_backward（pod 仍在，跳过 provision）

**步骤**（DM2 结束后 pod 仍存活）：
1. 再次调用 `create_model`：
   ```bash
   curl -X POST http://<WEAVER_SERVER>/api/v1/sessions/<session_id>/models \
     -H "X-WEAVER-API-KEY: <key>" \
     -H "Content-Type: application/json" \
     -d '{"base_model": "debug-manual/pig-latin", "training_mode": "full_ft", "model_seq_id": 2}'
   ```
2. 再次调用 `forward_backward`（或 `forward`）触发训练。

**预期行为**：
- [ ] `create_model` 返回 `debug_info`（复用已有 pod 的信息）
- [ ] `forward_backward`/`forward` 时 **跳过 provision**，不新建 pod
- [ ] `kubectl exec` 命令仍然有效（pod 未变）

---

> ⏳ **等待 2 分钟后开始下一组**

---

## GROUP 6：边界 & 异常场景（E1 ~ E4）

### E1：full_ft pod 中途 crash，验证 stale 恢复

**步骤**：
1. 启动 full_ft 任务，等待 pod 创建：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py &
   ```
2. pod 创建后立即 kill：
   ```bash
   kubectl delete pod sunpeng-trainer-full_ft-<model_id>-master-0 -n <namespace>
   ```

**预期行为**：
- [ ] provisioner 检测到 stale instance（instance 状态异常）
- [ ] 使用 volc-tls skill 查看 provisioner 日志，确认有 stale 检测和重新 provision 的日志

---

### E2：LoRA pod 消失后重启

**步骤**：
1. LoRA 任务运行中，删除 trainer pod。
2. 下次启动新的 LoRA 任务（同 base_model）。

**预期行为**：
- [ ] `checkExistingLoRATrainer` 检测到 pod 不健康
- [ ] 重新触发 provision（不复用已死的 pod）
- [ ] 新 pod 正常创建

---

### E3：不存在的 base_model 启动

**步骤**：
```bash
curl -X POST http://<WEAVER_SERVER>/api/v1/sessions/<session_id>/models \
  -H "X-WEAVER-API-KEY: <key>" \
  -d '{"base_model": "nonexistent/model-xyz", "training_mode": "full_ft"}'
```

**预期行为**：
- [ ] 返回 4xx 错误（不是 500）
- [ ] **不产生悬空的 instance 记录**（数据库中无孤立记录）

---

### E4：full_ft + LoRA 同 base_model 混合并发

**步骤**：
1. Terminal A 启动 full_ft：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_fullft.py 2>&1 | tee /tmp/test_e4a.log
   ```
2. Terminal B 同时启动 LoRA：
   ```bash
   python3 /gpfs/users/sunpeng/code/bp/NexWeave/weaver/examples/pig_latin_lora.py 2>&1 | tee /tmp/test_e4b.log
   ```

**预期行为**：
- [ ] full_ft 产生 `sunpeng-trainer-full_ft-<model_id>` 和 `fullft-<model_id>`
- [ ] LoRA 产生 `trainer-lora-weaver-<base_model>` 和 `weaver-<base_model>`
- [ ] 两组 pod 互不干扰，各自正常运行
- [ ] 命名完全不同，无混淆

---

## 问题排查流程（全局）

任何 case 出现异常时，按以下顺序排查：

1. **查看脚本日志**：`/tmp/test_*.log`
2. **查 pod 状态**（infrawaves skill，按 model_id）
3. **查 provisioner 日志**（volc-tls skill）：
   - 关键词：`provisioning`, `terminate`, `skip`, `debug`, `stale`, `lora dedup`
4. **查 weaver-server 代码**：
   - auto-provision 逻辑：`internal/services/instance_orchestrator.go`
   - LoRA 去重：`checkExistingLoRATrainer()`
   - debug mode：`extractDebugMode()`, `provisionNewTrainer()`
   - terminate：`HandleTerminate()` → `getTrainingMode() == "lora"` 时跳过
5. **定位问题后**：在 `china-qijizhifeng/weaver-server` 创建 bug issue，关联 feedback repo

---

## 测试完成标准

| 分组 | Case 数 | 全通条件 |
|------|---------|----------|
| Full FT 基础 | F1, F2 | pod 正确命名、loss 下降、自动终止 |
| Full FT 并发 | F3, F4 | 所有 pod 独立存在、无干扰 |
| LoRA | L1~L4 | 共享去重正确、命名符合规范 |
| Debug Auto | DA1~DA4 | 首次正常、二次跳过、并发不重复 |
| Debug Manual | DM1~DM3 | create_model 不触发 provision、forward_backward 触发 provision、sleep infinity、跳过复用 |
| 边界异常 | E1~E4 | 错误处理正确、无数据污染 |
