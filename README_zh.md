# Weaver Python SDK

[![PyPI version](https://img.shields.io/pypi/v/nex-weaver)](https://pypi.org/project/nex-weaver/)
[![Python](https://img.shields.io/pypi/pyversions/nex-weaver)](https://pypi.org/project/nex-weaver/)
[![CI](https://github.com/nex-agi/weaver/actions/workflows/ci.yml/badge.svg)](https://github.com/nex-agi/weaver/actions/workflows/ci.yml)

[English](README.md) | 中文

NexWeave Weaver 服务端的 Python 客户端。SDK 封装了 `weaver-server` 暴露的 REST API，提供训练、采样、遥测和运维管理的便捷接口。

## 安装

```bash
pip install nex-weaver
```

## 配置

通过关键字参数或环境变量配置：
- `WEAVER_API_KEY`
- `WEAVER_ORGANIZATION_ID` / `WEAVER_PROJECT_ID`：规范 UUID
- `WEAVER_ORGANIZATION` / `WEAVER_PROJECT`：UUID、slug 或展示名称

规范 UUID 的优先级更高。组织和项目均为空时，服务端继续使用稳定的个人组织和默认项目回退逻辑。

### 训练张量传输

训练张量默认使用 inline JSON，以保持现有客户端的行为不变。要启用压缩的二进制传输，配置 service client：

```python
from weaver import ServiceClient

with ServiceClient(
    tensor_transport="http-binary",
    tensor_compression="zstd",  # 可省略：二进制 tensor pack 默认使用 Zstandard。
) as client:
    ...
```

`AsyncServiceClient` 支持相同的参数。可运行的 Pig Latin 示例也提供了对应的命令行参数：

```bash
python examples/pig_latin.py \
  --tensor-transport http-binary \
  --tensor-compression zstd
```

所有 client 同样支持环境变量，因此也可以不传命令行参数来配置同一个示例：

```bash
WEAVER_TENSOR_TRANSPORT=http-binary \
WEAVER_TENSOR_COMPRESSION=zstd \
python examples/pig_latin.py
```

| Transport | Compression | 行为 |
| --- | --- | --- |
| `default` | 忽略 | 兼容旧行为的 inline JSON（默认值） |
| `http-binary` | `raw` | 不压缩的二进制 tensor pack |
| `http-binary` | `zstd` | 使用 Zstandard 压缩的二进制 tensor pack |

`tensor_compression`（或 `WEAVER_TENSOR_COMPRESSION`）仅在 transport 为 `http-binary` 时生效。对于 `cross_entropy` 请求，SDK 只会将符合条件的稠密输入张量移入二进制 tensor pack；控制元数据和其他值仍使用 JSON。操作返回输出张量时，SDK 会自动下载并将其还原成原有的公开返回结构。因此无需修改 `Datum` 构造、训练调用或结果处理代码。连接到不支持二进制 tensor pack 的旧版 Weaver server/trainer 部署时，请继续使用 `default` transport。

## 快速开始

```python
from weaver import ServiceClient

def main():
    with ServiceClient() as client:
        session = client.ensure_session()
        print(session)

if __name__ == "__main__":
    main()
```

可以使用易读的引用选择范围，也可以通过 CLI 查询和解析：

```python
with ServiceClient(organization="research", project="alignment") as client:
    client.ensure_session()
```

```bash
weaver organizations list
weaver projects list --organization research
weaver scope resolve --organization research --project alignment
```

组织 slug 全局唯一；项目 slug 和名称在组织内唯一。展示名称存在歧义时会明确报错，不会猜测选择。

### 支持的模型与训练模式

为保持兼容，原调用仍返回模型名称；选择 LoRA 或 Full-FT 时可以请求包含两种独立价格的详情：

```python
models = client.list_supported_models(detailed=True)
for model in models:
    for mode in model.training_modes:
        print(model.name, mode.display_name, mode.prices)
```

`training_modes` 只包含模型明确支持且四项有效价格齐全的模式。缺少 Full-FT
报价就表示不支持 Full-FT；SDK 不会自行展示 10× 的估算价格。

CLI 默认让每种模式占一行，并紧凑展示训练、输入、缓存输入、输出四列价格；也可以只看一种模式、只输出名称，或输出稳定 JSON：

```bash
weaver list supported-models
weaver list supported-models --mode full-ft
weaver list supported-models --format names
weaver list supported-models --format json
```

### 托管数据集

托管数据集只暴露当前调用方有权使用的目录 metadata 和稳定的样本引用。每个版本会
标注 `content_visibility`；`protected` 响应隐藏 token 身份，`public` 响应可以保留真实
token。第一期对两种可见性都只支持 SFT。用户明确选择 dataset `name` 和 `version`，并在
整样本 packing 前通过绑定模型的 training client 查询有效长度：

```python
from weaver import ServiceClient
from weaver.types import Datum, SampleRef

with ServiceClient() as service:
    service.ensure_session()
    for dataset in service.datasets.list(compatible_model="Qwen/Qwen3-8B"):
        print(dataset.name, dataset.version, dataset.sample_count)

    ref = SampleRef(dataset="hq-math", version="2026-08", sample_idx=42)
    trainer = service.create_model(
        base_model="Qwen/Qwen3-8B",
        training_max_sequence_length=4096,
    )
    length = trainer.resolve_sample_ref_lengths([ref])[0].input_token_count
    print(length)

    datum = Datum.from_sample_ref(
        dataset=ref.dataset,
        version=ref.version,
        sample_idx=ref.sample_idx,
        datum_id="batch-7-item-3",
    )
    result = trainer.forward_backward([datum], "cross_entropy")
```

模型创建时会固定 tokenizer、chat template 和完整的 shift 前 token 上限，client 不能按
sample 覆盖。`input_token_count` 是 autoregressive shift 后的有效训练长度（小于固定的完整
token 上限），可用于 client 侧整样本 packing。

无论 `content_visibility` 是 `protected` 还是 `public`，`SampleRef` 第一期都只能用于内置
`cross_entropy` 的 `forward_backward`；不能用于 `forward`、自定义或 surrogate loss、
`sample`、`compute_logprobs`，也不提供数据集下载。普通 token-in Datum 不受这些限制。
managed Datum 要求 `loss_fn_config`、client `loss_fn_inputs` 和逐 datum `metadata` 均为空，
并使用默认 JSON tensor transport；model input、target、loss mask 和 weights 均由 server
提供。受保护响应中的 token 身份数组会按真实长度替换为 `-8`，并拒绝
logprobs、elementwise loss 等依赖 label 的逐 token 字段；不要把 `-8` 再传入
`ModelInput` 或 `target_tokens`。公开响应可以保留真实 token，但请求侧仍遵循同一 SFT-only
边界。

## 使用方法

参见 [`examples/weaver_walkthrough.ipynb`](examples/weaver_walkthrough.ipynb)，通过 Pig Latin 翻译任务交互式演示完整的 SDK 工作流——涵盖数据准备、LoRA / 全量微调、采样推理和 checkpoint 管理。

完整可运行脚本参见 [`examples/pig_latin.py`](examples/pig_latin.py)。

### 生成控制（仅限全量微调）

RL 权重热切需要在新权重落地前让正在进行的生成停下来。采样客户端可以冻结所在的推理引擎，并在之后恢复：

```python
with sampling_client.paused(mode=PauseMode.ABORT):
    path = training_client.save_weights_for_sampler(name="step-42")
    new_client = service.create_sampling_client(
        model_path=path, model_id=model_id, base_model=base_model
    )
```

使用前需要知道两件事：

- **暂停作用于整个引擎，而不是单个 sampling session。** 它会冻结该模型引擎上*所有*正在进行的请求，包括通过更早的 sampling session 发出的那些。这正是它能用于权重热切的原因——需要打断的恰恰是上一个权重版本的请求——但也意味着暂停永远不可能只影响"我自己的请求"。
- **仅支持全量微调。** 全量微调模型拥有独占的引擎；LoRA adapter 则共用同一个底模引擎，在那里暂停会打断其他租户的生成，因此这类调用在发出请求之前就会被拒绝。

建议使用 `paused()` 而不是直接调用 `pause_generation()` / `continue_generation()`：暂停之后如果没有走到恢复那一步，引擎会一直冻结下去，服务端不会自动恢复。异步客户端对应的形式是 `async with`。

### HuggingFace 权重导出

checkpoint 以 trainer 原生的分布式格式保存。`export_weights()` 会把它转换成 HuggingFace 目录——全量微调得到完整模型，LoRA 得到 PEFT adapter——再用 `download_weights()` 下载到本地：

```python
artifact = training_client.export_weights()          # 保存当前权重并导出
artifact = training_client.export_weights(checkpoint=ckpt)  # 导出已有的 checkpoint

service.download_weights(artifact, "./hf-weights")
```

```bash
weaver checkpoint export weaver://<model>/checkpoints/step-42
weaver checkpoint download weaver://<model>/checkpoints/step-42 -o ./hf-weights
```

需要知道三件事：

- **导出必须显式发起，下载不会隐式触发导出。** 转换一个完整模型是分钟级的算力开销和几十 GB 的存储，因此在没有已完成产物时 `download_weights()` 会直接报错提示先执行 `export_weights`，而不是悄悄启动一次转换。
- **产物的过期时间独立于源 checkpoint**（默认 7 天，可用 `ttl_seconds` 调整）。删除源 checkpoint 不会连带删除产物，反之亦然。
- **LoRA 默认导出 adapter。** 传 `merge_adapter=True` 可以把它合并进底模、导出完整的 HF 模型；全量微调模型传这个参数会被拒绝。

下载是并行的，支持断点续传、过期 URL 自动刷新，并在落盘前校验每个文件的 sha256。两个方法都有对应的 `Async*` 版本。

### 部署 checkpoint

`deploy_checkpoint()` 把一个 checkpoint 上架成公开的 OpenAI 兼容端点：服务端会把它转换成 HuggingFace 格式，拉起一个独立的推理负载，再用你指定的名字注册到 NorthGate 网关。

```python
deployment = training_client.deploy_checkpoint(ckpt, name="my-chat-model")
print(deployment.endpoint)          # OpenAI 兼容的 URL

service.list_deployments()
service.get_deployment(deployment.id)
service.delete_deployment(deployment.id)
```

```bash
weaver deployment create weaver://<model>/checkpoints/step-42 --name my-chat-model
weaver deployment list
weaver deployment get <deployment-id>
weaver deployment delete <deployment-id>
```

需要知道四件事：

- **上架有权限门，且默认关闭。** `deployment.publish` 这个能力不按 Weaver 角色授予，而是按主体来源：SSO 会话天然满足；API key 只有在铸造时使用的 IAM `biz_code` 位于服务端白名单里才满足；service credential 一律不满足。功能未开启的服务端会返回 503。两种情况都会抛出 `WeaverAPIError`，并在报错信息里说明该改什么。列出、查看、删除自己的部署不需要这个能力——谁上架的端点，谁始终可以下架。
- **部署是独立且长期存在的。** 它不复用训练用的推理实例，生命周期长于训练会话，在你删除它之前会一直占着 GPU；同时它会钉住源 checkpoint 和导出产物，使其不被回收。
- **名字是全局的，且必须在所有下游都合法。** 它同时是 served model name、网关的 `model_name` 和 Kubernetes label：最长 63 个字符，只能包含字母、数字、`.`、`-`、`_`，且首尾必须是字母或数字。`overwrite=True` 只会替换网关上的同名注册，不会释放 Weaver 侧已被占用的名字。
- **整个过程需要几十分钟**，主要耗时在转换。传 `wait=False` 可以拿到 `OperationHandle` 而不阻塞。所有方法都有对应的 `Async*` 版本。

## 生态

[NexRL](https://github.com/nex-agi/NexRL) 是配套的 RL 训练框架。在其 **training-service** 模式下，NexRL 负责编排完整的 RL 流程（rollout、轨迹收集、策略更新），Weaver 提供底层的训练和推理服务。

## 应用案例

**OpenClaw 自主学习** —
[MetaClaw](https://github.com/aiming-lab/MetaClaw) 已接入 Weaver 作为 RL 训练后端。通过设置 `rl.backend=weaver`，MetaClaw 将每次真实对话转化为学习信号，使用 Weaver 进行云端 LoRA 训练，让个人智能体在使用中持续进化，无需本地 GPU。

## 深入了解

更多技术细节请参考 [Deep Dive into Weaver](https://dawning-road.github.io/blog/deep-dive-weaver)。
