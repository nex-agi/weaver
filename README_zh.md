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

## 生态

[NexRL](https://github.com/nex-agi/NexRL) 是配套的 RL 训练框架。在其 **training-service** 模式下，NexRL 负责编排完整的 RL 流程（rollout、轨迹收集、策略更新），Weaver 提供底层的训练和推理服务。

## 应用案例

**OpenClaw 自主学习** —
[MetaClaw](https://github.com/aiming-lab/MetaClaw) 已接入 Weaver 作为 RL 训练后端。通过设置 `rl.backend=weaver`，MetaClaw 将每次真实对话转化为学习信号，使用 Weaver 进行云端 LoRA 训练，让个人智能体在使用中持续进化，无需本地 GPU。

## 深入了解

更多技术细节请参考 [Deep Dive into Weaver](https://dawning-road.github.io/blog/deep-dive-weaver)。
