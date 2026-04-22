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

## 交互式教程

参见 [`examples/weaver_walkthrough.ipynb`](examples/weaver_walkthrough.ipynb)，通过 Pig Latin 翻译任务演示完整的 Weaver SDK 工作流，包括：

- 数据准备与 token 级别的 Datum 构造
- LoRA 训练与全量微调
- 采样推理
- Checkpoint 保存与恢复

## 配合 NexRL 进行 RL 训练

对于强化学习场景，推荐将 Weaver 与 [NexRL](https://github.com/nex-agi/NexRL) 的 **training-service** 模式配合使用。NexRL 负责编排完整的 RL 流程（rollout、轨迹收集、策略更新），Weaver 提供底层的训练和推理服务，实现端到端的 RL 训练而无需直接管理 GPU 资源。

## 配合 MetaClaw 进行 OpenClaw 自主学习

[MetaClaw](https://github.com/aiming-lab/MetaClaw) 已接入 Weaver 作为 RL 训练后端，用于 OpenClaw 自主学习场景。MetaClaw 将每次真实对话转化为学习信号——通过设置 `rl.backend=weaver`，即可使用 Weaver 进行云端 LoRA 训练，让你的个人智能体在使用中持续进化，无需本地 GPU。

## 深入了解

更多技术细节请参考 [Deep Dive into Weaver](https://dawning-road.github.io/blog/deep-dive-weaver)。

## 完整示例

参见 `weaver/examples/pig_latin.py`。
