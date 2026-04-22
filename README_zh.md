# Weaver Python SDK

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

## 深入了解

更多技术细节请参考 [Deep Dive into Weaver](https://dawning-road.github.io/blog/deep-dive-weaver)。

## 完整示例

参见 `weaver/examples/pig_latin.py`。
