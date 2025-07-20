# PLE-PyTorch

这是一个基于 PyTorch 实现的 **Progressive Layered Extraction (PLE)** 模型。

实现主要参考了 2020 年 RecSys 的论文: *[Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://doi.org/10.1145/3383313.3412236)*。

## 实现的功能

* 实现了 `PLE` 模型的完整核心架构，包括多层的专家网络和门控机制。
* **支持为不同任务配置不同数量的专属专家**，比原始论文的设定更具灵活性。
* 包含了 `Expert`, `Tower`, `Gate` 等模块化组件。

## 快速使用

下面是如何实例化一个两层、双任务的 `PLE` 模型的简单示例。

```python
from PLE import PLE

# --- 模型超参数 ---
# 双任务场景
num_tasks = 2
# 共享专家数量
num_shared_experts = 4
# 任务专属专家数量列表：任务0有3个专属专家，任务1有2个
num_task_experts_list = [3, 2] 

# --- 实例化模型 ---
model = PLE(
    input_dim=128,
    num_tasks=num_tasks,
    expert_dim=32,
    tower_dim=16,
    num_shared_experts=num_shared_experts,
    num_task_experts_list=num_task_experts_list,
    num_layers=2
)

# 打印模型结构
print(model)
```

## 运行验证
项目内包含一个 test_ple.py 脚本，用于快速验证模型结构是否正确、能否正常完成前向和反向传播。
```bash
python test_ple.py
```