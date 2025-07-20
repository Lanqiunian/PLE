# test_ple.py

import torch
import torch.nn as nn

# 从我们的PLE.py文件中导入所有需要的类
from PLE import PLE, Expert, Tower, Gate

def test_ple_model():
    """
    一个简单的测试函数，用于验证PLE模型的基本功能。
    """
    print("--- 开始测试PLE模型 ---")

    # 1. 定义测试用的超参数
    # =======================================================
    BATCH_SIZE = 4
    INPUT_DIM = 128
    EXPERT_DIM = 32
    TOWER_DIM = 16
    NUM_LAYERS = 2
    
    # 模拟一个3任务场景
    NUM_TASKS = 3
    
    # 共享专家数量
    NUM_SHARED_EXPERTS = 2
    
    # [关键] 测试我们灵活的专家数量配置
    # 任务0有4个专家, 任务1有2个专家, 任务2有3个专家
    NUM_TASK_EXPERTS_LIST = [4, 2, 3]
    
    print("测试配置:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Input Dim: {INPUT_DIM}")
    print(f"  Num Tasks: {NUM_TASKS}")
    print(f"  Num Shared Experts: {NUM_SHARED_EXPERTS}")
    print(f"  Num Task Experts List: {NUM_TASK_EXPERTS_LIST}")
    print(f"  Num PLE Layers: {NUM_LAYERS}")
    
    # 2. 验证模型能否成功实例化
    # =======================================================
    try:
        model = PLE(
            input_dim=INPUT_DIM,
            num_tasks=NUM_TASKS,
            expert_dim=EXPERT_DIM,
            tower_dim=TOWER_DIM,
            num_shared_experts=NUM_SHARED_EXPERTS,
            num_task_experts_list=NUM_TASK_EXPERTS_LIST,
            num_layers=NUM_LAYERS
        )
        print("\n[PASS] 1. 模型成功实例化。")
    except Exception as e:
        print(f"\n[FAIL] 1. 模型实例化失败: {e}")
        return

    # 3. 验证前向传播能否成功执行
    # =======================================================
    try:
        # 创建一个假的输入tensor
        dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM)
        # 执行前向传播
        outputs = model(dummy_input)
        print("[PASS] 2. 前向传播成功执行。")
    except Exception as e:
        print(f"[FAIL] 2. 前向传播失败: {e}")
        return

    # 4. 验证输出的形状和数量是否正确
    # =======================================================
    # 输出应该是一个列表，长度等于任务数量
    assert len(outputs) == NUM_TASKS, f"输出数量错误，应为{NUM_TASKS}，实际为{len(outputs)}"
    
    # 每个任务的输出形状应该是 (BATCH_SIZE, 1)
    for i, output in enumerate(outputs):
        expected_shape = (BATCH_SIZE, 1)
        assert output.shape == expected_shape, \
            f"任务{i}的输出形状错误，应为{expected_shape}，实际为{output.shape}"
            
    print("[PASS] 3. 模型输出的形状和数量正确。")

    # 5. 验证模型能否进行反向传播
    # =======================================================
    try:
        # 创建假的标签和损失函数
        dummy_labels = [torch.rand(BATCH_SIZE, 1) for _ in range(NUM_TASKS)]
        # 假设所有任务都是回归任务，使用MSELoss
        loss_fns = [nn.MSELoss() for _ in range(NUM_TASKS)]
        
        # 计算总损失
        total_loss = 0
        for i in range(NUM_TASKS):
            total_loss += loss_fns[i](outputs[i], dummy_labels[i])
            
        # 执行反向传播
        total_loss.backward()
        
        # 检查一个参数是否有梯度，以确认反向传播生效
        assert model.towers[0].net[0].weight.grad is not None
        
        print("[PASS] 4. 反向传播成功，梯度已计算。")

    except Exception as e:
        print(f"[FAIL] 4. 反向传播失败: {e}")
        return

    print("\n--- 所有基本测试通过！PLE模型结构验证成功。 ---")

if __name__ == "__main__":
    test_ple_model()