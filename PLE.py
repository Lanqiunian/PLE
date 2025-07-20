import torch
import torch.nn as nn

class Expert(nn.Module):
    """
    定义一个专家网络，使用MLP结构
    """
    def __init__(self, input_dim, expert_dim):
        super(Expert,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.net(x)
    
class Tower(nn.Module):
    """
    Tower网络
    接受融合后的Expert输出，为特定任务产出最后logits
    """
    def __init__(self,expert_dim,tower_dim,output_dim):
        super(Tower,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(expert_dim,tower_dim),
            nn.ReLU(),
            nn.Linear(tower_dim,output_dim)
        )
    def forward(self,x):
        return self.net(x)
    

class Gate(nn.Module):
    def __init__(self, input_dim,num_experts):
        super(Gate,self).__init__()
        self.gate_layer = nn.Linear(input_dim, num_experts)

    def forward(self,x):
        return nn.functional.softmax(self.gate_layer(x), dim=1)


class CGC(nn.Module):
    def __init__(self,input_dim, num_tasks,expert_dim, tower_dim, 
                 num_shared_experts,num_task_experts):
        super(CGC,self).__init__()
        self.num_tasks = num_tasks 
        #定于专属、任务专家网络
        self.shared_experts = nn.ModuleList([
            Expert(input_dim,expert_dim) for _ in range(num_shared_experts)
        ])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                Expert(input_dim,expert_dim) for _ in range(num_task_experts)
            ])
        ])
        #定义Gate
        self.gates = nn.ModuleList([
            Gate(input_dim,num_shared_experts + num_task_experts) for _ in range(num_tasks)
        ])
        #Towers
        self.towers = nn.ModuleList([
            Tower(expert_dim,tower_dim,1) for _ in range(num_tasks)
        ])

    def forward(self,x):
        # 1.计算所有experts的输出
        shared_expert_outputs = [expert(x) for expert in self.shared_experts]
        task_expert_outputs = [
            [expert(x) for expert in task_expert_group]
            for task_expert_group in self.task_experts
        ]
        # 2.对每个任务，融合experts 并送入 tower
        final_outputs = []
        for i in range(self.num_tasks):
            # 2.1 获取当前任务相关的experts输出
            # S^k(x)
            current_experts = shared_expert_outputs + task_expert_outputs[i]
            # (batch_size, num_experts, expert_dim)
            current_experts_tensor = torch.stack(current_experts, dim = 1)
            # 2.2 通过gate计算权重
            # w^k(x)
            gate_weights = self.gates[i](x)
            gate_weights = gate_weights.unsqueeze(-1)

            # 2.3 加权求和
            # g^k(x)
            # (batch_size, expert_dim)
            weighted_experts_output = torch.sum(current_experts_tensor * gate_weights,dim = 1)

            # 2.4 送入Tower
            tower_output = self.towers[i](weighted_experts_output)
            final_outputs.append(tower_output)

        return final_outputs
    
class PLE(nn.Module):
    def __init__(self, input_dim, num_tasks, expert_dim, tower_dim,
                 num_shared_experts, num_task_experts_list, num_layers=2):
        """
        PLE 初始化函数
        
        Args:
            input_dim (int): 输入特征的维度
            num_tasks (int): 任务数量
            expert_dim (int): 每个专家网络的输出维度
            tower_dim (int): 每个塔网络的隐藏层维度
            num_shared_experts (int): 共享专家的数量
            num_task_experts_list (list of int): 一个列表，长度等于num_tasks, 
                                                  其中第i个元素代表任务i的专属专家数量。
            num_layers (int): PLE的层数
        """
        super(PLE, self).__init__()

        # --- 断言检查，确保输入参数的正确性 ---
        assert num_tasks == len(num_task_experts_list), \
            "任务数量 (num_tasks) 必须与任务专家数量列表 (num_task_experts_list) 的长度相等"

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.num_shared_experts = num_shared_experts
        self.num_task_experts_list = num_task_experts_list

        # --- 第一层 ---
        self.first_layer_experts = nn.ModuleDict()
        self.first_layer_experts['shared'] = nn.ModuleList([
            Expert(input_dim, expert_dim) for _ in range(self.num_shared_experts)
        ])
        
        # 根据 num_task_experts_list 来创建不同数量的任务专家
        self.first_layer_experts['task'] = nn.ModuleList([
            nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(num_experts)])
            for num_experts in self.num_task_experts_list
        ])

        self.first_layer_gates = nn.ModuleDict()
        # 每个任务的门控网络，其输出维度根据该任务的专家数量动态计算
        self.first_layer_gates['task'] = nn.ModuleList([
             Gate(input_dim, self.num_shared_experts + num_experts) 
             for num_experts in self.num_task_experts_list
        ])
        
        # 共享门控网络的输出维度，是所有专家的总和
        total_experts_at_first_layer = self.num_shared_experts + sum(self.num_task_experts_list)
        self.first_layer_gates['shared'] = Gate(input_dim, total_experts_at_first_layer)

        # --- 中间层 ---
        self.middle_layers = nn.ModuleList()
        if num_layers > 1:
            for _ in range(num_layers - 1):
                layer = nn.ModuleDict()
                layer['shared_experts'] = nn.ModuleList([
                    Expert(expert_dim, expert_dim) for _ in range(self.num_shared_experts)
                ])
                layer['task_experts'] = nn.ModuleList([
                    nn.ModuleList([Expert(expert_dim, expert_dim) for _ in range(num_experts)])
                    for num_experts in self.num_task_experts_list
                ])
                layer['task_gates'] = nn.ModuleList([
                    Gate(expert_dim, self.num_shared_experts + num_experts) 
                    for num_experts in self.num_task_experts_list
                ])
                total_experts_at_middle_layer = self.num_shared_experts + sum(self.num_task_experts_list)
                layer['shared_gate'] = Gate(expert_dim, total_experts_at_middle_layer)

                self.middle_layers.append(layer)

        # --- Towers ---
        self.towers = nn.ModuleList([
            Tower(expert_dim, tower_dim, 1) for _ in range(num_tasks)
        ])


    def forward(self, x):
        # --- 第一层处理 ---
        shared_expert_outputs = [exp(x) for exp in self.first_layer_experts['shared']]
        task_expert_outputs = [[exp(x) for exp in group] for group in self.first_layer_experts['task']]

        task_gate_outputs = []
        for i in range(self.num_tasks):
            # experts_for_task 会自动包含 num_shared_experts + num_task_experts_list[i] 个专家
            experts_for_task = torch.stack(shared_expert_outputs + task_expert_outputs[i], dim=1)
            gate_weights = self.first_layer_gates['task'][i](x).unsqueeze(-1)
            task_gate_outputs.append(torch.sum(experts_for_task * gate_weights, dim=1))

        all_experts = torch.stack(shared_expert_outputs + [item for group in task_expert_outputs for item in group], dim=1)
        shared_gate_weights = self.first_layer_gates['shared'](x).unsqueeze(-1)
        shared_gate_output = torch.sum(all_experts * shared_gate_weights, dim=1)
        
        last_layer_gate_outputs = task_gate_outputs + [shared_gate_output]

        # --- 中间层处理 ---
        for layer in self.middle_layers:
            next_layer_gate_outputs = []
            
            shared_expert_outputs = [exp(last_layer_gate_outputs[-1]) for exp in layer['shared_experts']]
            task_expert_outputs = [[exp(last_layer_gate_outputs[i]) for exp in group] for i, group in enumerate(layer['task_experts'])]

            for i in range(self.num_tasks):
                selector = last_layer_gate_outputs[i]
                experts_for_task = torch.stack(shared_expert_outputs + task_expert_outputs[i], dim=1)
                gate_weights = layer['task_gates'][i](selector).unsqueeze(-1)
                next_layer_gate_outputs.append(torch.sum(experts_for_task * gate_weights, dim=1))

            selector = last_layer_gate_outputs[-1]
            all_experts = torch.stack(shared_expert_outputs + [item for group in task_expert_outputs for item in group], dim=1)
            shared_gate_weights = layer['shared_gate'](selector).unsqueeze(-1)
            next_layer_gate_outputs.append(torch.sum(all_experts * shared_gate_weights, dim=1))
            
            last_layer_gate_outputs = next_layer_gate_outputs

        # --- Towers ---
        final_outputs = []
        for i in range(self.num_tasks):
            tower_input = last_layer_gate_outputs[i]
            final_outputs.append(self.towers[i](tower_input))
            
        return final_outputs














        