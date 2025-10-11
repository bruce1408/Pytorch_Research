import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(x)    # 模块式
        x = F.relu(x)       # 函数式（没有 fx 很难统一替换）
        return x

m = M().eval()
gm = fx.symbolic_trace(m)


print("="*20, "BEFORE TRANSFORMATION", "="*20)
print("--- Graph Representation ---")
gm.graph.print_tabular()
print("\n--- Python Code ---")
print(gm.code)


for n in gm.graph.nodes:
    # A) 模块式 ReLU → 改写成 SiLU 模块（替换目标指针）
    if n.op == "call_module" and isinstance(gm.get_submodule(n.target), nn.ReLU):
        name = f"{n.target}_silu"; gm.add_submodule(name, nn.SiLU()); n.target = name
    
    # B) 函数式 ReLU → 改写成 nn.SiLU 模块调用（新增、插入节点）
    if n.op == "call_function" and n.target is F.relu:
        # 把 F.relu 改写成 nn.SiLU 模块调用
        name = f"silu_{n.name}"; gm.add_submodule(name, nn.SiLU())
        with gm.graph.inserting_after(n):
            new_node = gm.graph.call_module(name, args=n.args, kwargs={})
        n.replace_all_uses_with(new_node); gm.graph.erase_node(n)

gm.recompile()

print("\n" + "="*20, "AFTER TRANSFORMATION", "="*20)
print("--- Graph Representation ---")
gm.graph.print_tabular()
print("\n--- Python Code ---")
print(gm.code)
