import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from common import enter_workspace

enter_workspace()

def export_onnx(gm, onnx_file_path):

    dummy_input = torch.randn(1, 4)

    # onnx_file_path = "transformed_model.onnx"

    # c) 调用 torch.onnx.export() 函数
    print(f"\nExporting model to {onnx_file_path}...")
    torch.onnx.export(
        gm,                         # 你要导出的模型 (我们修改后的 GraphModule)
        (dummy_input,),             # 虚拟输入
        onnx_file_path,             # 导出的文件路径和名称
        verbose=False,              # 打印详细的导出信息
        input_names=['input'],      # 为输入节点命名，方便后续使用
        output_names=['output']     # 为输出节点命名
    )
    print("Model exported successfully!")

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(x)    # 模块式
        x = F.relu(x)       # 函数式（没有 fx 很难统一替换）
        x = F.relu(x)       # 函数式（没有 fx 很难统一替换）
        return x

m = M().eval()

export_onnx(m, "./before_modify_model.onnx")
gm = fx.symbolic_trace(m)


print("="*20, "BEFORE TRANSFORMATION", "="*20)
print("--- Graph Representation ---")
gm.graph.print_tabular()
print("\n--- Python Code ---")
print(gm.code)


for n in gm.graph.nodes:
    # A) 模块式 ReLU → 改写成 SiLU 模块（替换目标指针）
    # if n.op == "call_module" and isinstance(gm.get_submodule(n.target), nn.ReLU):
    if n.op == "call_module" and isinstance(n.target, str):
        submod = gm.get_submodule(n.target)
        
        # 因为都是module，可以直接替换，修改目标指针
        if isinstance(submod, nn.ReLU):
            name = f"{n.target}_silu"
            gm.add_submodule(name, nn.SiLU())
            n.target = name
    
    # B) 函数式 ReLU → 改写成 nn.SiLU 模块调用（新增、插入节点）
    if n.op == "call_function" and n.target is F.relu:
        
        # 把 F.relu 改写成 nn.SiLU 模块调用，但是函数无法替换模块，所以需要新增、插入节点
        name = f"silu_{n.name}"
        
        gm.add_submodule(name, nn.SiLU())
        
        # 插入一个新的节点
        with gm.graph.inserting_after(n):
            new_node = gm.graph.call_module(name, args=n.args, kwargs={})
            
        # 把原来的输入输出接入到新节点
        n.replace_all_uses_with(new_node)
        
        # 旧节点删除
        gm.graph.erase_node(n)


gm.recompile()

print("\n" + "="*20, "AFTER TRANSFORMATION", "="*20)
print("--- Graph Representation ---")
gm.graph.print_tabular()
print("\n--- Python Code ---")
print(gm.code)

export_onnx(gm, "./after_modify_model.onnx")



