import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
import os

# 尝试导入 thop，如果没有安装则跳过，避免报错
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' library not found. FLOPs calculation will be skipped.")

def get_model(device):
    """初始化模型并移动到指定设备"""
    # 新版 torchvision 推荐写法，替代 pretrained=True
    # 如果你想加载本地权重，请先初始化 weights=None，然后使用 load_state_dict
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) 
    model.to(device)
    return model

def print_model_summary(model, input_size=(3, 224, 224)):
    """打印模型层级结构"""
    print("-" * 20 + " Model Summary " + "-" * 20)
    summary(model, input_size)

def calc_flops_and_params(model, input_size=(1, 3, 224, 224)):
    """使用 thop 计算 FLOPs 和 参数量"""
    if not THOP_AVAILABLE:
        return
    
    print("-" * 20 + " THOP Calculation " + "-" * 20)
    # thop 需要输入在同一个 device 上
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_size).to(device)
    
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    print(f"MACs (计算量): {macs / 1e9:.2f} G (Giga Operations)")
    print(f"Params (参数量): {params / 1e6:.2f} M (Million)")

def manual_params_count(net):
    """手动统计参数量 (详细版)"""
    print("-" * 20 + " Manual Count " + "-" * 20)
    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    # 仅打印前5层作为示例，避免刷屏
    print("Printing shapes of first 5 layers:")
    for index, (name, p) in enumerate(net.named_parameters()):
        if index < 5:
            print(f' Layer {index}: {name} | Shape: {p.shape} | Params: {p.numel()}')
    
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {total_trainable_params:,}')
    return total_params, total_trainable_params

def load_checkpoint_demo2(pth_path):
    """
    演示如何加载 .pth 文件，并使用 tabulate 库漂亮地打印其内容。
    """
    print("-" * 20 + " Checkpoint Inspector " + "-" * 20)
    if not os.path.exists(pth_path):
        print(f"Error: Checkpoint file not found at {pth_path}")
        return

    # 动态导入 tabulate，如果失败则提供友好提示
    try:
        from tabulate import tabulate
    except ImportError:
        print("\nWarning: 'tabulate' is not installed. Run 'pip install tabulate' for formatted table output.")
        # 如果没有 tabulate，提供一个基础的回退打印方案
        def tabulate(table_data, headers, **kwargs):
            # 打印表头
            print(" | ".join(map(str, headers)))
            print("-" * (len(" | ".join(map(str, headers))) + 2))
            # 打印数据行
            for row in table_data:
                print(" | ".join(map(str, row)))
            return "" # 返回空字符串以匹配原函数的行为

    # 使用 map_location 确保模型可以在任何设备上加载
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    
    if not isinstance(checkpoint, dict):
        print(f"Loaded object is of type '{type(checkpoint).__name__}', not a dictionary.")
        print("This is likely a full model object, which is not the recommended format for saving.")
        return

    print(f"Inspecting checkpoint file: {os.path.basename(pth_path)}\n")

    # 准备用于 tabulate 的数据
    table_data = []
    for key, value in checkpoint.items():
        value_info = ""
        # 智能地判断值的类型并生成描述信息
        if key in ('state_dict', 'model', 'model_state_dict'):
            num_layers = len(value)
            first_layer_name = next(iter(value), "N/A")
            value_info = f"dict (Model State) with {num_layers} layers, starts with '{first_layer_name}'"
        elif key in ('optimizer', 'optimizer_state_dict'):
            value_info = f"dict (Optimizer State) with {len(value.get('state', {}))} tracked tensors"
        elif isinstance(value, torch.Tensor):
            value_info = f"Tensor, shape: {value.shape}, dtype: {value.dtype}"
        else:
            value_info = f"{value} (type: {type(value).__name__})"
        
        table_data.append([key, value_info])

    # 定义表头
    headers = ["Key", "Value / Type Description"]

    # 使用 tabulate 生成并打印表格
    # tablefmt="fancy_grid" 是一种美观的带内外框线的格式
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
            
            
            
            
            
            
def load_checkpoint_demo(pth_path):
    """演示如何加载 .pth 文件"""
    print("-" * 20 + " Checkpoint Loading " + "-" * 20)
    if not os.path.exists(pth_path):
        print(f"Error: File not found at {pth_path}")
        return

    # map_location 确保在没有 GPU 的机器上也能加载 GPU 训练的模型
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    
    print(f"Loaded object type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys in checkpoint: {checkpoint.keys()}")
        # 通常 checkpoint 包含 'model', 'optimizer', 'epoch' 等 key
        # 如果是纯参数字典，可以直接 load
        # model.load_state_dict(checkpoint) 
    else:
        print("Loaded object is likely a full model object (not recommended format).")

if __name__ == '__main__':
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化模型
    net = get_model(device)

    # 3. 查看结构 (TorchSummary)
    print_model_summary(net)

    # 4. 计算 FLOPs (THOP)
    calc_flops_and_params(net)

    # 5. 手动统计参数
    manual_params_count(net)

    # 6. 加载权重演示 (修改为你实际的文件路径)
    pthfile = '/home/bruce_ultra/workspace/Research_Experiments/outputs/resnet50-0676ba61.pth'
    load_checkpoint_demo2(pthfile)