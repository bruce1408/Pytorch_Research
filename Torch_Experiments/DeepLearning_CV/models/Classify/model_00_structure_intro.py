import torch
import torch.nn as nn
import torch.onnx
import netron
from torchvision.models import resnet50, resnet18
import os


def inspect_model_structure(model):
    """
    打印和检查一个 PyTorch 模型的内部结构、子模块和状态字典。
    """
    print("--- 1. 打印完整的模型结构 ---")
    print(model)
    print("\n" + "="*100 + "\n")

    print("--- 2. 打印模型的子模块 (不包括最后一层) ---")
    # 这在迁移学习中常用于获取特征提取器
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    print(feature_extractor)
    print("\n" + "="*100 + "\n")

    print("--- 3. 打印模型状态字典的键 (所有可学习参数的名称) ---")
    print(list(model.state_dict().keys()))
    print("\n" + "="*100 + "\n")

    # 创建一个模拟输入来测试模型的前向传播
    # ResNet 通常需要 224x224 的输入
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"--- 4. 模型输出的形状 ---")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")


def create_and_visualize_model(model, dummy_input, onnx_filename="temp_model.onnx"):
    """
    将一个 PyTorch 模型导出为 ONNX 格式，并使用 Netron 进行可视化。
    适用于你刚刚在代码里定义好一个模型的情况。
    """
    print(f"\n--- 正在将模型导出到 {onnx_filename} 并用 Netron 打开 ---")
    # 将模型导出为 ONNX 格式
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        verbose=False,  # 设置为 True 可以看到详细的导出过程
        input_names=['input'],    # 为输入和输出命名，在 Netron 中会显示
        output_names=['output']
    )
    print(f"模型已成功导出。")
    
    # 使用 Netron 启动可视化
    # 如果文件已经存在，Netron 会直接打开它
    if os.path.exists(onnx_filename):
        # netron.start(onnx_filename)
        print(f"Netron 已在浏览器中启动，用于可视化 {onnx_filename}。")
    else:
        print(f"错误: ONNX 文件 {onnx_filename} 未能创建。")


if __name__ == '__main__':
    # --- 任务一：深入检查一个预训练的 ResNet50 模型 ---
    print("========== 任务一：检查 ResNet50 模型结构 ==========")
    pretrained_model = resnet50(pretrained=True)
    inspect_model_structure(pretrained_model)

    # --- 任务二：可视化一个我们自己创建的 ResNet18 模型 ---
    print("\n\n========== 任务二：创建并可视化 ResNet18 模型 ==========")
    # 1. 实例化一个模型 (这里以 ResNet18 为例)
    my_net = resnet18(pretrained=False) # 我们不需要预训练权重，因为只关心结构

    # 2. 准备一个符合模型输入尺寸的模拟输入张量
    # ResNet18 和 ResNet50 一样，通常处理 224x224 的输入
    # 我们用批量大小为 1 来模拟
    mock_input = torch.randn(1, 3, 224, 224)
    
    # 3. 指定导出的 ONNX 文件名
    onnx_path = "resnet18_structure.onnx"

    # 4. 调用函数，导出并用 Netron 可视化
    create_and_visualize_model(my_net, mock_input, onnx_path)
    
    # 你也可以直接用 netron.start("resnet18_structure.onnx") 来打开一个已经存在的文件