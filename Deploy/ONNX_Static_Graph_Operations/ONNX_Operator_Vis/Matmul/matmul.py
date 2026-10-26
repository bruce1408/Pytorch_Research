import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import get_onnx_operator_output_dir

import torch.onnx as onnx
import torch
import torch.nn as nn
import os

# 创建一个包含torch.matmul操作的PyTorch模型, 广播乘法
class MatMulModel(nn.Module):
    def __init__(self):
        super(MatMulModel, self).__init__()

    def forward(self, x, y):
        output = torch.matmul(x, y)
        # FP16 MatMul 的结果再转为 FP32；ONNX 对应 MatMul -> Cast。
        # 这只改变输出类型，不能恢复 FP16 结果已经丢失的精度。
        output = output.to(torch.float32)
        print(output.shape)
        return output
# 创建模型实例
model = MatMulModel()

# 创建 FP16 示例输入张量
input_data1 = torch.randn(1, 3, 4, dtype=torch.float16)  # 输入1形状为(1, 3, 4)
input_data2 = torch.randn(1, 4, 2, dtype=torch.float16)  # 输入2形状为(1, 4, 2)

# 导出模型到ONNX格式


if __name__ == "__main__":
    # 先运行一次，查看 FP16 输入、FP32 输出的类型、形状和数值。
    model.eval()
    with torch.no_grad():
        output = model(input_data1, input_data2)
    print(f"输入1: dtype={input_data1.dtype}, shape={tuple(input_data1.shape)}")
    print(f"输入2: dtype={input_data2.dtype}, shape={tuple(input_data2.shape)}")
    print(f"输出: dtype={output.dtype}, shape={tuple(output.shape)}")
    print("输出数值:\n", output)

    # 导出模型为 ONNX 格式。
    model_dir = get_onnx_operator_output_dir("onnx_operators")
    onnx_file_path = "matmul.onnx"
    output_path = os.path.join(model_dir, onnx_file_path)
    onnx.export(
        model,
        (input_data1, input_data2),
        output_path,
        opset_version=18,
        verbose=True)
    print("ONNX模型已导出到:", output_path)
