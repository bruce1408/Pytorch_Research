import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import get_onnx_operator_output_path

import torch.onnx
import torch
import torch.nn as nn

# gather 就是按照索引去拿值
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.upsampe = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, a1):
            
        output = self.upsampe(a1)
        print(output.shape)
        return output



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)
# input_data2 = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = get_onnx_operator_output_path("unsample_scale_factor.onnx", "onnx_operators")
torch.onnx.export(model, (input_data), onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
