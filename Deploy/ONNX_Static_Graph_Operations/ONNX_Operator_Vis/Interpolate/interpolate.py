import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import get_onnx_operator_output_path

import torch.onnx
import torch
import torch.nn as nn

# interpolate 没有缩放因子的话，就会有unsqueeze这个算子产生
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
    def forward(self, a1):
        print(a1.shape)
        
        # 没有scale_factor会有unsqueeze这个算子
        # output = torch.nn.functional.interpolate(a1, size=a1.shape[2:], mode='nearest', align_corners=None)

        # 加上scale_factor
        output = torch.nn.functional.interpolate(a1, scale_factor=(2, 2), mode='nearest', align_corners=None)
        
        print(output.shape)
        return output



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 224, 224)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = get_onnx_operator_output_path("interpolate_with_scale_factor.onnx", "onnx_operators")
torch.onnx.export(model, (input_data), onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
