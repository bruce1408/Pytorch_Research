import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import get_onnx_operator_output_path

import torch.onnx
import torch
import torch.nn as nn

# 创建一个包含ConvTranspose2d操作的PyTorch模型
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3)

    def forward(self, x):
        output = self.conv_transpose(x)
        print(output.shape)
        return output
        
# 创建模型实例
model = ConvTranspose2dModel()

for name, module in model.named_modules():
    if hasattr(module, "weight") and isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
        print(name)
        
# 创建示例输入张量
input_data = torch.randn(1, 3, 224, 224)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = get_onnx_operator_output_path("convtranspose2d.onnx", "onnx_operators")
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
