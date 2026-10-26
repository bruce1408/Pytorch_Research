import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import get_onnx_operator_output_path

import torch.onnx
import torch
import torch.nn as nn

# 创建一个包含unfold操作的PyTorch模型
class UnfoldModel(nn.Module):
    def __init__(self):
        super(UnfoldModel, self).__init__()
        self.unfold = nn.Unfold((2, 2), stride=2)

    def forward(self, x):
        return self.unfold(x)

# 创建模型实例
model = UnfoldModel()

# 创建示例输入张量
input_data = torch.randn(1, 1, 5, 5)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
# onnx_path = "unfold_model.onnx"
onnx_path = get_onnx_operator_output_path("unfold.onnx", "onnx_operators")
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
