import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import get_onnx_operator_output_path

import os
import torch
import torch.nn as nn
import torch.onnx as onnx


# 可有可无，不是重点。定义一个简单的PyTorch模型，可以换成你自己的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.unfold = nn.Unfold((2,2), stride=2)
        
    def forward(self, x):
        print(x.shape)
        x  = self.unfold(x)
        print(x.shape)
        
        return x



x = torch.Tensor([[[[  1,  2,  3,  4],
   					[  5,  6,  7,  8],
   					[  9, 10, 11, 12],
   					[ 13, 14, 15, 16]]]])


# 创建模型实例
model = SimpleModel()

# 导出模型到ONNX格式
onnx_path = get_onnx_operator_output_path("convtranspose2d.onnx", "onnx_operators")
torch.onnx.export(model, x, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)

