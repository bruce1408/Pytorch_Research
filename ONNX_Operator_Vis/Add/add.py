import torch.onnx
import torch
import torch.nn as nn

# gather 就是按照索引去拿值
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        
    def forward(self, a1, a2):
            
        output = torch.add(a1, a2)
        print(output.shape)
        return output



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)
input_data2 = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/add.onnx"
torch.onnx.export(model, (input_data, input_data2), onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
