import onnx
from onnx import helper
from onnx import TensorProto

# 创建一个输入和输出张量的 TensorValueInfo
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 112, 112])

# 创建一个卷积操作节点
conv_node = helper.make_node(
    'Conv',              # 操作类型
    inputs=['input'],    # 输入张量
    outputs=['output'],  # 输出张量
    kernel_shape=[3, 3], # 卷积核大小
    strides=[1, 1],      # 步长
    pads=[1, 1, 1, 1],   # 填充
)

# 创建图
graph = helper.make_graph(
    [conv_node],          # 操作节点列表
    'simple_conv_graph',  # 图名称
    [input_tensor],       # 输入张量列表
    [output_tensor],      # 输出张量列表
)

# 创建模型
model = helper.make_model(graph)

# 保存模型
onnx.save(model, "simple_conv_model.onnx")
print("Model saved as 'simple_conv_model.onnx'")
