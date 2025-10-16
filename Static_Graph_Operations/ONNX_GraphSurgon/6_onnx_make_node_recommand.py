import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
import numpy as np
from common import enter_workspace
enter_workspace()

# 创建一个输入和输出张量的 TensorValueInfo
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 5, 5])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 5, 5])

# --- 2. 创建权重和偏置的 NumPy 数组 ---
# 创建一个 3x3 的卷积核，输入通道为1，输出通道为1
# 形状为 (C_out, C_in, Kernel_H, Kernel_W)
conv_weight_data = np.array([
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
], dtype=np.float32)

conv_bias_data = np.array([0.0], dtype=np.float32)

# --- 3. 将 NumPy 数组转换为 Initializer (TensorProto) ---
# 使用 from_array 创建 initializer，名字必须和下面 make_node 的 inputs 对应！
conv_weight_initializer = numpy_helper.from_array(conv_weight_data, name='conv1.weight')
conv_bias_initializer = numpy_helper.from_array(conv_bias_data, name='conv1.bias')

# 创建一个卷积操作节点,虽然有conv骨架，但是没有conv提供权重；
conv_node = helper.make_node(
    op_type='Conv',              # 操作类型
    inputs=['input', 'conv1.weight', 'conv1.bias'],    # 输入张量
    outputs=['output'],  # 输出张量
    kernel_shape=[3, 3], # 卷积核大小
    strides=[1, 1],      # 步长
    pads=[1, 1, 1, 1],   # 填充
)

# 创建图,推荐使用make_graph而不是GraphProto,因为make_graph会自动添加初始化器
graph = helper.make_graph(
    nodes=[conv_node],          # 操作节点列表
    name='simple_conv_graph',  # 图名称
    inputs=[input_tensor],       # 输入张量列表
    outputs=[output_tensor],      # 输出张量列表
    initializer=[conv_weight_initializer, conv_bias_initializer], # 初始化器列表
)

# 创建模型
model = helper.make_model(graph)

# 保存模型
onnx.save(model, "simple_conv_model.onnx")
print("Model saved as 'simple_conv_model.onnx'")
