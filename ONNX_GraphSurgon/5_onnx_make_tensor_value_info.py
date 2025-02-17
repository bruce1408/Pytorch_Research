import onnx
from onnx import helper
from onnx import TensorProto
import onnx
from onnx.helper import make_tensor_value_info as mtvi
from onnx import GraphProto, NodeProto

# 它用于创建 TensorValueInfo 对象，该对象用于定义张量的数据类型和形状。通常在创建 ONNX 模型时，会使用这个函数来描述输入、输出或中间张量的属性
def demo_1_create_tensor_value_info_example():
    # 定义张量的名称、数据类型和形状
    tensor_name = "input_tensor"
    tensor_type = TensorProto.FLOAT  # 数据类型为浮点型
    tensor_shape = [1, 3, 224, 224]  # 形状为 (1, 3, 224, 224)，通常用于图像数据

    # 使用 make_tensor_value_info 创建 ValueInfoProto 对象
    tensor_value_info = helper.make_tensor_value_info(
        name=tensor_name,
        elem_type=tensor_type,
        shape=tensor_shape
    )

    # 打印生成的 ValueInfoProto 对象
    print(tensor_value_info)



def demo_2_create_tensor_value_info_example():
    
    # 创建输入张量的 TensorValueInfo
    input_tensor = mtvi("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])  # name, type, shape

    # 创建输出张量的 TensorValueInfo
    output_tensor = mtvi("output", onnx.TensorProto.FLOAT, [1, 1000])  # name, type, shape

    # 创建一个节点
    node = NodeProto(
        name="MyNode",
        op_type="Identity",  # 使用一个简单的Identity操作
        input=["input"],  # 输入
        output=["output"],  # 输出
    )

    # 创建图
    graph = GraphProto(
        name="SimpleGraph",
        node=[node],  # 使用上面的节点
        input=[input_tensor],  # 输入张量
        output=[output_tensor],  # 输出张量
    )

    # 创建一个模型
    model = onnx.helper.make_model(graph)

    # 打印模型信息
    onnx.save(model, "simple_model.onnx")
    print("Model saved as 'simple_model.onnx'")





# 调用示例函数
demo_1_create_tensor_value_info_example()

demo_2_create_tensor_value_info_example()