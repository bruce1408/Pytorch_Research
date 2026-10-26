import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper
from common import enter_workspace
enter_workspace()

# 1) 构造两个布尔常量（Constant 节点）
A_val = np.array([[True,  False],
                  [True,  True ]], dtype=bool)

B_val = np.array([[False, True ],
                  [True,  False]], dtype=bool)

A_tensor = numpy_helper.from_array(A_val, name="A_const")
B_tensor = numpy_helper.from_array(B_val, name="B_const")

# Constant 节点通过属性 value 携带 TensorProto
constA = helper.make_node(
    "Constant",
    inputs=[],
    outputs=["A"],          # 这个名字会被后续节点引用
    value=A_tensor
)

constB = helper.make_node(
    "Constant",
    inputs=[],
    outputs=["B"],
    value=B_tensor
)

# 2) And 节点做按元素逻辑与
and_node = helper.make_node(
    "And",
    inputs=["A", "B"],
    outputs=["Y"]
)

output = helper.make_tensor_value_info("Y", TensorProto.BOOL, [2, 2])

# 3) 组图
graph = helper.make_graph(
    nodes=[constA, constB, and_node],
    name="ConstAndGraph",
    inputs=[],              # 全图没有外部输入
    outputs=[output]
)

# 4) 生成模型（注意 opset 版本要支持 And/Constant，常见 >= 13 即可）
opset = helper.make_operatorsetid("", 13)
model = helper.make_model(graph, opset_imports=[opset])
onnx.checker.check_model(model)

onnx.save(model, "const_and.onnx")
print("保存到 const_and.onnx")
