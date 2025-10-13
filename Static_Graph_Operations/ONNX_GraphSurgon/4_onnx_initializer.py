import onnx
from tabulate import tabulate
# 加载 ONNX 模型
onnx_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/resnet18.onnx"
model = onnx.load(onnx_path)

# 获取模型的 initializer 初始化值，即有权重参与计算的都可以是initializer
initializers = model.graph.initializer

initializers_list = {
    "name": [],
    "dims": [],
    "data_type": []
}

for init in initializers:
    initializers_list["name"].append(init.name)
    initializers_list["dims"].append(init.dims)
    initializers_list["data_type"].append(init.data_type)

print(tabulate(initializers_list, headers="keys", tablefmt="fancy_grid"))

print("="*30)
# exit(0)

node_list = {
    "name": [],
    "op_type": [],
    # "input": [],
    # "output": [],
    # "attribute": [],
    # "domain": [],
    # "doc_string": []
}
for node in model.graph.node:
    node_list["name"].append(node.name)
    node_list["op_type"].append(node.op_type)
    # node_list["input"].append(node.input)
    # node_list["output"].append(node.output)
    # node_list["attribute"].append(node.attribute)
    # node_list["domain"].append(node.domain)
    # node_list["doc_string"].append(node.doc_string)
    
print(tabulate(node_list, headers="keys", tablefmt="fancy_grid"))