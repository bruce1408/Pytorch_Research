import onnx

# 加载 ONNX 模型
model = onnx.load("/mnt/share_disk/bruce_trie/onnx_models/od_bev_1110.onnx")

# 获取模型的initializer
initializers = model.graph.initializer

for init in initializers:
    print(init.name)
    print(init.dims)  # 查看张量的形状
