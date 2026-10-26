1. make_tensor_value_info
用途：用于创建张量的信息，描述张量的名称、数据类型（如 FLOAT、INT32 等）和形状。
示例：

```python
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1000])
```

2. make_node
用途：用于创建一个操作节点（如卷积、矩阵乘法、加法等）。它表示 ONNX 图中的一个运算步骤，接受输入张量并生成输出张量。
示例：
```python
conv_node = helper.make_node(
    'Conv',
    inputs=['input'],
    outputs=['output'],
    kernel_shape=[3, 3],
    strides=[1, 1],
    pads=[1, 1, 1, 1]
)
```

3. make_graph
用途：用于创建一个 ONNX 计算图，包含节点、输入、输出以及图的名称等。
示例：
```python
graph = helper.make_graph(
    [conv_node],  # 图中的节点
    'simple_graph',  # 图的名称
    [input_tensor],  # 输入
    [output_tensor]  # 输出
)
```

4. make_model
用途：用于创建一个完整的 ONNX 模型。它使用已经创建的图对象，将其包装在模型中，并添加一些元数据（如版本号、作者等）。
示例：
```python
model = helper.make_model(graph)
```

5. make_tensor
用途：用于创建一个张量对象，通常用于初始化张量的值。这个函数会生成一个包含实际数据值的张量。它通常用于作为模型初始化时的常量张量（initializer）提供给节点。
示例：
```python

tensor = helper.make_tensor(
    name='weights',
    data_type=TensorProto.FLOAT,
    dims=[3, 3],
    vals=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
```

6. make_attribute
用途：用于创建一个属性对象，通常用于操作节点的配置。例如，卷积操作的步长、卷积核的大小等都可以通过这个属性来指定。
示例：
```python
kernel_shape_attr = helper.make_attribute('kernel_shape', [3, 3])
```

7. make_onnx_tensor
用途：用于创建一个张量（Tensor）并填充其数值。make_onnx_tensor 生成的是一个完全初始化的张量，常用于模型的常量初始化或加载数据。
示例：
```python
tensor = helper.make_onnx_tensor(
    name="tensor_name",
    dtype=TensorProto.FLOAT,
    shape=[2, 2],
    data=[1.0, 2.0, 3.0, 4.0]
)
```

8. make_model_proto
用途：创建一个 ONNX 模型协议对象（ModelProto），这是构建 ONNX 模型的底层结构。它通常用于将模型对象序列化为文件。
示例：
```python
model_proto = helper.make_model_proto(
    graph=graph,
    producer_name="my_model_creator",
    producer_version="1.0.0",
    domain="my_domain",
    model_version=1
)
```


- make_tensor_value_info：创建张量的元数据信息，描述其名称、类型和形状。
- make_node：创建操作节点（如 Conv、Add 等）。
- make_graph：创建一个 ONNX 计算图，包含节点、输入、输出等。
- make_model：创建完整的 ONNX 模型。
- make_tensor：创建包含数据的张量，常用于初始化。
- make_attribute：创建操作节点的属性，如卷积操作的步长、卷积核大小等。
- make_onnx_tensor：创建带有实际数据的张量，通常用于常量初始化。
- make_model_proto：创建一个模型协议对象，通常用于序列化和保存模型。