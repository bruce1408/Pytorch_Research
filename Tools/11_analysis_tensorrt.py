import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime as ort
import os

class TensorRTLayerDebugger:
    def __init__(self, onnx_model_path, input_data, fp16_mode=True):
        """
        初始化调试器
        
        Args:
            onnx_model_path: ONNX模型路径
            input_data: 模型输入数据,numpy数组或字典(多输入)
            fp16_mode: 是否使用FP16模式
        """
        self.onnx_model_path = onnx_model_path
        self.input_data = input_data
        self.fp16_mode = fp16_mode
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        
        # 获取所有节点名称
        self.node_names = [node.name for node in self.graph.node]
        
    def compare_layer_outputs(self):
        """比较ONNX和TensorRT每层的输出,找出NaN值出现的位置"""
        # 1. 获取ONNX模型每层输出
        onnx_outputs = self._get_onnx_layer_outputs()
        
        # 2. 获取TensorRT模型每层输出
        trt_outputs = self._get_trt_layer_outputs()
        
        # 3. 比较输出并找出问题
        problematic_layers = []
        
        for name in trt_outputs:
            if name in onnx_outputs:
                onnx_out = onnx_outputs[name]
                trt_out = trt_outputs[name]
                
                # 检查NaN值
                if np.isnan(trt_out).any():
                    nan_percentage = np.isnan(trt_out).sum() / trt_out.size * 100
                    problematic_layers.append({
                        'layer_name': name,
                        'has_nan': True,
                        'nan_percentage': f"{nan_percentage:.2f}%",
                        'shape': trt_out.shape
                    })
                    print(f"问题层: {name} - 包含 {nan_percentage:.2f}% NaN值")
                
                # 检查数值差异
                else:
                    # 计算相对误差
                    abs_diff = np.abs(onnx_out - trt_out)
                    max_abs_diff = np.max(abs_diff)
                    mean_abs_diff = np.mean(abs_diff)
                    
                    if max_abs_diff > 1e-2:  # 设置一个阈值来检测显著差异
                        problematic_layers.append({
                            'layer_name': name,
                            'has_nan': False,
                            'max_diff': max_abs_diff,
                            'mean_diff': mean_abs_diff,
                            'shape': trt_out.shape
                        })
                        print(f"可能有问题的层: {name} - 最大绝对差异: {max_abs_diff}, 平均差异: {mean_abs_diff}")
        
        return problematic_layers
    
    def _get_onnx_layer_outputs(self):
        """获取ONNX模型每层的输出"""
        outputs = {}
        
        # 创建一个临时ONNX模型,将所有中间层设为输出
        temp_model = onnx.ModelProto()
        temp_model.CopyFrom(self.model)
        
        # 收集所有中间输出
        output_names = []
        for node in temp_model.graph.node:
            for output in node.output:
                if output:  # 有些节点可能没有输出
                    output_names.append(output)
        
        # 创建会话
        options = ort.SessionOptions()
        session = ort.InferenceSession(
            self.onnx_model_path, 
            providers=['CPUExecutionProvider'], 
            sess_options=options
        )
        
        # 准备输入
        input_feed = {}
        if isinstance(self.input_data, dict):
            input_feed = self.input_data
        else:
            # 假设只有一个输入
            input_name = session.get_inputs()[0].name
            input_feed = {input_name: self.input_data}
        
        # 逐层运行并获取输出
        for output_name in output_names:
            try:
                # 创建一个只输出当前层的会话
                temp_session = ort.InferenceSession(
                    self.onnx_model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=options
                )
                layer_output = temp_session.run([output_name], input_feed)
                outputs[output_name] = layer_output[0]
            except Exception as e:
                print(f"无法获取ONNX层 {output_name} 的输出: {e}")
        
        return outputs
    
    def _get_trt_layer_outputs(self):
        """获取TensorRT模型每层的输出"""
        outputs = {}
        
        # 创建TensorRT引擎
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(self.onnx_model_path, 'rb') as model:
            parser.parse(model.read())
        
        config = builder.create_builder_config()
        if self.fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # 启用逐层输出
        config.set_flag(trt.BuilderFlag.DEBUG)
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        context = engine.create_execution_context()
        
        # 分配内存
        inputs = []
        outputs = []
        bindings = []
        
        for i in range(engine.num_bindings):
            binding_dims = engine.get_binding_shape(i)
            binding_size = trt.volume(binding_dims) * np.dtype(np.float32).itemsize
            device_mem = cuda.mem_alloc(binding_size)
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(i):
                inputs.append({"index": i, "mem": device_mem, "size": binding_size})
            else:
                outputs.append({"index": i, "mem": device_mem, "size": binding_size})
        
        # 复制输入数据到GPU
        stream = cuda.Stream()
        if isinstance(self.input_data, dict):
            for i, (name, data) in enumerate(self.input_data.items()):
                cuda.memcpy_htod_async(inputs[i]["mem"], data.astype(np.float32), stream)
        else:
            cuda.memcpy_htod_async(inputs[0]["mem"], self.input_data.astype(np.float32), stream)
        
        # 执行推理
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # 获取输出
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            
            # 为每个层创建输出绑定
            for j in range(layer.num_outputs):
                output_tensor = layer.get_output(j)
                output_name = f"{layer_name}_output_{j}"
                
                # 获取输出形状
                output_shape = tuple(output_tensor.shape)
                output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
                
                # 分配主机和设备内存
                host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
                device_output = cuda.mem_alloc(output_size)
                
                # 执行推理并获取输出
                context.execute_v2(bindings=[int(device_output)])
                cuda.memcpy_dtoh(host_output, device_output)
                
                # 重塑输出并存储
                outputs[output_name] = host_output.reshape(output_shape)
        
        return outputs

    def visualize_problematic_layers(self, problematic_layers):
        """可视化问题层的输入和输出分布"""
        import matplotlib.pyplot as plt
        
        for layer_info in problematic_layers:
            layer_name = layer_info['layer_name']
            
            # 获取该层的输入和输出
            if layer_name in self.trt_outputs:
                output_data = self.trt_outputs[layer_name]
                
                # 绘制直方图
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(output_data.flatten(), bins=50)
                plt.title(f"Layer: {layer_name} - Output Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                
                plt.subplot(1, 2, 2)
                plt.imshow(np.isnan(output_data).any(axis=tuple(range(len(output_data.shape)-2))), 
                          cmap='hot', interpolation='nearest')
                plt.title(f"Layer: {layer_name} - NaN Map")
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(f"{layer_name}_debug.png")
                plt.close()

# 使用示例
def debug_onnx_to_trt_conversion(onnx_path, sample_input, fp16=True):
    """
    调试ONNX到TensorRT的转换过程
    
    Args:
        onnx_path: ONNX模型路径
        sample_input: 样本输入数据
        fp16: 是否使用FP16模式
    """
    debugger = TensorRTLayerDebugger(onnx_path, sample_input, fp16_mode=fp16)
    problematic_layers = debugger.compare_layer_outputs()
    
    if problematic_layers:
        print(f"\n发现 {len(problematic_layers)} 个可能有问题的层:")
        for i, layer in enumerate(problematic_layers):
            print(f"{i+1}. {layer['layer_name']}")
            if 'has_nan' in layer and layer['has_nan']:
                print(f"   - 包含NaN值: {layer['nan_percentage']}")
            elif 'max_diff' in layer:
                print(f"   - 最大差异: {layer['max_diff']}, 平均差异: {layer['mean_diff']}")
            print(f"   - 形状: {layer['shape']}")
        
        # 可视化问题层
        debugger.visualize_problematic_layers(problematic_layers)
        
        # 提供修复建议
        print("\n可能的解决方案:")
        print("1. 对于包含NaN值的层,考虑在ONNX模型中添加Clip操作来限制值范围")
        print("2. 检查是否有除以零或对负数取平方根等操作")
        print("3. 对于某些层,可能需要强制使用FP32精度")
        print("4. 考虑使用TensorRT的混合精度模式,只对特定层使用FP32")
    else:
        print("未发现问题层,模型转换应该正常")

# 使用示例
if __name__ == "__main__":
    # 假设您有一个ONNX模型和样本输入
    onnx_model_path = "your_model.onnx"
    
    # 创建一个随机样本输入(根据您的模型输入形状调整)
    sample_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 调试转换过程
    debug_onnx_to_trt_conversion(onnx_model_path, sample_input, fp16=True)