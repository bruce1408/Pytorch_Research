import tensorrt as trt
import os
import pycuda.driver as cuda
import numpy as np
import onnx
import onnxruntime as ort


def get_onnx_input_shapes(onnx_file_path):
    """
    获取ONNX模型的所有输入形状
    """
    model = onnx.load(onnx_file_path)
    input_shapes = {}
    for input in model.graph.input:
        shape = [dim.dim_value if dim.dim_value != 0 else 1 for dim in input.type.tensor_type.shape.dim]
        input_shapes[input.name] = shape
    return input_shapes

def mock_input_data(input_shapes):
    """
    根据输入形状生成模拟数据
    """
    mock_inputs = {}
    for name, shape in input_shapes.items():
        mock_inputs[name] = np.random.rand(*shape).astype(np.float32)
    return mock_inputs


def save_inputs(input_dict, input_dir):
    """
    将输入字典中的每个张量的值保存到单独的文本文件
    
    :param input_dict: 包含输入张量的字典
    :param input_dir: 输入文件夹的路径
    """
    os.makedirs(input_dir, exist_ok=True)
    for name, tensor in input_dict.items():
        input_file = os.path.join(input_dir, f"{name}.txt")
        np.savetxt(input_file, tensor.flatten(), fmt='%.6f')
        print(f"输入 '{name}' 已保存到文件: {input_file}")
        
def save_outputs(output_dict, output_dir):
    """
    将输出字典中的每个张量保存到单独的文本文件
    
    :param output_dict: 包含输出张量的字典
    :param output_dir: 输出文件夹的路径
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, tensor in output_dict.items():
        output_file = os.path.join(output_dir, f"{name}.txt")
        np.savetxt(output_file, tensor.flatten(), fmt='%.6f')
        print(f"输出 '{name}' 已保存到文件: {output_file}")

    
    
def build_engine(onnx_file_path, engine_file_path, precision='fp32'):
    """
    将ONNX模型转换为TensorRT引擎
    
    :param onnx_file_path: ONNX模型文件路径
    :param engine_file_path: 保存TensorRT引擎的文件路径
    :param precision: 精度模式, 'fp32'或'fp16'
    """
    
    # 启用CUDA懒加载以减少设备内存使用并加速TensorRT初始化
    # cuda.set_lazy_loading(True)
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    success = parser.parse_from_file(onnx_file_path)
    if not success:
        raise ValueError("无法解析ONNX文件")

    config = builder.create_builder_config()
    # 修复 max_workspace_size 已弃用的警告
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 修复 build_engine 已弃用的警告
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT引擎已保存到: {engine_file_path}")

def load_inputs_from_txt(input_dir, input_shapes):
    """
    从文本文件加载输入数据
    
    :param input_dir: 输入文件夹的路径
    :param input_shapes: 输入形状字典
    :return: 包含输入张量的字典
    """
    input_dict = {}
    for name, shape in input_shapes.items():
        input_file = os.path.join(input_dir, f"{name}.txt")
        if os.path.exists(input_file):
            data = np.loadtxt(input_file)
            input_dict[name] = data.reshape(shape).astype(np.float32)
            print(f"已加载输入 '{name}', 形状为: {input_dict[name].shape}")
        else:
            print(f"警告: 未找到输入文件 '{input_file}'")
    return input_dict


def infer_onnxruntime(onnx_file, input_dir, output_dir):
    """
    使用ONNX Runtime进行推理，从文本文件加载输入数据
    
    :param onnx_file: ONNX模型文件路径
    :param input_dir: 输入文件夹的路径
    :return: 包含输出张量的字典
    """
    # 获取ONNX模型输入形状
    input_shapes = get_onnx_input_shapes(onnx_file)
    print(f"ONNX模型输入形状: {input_shapes}")

    # 从文本文件加载输入数据
    input_dict = load_inputs_from_txt(input_dir, input_shapes)

    # 使用ONNX Runtime运行模型并获取输出
    session = ort.InferenceSession(onnx_file)
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, input_dict)

    # 创建包含输出张量的字典
    output_dict = {}
    for name, output in zip(output_names, outputs):
        output_dict[name] = output
        print(f"输出张量 '{name}' 的形状: {output.shape}")
    
    if output_dict is not None:
        print("输出字典的键:", list(output_dict.keys()))
        
        # 保存每个输出到单独的文本文件
        save_outputs(output_dict, onnx_output_dir)
    else:
        print("推理函数返回了 None")
        

    return output_dict

def load_engine(engine_file_path):
    """
    加载 TensorRT 引擎
    """
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer_tensorrt(engine_file, input_dir, output_dir):
    """
    使用 TensorRT 进行推理，从文本文件加载输入数据，并将结果保存到文本文件
    
    :param engine_file: TensorRT 引擎文件路径
    :param input_dir: 输入文件夹的路径
    :param output_dir: 输出文件夹的路径
    """
    # 初始化CUDA上下文
    cuda.init()
    device = cuda.Device(0)  # 使用第一个GPU设备
    cuda_ctx = device.make_context()
    
    try:
        # 加载 TensorRT 引擎
        with open(engine_file, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        trt_ctx = engine.create_execution_context()
        
        # 获取输入和输出的名称和形状
        input_names = []
        output_names = []
        input_shapes = {}
        output_shapes = {}
        
        for binding in engine:
            shape = tuple(engine.get_binding_shape(binding))
            if engine.binding_is_input(engine.get_binding_index(binding)):
                input_names.append(binding)
                input_shapes[binding] = shape
            else:
                output_names.append(binding)
                output_shapes[binding] = shape
        
        print(f"输入名称: {input_names}")
        print(f"输出名称: {output_names}")
        
        # 创建存储输入和输出数据的字典
        host_inputs = {}
        host_outputs = {}
        cuda_inputs = {}
        cuda_outputs = {}
        bindings = []
        
        # 准备输入数据
        for name in input_names:
            input_file = os.path.join(input_dir, f"{name}.txt")
            if os.path.exists(input_file):
                data = np.loadtxt(input_file)
                shape = input_shapes[name]
                data = data.reshape(shape).astype(np.float32)
                host_inputs[name] = data
                
                # 分配GPU内存并复制数据
                cuda_inputs[name] = cuda.mem_alloc(host_inputs[name].nbytes)
                cuda.memcpy_htod(cuda_inputs[name], host_inputs[name])
                bindings.append(int(cuda_inputs[name]))
                
                print(f"已加载输入 '{name}', 形状为: {host_inputs[name].shape}")
            else:
                print(f"警告: 未找到输入文件 '{input_file}'")
                return
        
        # 准备输出内存
        for name in output_names:
            shape = output_shapes[name]
            size = trt.volume(shape)
            dtype = np.float32  # 假设输出是float32
            host_outputs[name] = np.zeros(shape, dtype=dtype)
            cuda_outputs[name] = cuda.mem_alloc(host_outputs[name].nbytes)
            bindings.append(int(cuda_outputs[name]))
            print(f"为输出 '{name}' 分配内存, 形状为: {shape}")
        
        # 执行推理
        print("开始执行推理...")
        cuda_ctx.push()
        trt_ctx.execute_v2(bindings=bindings)
        
        # 将输出从GPU复制到CPU
        for name in output_names:
            cuda.memcpy_dtoh(host_outputs[name], cuda_outputs[name])
        
        cuda_ctx.pop()
        
        # 保存输出
        os.makedirs(output_dir, exist_ok=True)
        for name, output in host_outputs.items():
            output_file = os.path.join(output_dir, f"{name}.txt")
            np.savetxt(output_file, output.flatten(), fmt='%.6f')
            print(f"输出 '{name}' 已保存到文件: {output_file}")
    
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 清理所有CUDA资源
        try:
            # 释放所有分配的GPU内存
            for tensor in list(cuda_inputs.values()) + list(cuda_outputs.values()):
                if tensor:
                    tensor.free()
            
            # 清理TensorRT上下文
            if 'trt_ctx' in locals():
                del trt_ctx
            
            # 清理TensorRT引擎
            if 'engine' in locals():
                del engine
            
            # 确保CUDA上下文被正确清理
            if cuda_ctx:
                cuda_ctx.pop()
                cuda_ctx.detach()
        except Exception as e:
            print(f"清理资源时发生错误: {e}")
        finally:
            # 确保CUDA上下文被完全清理
            while cuda.Context.get_current() is not None:
                try:
                    cuda.Context.pop()
                except cuda.LogicError:
                    break

        
        

if __name__ == "__main__":
    onnx_file = "/share/cdd/onnx_models/od_bev_0306.onnx"
    engine_file_fp32 = "/share/cdd/onnx_models/od_bev_fp32_pytensorrt.trt"
    engine_file_fp16 = "/share/cdd/onnx_models/od_bev_fp16_pytensorrt.trt"
    # output_file = "/share/cdd/onnx_models/od_bev_output.npy"
    input_dir = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/Tensorrt/od_bev_inputs"
    onnx_output_dir = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/Tensorrt/od_bev_onnx_outputs"
    tensorrt_output_dir = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/Tensorrt/od_bev_tensorrt_outputs"

    # 转换为FP32精度
    # build_engine(onnx_file, engine_file_fp32, precision='fp32')
    
    # 转换为FP16精度
    # build_engine(onnx_file, engine_file_fp16, precision='fp16')
    
     # 调用函数进行推理
    # output_dict = infer_onnxruntime(onnx_file, input_dir,onnx_output_dir )
    
    infer_tensorrt(engine_file_fp32, input_dir, tensorrt_output_dir)
