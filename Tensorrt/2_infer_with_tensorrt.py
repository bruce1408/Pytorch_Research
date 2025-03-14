import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 必须保留此项以保证上下文管理

def load_engine(engine_file_path):
    """安全加载TensorRT引擎并验证有效性"""
    if not os.path.exists(engine_file_path):
        raise FileNotFoundError(f"引擎文件 {engine_file_path} 不存在")
    
    with open(engine_file_path, "rb") as f:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
        
    if not engine:
        raise RuntimeError("引擎反序列化失败")
    return engine

def validate_shapes(engine, inputs):
    """严格验证输入形状与模型期望的匹配性"""
    for name in inputs:
        binding_idx = engine.get_binding_index(name)
        if binding_idx == -1:
            raise KeyError(f"模型未定义输入绑定: {name}")
        
        expected_shape = tuple(engine.get_tensor_shape(name))
        actual_shape = inputs[name].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"输入 {name} 形状不匹配！期望 {expected_shape}，实际 {actual_shape}"
            )

def infer_tensorrt(engine_file, input_dir, output_dir):
    """
    TensorRT推理完整流程
    :param engine_file: TensorRT引擎文件路径
    :param input_dir: 输入文件夹路径（需包含input_0.txt, input_1.txt等）
    :param output_dir: 输出文件夹路径
    """
    # CUDA设备初始化
    cuda.init()
    device = cuda.Device(0)
    cuda_ctx = device.make_context()
    
    engine = None
    host_outputs = {}
    device_outputs = {}
    device_inputs = {}

    try:
        # 阶段1：加载引擎
        engine = load_engine(engine_file)
        print("成功加载TensorRT引擎")
        
        # 阶段2：创建执行上下文
        context = engine.create_execution_context()
        if not context:
            raise RuntimeError("无法创建执行上下文")
            
        # 阶段3：准备输入输出绑定
        input_bindings = []
        output_bindings = []
        for binding in engine:
            name = binding
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_bindings.append(name)
            else:
                output_bindings.append(name)
        
        print(f"输入绑定: {input_bindings}")
        print(f"输出绑定: {output_bindings}")

        # 阶段4：加载并验证输入数据
        inputs = {}
        for name in input_bindings:
            input_path = os.path.join(input_dir, f"{name}.txt")
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"输入文件 {input_path} 不存在")
            
            # 加载并强制对齐形状
            data = np.loadtxt(input_path).astype(np.float32)
            expected_shape = tuple(engine.get_tensor_shape(name))
            actual_shape = data.shape
            if actual_shape != expected_shape:
                data = data.reshape(expected_shape)
                print(f"调整输入 {name} 形状: {actual_shape} -> {expected_shape}")
            
            inputs[name] = data
            print(f"加载输入 {name}: 形状 {data.shape} 类型 {data.dtype}")

        # 严格形状验证
        validate_shapes(engine, inputs)

        # 创建CUDA流
        stream = cuda.Stream()

        # 分配输出内存
        outputs = {}
        output_host = {}
        bindings = []

        # 对每个绑定的张量，确保输入/输出分配和绑定正确
        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                # 确保输入数据正确绑定
                if binding not in inputs:
                    raise ValueError(f"未找到输入绑定: {binding}")
                bindings.append(inputs[binding].ctypes.data)
            else:
                shape = engine.get_tensor_shape(binding)
                shape = tuple(shape)  # 解决 Dims 对象问题
                dtype = trt.nptype(engine.get_tensor_dtype(binding))
                np_dtype = np.dtype(dtype)
                size = trt.volume(shape) * np_dtype.itemsize
                output = cuda.mem_alloc(size)
                output_host[binding] = cuda.pagelocked_empty(shape, np_dtype)
                outputs[binding] = output
                bindings.append(int(output))

        # 执行推理
        print("开始执行推理...")
        cuda_ctx.push()
        
        try:
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        except Exception as e:
            print(f"推理执行时发生错误: {e}")
            raise

        # 同步CUDA流
        stream.synchronize()

        # 将输出从 GPU 复制到 CPU
        for name, output in outputs.items():
            cuda.memcpy_dtoh_async(output_host[name], output, stream)

        # 再次同步CUDA流
        stream.synchronize()

        cuda_ctx.pop()

        # 保存输出
        os.makedirs(output_dir, exist_ok=True)
        for name, output in output_host.items():
            output_file = os.path.join(output_dir, f"{name}.txt")
            np.savetxt(output_file, output.flatten(), fmt='%.6f')
            print(f"输出 '{name}' 已保存到文件: {output_file}")

    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 清理所有 CUDA 资源
        try:
            # 释放所有分配的 GPU 内存
            for tensor in outputs.values():
                if tensor is not None:
                    tensor.free()

            # 确保上下文被正确弹出和清理
            while cuda_ctx.get_current() is not None:
                cuda_ctx.pop()

            cuda_ctx.detach()
        except Exception as e:
            print(f"清理资源时发生错误: {e}")

# 示例调用
engine_file = "/share/cdd/onnx_models/od_bev_fp32_pytensorrt.trt"  # TensorRT 引擎文件路径
input_dir = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/Tensorrt/od_bev_inputs"      # 输入文件夹路径
output_dir = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/Tensorrt/od_bev_tensorrt_outputs"    # 输出文件夹路径

infer_tensorrt(engine_file, input_dir, output_dir)