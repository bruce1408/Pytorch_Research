# demo.py
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

def main():
    """
    一个演示 onnx.numpy_helper 和 onnx.helper 用法的 Demo.
    """
    print("="*50)
    print("Part 1: 使用 numpy_helper.from_array 和 to_array")
    print("="*50)

    # 1. 准备一个原始的 NumPy 数组
    # 假设这是我们从一个 PyTorch 模型中提取的 FP32 权重
    original_weights_np = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float32)
    print(f"[*] 步骤 1: 原始的 NumPy 数组 (FP32 权重):\n{original_weights_np}\n")
    print(f"    - 类型: {type(original_weights_np)}")
    print(f"    - 数据类型: {original_weights_np.dtype}\n")

    # 2. 使用 onnx.numpy_helper.from_array "打包"
    # 这是将 NumPy 数组转换为 ONNX TensorProto 格式最简单、最常用的方法。
    # 我们需要给这个 tensor 起一个名字，比如 "conv1.weight"。
    tensor_proto_from_helper = numpy_helper.from_array(
        original_weights_np, 
        name="conv1.weight"
    )
    print(f"[*] 步骤 2: 使用 from_array '打包' 后的 TensorProto 对象:\n{tensor_proto_from_helper}")
    print(f"    - 类型: {type(tensor_proto_from_helper)}")
    print(f"    - 这是一个 ONNX 内部格式，不方便直接计算。\n")

    # 3. 使用 onnx.numpy_helper.to_array "解包"
    # 这是将 TensorProto 转换回 NumPy 数组的标准方法。
    unpacked_weights_np = numpy_helper.to_array(tensor_proto_from_helper)
    print(f"[*] 步骤 3: 使用 to_array '解包' 回 NumPy 数组:\n{unpacked_weights_np}\n")
    print(f"    - 类型: {type(unpacked_weights_np)}")
    print(f"    - 数据类型: {unpacked_weights_np.dtype}")

    # 验证解包后的数据和原始数据是否完全一致
    assert np.array_equal(original_weights_np, unpacked_weights_np)
    print("\n[成功] 解包后的数组与原始数组完全一致！\n")


    print("="*50)
    print("Part 2: 使用 helper.make_tensor (更手动的方式)")
    print("="*50)

    # 1. 准备另一个 NumPy 数组
    # 假设这是我们量化后的 INT8 权重
    quantized_weights_np = np.array([
        -10, 0, 20, 127
    ], dtype=np.int8)
    print(f"[*] 步骤 1: 原始的 NumPy 数组 (INT8 权重):\n{quantized_weights_np}\n")

    # 2. 使用 onnx.helper.make_tensor "打包"
    # 这种方法更手动，你需要明确提供所有信息：name, data_type, dims, vals
    tensor_proto_from_make = helper.make_tensor(
        name="fc1.weight_quantized",
        data_type=TensorProto.INT8,       # 必须明确指定数据类型
        dims=quantized_weights_np.shape,  # 必须提供形状
        vals=quantized_weights_np.flatten().tolist() # 必须提供一个平铺后的一维列表
    )
    print(f"[*] 步骤 2: 使用 make_tensor '打包' 后的 TensorProto 对象:\n{tensor_proto_from_make}")

    # 3. 同样使用 to_array "解包" 来验证
    unpacked_quantized_np = numpy_helper.to_array(tensor_proto_from_make)
    print(f"[*] 步骤 3: '解包' 回 NumPy 数组:\n{unpacked_quantized_np}\n")

    assert np.array_equal(quantized_weights_np, unpacked_quantized_np)
    print("[成功] 使用 make_tensor 创建并解包后，数据依然完全一致！\n")


if __name__ == "__main__":
    main()