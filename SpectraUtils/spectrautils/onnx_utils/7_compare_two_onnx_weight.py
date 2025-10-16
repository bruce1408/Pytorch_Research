import onnx
from onnx import numpy_helper
import numpy as np

def load_onnx_weights(onnx_file):
    """
    加载 ONNX 模型并提取权重
    """
    model = onnx.load(onnx_file)
    
    # 检查模型的有效性
    try:
        onnx.checker.check_model(model)
        print(f"模型 {onnx_file} 验证通过，结构有效")
    except Exception as e:
        print(f"模型 {onnx_file} 验证失败: {e}")
    
    weights = {}

    # 遍历所有的权重，提取 name 和对应的权重值
    for tensor in model.graph.initializer:
        # 根据数据类型选择正确的numpy数据类型
        if tensor.data_type == 1:  # FLOAT
            dtype = np.float32
        elif tensor.data_type == 2:  # UINT8
            dtype = np.uint8
        elif tensor.data_type == 3:  # INT8
            dtype = np.int8
        elif tensor.data_type == 6:  # INT32
            dtype = np.int32
        elif tensor.data_type == 7:  # INT64
            dtype = np.int64
        elif tensor.data_type == 11:  # DOUBLE
            dtype = np.float64
        else:
            print(f"未处理的数据类型: {tensor.data_type}，跳过 {tensor.name}")
            continue
            
        # 使用onnx提供的numpy_helper更安全地转换张量
        try:
            weights[tensor.name] = numpy_helper.to_array(tensor)
        except Exception as e:
            print(f"无法转换张量 {tensor.name}: {e}")
            continue

    return weights

def compare_weights(weights1, weights2, output_file=None):
    """
    比较两个 ONNX 模型的权重差异，并可选择将结果写入文件
    """
    all_layers_same = True
    results = []  # 存储基本比较结果
    diff_results = []  # 存储带有差异值的结果，用于排序
    
    for name in weights1:
        if name not in weights2:
            message = f"Layer '{name}' is missing in the second model!"
            print(message)
            results.append(message)
            all_layers_same = False
            continue
        
        # 计算权重差异
        diff = np.abs(weights1[name] - weights2[name])
        max_diff = np.max(diff)
        
        # 将差异和对应的层名称存储为元组，用于后续排序
        diff_results.append((name, max_diff))
        
        if max_diff > 1e-5:  # 设置一个阈值，考虑浮动的误差
            message = f"Layer '{name}' has a significant difference (max diff: {max_diff})"
            print(message)
            results.append(message)
            all_layers_same = False
        else:
            message = f"Layer '{name}' is the same between the two models."
            print(message)
            results.append(message)

    # 检查是否有额外的权重在第二个模型中
    for name in weights2:
        if name not in weights1:
            message = f"=========== Layer '{name}' is missing in the first model!============"
            print(message)
            results.append(message)
            all_layers_same = False
    
    # 对差异结果按max_diff从大到小排序
    sorted_diff_results = sorted(diff_results, key=lambda x: x[1], reverse=True)
    
    # 如果提供了输出文件，将结果写入文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ONNX模型权重比较结果\n")
            f.write("="*50 + "\n")
            
            # 首先写入按差异大小排序的结果
            f.write("\n按最大差异从大到小排序的结果:\n")
            f.write("-"*50 + "\n")
            for name, max_diff in sorted_diff_results:
                f.write(f"Layer '{name}' - 最大差异: {max_diff}\n")
            
            # 然后写入详细的比较结果
            f.write("\n详细比较结果:\n")
            f.write("-"*50 + "\n")
            for result in results:
                f.write(result + "\n")
            
            if all_layers_same:
                f.write("\n两个模型的权重没有显著差异。\n")
            else:
                f.write("\n两个模型的权重存在一些差异。\n")
        
        print(f"比较结果已写入文件: {output_file}")

    return all_layers_same

def main():
    # 需要比较的两个 ONNX 模型文件
    onnx_file1 = '/home/bruce_ultra/workspace/8620_code_repo/8620_code_x86/onnx_models/od_bev_250220.onnx'
    onnx_file2 = '/home/bruce_ultra/workspace/8620_code_repo/8620_code_x86/onnx_models/od_bev_0306.onnx'

    # 加载权重
    weights1 = load_onnx_weights(onnx_file1)
    weights2 = load_onnx_weights(onnx_file2)

    output_file = 'onnx_weight_comparison_result.txt'

    # 比较权重差异
    all_same = compare_weights(weights1, weights2, output_file)

    if all_same:
        print("The models have no significant weight differences.")
    else:
        print("The models have some differences in their weights.")

if __name__ == "__main__":
    main()
