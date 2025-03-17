import time, os
import numpy as np
import onnxruntime as ort

def get_model_io_info(onnx_path):
    """
    获取ONNX模型的输入输出信息
    
    Args:
        onnx_path: ONNX模型文件的路径
        
    Returns:
        tuple: 包含两个字典的元组 (input_info, output_info)
            - input_info: 包含所有输入节点信息的字典，格式为:
                {
                    'input_name': {
                        'name': 输入节点名称,
                        'shape': 输入形状(可能包含动态维度),
                        'type': 输入数据类型
                    }
                }
            - output_info: 包含所有输出节点信息的字典，格式与input_info类似
    """
    # 创建ONNX运行时的推理会话
    session = ort.InferenceSession(onnx_path)
    
    # 获取模型的所有输入节点信息， 使用字典推导式构建输入信息字典
    input_info = {
        input_node.name: {
            'name': input_node.name,  # 输入节点的名称
            'shape': input_node.shape,  # 输入张量的形状，可能包含动态维度(如None)
            'type': input_node.type  # 输入数据类型，如'tensor(float)'
        } for input_node in session.get_inputs()  # 遍历所有输入节点
    }
    
    # 获取模型的所有输出节点信息 使用字典推导式构建输出信息字典
    output_info = {
        output_node.name: {
            'name': output_node.name,  # 输出节点的名称
            'shape': output_node.shape,  # 输出张量的形状
            'type': output_node.type  # 输出数据类型
        } for output_node in session.get_outputs()  # 遍历所有输出节点
    }
    
    # 返回输入和输出信息
    return input_info, output_info


def get_results(result_dir, dtype=np.float32):
    """
    递归读取指定目录下的结果文件
    
    Args:
        result_dir: 结果文件目录
        dtype: 数据类型，默认为np.float32
        
    Returns:
        包含所有结果的嵌套字典
    """
    if result_dir is None:
        raise ValueError("result_dir cannot be None")
    
    results = {}
    
    for top_dir in sorted(os.listdir(result_dir)):
        top_path = os.path.join(result_dir, top_dir)
        if not os.path.isdir(top_path):
            continue
            
        mid_level_results = {}
        for mid_dir in sorted(os.listdir(top_path)):
            mid_path = os.path.join(top_path, mid_dir)
            if not os.path.isdir(mid_path):
                continue
                
            file_results = {}
            for file_name in sorted(os.listdir(mid_path)):
                file_path = os.path.join(mid_path, file_name)
                if os.path.isfile(file_path):
                    file_results[file_name] = np.fromfile(file_path, dtype=dtype)
            
            if file_results:  # 只有当有结果时才添加
                mid_level_results[mid_dir] = file_results
        
        if mid_level_results:  # 只有当有结果时才添加
            results[top_dir] = mid_level_results

    return results
    
    