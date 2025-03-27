import onnx
import numpy as np
import config_spectrautils as config
from typing import Dict, List, Union
from collections import OrderedDict
import pandas as pd


def convert_weights_matrix(weight_data):
    
    # 这样就是把权重参数 变成 [output_channel, input_channel * kernel_h * kernel_w]
    reshaped = weight_data.reshape(weight_data.shape[0], -1)
    
    # numpy 数组的转置
    return reshaped.T


def process_layer_data(name_value):
    name, value = name_value
    layer_weights = pd.DataFrame(convert_weights_matrix(value))
    layer_weights_summary_statistics = layer_weights.describe().T
    return name, layer_weights_summary_statistics


def get_onnx_model_weights(onnx_path: str) -> Dict[str, np.ndarray]:
    """
    get_onnx_model_weights 
    Extract weights from an ONNX model.
    
    :param onnx_path: Path to the ONNX model file
    :return: Dictionary of weight names and their corresponding numpy arrays
    """
    model = onnx.load(onnx_path)
    
    # 验证模型有效性
    onnx.checker.check_model(model)  
    
    # 创建一个字典来存储所有初始化器
    initializers = {i.name: i for i in model.graph.initializer}
    
    weights = OrderedDict()
    weight_tensor = OrderedDict()
    need_transpose = []   
    
    # 然后补充处理通过节点获取的权重
    for node in model.graph.node:
        if node.op_type in config.config_spectrautils["LAYER_HAS_WEIGHT_ONNX"]:
            if len(node.input) > 1:
                
                # 从 这里只选择 第2个输入，也就是权重，bias不考虑 
                for in_tensor_name in node.input[1:2]: 
                    weight_tensor[in_tensor_name] = onnx.numpy_helper.to_array(initializers[in_tensor_name])
                if node.op_type == 'ConvTranspose':
                    need_transpose.append(in_tensor_name)
                        
    
    # 合并权重并处理需要转置的情况
    for name, tensor in weight_tensor.items():
        if len(tensor.shape) >= 1:
            if name in need_transpose:
                tensor = tensor.transpose([1, 0, 2, 3])
            weights[name] = tensor
        
    return weights