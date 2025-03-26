import onnx
import onnxruntime as ort
from collections import OrderedDict


def get_onnx_model_io_info(onnx_path):
    """
    获取ONNX模型的输入输出信息
    
    Args:
        onnx_path: ONNX模型文件的路径
        
    Returns:
        tuple: 包含两个字典的元组 (input_info, output_info)
    """
    
    # 创建ONNX运行时的推理会话
    session = ort.InferenceSession(onnx_path)
    
    input_info = OrderedDict((input_node.name, {
        'shape': input_node.shape,
        'type': input_node.type
    }) for input_node in session.get_inputs())
    
    output_info = OrderedDict((output_node.name, {
        'shape': output_node.shape,
        'type': output_node.type
    }) for output_node in session.get_outputs())
    
    
    # 返回输入和输出信息
    return input_info, output_info