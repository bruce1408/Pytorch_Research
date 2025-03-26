import onnx
import onnxruntime as ort
from collections import OrderedDict


def get_onnx_model_input_output_info(onnx_path:str)->OrderedDict:
    """
    获取ONNX模型的输入输出信息
    
    Args:
        onnx_path (str): ONNX模型文件的路径
        
    Returns:
        tuple[OrderedDict, OrderedDict]: 返回两个OrderedDict的元组，分别包含:
            - input_info: 模型输入节点信息，包含shape和type
            - output_info: 模型输出节点信息，包含shape和type
            
    Example:
        >>> input_info, output_info = get_onnx_model_input_output_info("model.onnx")
        >>> print(input_info)  # 查看输入节点信息
        >>> print(output_info)  # 查看输出节点信息
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