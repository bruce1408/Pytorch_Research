import onnx, torch
import onnxruntime as ort
from collections import OrderedDict
from typing import Dict, Union, List
import torchvision.models as models


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


def export_model_onnx(
    model:torch.nn.Module, 
    input_info: Dict[str, Union[tuple, torch.Size]], 
    export_path: str, 
    output_names: List[str],
    opset_version=13
):
    
    """
    将PyTorch模型导出为ONNX格式,支持多个输入和输出

    Args:
        model (nn.Module): 要导出的PyTorch模型
        input_info (Dict[str, Union[tuple, torch.Size]]): 输入名称到形状的映射
        path (str): 保存ONNX模型的路径
        output_names (List[str]): 输出节点的名称列表
        dynamic_axes (Dict[str, Dict[int, str]], optional): 动态轴的配置
        opset_version (int, optional): ONNX操作集版本,默认为11
        export_params (bool, optional): 是否导出模型参数,默认为True
        validate (bool, optional): 是否验证导出的ONNX模型,默认为True

    Returns:
        None

    Example:
        >>> model = YourMultiInputOutputModel()
        >>> export_model_onnx(model, 
        ...                   {'input1': (1, 3, 224, 224), 'input2': (1, 1, 28, 28)}, 
        ...                   "multi_io_model.onnx",
        ...                   output_names=['output1', 'output2'])
    """
    
    
    dummy_inputs = {name: torch.randn(*shape) for name, shape in input_info.items()}

    # 确保模型处于评估模式
    model.eval()
    
    
    # 导出模型到ONNX
    torch.onnx.export(
        model,                    
        tuple(dummy_inputs.values()),      
        export_path,                     
        opset_version=opset_version,
        do_constant_folding=True, 
        input_names=list(input_info.keys()),  
        output_names=output_names,
        verbose=False
    )
    
    print("onnx model export to: ", export_path)
    

import hashlib
import os

def show_file_md5(file_path):
    """
    计算并显示指定文件的MD5值
    
    Args:
        file_path (str): 文件的路径
    
    Returns:
        str: 文件的MD5值，如果文件不存在则返回None
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。")
        return None
    
    # 初始化MD5对象
    md5_hash = hashlib.md5()
    
    try:
        # 以二进制模式打开文件
        with open(file_path, "rb") as f:
            # 逐块读取文件内容并更新MD5值
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        # 获取MD5值的十六进制表示
        md5_value = md5_hash.hexdigest()
        
        # print(f"文件 '{file_path}' 的MD5值是: \n{md5_value}")
        print(f"The MD5 value of file '{file_path}' is: \n{md5_value}")

        return md5_value
    
    except IOError as e:
        print(f"错误: 无法读取文件 '{file_path}'. {str(e)}")
        return None


if __name__ == "__main__":
    
    # 加载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    
    # 定义输入信息
    input_info = {'input': (1, 3, 224, 224)}  # ResNet18 期望的输入尺寸
    
    # 定义输出名称
    output_names = ['output']
    # export_model_onnx(model, 
    #                   input_info,
    #                   "./resnet18.onnx",
    #                   output_names
    #                 )
    
    show_file_md5("/mnt/share_disk/bruce_trie/workspace/outputs/parking_space_models/psd_v2_1_2_simplifier.onnx")