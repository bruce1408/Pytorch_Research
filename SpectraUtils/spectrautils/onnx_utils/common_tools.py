import onnx
import numpy as np
from .config_spectrautils import config_spectrautils as config
from typing import Dict, List, Union
from collections import OrderedDict
import pandas as pd

import holoviews as hv
import hvplot.pandas  # pylint:disable=unused-import


from bokeh import plotting
from bokeh.layouts import row, column, Spacer
from bokeh.plotting import figure
from bokeh.models import Div, WheelZoomTool
from bokeh.models import  WheelZoomTool


def add_title(layout, title):
    """
    Add a title to the layout.
    :return: layout wrapped with title div.
    """
    text_str = "<b>" + title + "</b>"
    wrap_layout_with_div = column(Div(text=text_str), layout)
    return wrap_layout_with_div


def get_layer_by_name(model, layer_name):
    """
    Helper function to get layer reference given layer name
    :param model        : model (nn.Module)
    :param layer_name   : layer_name
    :return:
    """
    try:
        return dict(model.named_modules())[layer_name]
    except KeyError as e:
        raise KeyError(f"Couldn't find layer named {layer_name}") from e


def get_device(model):
    """
    Function to find which device is model on
    Assumption : model is on single device
    :param model:
    :return: Device on which model is present
    """
    return next(model.parameters()).device


def histogram(data_frame, column_name, num_bins, x_label=None, y_label=None, title=None):
    """
    Creates a histogram of the column in the input data frame.
    :param data_frame: pandas data frame
    :param column_name: column in data frame
    :param num_bins: number of bins to divide data into for histogram
    :return: bokeh figure object
    """
    hv_plot_object = data_frame.hvplot.hist(column_name, bins=num_bins, height=400, tools="", xlabel=x_label,
                                            ylabel=y_label,
                                            title=title, fill_alpha=0.5)

    bokeh_plot = hv.render(hv_plot_object)
    style(bokeh_plot)
    return bokeh_plot


def get_torch_weights(module):
    """
    Args:
        module (torch.nn.Module): PyTorch模块
        通常是nn.Conv2d或nn.Linear等带有权重的层。

    Returns:
        numpy.ndarray: 形状为(num_input_features, num_output_channels)的二维numpy数组。
                       每列代表一个输出通道的权重向量。
    """
    weights = module.weight.data
    
    # 这样就是把权重参数 变成 [output_channel, input_channel * kernel_h * kernel_w]
    reshaped = weights.view(weights.shape[0], -1)
    
    # 将张量移动到 CPU 并转换为 numpy 数组
    return reshaped.detach().cpu().numpy().T


def style(p:figure) -> figure:
    """
    Style bokeh figure object p and return the styled object
    :param p: Bokeh figure object
    :return: St Styled Bokeh figure
    """
    # Title
    p.title.align = 'center'
    p.title.text_font_size = '14pt'
    p.title.text_font = 'serif'

    # Axis titles
    p.xaxis.axis_label_text_font_size = '12pt'
    # p.xaxis.axis_label_text_font_style = 'bold'
    
    p.yaxis.axis_label_text_font_size = '12pt'
    # p.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
    p.xaxis.major_label_text_font_size = '10pt'
    p.yaxis.major_label_text_font_size = '10pt'

    p.add_tools(WheelZoomTool())

    return p


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
        if node.op_type in config["LAYER_HAS_WEIGHT_ONNX"]:
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