import time, os
import numpy as np
from collections import OrderedDict
import onnxruntime as ort
from typing import List, Dict
import torch, torchvision, onnx
from  torchvision.models import ResNet18_Weights
from bokeh import plotting
from bokeh.layouts import column
import holoviews as hv
import pandas as pd
import hvplot.pandas  # pylint:disable=unused-import

from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.models import HoverTool, WheelZoomTool, ColumnDataSource
from bokeh.models import HoverTool, ColumnDataSource, Span, TableColumn, DataTable
from spectrautils import logging_utils, print_utils


class PlotsLayout:
    """
    Keeps track of a layout (rows and columns of plot objects) and pushes them to a bokeh session once the layout is complete
    """

    def __init__(self):
        self.title = None
        self.layout = []

    def add_row(self, figures_list):
        """
        adds a row to self.layout
        :param figures_list: list of figure objects.
        :return: None.
        """
        self.layout.append(figures_list)

    def complete_layout(self):
        """
        complete a layout by adding self.layout to a server session document.
        :return:
        """
        if self.title is None:
            print(type(self.layout))
            plot = self.layout if isinstance(self.layout, list) else column(self.layout)
        else:
            my_session_with_title = self.add_title()
            return my_session_with_title
        return plot

    def add_title(self):
        """
        Add a title to the current layout.
        :return: layout wrapped with title div.
        """
        text_str = "<b>" + self.title + "</b>"
        wrap_layout_with_div = column(Div(text=text_str), column(self.layout))
        return wrap_layout_with_div



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


def get_weights(conv_module):
    """
    Returns the weights of a conv_module in a 2d matrix, where each column is an output channel.

    :param conv_module: convNd module
    :return: 2d numpy array
    """
    output_channel_nums = conv_module.weight.shape[0]
    
    # input_channel * kernel_h * kernel_w
    axis_1_length = np.prod(conv_module.weight.shape[1:])
    
    reshaped = conv_module.weight.reshape(int(output_channel_nums), int(axis_1_length))
    
    # 转置
    weights = reshaped.detach().numpy().T
    return weights


def style(p):
    """
    Style bokeh figure object p and return the styled object
    :param p: Bokeh figure object
    :return: Bokeh figure object
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


def add_vertical_line_to_figure(x_coordinate, figure_object):
    """
    adds a vertical line to a bokeh figure object
    :param x_coordinate: x_coordinate to add line
    :param figure_object: bokeh figure object
    :return: None
    """
    # Vertical line
    vertical_line = Span(location=x_coordinate, dimension='height', line_color='red', line_width=1)
    figure_object.add_layout(vertical_line)
    
def convert_pandas_data_frame_to_bokeh_column_data_source(data):
    """
    Converts a pandas data frame to a bokeh column data source object so that it can be pushed to a server document
    :param data: pandas data frame
    :return: data table that can be displayed on a bokeh server document
    """
    data["index"] = data.index
    data = data[['index'] + data.columns[:-1].tolist()]

    data.columns.map(str)
    source = ColumnDataSource(data=data)
    return source


def line_plot_changes_in_summary_stats(data_before, data_after, x_axis_label=None, y_axis_label=None, title=None):
    """
    Returns a bokeh figure object showing a lineplot of min, max, and mean per output channel, shading in the area
    difference between before and after.
    :param data_before: pandas data frame with columns min, max, and mean.
    :param data_after: pandas data frame with columns min, max, and mean
    :param x_axis_label: string description of x axis
    :param y_axis_label: string description of y axis
    :param title: title for the plot
    :return: bokeh figure object
    """
    layer_weights_old_model = convert_pandas_data_frame_to_bokeh_column_data_source(data_before)
    layer_weights_new_model = convert_pandas_data_frame_to_bokeh_column_data_source(data_after)

    plot = figure(x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                  title=title,
                  tools="pan, box_zoom, crosshair, reset, save",
                  width=950, height=600, sizing_mode='stretch_both', output_backend="webgl")
    plot.line(x='index', y='min', line_width=2, line_color="#2171b5", line_dash='dotted', legend_label="Before Optimization",
              source=layer_weights_old_model, name="old model")
    plot.line(x='index', y='max', line_width=2, line_color="green", line_dash='dotted', source=layer_weights_old_model,
              name="old model")
    plot.line(x='index', y='mean', line_width=2, line_color="orange", line_dash='dotted',
              source=layer_weights_old_model, name="old model")

    plot.line(x='index', y='min', line_width=2, line_color="#2171b5",
              legend_label="After Optimization", source=layer_weights_new_model, name="new model")
    plot.line(x='index', y='max', line_width=2, line_color="green",
              source=layer_weights_new_model, name="new model")
    plot.line(x='index', y='mean', line_width=2, line_color="orange",
              source=layer_weights_new_model, name="new model")

    plot.varea(x=data_after.index,
               y1=data_after['min'],
               y2=data_before['min'], fill_alpha=0.3, legend_label="shaded region", name="new model")

    plot.varea(x=data_after.index,
               y1=data_after['max'],
               y2=data_before['max'], fill_color="green", fill_alpha=0.3, legend_label="shaded region")

    plot.varea(x=data_after.index,
               y1=data_after['mean'],
               y2=data_before['mean'], fill_color="orange", fill_alpha=0.3, legend_label="shaded region")

    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    plot.legend.background_fill_alpha = 0.3

    if not x_axis_label or not y_axis_label or not title:
        layout = row(plot)
        return layout

    # display a tooltip whenever the cursor in line with a glyph
    hover1 = HoverTool(tooltips=[("Output Channel", "$index"),
                                 ("Mean Before Optimization", "@mean{0.00}"),
                                 ("Minimum Before Optimization", "@min{0.00}"),
                                 ("Maximum Before Optimization", "@max{0.00}"),
                                 ("25 Percentile Before Optimization", "@{25%}{0.00}"),
                                 ("75 Percentile Before Optimization", "@{75%}{0.00}")], name='old model',
                       mode='mouse'
                       )
    hover2 = HoverTool(tooltips=[("Output Channel", "$index"),
                                 ("Mean After Optimization", "@mean{0.00}"),
                                 ("Minimum After Optimization", "@min{0.00}"),
                                 ("Maximum After Optimization", "@max{0.00}"),
                                 ("25 Percentile After Optimization", "@{25%}{0.00}"),
                                 ("75 Percentile After Optimization", "@{75%}{0.00}")], name='new model',
                       mode='mouse'
                       )
    plot.add_tools(hover1)
    plot.add_tools(hover2)
    style(plot)

    layout = row(plot)
    return layout


def scatter_plot_summary_stats(data_frame, x_axis_label_mean="mean", y_axis_label_mean="standard deviation",
                               title_mean="Mean vs Standard Deviation",
                               x_axis_label_min="Minimum",
                               y_axis_label_min="Maximum", title_min="Minimum vs Maximum"):
    """
    Creates a scatter plot, plotting min vs max, and mean vs std side by side.
    :param data_frame: pandas data frame object
    :param x_axis_label_mean: string description of x axis in plot showing mean vs std
    :param y_axis_label_mean: string description of y axis in plot showing mean vs std
    :param x_axis_label_min: string description of x axis in plot showing min vs max
    :param y_axis_label_min: string description of y axis in plot showing min vs max
    :return: bokeh figure
    """
    plot1 = figure(x_axis_label=x_axis_label_mean, y_axis_label=y_axis_label_mean,
                   title=title_mean,
                   tools="box_zoom, crosshair,reset", output_backend="webgl")
    plot1.circle(x=data_frame['mean'], y=data_frame['std'], size=10, color="orange", alpha=0.4)

    plot2 = figure(x_axis_label=x_axis_label_min, y_axis_label=y_axis_label_min,
                   title=title_min,
                   tools="box_zoom, crosshair,reset", output_backend="webgl")
    plot2.circle(x=data_frame['min'], y=data_frame['max'], size=10, color="#2171b5", alpha=0.4)
    style(plot1)
    style(plot2)
    # layout = row(plot1, plot2)
    return plot1, plot2


def visualize_changes_after_optimization_single_layer(name, old_model_module, new_model_module, scatter_plot=False):
    """
    Creates before and after plots for a given layer.
    :param name: name of module
    :param old_model_module: the module of the model before optimization
    :param new_model_module: the module of the model after optimization
    :param scatter_plot: Include scatter plot in plots
    :return: None
    """

    device_old_module = get_device(old_model_module)
    device_new_module = get_device(new_model_module)

    old_model_module.cpu()
    new_model_module.cpu()

    layout = PlotsLayout()
    layout.title = name
    layer_weights_summary_statistics_old = pd.DataFrame(get_weights(old_model_module)).describe().T
    layer_weights_summary_statistics_new = pd.DataFrame(get_weights(new_model_module)).describe().T

    summary_stats_line_plot = line_plot_changes_in_summary_stats(layer_weights_summary_statistics_old,
                                                                 layer_weights_summary_statistics_new,
                                                                 x_axis_label="Output Channel",
                                                                 y_axis_label="Summary statistics",
                                                                 title="Changes in Key Stats Per Output Channel")

    if scatter_plot:
        plot_mean_old_model, plot_min_old_model = scatter_plot_summary_stats(layer_weights_summary_statistics_old,
                                                                             x_axis_label_mean="Mean Weights Per Output Channel",
                                                                             y_axis_label_mean="Std Per Output Channel",
                                                                             title_mean="Mean vs Std After Optimization",
                                                                             x_axis_label_min="Min Weights Per Output Channel",
                                                                             y_axis_label_min="Max Weights Per Output Channel",
                                                                             title_min="Min vs Max After Optimization")

        plot_mean_new_model, plot_min_new_model = scatter_plot_summary_stats(layer_weights_summary_statistics_new,
                                                                             x_axis_label_mean="Mean Weights Per Output Channel",
                                                                             y_axis_label_mean="Std Per Output Channel",
                                                                             title_mean="Mean vs Std Before Optimization",
                                                                             x_axis_label_min="Min Weights Per Output Channel",
                                                                             y_axis_label_min="Max Weights Per Output Channel",
                                                                             title_min="Min vs Max Before Optimization")
        layout.add_row(row(plot_mean_old_model, plot_mean_new_model, plot_min_old_model))

        layout.add_row(row(summary_stats_line_plot, plot_min_new_model))
    else:
        layout.add_row(summary_stats_line_plot)

    old_model_module.to(device=device_old_module)
    new_model_module.to(device=device_new_module)

    return layout.complete_layout()


def detect_outlier_channels(data_frame_with_relative_ranges, column="relative range", factor=1.5):
    """
    检测相对权重范围的异常值。
    
    Args:
        data_frame_with_relative_ranges: 包含"relative range"列的pandas数据框
    
    Returns:
        list: 具有非常大的权重范围的输出通道列表
    """
    # 计算第一四分位数（Q1）
    Q1 = data_frame_with_relative_ranges.quantile(0.25)
    
    # 计算第三四分位数（Q3）
    Q3 = data_frame_with_relative_ranges.quantile(0.75)
    
    # 计算四分位距（IQR）
    IQR = Q3 - Q1
    
    # 定义上侧异常值的阈值：大于 Q3 + 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 定义下侧异常值的阈值：小于 Q1 - 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    # 检测上侧异常值
    upper_outliers = data_frame_with_relative_ranges[data_frame_with_relative_ranges['relative range'] > upper_bound]
    
    # 检测下侧异常值
    lower_outliers = data_frame_with_relative_ranges[data_frame_with_relative_ranges['relative range'] < lower_bound]
    
    # 获取上侧异常值的索引（即异常通道的标识）
    upper_output_channels_list = upper_outliers.index.tolist()
    
    # 获取下侧异常值的索引（即异常通道的标识）
    lower_output_channels_list = lower_outliers.index.tolist()
    
    # 返回上侧和下侧异常通道列表
    return upper_output_channels_list + lower_output_channels_list

    
    # # 定义异常值的阈值：大于 Q3 + 1.5 * IQR
    # v = (data_frame_with_relative_ranges > (Q3 + 1.5 * IQR))
    
    # # 将布尔序列转换为数据框
    # v_df = v.to_frame()
    
    # # 只保留异常值（True）的行
    # keep_only_outliers = v_df.loc[v_df['relative range']]
    
    # # 获取异常值的索引（即异常通道的标识）
    # output_channels_list = keep_only_outliers.index
    
    # # 返回异常通道列表
    # return output_channels_list


def identify_problematic_output_channels(weights_stats):
    """
    return a list of output channels that have large weight ranges
    :param module_weights_data_frame: pandas data frame where each column are summary statistics for each row, output channels
    :param largest_ranges_n: number of output channels to return
    识别卷积层中可能存在问题的输出通道
    Args:
        conv_module: 卷积模块
        threshold: 判断问题通道的阈值（默认0.1）
    Returns:
        list: 存在问题的通道索引列表
    """
    
    # 计算每个通道的权重范围
    weights_stats['range'] = weights_stats['max'] -  weights_stats['min']
    
    # 计算权重范围的绝对值
    weights_stats["abs range"] = weights_stats["range"].abs()
    
    # 找到最小的绝对范围值
    variable = weights_stats["abs range"].min()
    
    # 计算相对的范围，每个通道的绝对范围除以最小绝对范围
    weights_stats["relative range"] = weights_stats["abs range"] / variable
    
    # 按相对范围降序排序
    described_df = weights_stats.sort_values(by=['relative range'], ascending=False)
    
    # 提取所有通道的相对范围
    all_output_channel_ranges = described_df["relative range"]
    
    # 使用detect_outlier_channels 函数检测异常通道
    output_channels_needed = detect_outlier_channels(all_output_channel_ranges)

    # 返回异常通道列表和所有通道的相对范围
    return output_channels_needed, all_output_channel_ranges


def convert_pandas_data_frame_to_bokeh_data_table(data):
    """
    Converts a pandas data frame to a bokeh column data source object so that it can be pushed to a server document
    :param data: pandas data frame
    :return: data table that can be displayed on a bokeh server document
    """
    data["index"] = data.index
    data = data[['index'] + data.columns[:-1].tolist()]

    data.columns.map(str)
    source = ColumnDataSource(data=data)
    columns = [TableColumn(field=column_str, title=column_str) for column_str in data.columns]  # bokeh columns
    data_table = DataTable(source=source, columns=columns)
    layout = add_title(data_table, "Table Summarizing Weight Ranges")
    return layout



def line_plot_summary_statistics_model(layer_name, layer_weights_data_frame, height, width):
    """
    Given a layer
    :param layer_name:
    :param layer_weights_data_frame:
    :return:
    """
    layer_weights = convert_pandas_data_frame_to_bokeh_column_data_source(layer_weights_data_frame)
    plot = figure(x_axis_label="Output Channels", y_axis_label="Summary Statistics",
                  title="Weight Ranges per Output Channel: " + layer_name,
                  tools="pan, box_zoom, crosshair, reset, save",
                  width=width, height=height, output_backend="webgl")
    plot.line(x='index', y='min', line_width=2, line_color="#2171b5",
              legend_label="Minimum", source=layer_weights)
    plot.line(x='index', y='max', line_width=2, line_color="green",
              legend_label="Maximum", source=layer_weights)
    plot.line(x='index', y='mean', line_width=2, line_color="orange",
              legend_label="Average", source=layer_weights)

    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    plot.legend.background_fill_alpha = 0.3

    plot.add_tools(HoverTool(tooltips=[("Output Channel", "$index"),
                                       ("Mean", "@mean{0.00}"),
                                       ("Min", "@min{0.00}"),
                                       ("Max", "@max{0.00}"),
                                       ("25 percentile", "@{25%}{0.00}"),
                                       ("75 percentile", "@{75%}{0.00}")],
                             # display a tooltip whenever the cursor is vertically in line with a glyph
                             mode='mouse'
                             ))
    style(plot)
    return plot


def visualize_relative_weight_ranges_single_layer(layer, layer_name):
    """
    publishes a line plot showing  weight ranges for each layer, summary statistics
    for relative weight ranges, and a histogram showing weight ranges of output channels

    :param model: p
    :return:
    """
    layer_weights_data_frame = pd.DataFrame(get_weights(layer)).describe().T
    plot = line_plot_summary_statistics_model(layer_name, layer_weights_data_frame, width=1150, height=700)

    # list of problematic output channels, data frame containing magnitude of range in each output channel
    problematic_output_channels, output_channel_ranges_data_frame = identify_problematic_output_channels(
        layer_weights_data_frame)

    histogram_plot = histogram(output_channel_ranges_data_frame, "relative range", 75,
                               x_label="Weight Range Relative to Smallest Output Channel",
                               y_label="Count",
                               title="Relative Ranges For All Output Channels")
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.describe().T.to_frame()
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.drop("count")

    output_channel_ranges_as_column_data_source = convert_pandas_data_frame_to_bokeh_data_table(
        output_channel_ranges_data_frame)

    # add vertical lines to highlight problematic channels
    for channel in problematic_output_channels:
        add_vertical_line_to_figure(channel, plot)

    # push plot to server document
    column_layout = column(histogram_plot, output_channel_ranges_as_column_data_source)
    layout = row(plot, column_layout)
    layout_with_title = add_title(layout, layer_name)

    return layout_with_title

def visualize_changes_after_optimization(
        old_model: torch.nn.Module,
        new_model: torch.nn.Module,
        results_dir: str,
        selected_layers: List = None
) -> List[plotting.figure]:
    """
    Visualizes changes before and after some optimization has been applied to a model.

    :param old_model: pytorch model before optimization
    :param new_model: pytorch model after optimization
    :param results_dir: Directory to save the Bokeh plots
    :param selected_layers: a list of layers a user can choose to have visualized. If selected layers is None,
        all Linear and Conv layers will be visualized.
    :return: A list of bokeh plots
    """
    file_path = os.path.join(results_dir, 'visualize_changes_after_optimization.html')
    plotting.output_file(file_path)
    subplots = []
    if selected_layers:
        for name, module in new_model.named_modules():
            if name in selected_layers and hasattr(module, "weight"):
                old_model_module = get_layer_by_name(old_model, name)
                new_model_module = module
                subplots.append(
                    visualize_changes_after_optimization_single_layer(
                        name, old_model_module, new_model_module
                    )
                )

    else:
        for name, module in new_model.named_modules():
            if hasattr(module, "weight") and\
                    isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
                old_model_module = get_layer_by_name(old_model, name)
                new_model_module = module
                subplots.append(
                    visualize_changes_after_optimization_single_layer(
                        name, old_model_module, new_model_module
                    )
                )
    plotting.save(column(subplots))
    return subplots



def visualize_weight_ranges_single_layer(layer, layer_name, scatter_plot=False):
    """
    Given a layer, visualizes weight ranges with scatter plots and line plots
    :param layer: layer with weights
    :param layer_name: layer name
    :param scatter_plot: Include scatter plot in plots
    :return: None
    """
    device = get_device(layer)
    layer.cpu()
    
    # 获取当前层的权重
    layer_weights = pd.DataFrame(get_weights(layer))
    
    # 得到每一个权重的通道的统计量
    layer_weights_summary_statistics = layer_weights.describe().T

    line_plots = line_plot_summary_statistics_model(layer_name=layer_name,
                                                    layer_weights_data_frame=layer_weights_summary_statistics,
                                                    width=1000, height=700)

    if scatter_plot:
        scatter_plot_mean, scatter_plot_min = scatter_plot_summary_stats(layer_weights_summary_statistics,
                                                                         x_axis_label_mean="Mean Weights Per Output Channel",
                                                                         y_axis_label_mean="Std Per Output Channel",
                                                                         title_mean="Mean vs Standard Deviation: " + layer_name,
                                                                         x_axis_label_min="Min Weights Per Output Channel",
                                                                         y_axis_label_min="Max Weights Per Output Channel",
                                                                         title_min="Minimum vs Maximum: " + layer_name)

        scatter_plots_layout = row(scatter_plot_mean, scatter_plot_min)

        layout = column(scatter_plots_layout, line_plots)
    else:
        layout = line_plots
    layout_with_title = add_title(layout, layer_name)

    # Move layer back to device
    layer.to(device=device)
    return layout_with_title


def visualize_weight_ranges(
        model: torch.nn.Module,
        results_dir: str,
        selected_layers: List = None
) -> List[plotting.figure]:
    """
    Visualizes weight ranges for each layer through a scatter plot showing mean plotted against the standard deviation,
    the minimum plotted against the max, and a line plot with min, max, and mean for each output channel.

    :param model: pytorch model
    :param selected_layers:  a list of layers a user can choose to have visualized. If selected layers is None,
        all Linear and Conv layers will be visualized.
    :param results_dir: Directory to save the Bokeh plots
    :return: A list of bokeh plots
    """

    file_path = os.path.join(results_dir, 'visualize_weight_ranges.html')
    plotting.output_file(file_path)
    subplots = []
    if selected_layers:
        for name, module in model.named_modules():
            if name in selected_layers and hasattr(module, "weight"):
                subplots.append(visualize_weight_ranges_single_layer(module, name))
    else:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
                subplots.append(visualize_weight_ranges_single_layer(module, name))

    plotting.save(column(subplots))
    return subplots


def visualize_relative_weight_ranges_to_identify_problematic_layers(
        model: torch.nn.Module,
        results_dir: str,
        selected_layers: List = None
) -> List[plotting.figure]:
    """
    For each of the selected layers, publishes a line plot showing  weight ranges for each layer, summary statistics
    for relative weight ranges, and a histogram showing weight ranges of output channels
    with respect to the minimum weight range.

    :param model: pytorch model
    :param results_dir: Directory to save the Bokeh plots
    :param selected_layers: a list of layers a user can choose to have visualized. If selected layers is None,
        all Linear and Conv layers will be visualized.
    :return: A list of bokeh plots
    """

    file_path = os.path.join(results_dir, 'visualize_relative_weight_ranges_to_identify_problematic_layers.html')
    plotting.output_file(file_path)
    subplots = []
    # layer name -> module weights data frame mapping
    if not selected_layers:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and \
                isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
                subplots.append(visualize_relative_weight_ranges_single_layer(module, name))
    else:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and \
                    isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)) and \
                    name in selected_layers:
                subplots.append(visualize_relative_weight_ranges_single_layer(module, name))

    plotting.save(column(subplots))
    return subplots



def visualize_torch_model_weights(model: torch.nn.Module , model_name: str, results_dir: str = None):
    """
    Load a model and visualize its weight distributions.
    
    :param model_name: Name of the model to load (e.g., 'resnet18', 'vgg16', 'densenet121')
    :param results_dir: Directory to save the visualization results. If None, will create based on model name
    :param pretrained: Whether to load pretrained weights
    """
    # Set default results directory if none provided
    if results_dir is None:
        results_dir = f"{model_name}_visualization_results"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model
    try:
        model.eval()
    except AttributeError:
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    
    print(f"Loaded {model_name} model")
    print("Generating weight range visualizations...")
    
    # Visualize weight ranges for all layers
    visualize_weight_ranges(
        model=model,
        results_dir=results_dir
    )
    print("Generating relative weight range visualizations...")
    
    # Visualize relative weight ranges to identify potential problematic layers
    visualize_relative_weight_ranges_to_identify_problematic_layers(
        model=model,
        results_dir=results_dir
    )
    
    print_utils.print_colored_box(f"Visualization results have been saved to: {results_dir}")



def get_onnx_model_weights(onnx_path: str) -> Dict[str, np.ndarray]:
    """
    Extract weights from an ONNX model.
    
    :param onnx_path: Path to the ONNX model file
    :return: Dictionary of weight names and their corresponding numpy arrays
    """
    model = onnx.load(onnx_path)
    weights = {}
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT:
            weights[init.name] = onnx.numpy_helper.to_array(init)
    return weights



def visualize_onnx_model_weights(onnx_path: str, model_name: str, results_dir: str = None):
    """
    Load an ONNX model and visualize its weight distributions.
    
    :param onnx_path: Path to the ONNX model file
    :param model_name: Name of the model (for naming the results directory)
    :param results_dir: Directory to save the visualization results. If None, will create based on model name
    """
    # Set default results directory if none provided
    if results_dir is None:
        results_dir = f"{model_name}_onnx_visualization_results"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load ONNX model weights
    try:
        weights = get_onnx_model_weights(onnx_path)
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model from {onnx_path}. Error: {str(e)}")
    
    print(f"Loaded {model_name} ONNX model from {onnx_path}")
    print(f"Found {len(weights)} weight tensors")
    
    print("Generating weight range visualizations...")
    
    # 修改这里：创建一个更合适的伪模型类
    class PseudoModel:
        def __init__(self, weights):
            self._weights = weights
            
        def named_modules(self):
            # 返回权重字典的items
            return self._weights.items()
            
        def parameters(self):
            # 为了兼容某些可能需要parameters的函数
            return iter(self._weights.values())
    
    # 使用新的PseudoModel类
    pseudo_model = PseudoModel(weights)
    
    # Visualize weight ranges for all layers
    visualize_weight_ranges(
        model=pseudo_model,
        results_dir=results_dir
    )
    
    print("Generating relative weight range visualizations...")
    # Visualize relative weight ranges to identify potential problematic layers
    visualize_relative_weight_ranges_to_identify_problematic_layers(
        model=pseudo_model,
        results_dir=results_dir
    )
    
    print_utils.print_colored_box(f"Visualization results have been saved to: {results_dir}")

   
if __name__ == "__main__":
    onnx_path = "/share/cdd/onnx_models/od_bev_0324.onnx"
    input_name, output_name = get_onnx_model_io_info(onnx_path)
    
    
    # Example usage with different models
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    visualize_torch_model_weights(model, "resnet18")
    # visualize_onnx_model_weights(onnx_path, "od_bev_0324")