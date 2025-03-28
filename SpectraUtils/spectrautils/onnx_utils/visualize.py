import time, os
import numpy as np
from collections import OrderedDict
# import onnxruntime as ort
from tqdm import tqdm
from typing import List, Dict, Union
import torch, torchvision, onnx
from  torchvision.models import ResNet18_Weights
import pandas as pd
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from .common_tools import *
from .config_spectrautils import config_spectrautils as config

from datetime import datetime
import holoviews as hv
import hvplot.pandas  # pylint:disable=unused-import

from bokeh import plotting
from bokeh.layouts import row, column, Spacer
from bokeh.plotting import figure
from bokeh.models import DataTable, TableColumn, CustomJS, Div
from bokeh.models import HoverTool, ColumnDataSource, Span, WheelZoomTool
from spectrautils import logging_utils, print_utils


# os.environ["CUDA_VISIBLE_DEVICES"]="2"

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
    
    layout = PlotsLayout()
    
    with torch.no_grad():
        old_model_module.cpu()
        new_model_module.cpu()

    
    layout.title = name
    layer_weights_summary_statistics_old = pd.DataFrame(get_torch_weights(old_model_module)).describe().T
    layer_weights_summary_statistics_new = pd.DataFrame(get_torch_weights(new_model_module)).describe().T

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


def detect_outlier_channels(data_frame, column="relative range", factor=1.5):
    """
    检测相对权重范围的异常值。
    
    Args:
        data_frame_with_relative_ranges: 包含"relative range"列的pandas数据框
    
    Returns:
        list: 具有非常大的权重范围的输出通道列表
    """
    # 计算四分位数和四分位距
    Q1 = data_frame.quantile(0.25)
    Q3 = data_frame.quantile(0.75)
    IQR = Q3 - Q1
    
    # 计算上下界
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # 检测异常值
    outliers = {
        'upper': data_frame[data_frame > upper_bound].index.tolist(),
        'lower': data_frame[data_frame < lower_bound].index.tolist()
    }
    
    return outliers

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
    output_channels_needed = detect_outlier_channels(all_output_channel_ranges, "relative range")

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
    layer_weights_data_frame = pd.DataFrame(get_torch_weights(layer)).describe().T
    plot = line_plot_summary_statistics_model(layer_name, layer_weights_data_frame, width=1150, height=700)

    # list of problematic output channels, data frame containing magnitude of range in each output channel
    problematic_output_channels, output_channel_ranges_data_frame = identify_problematic_output_channels(layer_weights_data_frame)

    histogram_plot = histogram(output_channel_ranges_data_frame, "relative range", 75,
                               x_label="Weight Range Relative to Smallest Output Channel",
                               y_label="Count",
                               title="Relative Ranges For All Output Channels")
    
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.describe().T.to_frame()
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.drop("count")

    output_channel_ranges_as_column_data_source = convert_pandas_data_frame_to_bokeh_data_table(
        output_channel_ranges_data_frame)

    # add vertical lines to highlight problematic channels
    for channel in problematic_output_channels["upper"]:
        add_vertical_line_to_figure(channel, plot)
    
    for channel in problematic_output_channels["lower"]:
        add_vertical_line_to_figure(channel, plot)

    # push plot to server document
    column_layout = column(histogram_plot, output_channel_ranges_as_column_data_source)
    layout = row(plot, column_layout)
    layout_with_title = add_title(layout, layer_name)

    return layout_with_title


def visualize_relative_onnx_weight_ranges_single_layer(layer_weights_data_frame, layer_name):
    """
    publishes a line plot showing  weight ranges for each layer, summary statistics
    for relative weight ranges, and a histogram showing weight ranges of output channels
    :param model: p
    :return:
    """
    # layer_weights_data_frame = pd.DataFrame(get_torch_weights(layer)).describe().T
    plot = line_plot_summary_statistics_model(layer_name, layer_weights_data_frame, width=1150, height=700)

    # list of problematic output channels, data frame containing magnitude of range in each output channel
    problematic_output_channels, output_channel_ranges_data_frame = identify_problematic_output_channels(layer_weights_data_frame)

    histogram_plot = histogram(output_channel_ranges_data_frame,
                               "relative range", 
                               75,
                               x_label="Weight Range Relative to Smallest Output Channel",
                               y_label="Count",
                               title="Relative Ranges For All Output Channels")
    
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.describe().T.to_frame()
    output_channel_ranges_data_frame = output_channel_ranges_data_frame.drop("count")

    output_channel_ranges_as_column_data_source = convert_pandas_data_frame_to_bokeh_data_table(
        output_channel_ranges_data_frame)

        
    # add vertical lines to highlight problematic channels
    for channel in problematic_output_channels["upper"]:
        add_vertical_line_to_figure(channel, plot)
    
    for channel in problematic_output_channels["lower"]:
        add_vertical_line_to_figure(channel, plot)


    # push plot to server document
    column_layout = column(histogram_plot, output_channel_ranges_as_column_data_source)
    layout = row(plot, column_layout)
    layout_with_title = add_title(layout, layer_name)

    return layout_with_title


def visualize_relative_onnx_weight_ranges_single_layer_quick(layer_weights_data_frame, layer_name):
    """
    Creates a static-like Bokeh plot showing weight ranges for each layer and a histogram.
    The histogram is placed on the right side of the line plot.
    """
    
    problematic_output_channels, output_channel_ranges_data_frame = identify_problematic_output_channels(layer_weights_data_frame)

    # Create line plot
    source = ColumnDataSource(layer_weights_data_frame)
    p = figure(title=f"Weight Ranges per Output Channel: {layer_name}", 
               x_axis_label="Output Channels", 
               y_axis_label="Summary Statistics",
               width=1000, 
               height=500,
               tools="")  # Empty string means no tools

    p.line('index', 'min', line_color='#2171b5', line_width=2, legend_label="Min", source=source)
    p.line('index', 'max', line_color='green', line_width=2, legend_label="Max", source=source)
    p.line('index', 'mean', line_color='orange', line_width=2, legend_label="Mean", source=source)
    
    p.legend.click_policy = "hide"

    # Add hover tool (this will still work even without other interactive tools)
    hover = HoverTool(tooltips=[
        ("Channel", "@index"),
        ("Min", "@min{0.00}"),
        ("Max", "@max{0.00}"),
        ("Mean", "@mean{0.00}")
    ])
    p.add_tools(hover)
    
    # 添加垂直线标记异常通道
    for channel in problematic_output_channels["upper"]:
        p.add_layout(Span(location=channel, dimension='height', line_color='#F01DA9', 
                         line_dash='dashed', line_width=1))
    
    for channel in problematic_output_channels["lower"]:
        p.add_layout(Span(location=channel, dimension='height', line_color='#1322ED', 
                         line_dash='dashed', line_width=1))
        

    # Create histogram 直方图
    hist, edges = np.histogram(output_channel_ranges_data_frame, bins=50)
    hist_df = pd.DataFrame({'count': hist, 'left': edges[:-1], 'right': edges[1:]})
    hist_source = ColumnDataSource(hist_df)

    h = figure(title="Relative Ranges For All Output Channels",
               x_axis_label="Weight Range Relative to Smallest Output Channel",
               y_axis_label="Count",
               width=600, height=500,
               tools="")

    h.quad(bottom=0, top='count', left='left', right='right', source=hist_source,
           fill_color="navy", line_color="white", alpha=0.5)

    
    # 在直方图中添加垂直线标记异常值的阈值
    Q1 = layer_weights_data_frame["relative range"].quantile(0.25)
    Q3 = layer_weights_data_frame["relative range"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    h.add_layout(Span(location=upper_bound, dimension='height', line_color='#F01DA9',
                     line_dash='dashed', line_width=2))
    h.add_layout(Span(location=lower_bound, dimension='height', line_color='#1322ED',
                     line_dash='dashed', line_width=2))
    
    
    # ==============================================
    # 创建一个居中的布局
    layout = row(
        p, h,
        align="center"  # 使用 align="center" 来确保布局居中
    )

    # 使用column包装row，以便应用全局样式
    final_layout = column(
        layout,
        sizing_mode='stretch_width',  # 使布局在水平方向上填充
        align="center",  # 在 column 层级也确保居中
        styles={'margin': '0 auto', 'max-width': '2600px'}  # 设置最大宽度并居中
    )
    # ==============================================
    return final_layout


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


def visualize_weight_ranges_single_layer(layer, layer_name, use_dynamic=True, is_onnx=False):
    """
    Given a layer, visualizes weight ranges with scatter plots and line plots
    :param layer: layer with weights or layer weights summary statistics for ONNX
    :param layer_name: layer name
    :param is_onnx: Boolean flag to indicate if the input is from an ONNX model
    :return: Bokeh layout
    """
    if not is_onnx:
        device = get_device(layer)
        layer.cpu()
        layer_weights = pd.DataFrame(get_torch_weights(layer))
        layer_weights_summary_statistics = layer_weights.describe().T
    else:
        layer_weights_summary_statistics = layer
    
    if use_dynamic:   
        line_plots = line_plot_summary_statistics_model(
            layer_name=layer_name,
            layer_weights_data_frame=layer_weights_summary_statistics,
            width=1500,
            height=700,
        )

        scatter_plot_mean, scatter_plot_min = scatter_plot_summary_stats(
            layer_weights_summary_statistics,
            x_axis_label_mean="Mean Weights Per Output Channel",
            y_axis_label_mean="Std Per Output Channel",
            title_mean="Mean vs Standard Deviation: " + layer_name,
            x_axis_label_min="Min Weights Per Output Channel",
            y_axis_label_min="Max Weights Per Output Channel",
            title_min="Minimum vs Maximum: " + layer_name
        )

        scatter_plots_layout = row(scatter_plot_mean, scatter_plot_min, align="center")
        layout = column(scatter_plots_layout, Spacer(height=20), line_plots, align="center", sizing_mode='scale_width')
    
    else:
        # 静态模式 - 使用基本的figure而不添加交互工具
        plot = figure(
            title=f"Weight Ranges per Output Channel: {layer_name}",
            x_axis_label="Output Channels",
            y_axis_label="Summary Statistics",
            width=1500,
            height=500,
            tools="",  # 不添加任何交互工具
            toolbar_location=None  # 移除工具栏
        )

        # 静态模式 - 创建线图和散点图
        line_plot = figure(
            title=f"Weight Ranges per Output Channel: {layer_name}",
            x_axis_label="Output Channels",
            y_axis_label="Summary Statistics",
            width=1500,
            height=500,
            tools="",
            toolbar_location=None
        )

        # 添加线图
        line_plot.line(layer_weights_summary_statistics.index, layer_weights_summary_statistics['min'], line_color='#2171b5', legend_label="Min", line_width=1)
        line_plot.line(layer_weights_summary_statistics.index, layer_weights_summary_statistics['max'], line_color='green', legend_label="Max", line_width=1)
        line_plot.line(layer_weights_summary_statistics.index, layer_weights_summary_statistics['mean'], line_color='orange', legend_label="Mean", line_width=1)

        line_plot.legend.location = "top_right"

        # 创建散点图 - Mean vs Std
        scatter_mean = figure(
            title="Mean vs Standard Deviation",
            x_axis_label="Mean Weights Per Output Channel",
            y_axis_label="Std Per Output Channel",
            width=500,
            height=500,
            tools="",
            toolbar_location=None
        )
        scatter_mean.circle(layer_weights_summary_statistics['mean'], layer_weights_summary_statistics['std'], size=5, color="navy", alpha=0.5)

        # 创建散点图 - Min vs Max
        scatter_min_max = figure(
            title="Minimum vs Maximum",
            x_axis_label="Min Weights Per Output Channel",
            y_axis_label="Max Weights Per Output Channel",
            width=500,
            height=500,
            tools="",
            toolbar_location=None
        )
        scatter_min_max.circle(layer_weights_summary_statistics['min'], layer_weights_summary_statistics['max'], size=5, color="firebrick", alpha=0.5)

        # 组合所有图表
        scatter_plots_layout = row(scatter_mean, scatter_min_max, align="center")
        layout = column(scatter_plots_layout, Spacer(height=20), line_plot, align="center", sizing_mode='scale_width')

        
    layout_with_title = add_title(layout, layer_name)

    if not is_onnx:
        layer.to(device=device)

    return layout_with_title



def visualize_weight_ranges(
        model: Union[torch.nn.Module, OrderedDict],
        results_dir: str,
        num_processes: int = 16,
        selected_layers: List = None,
        is_onnx: bool = False
) -> List[plotting.figure]:
    """
    Visualizes weight ranges for each layer through a scatter plot showing mean plotted against the standard deviation,
    the minimum plotted against the max, and a line plot with min, max, and mean for each output channel.

    :param model: pytorch model or OrderedDict of ONNX weights
    :param results_dir: Directory to save the Bokeh plots
    :param num_processes: Number of processes to use for parallel processing (only for ONNX)
    :param selected_layers: a list of layers a user can choose to have visualized. If selected layers is None,
        all Linear and Conv layers will be visualized.
    :param is_onnx: Boolean flag to indicate if the input is an ONNX model
    :return: A list of bokeh plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'{timestamp}_visualize_{"onnx" if is_onnx else "pytorch"}_weight_ranges.html'
    file_path = os.path.join(results_dir, file_name)
    plotting.output_file(file_path)
    
    use_dynamic = True
    subplots = []
    
    if is_onnx:
        tensor_nums = len(model)
        
        
        if tensor_nums > 300:
            use_dynamic = False
            
        if selected_layers:
            raise NotImplementedError("Visualization for selected ONNX layers is not implemented yet.")
        
        # 使用ProcessPoolExecutor进行并行处理
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_layer_data, item) for item in model.items()]
            
            # 使用tqdm显示进度
            for future in tqdm(futures, total=len(model), desc="Processing layers", unit="layer"):
                name, layer_weights_summary_statistics = future.result()
                
                # 在主进程中创建Bokeh图表
                subplot = visualize_weight_ranges_single_layer(layer_weights_summary_statistics, name, use_dynamic=use_dynamic, is_onnx=True)
                    
                subplots.append(subplot)
    else:
        if selected_layers:
            for name, module in model.named_modules():
                if name in selected_layers and hasattr(module, "weight"):
                    subplots.append(visualize_weight_ranges_single_layer(module, name))
        else:
            for name, module in model.named_modules():
                if hasattr(module, "weight") and isinstance(module, tuple(config["LAYER_HAS_WEIGHT_TORCH"])):
                    subplots.append(visualize_weight_ranges_single_layer(module, name))
                    
    
    # =========================================================
    # 创建一个居中的布局
    layout = column(
        subplots,
        sizing_mode='stretch_width',  # 使布局在水平方向上填充
        styles={'margin': '0 auto', 'max-width': '1600px'}  # 设置最大宽度并居中
    )

    plotting.save(layout)
    print_utils.print_colored_box(f"Visualization saved to: {file_path}")
    # =========================================================
    return subplots

def visualize_relative_weight_ranges_to_identify_problematic_layers(
    model: Union[torch.nn.Module, OrderedDict],
    results_dir: str,
    num_processes: int = 16,
    selected_layers: List = None,
    is_onnx: bool = False
) -> List[plotting.figure]:
    """
    Visualizes relative weight ranges for each layer to help identify problematic layers.
    Creates scatter plots showing the ratio of standard deviation to mean against the ratio of max to min.

    Args:
        model: PyTorch model or OrderedDict of ONNX weights
        results_dir: Directory to save the Bokeh plots
        num_processes: Number of processes for parallel processing (ONNX only)
        selected_layers: List of layer names to visualize. If None, visualizes all layers with weights
        is_onnx: Boolean flag to indicate if input is an ONNX model

    Returns:
        List of Bokeh plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'{timestamp}_visualize_{"onnx" if is_onnx else "pytorch"}_relative_weight_ranges.html'
    file_path = os.path.join(results_dir, file_name)
    plotting.output_file(file_path)

    subplots = []
    
    if is_onnx:
        weights = model
        tensor_weights_num = len(weights)
        if selected_layers:
            raise NotImplementedError("Visualization for selected ONNX layers is not implemented yet.")
        
        # 使用ProcessPoolExecutor进行并行处理
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_layer_data, item) for item in weights.items()]
            
            # 使用tqdm显示进度
            for future in tqdm(futures, total=len(weights), desc="Processing relative range", unit="layer"):
                name, layer_weights_summary_statistics = future.result()
                
                # 在主进程中创建Bokeh图表
                if tensor_weights_num > 300 :
                    subplot = visualize_relative_onnx_weight_ranges_single_layer_quick(layer_weights_summary_statistics, name)
                else:
                    subplot = visualize_relative_onnx_weight_ranges_single_layer(layer_weights_summary_statistics, name)
                subplots.append(subplot)
        # plotting.save(column(subplots, sizing_mode='scale_width', align='center'))
    else:
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

    # Create centered layout
    layout = column(
        subplots,
        sizing_mode='stretch_width',
        styles={'margin': '0 auto', 'max-width': '1600px'}
    )

    plotting.save(layout)
    print(f"Visualization saved to: {file_path}")
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
        results_dir=results_dir,
        is_onnx=False
    )
        
    # Visualize relative weight ranges to identify potential problematic layers
    visualize_relative_weight_ranges_to_identify_problematic_layers(
        model=model,
        results_dir=results_dir,
        is_onnx=False
    )
    
    print_utils.print_colored_box(f"Visualization results have been saved to: {results_dir}")


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
    
    try:
        weights = get_onnx_model_weights(onnx_path)
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model from {onnx_path}. Error: {str(e)}")
    
    print(f"Loaded {model_name} ONNX model from {onnx_path}")
    
    print_utils.print_colored_box(f"Found {len(weights)} weight tensors")
    
    # 只可视化权重
    visualize_weight_ranges(weights, results_dir, is_onnx=True)
    
    # 可视化权重和有问题的输出
    visualize_relative_weight_ranges_to_identify_problematic_layers(weights, results_dir, is_onnx=True)
    
    print_utils.print_colored_box(f"Visualization results have been saved to: {results_dir}")

    
if __name__ == "__main__":
    # onnx_path = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/SpectraUtils/spectrautils/resnet18_official.onnx"
    onnx_path = "/share/cdd/onnx_models/od_bev_0317.onnx"
    
    # Example usage with different models
    model_old = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # 加载你的本地模型
    model_new = torch.load('/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/SpectraUtils/spectrautils/resnet_model_cle_bc.pt')

    # model_new = torchvision.models.resnet18(pretrained=False)

    
    # visualize_torch_model_weights(model_new, "resnet18_new")
    # visualize_changes_after_optimization(model_old, model_new, "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/SpectraUtils")
    # visualize_onnx_model_weights(onnx_path, "resnet18")
    visualize_onnx_model_weights(onnx_path, "od_bev")
    
    