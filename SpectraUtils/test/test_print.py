'''
version: 1.0.0
Author: BruceCui
Date: 2024-05-12 23:15:38
LastEditors: BruceCui
LastEditTime: 2025-02-03 16:24:47
'''

import os
import sys


# 获取项目的根目录并添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from printk.printk import print_colored_box, print_colored_box_line
print_colored_box("hello world", 60, text_color='green', box_color='yellow', align='center')
print_colored_box("请在此脚本目录运行该脚本", align='center')
print_colored_box_line("警告", "请立即检查系统！", attrs=['bold'], text_color='red', box_color='yellow', box_width=50)


onnx_name = ["1.onnx", "2.onnx", "3.onnx", "4.onnx", "5.onnx", "6.onnx", "7.onnx", "8.onnx", "9.onnx", "10.onnx"]
print_colored_box(onnx_name, attrs=['bold'], text_color='red', box_color='yellow')



import torch.distributed as dist

idx = 1

try:
    # 尝试初始化分布式环境（如果需要的话）
    # dist.init_process_group(backend="nccl")  # 取消注释并配置适当的后端
    
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print_colored_box("在分布式环境中：Epoch {} 已完成！".format(idx + 1))
    else:
        print_colored_box("不在分布式环境中：Epoch {} 已完成！".format(idx + 1))
except Exception as e:
    print_colored_box("发生异常：{}".format(str(e)))
    print_colored_box("在非分布式环境中：Epoch {} 已完成！".format(idx + 1))
