'''
version: 1.0.0
Author: BruceCui
Date: 2024-05-12 23:15:38
LastEditors: BruceCui
LastEditTime: 2025-02-03 16:24:47
'''

import os
import sys
import torch.distributed as dist


# 获取项目的根目录并添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from  spectrautils.print_utils import print_colored_box, print_colored_box_line, print_colored_text

print_colored_text("\n--- Testing print_colored_box ---", 'cyan')
# 测试简单字符串和不同对齐方式
print_colored_box("Hello World! (left aligned)", bbox_width=60, text_color='green', box_color='yellow', align='left')
print_colored_box("Hello World! (center aligned)", bbox_width=60, text_color='green', box_color='yellow', align='center')
print_colored_box("Hello World! (right aligned)", bbox_width=60, text_color='green', box_color='yellow', align='right')

# 测试列表输入和中文字符，这对于显示文件名或步骤列表很有用
onnx_name = ["1.onnx", "2.onnx", "3.onnx", "4.onnx", "5.onnx", "这是一个中文文件名.onnx"]
print_colored_box(onnx_name, attrs=['bold'], text_color='red', box_color='yellow', align='center')

# 测试一个多行的警告信息
warning_message = [
    "警告: 请立即检查系统！",
    "请按照如下顺序操作:",
    "1. clone the code",
    "2. debug the code",
    "3. run the code"
]
print_colored_box(warning_message, text_color='red', box_color='yellow', align='left')


# --- 测试 print_colored_box_line ---
print_colored_text("\n--- Testing print_colored_box_line ---", 'cyan')
# 这个函数适合打印标题和单行消息
print_colored_box_line("任务状态", "模型训练已完成", 
                       attrs=['bold'], text_color='white', box_color='blue', box_width=60)


# --- 测试 print_colored_text ---
print_colored_text("\n--- Testing print_colored_text ---", 'cyan')
# 这个函数适合打印简单的彩色文本，比如成功或失败的消息
conversion_info = \
f"""
{'-'*60}
PyTorch to ONNX Conversion Details:
- Input Shape      : 1, 3, 1200, 1200
- Model Config     : /path/to/config.py
- Model Weight     : /path/to/weights.pth
- Output ONNX File : /path/to/output.onnx
{'-'*60}
"""
print_colored_text(f"✔ Conversion Successful!", text_color='green', attrs=['bold'])
print_colored_text(conversion_info, text_color='white')
print_colored_text(f"❌ Conversion Failed! Check logs for details.", text_color='red', attrs=['bold'])


idx = 1

try:
    # 尝试初始化分布式环境（如果需要的话）
    # dist.init_process_group(backend="nccl")  # 取消注释并配置适当的后端
    
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print_colored_box("在分布式环境中: Epoch {} 已完成！".format(idx + 1))
    else:
        print_colored_box("不在分布式环境中: Epoch {} 已完成！".format(idx + 1))
except Exception as e:
    print_colored_box("发生异常: {}".format(str(e)))
    print_colored_box("在非分布式环境中: Epoch {} 已完成！".format(idx + 1))

