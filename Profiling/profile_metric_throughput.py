import torch
import numpy as np
import time
# from efficientnet_pytorch import EfficientNet
import torchvision.models as models

from spectrautils.print_utils import *

# ===================================================================
# 1. 将配置参数提取为变量，避免使用“魔法数字”
# ===================================================================
BATCH_SIZE = 64
WARMUP_ITER = 50
TEST_ITER = 100
MODEL_NAME = 'resnet50'
# ===================================================================


# ===================================================================
# 2. 自动选择计算设备，让代码更具移植性
# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 3. 加载预训练模型，并使用 .to(device) 将其移动到目标设备
model = models.__dict__[MODEL_NAME](weights='IMAGENET1K_V1')
model.to(device)
model.eval()

# 创建假的输入数据，同样移动到目标设备
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, dtype=torch.float).to(device)

with torch.no_grad():
    for rep in range(WARMUP_ITER):
        _ = model(dummy_input)
torch.cuda.synchronize()
print("Warm-up finished.")

total_time = 0
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
with torch.no_grad():
    for rep in range(TEST_ITER):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) / 1000
        total_time += curr_time

Throughput = (TEST_ITER * BATCH_SIZE) / total_time
print_colored_text(f'Batch Size: {BATCH_SIZE}', "green")
print_colored_text(f'Average inference time: {total_time / TEST_ITER * 1000:.2f} ms', "green")
print_colored_text(f'Final Throughput: {Throughput:.2f} images/sec', "green")

