import torch
import time
import sys
import numpy as np
import torchvision.models as models
from spectrautils.print_utils import *

# ===================================================================
# 1. 配置区域 (Configuration)
# ===================================================================
CONFIG = {
    'BATCH_SIZE': 64,
    'WARMUP_ITER': 50,
    'TEST_ITER': 100,
    'MODEL_NAME': 'resnet50',
    'IMG_SIZE': (3, 224, 224)  # 显式定义输入尺寸
}

# ===================================================================
# 3. 核心评测函数
# ===================================================================
def benchmark_model():
    # ----------------------
    # A. 环境与设备准备
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 如果是 GPU，开启 cudnn benchmark 加速 (针对固定输入尺寸优化)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ----------------------
    # B. 模型加载
    # ----------------------
    print(f"Loading model: {CONFIG['MODEL_NAME']} ...")
    try:
        model = models.__dict__[CONFIG['MODEL_NAME']](weights='IMAGENET1K_V1')
    except KeyError:
        print_colored_text(f"Error: Model {CONFIG['MODEL_NAME']} not found in torchvision.", "red")
        return

    model.to(device)
    model.eval()

    # ----------------------
    # C. 数据准备
    # ----------------------
    input_shape = (CONFIG['BATCH_SIZE'], *CONFIG['IMG_SIZE'])
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    print(f"Input Shape: {input_shape}")

    # ----------------------
    # D. 热身 (Warm-up)
    # ----------------------
    print(f"Warming up for {CONFIG['WARMUP_ITER']} iterations...")
    with torch.no_grad():
        for _ in range(CONFIG['WARMUP_ITER']):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print("Warm-up finished. Starting benchmark...")

    # ----------------------
    # E. 正式测试 (Benchmark)
    # ----------------------
    timings = [] # 存储每次迭代的时间，方便后续分析
    
    with torch.no_grad():
        for rep in range(CONFIG['TEST_ITER']):
            # 根据设备选择计时方式
            if device.type == 'cuda':
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize() # 等待 GPU 完成
                curr_time = starter.elapsed_time(ender) / 1000 # 转换为秒
            else:
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                curr_time = end_time - start_time
            
            timings.append(curr_time)

    # ----------------------
    # F. 结果计算与输出
    # ----------------------
    timings = np.array(timings)
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    
    # 吞吐量 = (Batch Size) / (单次迭代平均时间)
    throughput = CONFIG['BATCH_SIZE'] / avg_time

    print("-" * 30)
    print_colored_text(f"Model: {CONFIG['MODEL_NAME']}", "yellow")
    print_colored_text(f"Batch Size: {CONFIG['BATCH_SIZE']}", "green")
    print_colored_text(f"Inference Latency: {avg_time * 1000:.2f} ± {std_time * 1000:.2f} ms", "green")
    print_colored_text(f"Final Throughput: {throughput:.2f} images/sec", "green")

if __name__ == "__main__":
    benchmark_model()