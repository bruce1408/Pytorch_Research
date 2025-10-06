import os
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, datasets

from model import pyramidnet  # 假设您的模型定义在这个文件里
from spectrautils import logging_utils # 假设您的日志工具在这里
os.environ["OMP_NUM_THREADS"] = "4"
# --- MODIFIED: 简化 argparse，移除不再需要的分布式参数 ---
parser = argparse.ArgumentParser(description='cifar10 classification models with torchrun')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--resume', default=None, help='Checkpoint to resume training from')
parser.add_argument('--batch_size', type=int, default=128, help='Total batch size across all GPUs')
parser.add_argument('--num_workers', type=int, default=8, help='Total number of data loading workers across all GPUs')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument("--output", default="./dist_output_torchrun", help="Path to save model checkpoints")

# ... 您原来的 AverageMeter, ProgressMeter, accuracy 函数可以保持不变 ...
# (为了简洁，这里省略了这些辅助类和函数的代码，您可以直接从原文件复制)
# ...
# 假设 AverageMeter, ProgressMeter, accuracy, train 函数都已定义好
# ...

def main():
    args = parser.parse_args()
    
    # --- NEW: torchrun 初始化流程 ---
    # 1. 初始化进程组，torchrun 会自动设置 MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    dist.init_process_group(backend="nccl")
    
    # 2. 从环境变量获取当前进程的 local_rank，并设置当前设备
    # local_rank 是指在当前机器上的 GPU 编号
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Starting DDP training on {world_size} GPUs.")
        os.makedirs(args.output, exist_ok=True)
    
    # --- 数据准备 ---
    # batch_size 和 num_workers 需要根据 GPU 数量进行分配
    per_device_batch_size = args.batch_size // world_size
    per_device_num_workers = args.num_workers // world_size
    if rank == 0:
        print(f"Rank {rank}: Per-device batch size: {per_device_batch_size}, Workers: {per_device_num_workers}")

    # 使用伪数据以保证可运行
    train_dataset = datasets.FakeData(50000, (3, 32, 32), 10, transforms.ToTensor())
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        num_workers=per_device_num_workers,
        sampler=train_sampler,
        pin_memory=True
    )
    
    # --- 模型准备 ---
    print(f"Rank {rank}: Initializing model...")
    net = pyramidnet().to(local_rank)
    # 使用DDP包装模型，注意 device_ids 和 output_device 的设置
    
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # --- 训练循环 ---
    for epoch in range(args.epochs):
        # 必须在每个epoch开始前设置sampler的epoch，以保证shuffle正常工作
        train_sampler.set_epoch(epoch)
        # 调用您原来的 train 函数，但需要做一些适配
        # (为了演示，这里简化了train函数的调用)
        # train(net, criterion, optimizer, train_loader, local_rank, epoch)
        
        # --- 简化的训练循环示例 ---
        net.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if rank == 0:
            print(f"Epoch: {epoch+1}/{args.epochs} | Average Loss: {total_loss / len(train_loader):.4f}")

    # --- 训练结束 ---
    if rank == 0:
        print("Training complete. Saving model...")
        # 保存模型时，需要保存 model.module 的 state_dict
        state = {"model": net.module.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, os.path.join(args.output, "final.pth"))

    # 销毁进程组
    dist.destroy_process_group()

if __name__ == '__main__':
    # --- DELETED: 删除了整个 mp.spawn 逻辑 ---
    # `torchrun` 会负责启动多个进程来运行下面的 main 函数
    # torchrun --nproc_per_node=4 train_demo_2_torchrun.py --batch_size 128 --epochs 10
    main()