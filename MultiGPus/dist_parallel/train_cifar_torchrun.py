# -*- coding: UTF-8 -*-
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from model import pyramidnet  # 假设您的模型定义在这个文件里

def setup_distributed():
    """使用 torchrun 时，初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()

def main():
    # --- 1. 参数解析 (简化后) ---
    parser = argparse.ArgumentParser(description='Optimized CIFAR10 DDP Training')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--resume', default=None, type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('--batch_size', type=int, default=256, help='Total batch size across all GPUs')
    parser.add_argument('--num_workers', type=int, default=10, help='Total number of data loading workers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument("--output", default="./dist_output_optimized", help="Path to save model checkpoints")
    args = parser.parse_args()

    # --- 2. 初始化与环境设置 ---
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cudnn.benchmark = True
    
    if rank == 0:
        print(f"Starting optimized DDP training on {world_size} GPUs.")
        os.makedirs(args.output, exist_ok=True)

    # --- 3. 数据准备 ---
    per_device_batch_size = args.batch_size // world_size
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset_train = datasets.CIFAR10(root='./', train=True, download=True, transform=transforms_train)
    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset_train, batch_size=per_device_batch_size, 
                              num_workers=args.num_workers // world_size,
                              sampler=train_sampler, pin_memory=True)

    # --- 4. 模型与优化器准备 ---
    model = pyramidnet().to(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # 优化点：使用更平滑的余弦退火学习率调度器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 优化点：初始化混合精度训练的 GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # --- 5. 断点续训逻辑 (优化后) ---
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        # 加载时使用 model.module 访问原始模型
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        if rank == 0:
            print(f"Successfully resumed from epoch {start_epoch}")

    # --- 6. 训练循环 ---
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        # --- 调用训练函数 ---
        train_one_epoch(epoch, model, train_loader, optimizer, scheduler, scaler, local_rank, args)
        
        # 在主进程上保存检查点
        if rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.output, "last.pth"))
            print(f"Epoch {epoch+1} checkpoint saved.")

    cleanup()

def train_one_epoch(epoch, model, loader, optimizer, scheduler, scaler, rank, args):
    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(rank), targets.to(rank)

        # 优化点：使用自动混合精度 (AMP)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)

        optimizer.zero_grad()
        # 使用 scaler 进行反向传播
        scaler.scale(loss).backward()
        
        # 优化点：加入梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 使用 scaler 更新优化器
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    # 在主进程上打印日志
    if rank == 0:
        avg_loss = total_loss / len(loader)
        print(f"Epoch: {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 更新学习率
    scheduler.step()


if __name__ == '__main__':
    main()