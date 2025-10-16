import os
import time
import datetime
from typing import Tuple, Dict, Any, Optional
from torch.cuda.amp import autocast, GradScaler
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from spectrautils import time_utils
from model import pyramidnet


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR10 分类模型训练')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    parser.add_argument('--resume', default=None, help='恢复训练的检查点路径')
    parser.add_argument('--batch_size', type=int, default=768, help='训练批次大小')
    parser.add_argument('--num_worker', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--gpu_devices', default='0,1,2,3', help='使用的GPU设备ID')
    parser.add_argument('--amp', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='数据预取因子')
    parser.add_argument('--save_path', default='./checkpoints', help='模型保存路径')

    return parser.parse_args()


def setup_environment(gpu_devices: str) -> None:
    """设置运行环境"""
    # 设置可见GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    # 设置当前工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # 启用cudnn benchmark模式以加速训练
    cudnn.benchmark = True


def prepare_data(batch_size: int, num_workers: int, prefetch_factor: int = 2) -> Tuple[DataLoader, tuple]:
    """准备训练数据"""
    print('==> 准备数据...')
    # 定义数据增强和归一化
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载CIFAR10训练集
    dataset_train = CIFAR10(
        # root='/share/cdd', 
        root="/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/MultiGPus/dist_parallel",
        train=True, 
        download=False,
        transform=transforms_train
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True  # 加速数据从CPU到GPU的传输
    )

    # CIFAR10的10个类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, classes


def build_model(device: torch.device) -> nn.Module:
    """构建并初始化模型"""
    print('==> 创建模型...')
    # 创建基础模型
    net = pyramidnet()
    
    # 将模型转移到GPU
    net = net.to(device)
    
    # 使用DistributedDataParallel替代DataParallel以获得更好的性能
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU训练")
        net = nn.DataParallel(net)    
    
    # 打印模型参数量
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'模型参数数量: {num_params:,}')
    
    return net


@time_utils.time_it
def train_epoch(
    net: nn.Module, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    train_loader: DataLoader, 
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """训练一个epoch"""
    net.train()  # 设置为训练模式
    
    # 初始化统计变量
    train_loss = 0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        # 将数据转移到GPU
        inputs = inputs.to(device, non_blocking=True)  # non_blocking加速数据传输
        targets = targets.to(device, non_blocking=True)
        
        # 使用混合精度训练
        if use_amp:
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad(set_to_none=True)  # 更高效地清零梯度
            loss.backward()
            optimizer.step()
        
        # 更新统计信息
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 计算准确率
        acc = 100 * correct / total
        
        # 计算批次处理时间
        batch_time = time.time() - start
        
        # 打印训练信息
        if batch_idx % 20 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] | '
                  f'Loss: {train_loss/(batch_idx+1):.3f} | '
                  f'Acc: {acc:.3f}% | '
                  f'Batch time: {batch_time:.3f}s')
    
    # 计算总训练时间
    elapse_time = datetime.timedelta(seconds=time.time() - epoch_start)
    print(f"Epoch {epoch} 训练时间: {elapse_time}")
    
    # 返回训练指标
    return {
        'loss': train_loss / len(train_loader),
        'accuracy': acc
    }


def save_checkpoint(net: nn.Module, optimizer: optim.Optimizer, 
                   epoch: int, metrics: Dict[str, float], save_path: str) -> None:
    """保存检查点"""
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, f'{save_path}/checkpoint_epoch{epoch}.pth')
    print(f"模型已保存至 {save_path}/checkpoint_epoch{epoch}.pth")
    
    
def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置环境
    setup_environment(args.gpu_devices)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    train_loader, _ = prepare_data(args.batch_size, args.num_worker)
    
    # 构建模型
    net = build_model(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    # 添加学习率调度器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度训练设置
    scaler = GradScaler() if args.amp else None
    
    
    # 训练模型
    for epoch in range(args.epochs):
        print(f"\n开始训练 Epoch {epoch+1}/{args.epochs}")
        train_metrics = train_epoch(net, 
                                    criterion, 
                                    optimizer, 
                                    train_loader, 
                                    device, epoch+1, 
                                    args.amp, scaler)
        scheduler.step()  # 更新学习率
        
        print(f"Epoch {epoch+1} 结果 - 损失: {train_metrics['loss']:.4f}, "
              f"准确率: {train_metrics['accuracy']:.2f}%")
        
        # 保存检查点
        if args.save_path:
            save_checkpoint(net, optimizer, epoch+1, train_metrics, args.save_path)
    
    print()

if __name__ == '__main__':
    main()
