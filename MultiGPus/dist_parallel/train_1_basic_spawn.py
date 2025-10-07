import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp  # <--- 导入 multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models

# --- 这个文件现在是自包含的，可以直接用 python 命令运行 ---

def setup(rank: int, world_size: int):
    """
    为每个进程设置分布式环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3457' # 选择一个未被占用的端口
    
    # 初始化进程组，明确指定rank和world_size
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()

# (RandomDataset 类和辅助函数保持不变)
class RandomDataset(Dataset):
    def __init__(self, num_samples=10000, num_classes=100):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# =============================================================================
#  核心逻辑：main_worker 函数
#  mp.spawn 会为每个GPU进程调用这个函数
# =============================================================================
def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    """
    这个函数是每个GPU进程实际执行的训练代码。
    `rank` 参数由 mp.spawn 自动传入。
    """
    print(f"--> Running DDP on rank {rank}.")
    
    # 1. 初始化该进程的分布式环境
    setup(rank, world_size)

    # --- 数据准备 ---
    dataset = RandomDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # 注意：这里的 batch_size 是每个GPU上的批次大小
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        pin_memory=True, 
        num_workers=4
    )

    # --- 模型准备 ---
    model = models.resnet18(weights=None, num_classes=100).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=0.9)

    # --- 训练循环 ---
    for epoch in range(args.epochs):
        # 必须设置 sampler 的 epoch，以保证 shuffle 在多 epoch 训练中正常工作
        sampler.set_epoch(epoch)
        
        ddp_model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 只在主进程(rank 0)打印日志
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")

    # --- 保存模型 ---
    if rank == 0:
        # 保存时需要保存 .module 里的原始模型
        torch.save(ddp_model.module.state_dict(), "resnet18_ddp_spawn.pth")
        print("Training finished and model saved to resnet18_ddp_spawn.pth")

    # --- 清理 ---
    cleanup()

# =============================================================================
#  程序入口：使用 mp.spawn 启动 main_worker
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDP ResNet-18 Training with mp.spawn')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    # 设置要使用的GPU数量
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("需要至少2张GPU来进行分布式训练。")
    else:
        print(f"将在 {world_size} 张GPU上启动分布式训练...")
        # 使用 mp.spawn 启动训练
        # nprocs: 要启动的进程数，即GPU数量
        # args: 要传递给 main_worker 函数的额外参数元组
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True)