import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets

from model import pyramidnet
from spectrautils import logging_utils


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



# 后端采用nccl nvidia 进程之间通信方式
dist.init_process_group(backend="nccl")

# 从环境变量获取当前 local_rank 变量
local_rank  = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 获取全局唯一的rank编号
rank = dist.get_rank()

# 获取多个gpu数量
world_size = dist.get_world_size()

output = "/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/MultiGPus/dist_parallel/logs"
if rank == 0:
    print("ddp training on {} GPUs".format(world_size))
    os.makedirs(output, exist_ok=True)
    
# 数据准备
batch_size = 512
num_workers = 8
# 每个gpu上面跑的batch_size数目
per_device_batch_size = batch_size // world_size

# 每个gpu上面跑的num_workers数目
per_device_num_workers = num_workers // world_size

if rank == 0:
    print("Rank {}: Per-device batch size: {per_device_batch_size}, Workers: {per_device_num_workers}")

train_dataset = datasets.FakeData(5000, (3, 32, 32), 10, transforms.ToTensor())
train_sample = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
train_loader = DataLoader(
    train_dataset,
    batch_size = per_device_batch_size,
    num_workers = per_device_num_workers,
    sampler = train_sample,
    pin_memory = True
)

net = pyramidnet().to(local_rank)

net = DDP(net, device_ids=[local_rank], output_device=local_rank)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

for epoch in range(100):
    train_sample.set_epoch(epoch)
    
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
            print(f"Rank {rank}: Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        

dist.destroy_process_group()