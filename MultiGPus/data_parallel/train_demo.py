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
from spectrautils import time_utils
from model import pyramidnet
from torchvision.datasets import CIFAR10



os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

transforms_trian = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


transforms_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = CIFAR10(
    root="/mnt/share_disk/bruce_trie/workspace/Pytorch_Research/MultiGPus/dist_parallel",
    train = True,
    download = False,
    transform = transforms_trian
)

train_loader = DataLoader(
    dataset_train,
    batch_size = 512,
    shuffle = True,
    num_workers = 8,
    prefetch_factor = 2,
    pin_memory = True
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


model = pyramidnet()

model.to(device)

loss = nn.CrossEntropyLoss()

optimizer  = optim.Adam(model.parameters(), lr = 0.001)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100)

scaler = GradScaler()

for epoch in range(10):
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(inputs)
        loss_value = loss(outputs, targets)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100 * correct / total

        
        if batch_idx % 20 == 0:
            print(f'Epoch: [{epoch+1}][{batch_idx}/{len(train_loader)}] | '
                  f'Loss: {train_loss/(batch_idx+1):.3f} | '
                  f'Acc: {acc:.3f}% ')
    
    scheduler.step()  # 更新学习率
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} 结果 - 平均损失: {avg_loss:.4f}")
        
            
        
    
    
