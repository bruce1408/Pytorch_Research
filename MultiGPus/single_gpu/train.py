# 导入必要的库
import os
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import pyramidnet

# 指定使用的GPU设备号，这里使用第3号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 设置命令行参数
parser = argparse.ArgumentParser(description='CIFAR-10分类模型训练')
parser.add_argument('--lr', default=0.1, type=float, help='学习率')
parser.add_argument('--resume', default=None, help='恢复训练的模型路径')
parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
parser.add_argument('--num_worker', type=int, default=6, help='数据加载的工作进程数')
parser.add_argument('--epochs', type=int, default=200, help='训练的总轮数')  # 新增参数
args = parser.parse_args()


def main():
    # 初始化最佳准确率
    best_acc = 0
    
    # 检测可用设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备:", device)
    print('==> 准备数据中...')
    
    # 定义训练数据的转换操作（数据增强）
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 随机裁剪
        transforms.RandomHorizontalFlip(),         # 随机水平翻转
        transforms.ToTensor(),                     # 转换为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # 归一化
                             (0.2023, 0.1994, 0.2010))])

    # 加载CIFAR-10训练数据集
    dataset_train = CIFAR10(
        root='/share/cdd/',  # 修改为更通用的路径
        train=True,
        download=True,
        transform=transforms_train
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        shuffle=True,                # 打乱数据顺序
        num_workers=args.num_worker, # 多进程加载
        pin_memory=True              # 加速数据传输到GPU
    )

    # CIFAR-10的10个类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> 构建模型中...')

    # 创建PyramidNet模型
    net = pyramidnet()
    
    # 如需使用多GPU，取消下面的注释
    # net = nn.DataParallel(net)
    
    # 将模型转移到指定设备上
    net = net.to(device)
    
    # 计算模型参数数量
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('模型包含的参数数量:', num_params)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
                          momentum=0.9, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练模型
    for epoch in range(args.epochs):
        print(f"\n第 {epoch+1}/{args.epochs} 轮训练")
        train(net, criterion, optimizer, train_loader, device)
        scheduler.step()
        
        # 这里可以添加验证和模型保存代码
        # if current_acc > best_acc:
        #     best_acc = current_acc
        #     torch.save(net.state_dict(), f'checkpoint/best_model.pth')


def train(net, criterion, optimizer, train_loader, device):
    """训练一个epoch的模型"""
    # 设置为训练模式
    net.train()
    
    # 初始化统计变量
    train_loss = 0
    correct = 0
    total = 0

    # 记录epoch开始时间
    epoch_start = time.time()
    
    # 遍历批次数据
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 记录批次开始时间
        start = time.time()

        # 将数据转移到指定设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()  # 清除梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        # 更新统计信息
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 计算当前准确率
        acc = 100 * correct / total

        # 计算批次耗时
        batch_time = time.time() - start

        # 每20个批次打印一次训练信息
        if batch_idx % 20 == 0:
            print('批次: [{}/{}]| 损失: {:.3f} | 准确率: {:.3f}% | 批次用时: {:.3f}s '.format(
                batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc, batch_time))

    # 计算总训练时间
    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("总训练用时 {}".format(elapse_time))


if __name__ == '__main__':
    main()