import os
import sys
import torch
import argparse
import torch.nn as nn
from torchvision.datasets import FakeData
from torchvision import transforms
import config as config
from data_02_dog_cat import CustomData
from model_inception_v1 import Inception_v1
from spectrautils import logging_utils

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# parameters
os.environ['CUDA_VISIBLES_DEVICES'] = '0'
batchsize = 64
num_works = 4
epochs = 30
learning_rate = 0.01
gamma = 0.96

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--dummy", default=False, type=bool, help="use toy data")
args = parser.parse_args()

mean = [0.485, 0.456, 0.406]
std = [0.2459, 0.2424, 0.2603115]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224, 224), padding=4),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if args.dummy:
    trainData = FakeData(10000, (3, 224, 224), 2, transforms.ToTensor())
    valData = FakeData(2000, (3, 224, 224), 2, transforms.ToTensor())
else:
    dataset_root = '/home/bruce_ultra/workspace/Research_Experiments/cat_dog/'

    # 创建训练集实例
    trainData = CustomData(root_dir=dataset_root, transform=transform_train, mode='train', split_ratio=0.8)

    # 创建验证集实例
    valData = CustomData(root_dir=dataset_root, transform=transform_val, mode='val', split_ratio=0.8)

trainloader = torch.utils.data.DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=num_works)
valloader = torch.utils.data.DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=num_works)


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, epoch, scheduler):
    print("start training the models ")
    model.train()
    # lr_ = lr.get_last_lr()[0]
    for index, (img, label) in enumerate(trainloader):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        
         # --- 以下为修改部分 ---

        # 1. 获取 InceptionV1 的三个输出
        main_output, aux_output1, aux_output2 = model(img)

        # 2. 分别计算三个损失
        loss_main = criterion(main_output, label)
        loss_aux1 = criterion(aux_output1, label)
        loss_aux2 = criterion(aux_output2, label)

        # 3. 按照论文建议，将损失加权求和
        loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
        
        loss.backward()
        optimizer.step()
        train_acc = get_acc(main_output, label)
        current_lr = scheduler.get_last_lr()[0]

        print("Epoch:%d [%d|%d] loss:%.6f acc:%.6f, lr:%.6f" % (epoch, index, len(trainloader), loss.item(), train_acc, current_lr))
    # lr.step()


def val(model, epoch):
    print('begin to eval')
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for index, (img, label) in enumerate(valloader):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            out = out[0] + out[1] + out[2]
            _, pred = torch.max(out.data, 1)
            val_acc = get_acc(out, label)
            total += img.shape[0]
            correct += pred.eq(label.data).cpu().sum()
            print("Epoch:%d [%d|%d] total:%d correct:%d, get_acc %f" % (epoch, index, len(valloader), total, correct.numpy(), val_acc))
    print("Acc: %f " % (1.0 * correct.numpy() / total))


if __name__ == '__main__':
    model = Inception_v1(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(model, epoch, scheduler)
        val(model, epoch)
        scheduler.step()
    torch.save(model, 'model_cat_dog.pth')
