import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from utils.DataSet_train_val_test import CustomData
from data_02_dog_cat import CustomData

import torch.utils.data as data
# from utils.inception_advance import Inception_v1
from model_resnet_v1 import ResNet50

# parameters
os.environ['CUDA_VISIBLES_DEVICES'] = '0'
batchsize = 128
num_workers = 8
epochs = 5
learning_rate = 0.001
gamma = 0.96

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

dataset_root = '/home/bruce_ultra/workspace/Research_Experiments/cat_dog/'

# 创建训练集实例
trainset = CustomData(root_dir=dataset_root, transform=transform_train, mode='train', split_ratio=0.8)

# 创建验证集实例
valset = CustomData(root_dir=dataset_root, transform=transform_val, mode='val', split_ratio=0.8)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=num_workers) # 验证集通常不需要 shuffle


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(model, epoch, lr):
    print("start training the models ")
    model.train()

    # lr_ = lr.get_lr()[0]
    for index, (img, label) in enumerate(trainloader):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        # if index % 100 == 0 and index is not 0:
        #     lr.step()
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f, lr:%f" % (
        epoch, index, len(trainloader), loss.mean(), train_acc, lr.get_lr()[0]))


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
            _, pred = torch.max(out.data, 1)
            total += img.shape[0]
            correct += pred.data.eq(label.data).cpu().sum()
            print("Epoch:%d [%d|%d] total:%d correct:%d" % (epoch, index, len(valloader), total, correct.numpy()))
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == '__main__':
    model = ResNet50([3, 4, 6, 3], num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(model, epoch, lr)
        val(model, epoch)
        lr.step()
    torch.save(model, 'model_cat_dog.pt')
