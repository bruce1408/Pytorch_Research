import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data_02_dog_cat import CustomData
from torchvision.models import resnet50, ResNet50_Weights
from model_resnet_v1 import ResNet50

# parameters
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
save_path = "./self_resnet50.pt"
gamma = 0.96
num_workers = 8
batchsize = 128  # batch_size 不要太大
epochs = 10
learning_rate = 0.01

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

print("--- 正在创建数据集 ---")
dataset_root = '/home/bruce_ultra/workspace/Research_Experiments/cat_dog/'

# 创建训练集实例
trainset = CustomData(root_dir=dataset_root, transform=transform_train, mode='train', split_ratio=0.8)

# 创建验证集实例
valset = CustomData(root_dir=dataset_root, transform=transform_val, mode='val', split_ratio=0.8)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=num_workers) # 验证集通常不需要 shuffle


class Net(nn.Module):
    """
    实现一个简单的只有三层卷积的神经网络来做训练.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14
        )
        self.fc1 = nn.Linear(256*14*14, 1024)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, input):
        output = self.net(input)
        output = output.view(output.shape[0], -1)
        output = self.fc1(output)
        output = self.drop(output)
        output = self.fc2(output)
        output = self.drop(output)
        output = self.fc3(output)
        return output


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, epoch):
    blue = lambda x: '\033[94m' + x + '\033[0m'
    model.train()
    optimizer.zero_grad()

    for batch_idx, (img, label) in enumerate(trainloader):
        image = img.cuda()
        label = label.cuda()
        
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        current_lr = scheduler.get_last_lr()[0]

        # 显示进度条的另外一种写法, 带颜色的写法， lambda 是str连接起来的
        # sys.stdout.write('\033[1;36m \r>>Train Epoch:%d [%d|%d] loss:%f, lr:%f \033[0m' %
                        #  (epoch, batch_idx, len(trainloader), loss.mean(), scheduler.get_lr()[0]))

        sys.stdout.write('\033[1;36m \r>>Train Epoch:%d [%d|%d] loss:%.6f, lr:%.6f \033[0m' %
                         (epoch, batch_idx, len(trainloader), loss.mean(), current_lr))

        # print('\r Test %s: %f   ***  %s: %f' % (blue('Accuracy'), epoch, blue('Best Accuracy'), batch_idx))\
        # sys.stdout.write("\r train Epoch: %d [%d|%d] loss:%f, lr:%f " % (epoch, batch_idx, len(trainloader), loss.mean(),
        #                  scheduler.get_lr()[0]))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


def val(model, epoch):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valloader):
            image = img.cuda()
            label = label.cuda()
            out = model(image)
            _, predicted = torch.max(out.data, 1)
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
            sys.stdout.write('\033[1;35m \r>>Validation Epoch:%d [%d|%d] total:%d, corretc:%d \033[0m' %
                             (epoch, batch_idx, len(valloader), total, correct.numpy()))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == '__main__':
    model = Net().cuda()
    # model = Inception_v1(num_classes=2)
    model = ResNet50([3, 4, 6, 3], num_classes=2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    criterion = nn.CrossEntropyLoss().cuda()

    # ============ model structure ===========
    # print(model)
    # if torch.cuda.is_available():
    #     summary(model.cuda(), (3, 224, 224))
    # else:
    #     summary(model, (3, 224, 224))
    # ============= load the model ============
    # if os.path.exists(save_path):
    #     model.load_state_dict(torch.load(save_path))
    #     print("======== load the model from %s ========" % save_path)
    # else:
    #     print("======== train the net from srcatch ==========")
    # ============ train the model =============
    for epoch in range(epochs):
        train(model, epoch)
        val(model, epoch)
        # 只保存模型权重，如果是model则就是保存整个模型
        torch.save(model.state_dict(), save_path)

