import os
import torch
import torch.nn as tnn
import torchvision.transforms as transforms
import torch.optim as optim
from data_02_dog_cat import CustomData
import config as config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCH = 50
N_CLASSES = 2
num_workers = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 对训练集和验证集使用不同的 transform 是一个好习惯
# 验证集不需要 RandomResizedCrop 和 RandomHorizontalFlip 等数据增强操作
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# trainData = dsets.ImageFolder('/raid/bruce/datasets/dogs_cats/train', transform)
# testData = dsets.ImageFolder('/raid/bruce/datasets/dogs_cats/train', transform)

dataset_root = '/home/bruce_ultra/workspace/Research_Experiments/cat_dog/'

# 创建训练集实例
trainset = CustomData(root_dir=dataset_root, transform=transform_train, mode='train', split_ratio=0.8)

# 创建验证集实例
valset = CustomData(root_dir=dataset_root, transform=transform_val, mode='val', split_ratio=0.8)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers) # 验证集通常不需要 shuffle


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


vgg16 = VGG16(n_classes=N_CLASSES)
vgg16.cuda()

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

def train(model, train_loader, optimizer, criterion, epoch):
    # Train the model
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images) # model 现在只返回 outputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 10 == 0: # 每 10 个 batch 打印一次
            print(f'Epoch [{epoch+1}/{EPOCH}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/20:.4f}')
            running_loss = 0.0

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    print('=' * 30)
    print(f'Validation -> Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print('=' * 30)
    return avg_loss



if __name__ == '__main__':
    vgg16_model = VGG16(n_classes=N_CLASSES).to(device)
    
    criterion = tnn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    for epoch in range(EPOCH):
        train(vgg16_model, trainloader, optimizer, criterion, epoch)
        val_loss = validate(vgg16_model, valloader, criterion)
        scheduler.step(val_loss)

    print("Finished Training!")
    torch.save(vgg16_model.state_dict(), 'cnn_scratch.pt')
