
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义SE模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 无论输入特征图的大小是多少，输出都会是1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化, 1表示输出尺寸是1x1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个全连接层
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个全连接层
            nn.Sigmoid()  # Sigmoid激活函数
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # [1, 64, 32, 32]
        y = self.avg_pool(x).view(b, c)  # 挤压操作：全局平均池化
        y = self.fc(y).view(b, c, 1, 1)  # 激励操作：全连接层 + Sigmoid
        return x * y.expand_as(x)  # 重标定：将通道权重应用到输入特征图上 扩展为 [1, 64, 32, 32]

# 定义一个简单的卷积神经网络，集成SE模块
class SimpleCNNWithSE(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNNWithSE, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.se1 = SEBlock(64)  # 在第一个卷积层后加入SE模块

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)  # 在第二个卷积层后加入SE模块

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.se1(x)  # 应用SE模块

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.se2(x)  # 应用SE模块

        x = F.adaptive_avg_pool2d(x, (1, 1))  # 全局平均池化
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # 全连接层
        return x

# 测试代码
if __name__ == "__main__":
    # 创建一个随机输入张量，模拟一个batch的图像数据
    input_tensor = torch.randn(1, 3, 32, 32)  # 假设输入图像大小为32x32，3个通道

    # 实例化模型
    model = SimpleCNNWithSE(num_classes=10)

    # 前向传播
    output = model(input_tensor)

    # 打印输出
    print("Output shape:", output.shape)