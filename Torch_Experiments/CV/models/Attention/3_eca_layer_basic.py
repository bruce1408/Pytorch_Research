# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 定义ECA模块，包含自适应核大小选择
# class ECA(nn.Module):
#     def __init__(self, channel):
#         super(ECA, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 全局平均池化
#         b, c, h, w = x.size()
#         y = self.avg_pool(x).view(b, 1, c)  # 挤压操作：全局平均池化 [1, 1, 64]

#         # 自适应确定核大小
#         t = int(abs((torch.log2(torch.tensor(c)) / torch.tensor(2)).item() + 1))
#         k_size = max(t if t % 2 else t + 1, 3)  # 确保核大小至少为3

#         # 1D卷积
#         y = y.permute(0, 2, 1)  # 转换为 (b, c, 1)
#         y = F.conv1d(y, torch.ones(1, 1, k_size).to(x.device), padding=(k_size - 1) // 2)  # 局部跨通道交互
#         y = y.permute(0, 2, 1)  # 转换回 (b, 1, c)

#         # Sigmoid激活函数
#         y = self.sigmoid(y).view(b, c, 1, 1)  # 重标定：将通道权重应用到输入特征图上

#         return x * y.expand_as(x)  # 逐元素相乘

# # 定义一个简单的卷积神经网络，集成ECA模块
# class SimpleCNNWithECA(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNNWithECA, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.eca1 = ECA(64)  # 在第一个卷积层后加入ECA模块

#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.eca2 = ECA(128)  # 在第二个卷积层后加入ECA模块

#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.eca1(x)  # 应用ECA模块

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.eca2(x)  # 应用ECA模块

#         x = F.adaptive_avg_pool2d(x, (1, 1))  # 全局平均池化
#         x = x.view(x.size(0), -1)  # 展平
#         x = self.fc(x)  # 全连接层
#         return x

# # 测试代码
# if __name__ == "__main__":
#     # 创建一个随机输入张量，模拟一个batch的图像数据
#     input_tensor = torch.randn(1, 3, 32, 32)  # 假设输入图像大小为32x32，3个通道

#     # 实例化模型
#     model = SimpleCNNWithECA(num_classes=10)

#     # 前向传播
#     output = model(input_tensor)

#     # 打印输出
#     print("Output shape:", output.shape)



import torch
import torch.nn as nn
import math
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA) 模块：
    1. 对输入 x (N, C, H, W) 先做全局平均池化 → (N, C, 1, 1)
    2. squeeze 后 reshape → (N, C)，再 unsqueeze → (N, 1, C)
    3. 用一维卷积 (Conv1d) 计算通道间局部交互 → (N, 1, C)
    4. Sigmoid → (N, 1, C)，再 reshape → (N, C, 1, 1)
    5. 对原输入做逐通道加权
    """
    def __init__(self, channels, k_size=None):
        super().__init__()
        # 如果未显式指定 k_size，根据论文经验公式自动计算：k = |log2(C)/γ + b|_odd
        if k_size is None:
            t = int(abs((math.log2(channels) / 2) + 1))
            k_size = t if t % 2 else t + 1  # 保证是奇数
        # 全局平均池化到 (N, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 一维卷积，通道维度滑动窗口
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, C, H, W)
        # 1. 全局平均池化 → (N, C, 1, 1)
        y = self.avg_pool(x)
        # 2. reshape → (N, C)，再 unsqueeze → (N, 1, C)
        y = y.view(x.size(0), x.size(1))   # (N, C)
        y = y.unsqueeze(1)                 # (N, 1, C)
        # 3. 一维卷积 → (N, 1, C)
        y = self.conv(y)
        # 4. Sigmoid → (N, 1, C)，再 reshape → (N, C, 1, 1)
        y = self.sigmoid(y).view(x.size(0), x.size(1), 1, 1)
        # 5. 对原始特征做通道加权
        return x * y.expand_as(x)


class BasicBlockECA(BasicBlock):
    """在 BasicBlock 中插入 ECA 注意力"""
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        # 在第二个 3×3 卷积后插入 ECA
        # self.eca = ECAModule(planes)
        self.eca = ECAModule(planes)

    def forward(self, x):
        identity = x

        # 第一个卷积 + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积 + BN
        out = self.conv2(out)
        out = self.bn2(out)

        # ECA 注意力
        out = self.eca(out)

        # 如果需要下采样 (shortcut)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckECA(Bottleneck):
    """在 Bottleneck 中插入 ECA 注意力"""
    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                groups=1, base_width=64, dilation=1, norm_layer=None, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, 
                        groups, base_width, dilation, norm_layer)
        # Bottleneck 的输出通道 = planes * expansion
        self.eca = ECAModule(self.expansion * planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # ECA 注意力
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ECAResNet(ResNet):
    """
    ECA-ResNet：
    继承 torchvision.models.resnet.ResNet，将 BasicBlock/Bottleneck 替换为带 ECA 的版本
    支持 depth=18,34,50,101,152
    """
    def __init__(self, depth, **kwargs):
        assert depth in [18, 34, 50, 101, 152], "depth 必须是 18, 34, 50, 101 或 152"
        # 根据 depth 选择对应的 block 与层数
        if depth == 18:
            block = BasicBlockECA
            layers = [2, 2, 2, 2]
        elif depth == 34:
            block = BasicBlockECA
            layers = [3, 4, 6, 3]
        elif depth == 50:
            block = BottleneckECA
            layers = [3, 4, 6, 3]
        elif depth == 101:
            block = BottleneckECA
            layers = [3, 4, 23, 3]
        elif depth == 152:
            block = BottleneckECA
            layers = [3, 8, 36, 3]
        # 调用父类初始化，传入自定义的 block 和层数
        super().__init__(block=block, layers=layers, **kwargs)
        # 如果需要按经典 ResNet 预训练权重初始化，可以调用 init_weights

    def init_weights(self, pretrained=None):
        """
        重载 init_weights：如果提供了预训练模型路径，严格加载 ResNet 对应层，
        对于 ECA 层 (conv1d) 由于原始预训练权重没有，需要 strict=False
        """
        if pretrained is not None:
            from torch.hub import load_state_dict_from_url
            state_dict = torch.load(pretrained, map_location='cpu')
            # strict=False 会忽略 ECA 相关层的缺失
            self.load_state_dict(state_dict, strict=False)
        else:
            super().init_weights(pretrained=None)


def eca_resnet34(pretrained=False, **kwargs):
    """
    构造 ECA-ResNet-34
    :param pretrained: 如果为 True，需要传入预训练权重路径
    :param kwargs: 其他 ResNet 参数，例如 num_classes, zero_init_residual 等
    """
    model = ECAResNet(depth=34, **kwargs)
    if pretrained:
        # 这里假设你已经下载好了 eca_resnet34 的预训练权重文件，
        # 例如路径 '/path/to/eca_resnet34.pth'
        model.init_weights(pretrained='/path/to/eca_resnet34.pth')
    return model


def eca_resnet50(pretrained=False, **kwargs):
    """
    构造 ECA-ResNet-50
    """
    model = ECAResNet(depth=50, **kwargs)
    if pretrained:
        model.init_weights(pretrained='/path/to/eca_resnet50.pth')
    return model


if __name__ == "__main__":
    # 简单测试：构造 ECA-ResNet-34 并前向推理一个随机输入
    net = eca_resnet34(pretrained=False, num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print("ECA-ResNet-34 输出维度：", out.shape)  # (2, 1000)

    # 构造 ECA-ResNet-50
    net50 = eca_resnet50(pretrained=False, num_classes=1000)
    y = torch.randn(2, 3, 224, 224)
    out50 = net50(y)
    print("ECA-ResNet-50 输出维度：", out50.shape)  # (2, 1000)
