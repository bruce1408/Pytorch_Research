import torch
import torch.nn as nn
from torch.onnx import export

# from torchsummary import summary
"""
实现pytorch ResNet50网络结构,实现连接就是不用加卷积的情况
reference: https://blog.csdn.net/Cheungleilei/article/details/103610799
"""


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsampling=False, dim=4):
        """
        代码比tensorflow更加简洁和方便,
        :param in_channel: 输入维度
        :param out_channel: 输出维度
        :param stride: 步长
        :param downsampling: shortcut是否有卷积操作
        :param dim: 维度参数
        """
        super(Bottleneck, self).__init__()
        self.dim = dim
        self.downsampling = downsampling
        
        self.bottleneck = nn.Sequential(
            # 1x1卷积，降维，权重是 out_channel x in_channel x h x w, W * x, 所以 in_channel 在后面
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),  # 默认padding=0
            # 批归一化
            nn.BatchNorm2d(out_channel),
            # ReLU激活
            nn.ReLU(),
            # 3x3卷积
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            # 1x1卷积升维
            nn.Conv2d(out_channel, out_channel*dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel*dim)
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel*dim)
            )
        self.relu = nn.ReLU()

    def forward(self, x):

        x_shortCut = x
        out = self.bottleneck(x)
        # print(out.shape, x.shape)

        if self.downsampling:
            x_shortCut = self.downsample(x)

        out += x_shortCut
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, blocks, num_classes=2):
        super(ResNet50, self).__init__()
        
        # 初始卷积层
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )  # output shape=[55, 55, 3]
        # blocks = [3, 4, 6, 3]

        # 4个残差块
        self.layer2 = self.make_layer(64, 64, blocks[0], stride=1)
        self.layer3 = self.make_layer(256, 128, blocks[1], stride=2)
        self.layer4 = self.make_layer(512, 256, blocks[1], stride=2)
        self.layer5 = self.make_layer(1024, 512, blocks[1], stride=2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, in_channel, out_channel, block_num, stride=1):
        layers = list()
        layers.append(Bottleneck(in_channel, out_channel, stride, downsampling=True))  # 第一个单元是带有卷积的shortcut

        # 其余的部分不带卷积shortcut,直接把结果和输入进行叠加即可
        for i in range(1, block_num):
            layers.append(Bottleneck(out_channel*4, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = ResNet50([3, 4, 6, 3], num_classes=2)
    model.eval()
    # if torch.cuda.is_available():
    #     summary(net.cuda(), (3, 224, 224))
    # else:
    #     summary(net, (3, 224, 224))
    outputs = model(x)

    # net1 = Bottleneck(3, 64)
    # outputs = net1(x)
    print(outputs.shape)
    
    export(
        model,
        x,
        "./resnet50_manual_build.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )