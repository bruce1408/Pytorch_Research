import torch
import torch.nn as nn
import torch.nn.functional as F

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3, torch2caffe_flg=False):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if torch2caffe_flg:  ##使用2d卷积替换1d卷积的操作
            self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, (k_size - 1) // 2), bias=False)
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = F.sigmoid
        self.torch2caffe_flg = torch2caffe_flg

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # b,c,1,1

        # Two different branches of ECA module
        if self.torch2caffe_flg:  ##使用2d卷积替换1d卷积的操作
            y = self.conv(y.view(y.shape[0], 1, 1, y.shape[1])).view(y.shape[0], y.shape[1], 1, 1)
        else:
            y = self.conv(y.view(y.shape[0], 1, y.shape[1])).view(y.shape[0], y.shape[1], 1, 1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

# 测试代码
if __name__ == '__main__':
    # 1. 设置输入参数
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    k_size = 3

    # 2. 创建一个随机输入张量
    input_tensor = torch.randn(batch_size, channels, height, width)
    print("Input tensor shape:", input_tensor.shape)

    # 3. 实例化 eca_layer
    eca = eca_layer(channel=channels, k_size=k_size)

    # 4. 前向传播，并打印每层的输出
    print("--- ECA Layer Analysis ---")
    print("Input:", input_tensor.shape)

    # 平均池化
    avg_pool_output = eca.avg_pool(input_tensor)
    print("AvgPool Output:", avg_pool_output.shape)

    # 1D 卷积
    if eca.torch2caffe_flg:
        conv_input = avg_pool_output.view(batch_size, 1, 1, channels)
    else:
        conv_input = avg_pool_output.view(batch_size, 1, channels)
    conv_output = eca.conv(conv_input)
    print("Conv Output:", conv_output.shape)

    # Sigmoid 激活
    if eca.torch2caffe_flg:
        sigmoid_input = conv_output.view(batch_size, channels, 1, 1)
    else:
        sigmoid_input = conv_output.view(batch_size, channels, 1, 1)
    sigmoid_output = eca.sigmoid(sigmoid_input)
    print("Sigmoid Output:", sigmoid_output.shape)

    # 通道加权
    scaled_output = input_tensor * sigmoid_output.expand_as(input_tensor)
    print("Scaled Output:", scaled_output.shape)

    print("--- ECA Layer Analysis End ---")