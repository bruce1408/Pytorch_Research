import torch
import torch.nn as nn
from typing import List, Optional

class MobileNetV1(nn.Module):
    """
    MobileNet V1 架构实现
    使用深度可分离卷积减少计算量和参数数量
    
    论文: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    """
    
    def __init__(self, num_classes: int = 1000, input_size: int = 224, width_mult: float = 1.0):
        """
        Args:
            num_classes: 分类类别数
            input_size: 输入图像尺寸
            width_mult: 宽度乘数，用于调整模型大小
        """
        super(MobileNetV1, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.width_mult = width_mult
        
        # 计算通道数
        def make_divisible(channels: int, divisor: int = 8) -> int:
            """确保通道数是divisor的倍数"""
            new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
            if new_channels < 0.9 * channels:  # 防止通道数过小
                new_channels += divisor
            return int(new_channels)
        
        # 基础通道数设置
        base_channels = [
            (32, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),  # 重复5次
            (512, 1024, 2),
            (1024, 1024, 1)
        ]
        
        # 应用宽度乘数
        if width_mult != 1.0:
            base_channels = [
                (make_divisible(int(inp * width_mult)), 
                 make_divisible(int(oup * width_mult)), 
                 stride) 
                for inp, oup, stride in base_channels
            ]
        
        # 构建特征提取层
        layers = []
        
        # 第一层：标准卷积
        first_channels = make_divisible(int(32 * width_mult))
        layers.append(self._conv_bn(3, first_channels, stride=2))
        
        # 深度可分离卷积层
        input_channel = first_channels
        for output_channel, next_channel, stride in base_channels:
            layers.append(self._conv_dw(input_channel, output_channel, stride))
            input_channel = output_channel
        
        self.features = nn.Sequential(*layers)
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(input_channel, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _conv_bn(self, inp: int, oup: int, stride: int) -> nn.Sequential:
        """标准卷积 + 批归一化 + ReLU"""
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)  # 使用ReLU6更适合移动端
        )
    
    def _conv_dw(self, inp: int, oup: int, stride: int) -> nn.Sequential:
        """深度可分离卷积"""
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            
            # 点卷积
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # float32
            'num_classes': self.num_classes,
            'width_mult': self.width_mult
        }


# 测试和验证函数
def test_mobilenet():
    """测试MobileNet V1模型"""
    model = MobileNetV1(num_classes=1000)
    
    # 获取模型信息
    info = model.get_model_info()
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 测试前向传播
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(f"\n输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证输出
    assert output.shape == (1, 1000), f"期望输出形状 (1, 1000)，实际得到 {output.shape}"
    print("✓ 模型测试通过")
    
    return model


def create_mobilenet_versions():
    """创建不同版本的MobileNet"""
    versions = {
        'mobilenet_v1_1.0': MobileNetV1(width_mult=1.0),
        'mobilenet_v1_0.75': MobileNetV1(width_mult=0.75),
        'mobilenet_v1_0.5': MobileNetV1(width_mult=0.5),
        'mobilenet_v1_0.25': MobileNetV1(width_mult=0.25),
    }
    
    print("\n不同版本MobileNet对比:")
    print("-" * 60)
    print(f"{'版本':<20} {'参数量':<15} {'模型大小(MB)':<15}")
    print("-" * 60)
    
    for name, model in versions.items():
        info = model.get_model_info()
        print(f"{name:<20} {info['total_parameters']:<15,} {info['model_size_mb']:<15.2f}")


if __name__ == "__main__":
    
    # 运行测试
    model = test_mobilenet()
    
    # 显示不同版本对比
    create_mobilenet_versions()
