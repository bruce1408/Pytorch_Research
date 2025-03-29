import torch
from torch.autograd import Variable
import torch.optim as optim
from spectrautils import logging_utils
from spectrautils import print_utils
logger_manager = logging_utils.AsyncLoggerManager(work_dir='./torch_basic_logs')
logger = logger_manager.logger

# =========================================
# 这是我们的训练数据，x 和 y 呈线性关系，y = 2x。
# =========================================


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 0.4.0 版本 w 参数随机初始化一个值
# w = Variable(torch.Tensor([5.0]), requires_grad=True)  # Any random value
# 1.5.0版本写法
w = torch.Tensor([5.0])
w.requires_grad = True


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", 4, forward(4).data)

# Training loop
for epoch in range(20):
    for x_val, y_val in zip(x_data, y_data):
        # 求损失函数
        l = loss(x_val, y_val)
        
        # 方向传播自动求梯度
        l.backward()
        
        # logger.info("x_val is%f," x_val, y_val, w.grad.data)
        logger.info("grad: x_val=%.2f, y_val=%.2f, w.grad=%.2f, w is %.2f", x_val, y_val, w.grad.data.item(), w.item())
        
        # 参数更新
        w.data = w.data - 0.01 * w.grad.data

        # 每次参数更新之后梯度值清零
        w.grad.data.zero_()

print("predict (after training)", 4, forward(4).data)
