from spectrautils import logging_utils
from spectrautils import print_utils
logger_manager = logging_utils.AsyncLoggerManager(work_dir='./torch_basic_logs')
logger = logger_manager.logger

# =========================================
# 这是我们的训练数据，x 和 y 呈线性关系，y = 2x。
# =========================================

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value


def forward(x):
    return x * w

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):  # d_loss/d_w
    """
    梯度推导过程：
    1. 模型：y_pred = x * w
    2. 损失函数：L = (y_pred - y)^2
    3. 需要计算：dL/dw
    
    使用链式法则：
    dL/dw = dL/dy_pred * dy_pred/dw
    
    其中：
    dL/dy_pred = 2(y_pred - y)
    dy_pred/dw = x
    
    因此：
    dL/dw = 2(y_pred - y) * x
          = 2(xw - y) * x
          = 2x(xw - y)
    """
    
    return 2 * x * (x * w - y)


# Before training
print("predict (before training)",  4, forward(4))

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        # 沿着梯度的负方向更新参数
        w = w - 0.01 * grad
        
        logger.info(f"grad: {round(grad, 2)}, x_val: {x_val}, y_val: {y_val}")
        
        l = loss(x_val, y_val)

    logger.info("progress: epoch=%d, w=%.2f, loss=%.2f", epoch, w, l)


# After training
print("predict (after training)",  "if input is 4, the result is:", forward(4))
