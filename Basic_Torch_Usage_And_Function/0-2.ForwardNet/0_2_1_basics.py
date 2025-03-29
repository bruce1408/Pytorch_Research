import numpy as np
import matplotlib.pyplot as plt
from spectrautils.print_utils import print_colored_box
from spectrautils import logging_utils 

logger_manager = logging_utils.AsyncLoggerManager(work_dir='./torch_basic_logs')
logger = logger_manager.logger

# =========================================
# 这是我们的训练数据，x 和 y 呈线性关系，y = 2x。
# =========================================

# 定义数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义前向传播函数
def forward(x):
    return x * w


# 定义一个简单的损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        logger.info("the x_val is %.2f, y_val is %.2f, y_pred_val is %.2f, loss is %.2f" % (x_val, y_val, y_pred_val, l))
    logger.info(f"MSE : {l_sum / 3}, w is : {w}")
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.savefig('loss_vs_w.png')  # 保存图片
# plt.show()

# 找到最小loss对应的w值
min_mse_index = mse_list.index(min(mse_list))
min_w = w_list[min_mse_index]

print_colored_box(f"The minimum loss occurs at w = {min_w:.2f}")

