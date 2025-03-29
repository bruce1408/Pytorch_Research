import numpy as np
import matplotlib.pyplot as plt


# =========================================
# 这是我们的训练数据，x 和 y 呈线性关系，y = 2x。
# =========================================


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
    print("w = %.2f" % w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        print(x_val, y_val)
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("the x_val is %.2f, y_val is %.2f, y_pred_val is %.2f, loss is %.2f" % (x_val, y_val, y_pred_val, l))
    print("MSE= ", l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
