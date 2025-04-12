import torch
import numpy as np

def intro_deatch():
    # 创建一个张量，要求梯度
    a = torch.randn(3, 3, requires_grad=True)

    # 使用 clone().detach() 来避免共享计算图
    b = a.clone().detach()

    # 进行操作
    c = a + 2

    # 进行反向传播
    c.sum().backward()

    print(a.grad)  # a的梯度
    print(b.grad)  # b的梯度 (应该为None，因为b不在计算图中)


def intro_as_tensor():
    # torch.tensor() 创建新 tensor，是深拷贝；
    # torch.as_tensor() 在可能的情况下会共享内存，相当于浅拷贝。

    # NumPy 数组
    np_array = np.array([1, 2, 3])

    # torch.tensor() 创建新 tensor，是深拷贝；
    t1 = torch.tensor(np_array)
    np_array[0] = 100  # 修改 NumPy 数组
    print(t1)  # 输出: tensor([1, 2, 3])  # t1 不受影响

    # torch.as_tensor() 在可能的情况下会共享内存，相当于浅拷贝。如果输入已经是一个 tensor，则直接返回该 tensor（即没有复制）；
    t2 = torch.as_tensor(np_array)
    np_array[1] = 200  # 修改 NumPy 数组
    print(t2)  # 输出: tensor([100, 200, 3])  # t2 反映了 np_array 的变化


if __name__ == "__main__":
    intro_as_tensor()
    
