import torch

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
