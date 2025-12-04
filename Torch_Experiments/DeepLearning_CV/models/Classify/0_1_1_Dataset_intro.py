"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
print("x: ", x)
y = torch.linspace(10, 1, 10)
print("y: ", y)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=5,
    shuffle=False,
    num_workers=2,
)

print("打印 DataLoader 中的数据:")
for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  X: {batch_x}")
    print(f"  Y: {batch_y}")
    print()
    
print(loader)
loader = iter(loader)
input_var = next(loader)[0]
print("next: \n", input_var)
print(next(loader))


