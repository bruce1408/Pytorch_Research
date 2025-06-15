import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import mmcv
from mmcv.runner import Runner, Hook

# -------------------------------
# 定义一个简单的 Dataset
# -------------------------------
class DummyDataset(Dataset):
    def __init__(self, num_samples=20):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回一个简单的数据和标签
        x = torch.tensor([idx], dtype=torch.float32)
        y = torch.tensor([idx % 2], dtype=torch.long)
        return x, y

# -------------------------------
# 定义一个简单的模型
# -------------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# 定义自定义 Hook
# -------------------------------
class PrintHook(Hook):
    def before_run(self, runner):
        print(">> [Hook] Before run: Training is starting!")

    def after_run(self, runner):
        print(">> [Hook] After run: Training has ended!")

    def before_train_epoch(self, runner):
        print(f">> [Hook] Before train epoch {runner.epoch}!")

    def after_train_epoch(self, runner):
        print(f">> [Hook] After train epoch {runner.epoch}!")

    def before_train_iter(self, runner):
        print(f">> [Hook] Before train iter {runner.iter}!")

    def after_train_iter(self, runner):
        print(f">> [Hook] After train iter {runner.iter}!")


# -------------------------------
# 定义打印 Loss 的 Hook
# -------------------------------
class PrintLossHook(Hook):
    def before_run(self, runner):
        print(">> [LossHook] 训练开始，准备打印 loss!")

    def after_run(self, runner):
        print(">> [LossHook] 训练结束，停止打印 loss!")

    def after_train_iter(self, runner):
        cur_iter = runner.iter
        cur_epoch = runner.epoch
        loss = runner.outputs['loss']
        print(f">> [LossHook] Epoch: {cur_epoch + 1}, Iter: {cur_iter}, Loss: {loss.item():.4f}")


class ValidationHook(Hook):
    def __init__(self, val_dataloader):
        self.val_dataloader = val_dataloader
        
    def after_train_epoch(self, runner):
        if((runner.epoch + 1) % 2 == 0):
            runner.model.eval()
            total_loss = 0
            num_samples = 0
            
            print(f"\n====== 开始第 {runner.epoch + 1} epoch的验证 ======")

            with torch.no_grad():  # 不计算梯度
                for data in self.val_dataloader:
                    outputs = runner.model(data[0])
                    loss = nn.functional.cross_entropy(outputs, data[1].squeeze())
                    
                    total_loss += loss.item() * len(data[0])

                    num_samples += 1
                    
            avg_val_loss = total_loss / num_samples
            print(f"验证集平均Loss: {avg_val_loss:.4f}")
            print("====== 验证结束 ======\n")
            runner.model.train()        
    
    
# -------------------------------
# 定义批处理函数
# -------------------------------
def batch_processor(model, data, train_mode):
    # data 是 (input, label) 的元组
    inputs, labels = data
    outputs = model(inputs)

    loss = nn.functional.cross_entropy(outputs, labels.squeeze())
    return dict(loss=loss)


# -------------------------------
# 主函数
# -------------------------------
def main():
    # 准备训练数据
    dataset = DummyDataset(num_samples=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    

    # 准备验证数据
    val_dataset = DummyDataset(num_samples=10)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    
    # 初始化模型和优化器
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 创建 Runner
    runner = Runner(
        model=model,
        batch_processor=batch_processor,
        optimizer=optimizer,
        work_dir='./works_dir',
        logger=mmcv.get_logger('test')  # 添加一个名称参数
    )

    # 注册自定义 Hook
    runner.register_hook(PrintHook())
    runner.register_hook(PrintLossHook())  # 添加这一行来注册新的 PrintLossHook
    runner.register_hook(ValidationHook(val_dataloader))  # 添加验证Hook

    
    # 执行训练流程：这里只进行10个 epoch 的训练，workflow 中 'train' 表示训练阶段
    runner.run([dataloader], workflow=[('train', 1)], max_epochs=10)

if __name__ == '__main__':
    main()
