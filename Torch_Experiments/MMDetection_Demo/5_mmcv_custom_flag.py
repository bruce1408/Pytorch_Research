import numpy as np
from torch.utils.data import Dataset, DataLoader
from mmdet.datasets import GroupSampler

class MyImageDataset(Dataset):
    def __init__(self, img_sizes, samples_per_gpu=2):
        """
        img_sizes: 列表，每个元素是 (height, width)。
        """
        self.img_sizes = img_sizes
        # 1. 根据长宽比打 flag：横图→0，竖图→1
        self.flag = np.array(
            [0 if w/h > 1.0 else 1 for h, w in img_sizes],
            dtype=np.int64
        )
        self.samples_per_gpu = samples_per_gpu

    def __len__(self):
        return len(self.img_sizes)

    def __getitem__(self, idx):
        # 这里只返回图像尺寸，真实场景当然要返回 tensor/image_path + label 等
        return {'size': self.img_sizes[idx], 'flag': int(self.flag[idx])}

# 构造一个示例数据集：10 张图，尺寸随机
np.random.seed(0)
sizes = [(np.random.randint(200, 600), np.random.randint(200,600)) for _ in range(10)]
dataset = MyImageDataset(sizes, samples_per_gpu=2)

# 打印 flag
print("Image sizes and their flags:")
for i, size in enumerate(sizes):
    print(f"  idx={i:2d}, size={size}, flag={dataset.flag[i]}")

# 用 GroupSampler 来采样
sampler = GroupSampler(dataset, samples_per_gpu=2)
loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)

print("\nBatches produced by GroupSampler (all items in a batch have same flag):")
for batch in loader:
    flags = batch['flag']
    print(flags.tolist())
