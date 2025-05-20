# custom_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class CustomImageDataset(Dataset):
    """
    一个简单的图像分类数据集：
      - 假设 images/ 下有多张 .jpg 图片
      - annotations 是一个 list of (filename, label) 元组
    """
    def __init__(self, img_dir, annotations, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = annotations  # e.g. [('0001.jpg', 0), ('0002.jpg', 2), …]
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
        ])

    def __len__(self):
        # Dataset 的总样本数
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 读取第 idx 个样本
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


# 使用 Dataset 并创建 DataLoader
if __name__ == "__main__":
    # 假设我们有以下标注列表
    annotations = [
        ("0001.jpg", 0),
        ("0002.jpg", 1),
        ("0003.jpg", 2),
        # …
    ]

    # 1) 构建 Dataset
    dataset = CustomImageDataset(
        img_dir="path/to/images",
        annotations=annotations,
        transform=None,  # 可以传入任意 torchvision.transforms
    )

    # 2) 基于 Dataset 构建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,        # 每个 batch 8 张图
        shuffle=True,        # 打乱顺序
        num_workers=4,       # 用 4 个进程并行加载
        pin_memory=True,     # 如果用 GPU，能稍微提速
    )

    # 3) 迭代 DataLoader，拿到 batch
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images.shape={images.shape}, labels={labels}")
        # images.shape → [8, 3, 224, 224]
        if batch_idx == 0:
            break

