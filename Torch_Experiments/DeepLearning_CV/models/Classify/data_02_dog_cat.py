import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import random

class CustomData(data.Dataset):
    """
    一个更健壮的自定义数据类，能够自动将训练数据划分为训练集和验证集。

    - 在初始化时，读取所有数据并进行一次性的80/20分割。
    - 使用 mode='train' 或 mode='val' 来决定使用哪部分数据。
    - __getitem__ 始终返回 (image, label)。
    """
    def __init__(self, root_dir, transform=None, mode='train', split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        # 1. 一次性读取所有训练图片的路径和标签
        all_images = []
        all_labels = []
        train_folder = os.path.join(self.root_dir, 'train')
        
        print(f"正在从 {train_folder} 加载所有图片路径和标签...")
        for file in tqdm(os.listdir(train_folder)):
            all_images.append(os.path.join(train_folder, file))
            
            # 从文件名判断标签
            if file.startswith('cat'):
                all_labels.append(0)  # 猫 -> 0
            elif file.startswith('dog'):
                all_labels.append(1)  # 狗 -> 1
        
        # 2. 对数据进行洗牌和分割
        # 使用固定的随机种子保证每次分割结果一致
        temp = list(zip(all_images, all_labels))
        random.seed(seed)
        random.shuffle(temp)
        all_images, all_labels = zip(*temp)
        
        split_index = int(len(all_images) * split_ratio)
        
        # 3. 根据 mode 决定使用哪部分数据
        if self.mode == 'train':
            self.image_paths = all_images[:split_index]
            self.labels = all_labels[:split_index]
            print(f"训练集模式：共 {len(self.image_paths)} 张图片。")
        elif self.mode == 'val':
            self.image_paths = all_images[split_index:]
            self.labels = all_labels[split_index:]
            print(f"验证集模式：共 {len(self.image_paths)} 张图片。")
        else:
            raise ValueError("mode 参数必须是 'train' 或 'val'")

    def __getitem__(self, item):
        """
        __getitem__ 现在逻辑统一，始终返回处理后的图片和标签。
        """
        img_path = self.image_paths[item]
        label = self.labels[item]
        
        try:
            img = Image.open(img_path).convert('RGB') # 转换为RGB以防灰度图
        except Exception as e:
            print(f"警告：无法打开图片 {img_path}。将返回一个黑色图片。错误: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0)) # 返回一个默认图片
            label = 0 # 给一个默认标签

        if self.transform:
            img = self.transform(img)
            
        # 返回图片张量和 Python 数字标签
        # DataLoader 会自动将批次中的数字标签打包成一个 LongTensor
        return img, label

    def __len__(self):
        return len(self.image_paths)
