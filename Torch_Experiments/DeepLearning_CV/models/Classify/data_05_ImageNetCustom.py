import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageNetCustom(Dataset):
    """
    一个健壮的、能够正确处理 ImageNet 训练集和验证集的自定义 Dataset 类。
    这个版本假设 'train' 和 'val' 目录都采用了子文件夹以类别ID命名的结构。
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 指向 'train' 或 'val' 目录的路径。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.list_img = []
        self.list_label = []
        
        self.label2idx = {}
        self.idx2label = {}

        # 关键：类别映射文件通常在数据集的根目录，而不是在 train/val 目录里
        dataset_root = os.path.dirname(root_dir)
        self._build_label_map(dataset_root)

        print(f"Loading data from: {self.root_dir}")

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        # 统一的加载逻辑，适用于 train 和 val
        for class_name in sorted(os.listdir(self.root_dir)): # 使用 sorted() 保证顺序一致
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            if class_name not in self.label2idx:
                continue # 跳过不属于前1000类的文件夹
            
            label_idx = self.label2idx[class_name]
            
            for img_name in sorted(os.listdir(class_dir)): # 使用 sorted() 保证顺序一致
                img_path = os.path.join(class_dir, img_name)
                
                if os.path.isfile(img_path):
                    self.list_img.append(img_path)
                    self.list_label.append(label_idx)

    def _build_label_map(self, dataset_root):
        """
        从 LOC_synset_mapping.txt 构建类别名到索引的映射。
        严格限制只映射前1000个类别。
        
        Args:
            dataset_root (string): ImageNet 数据集的根目录 (e.g., '.../imagenet/')
        """
        map_file = os.path.join(dataset_root, 'LOC_synset_mapping.txt')
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Label mapping file not found: {map_file}")
            
        with open(map_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break # 严格限制1000个类别

                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ')
                if len(parts) < 2:
                    continue
                
                class_name = parts[0]
                self.label2idx[class_name] = i
                self.idx2label[i] = class_name
                
    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img_path = self.list_img[idx]
        label = self.list_label[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Returning a dummy image.")
            # 返回一个占位符，以防个别图片损坏导致训练中断
            dummy_img = torch.zeros(3, 224, 224)
            dummy_label = torch.tensor(-1, dtype=torch.long) # 使用一个无效标签
            return dummy_img, dummy_label

        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)


# ==============================================================================
#  这是一个可以直接运行的测试模块，用于验证该 Dataset 类的功能
#  你可以通过在终端运行 `python data_05_ImageNetCustom.py` 来执行它
# ==============================================================================
if __name__ == '__main__':
    
    # --- 1. 定义数据变换 ---
    # 定义用于验证集的标准变换
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    # --- 2. 设置你的数据集路径 ---
    # ！！！注意：请将这里的路径替换成你自己的真实路径！！！
    IMAGENET_ROOT = "/DataVault/datasets/imagenet" 
    train_dir = os.path.join(IMAGENET_ROOT, 'train')
    val_dir = os.path.join(IMAGENET_ROOT, 'val')

    # --- 3. 测试训练集加载 ---
    try:
        print("="*30)
        print("Testing Train Dataset Loader...")
        print("="*30)
        # 注意这里的 root_dir 直接指向 train 目录
        train_dataset = ImageNetCustom(root_dir=train_dir, transform=val_transform)
        print(f"Successfully loaded Train dataset.")
        print(f"Total train images: {len(train_dataset)}")
        
        # 随机抽取一个样本进行检查
        if len(train_dataset) > 0:
            sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
            img, label = train_dataset[sample_idx]
            class_name = train_dataset.idx2label[label.item()]
            print(f"Sample check PASSED: ")
            print(f"  - Random sample index: {sample_idx}")
            print(f"  - Image shape: {img.shape}")
            print(f"  - Label: {label.item()} ({class_name})")
        else:
            print("Warning: Train dataset is empty!")
            
    except Exception as e:
        print(f"Error while testing Train dataset: {e}")

    # --- 4. 测试验证集加载 ---
    try:
        print("\n" + "="*30)
        print("Testing Validation Dataset Loader...")
        print("="*30)
        # 注意这里的 root_dir 直接指向 val 目录
        val_dataset = ImageNetCustom(root_dir=val_dir, transform=val_transform)
        print(f"Successfully loaded Validation dataset.")
        print(f"Total validation images: {len(val_dataset)}")
        
        # 随机抽取一个样本进行检查
        if len(val_dataset) > 0:
            sample_idx = torch.randint(len(val_dataset), size=(1,)).item()
            img, label = val_dataset[sample_idx]
            class_name = val_dataset.idx2label[label.item()]
            print(f"Sample check PASSED: ")
            print(f"  - Random sample index: {sample_idx}")
            print(f"  - Image shape: {img.shape}")
            print(f"  - Label: {label.item()} ({class_name})")
        else:
            print("Warning: Validation dataset is empty!")

    except Exception as e:
        print(f"Error while testing Validation dataset: {e}")
