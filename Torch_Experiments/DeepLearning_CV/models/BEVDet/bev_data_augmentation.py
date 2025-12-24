import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BEVDataAugmentation(nn.Module):
    """
    [BDA] BEV 空间数据增强
    作用: 直接对生成的 BEV 特征图进行旋转、翻转、缩放
    原理: 使用 affine_grid 和 grid_sample
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, bda_mat):
        """
        x: BEV 特征图 (B, C, H, W)
        bda_mat: BEV 变换矩阵 (B, 3, 3) - 从 Dataset 传进来的
        """
        B, C, H, W = x.shape
        
        # PyTorch 的 affine_grid 需要变换矩阵是 2x3 的 (针对 2D 图像)
        # 并且是 "归一化坐标系" (-1 到 1)
        # bda_mat 通常是在物理坐标系下的 (比如旋转 90度)，我们需要在 Dataset 里处理好，
        # 或者在这里把它转换成 grid_sample 能吃的格式。
        
        # 为了简化演示，我们假设 bda_mat 已经是适配 PyTorch grid_sample 的 2x3 矩阵
        # 实际项目中，需要把物理坐标矩阵 T_phys 转换为 归一化坐标矩阵 T_norm
        # T_norm = Inv(K_norm) * T_phys * K_norm
        
        # 取前两行，构成 2x3 仿射矩阵
        rot_mat = bda_mat[:, :2] # (B, 2, 3)

        # 1. 生成采样网格
        # 输出尺寸和输入一样 (H, W)
        grid = F.affine_grid(rot_mat, x.size(), align_corners=True)
        
        # 2. 采样 (Warp)
        # 相当于把 BEV 特征图旋转了
        x_aug = F.grid_sample(x, grid, align_corners=True, mode='bilinear')
        
        return x_aug

# ================= 辅助函数：生成 BDA 矩阵 =================
def get_bda_matrices(batch_size):
    """
    模拟 Dataset 中生成 BDA 矩阵的过程
    """
    bda_mats = []
    for _ in range(batch_size):
        # 1. 随机旋转 (-45 ~ 45 度)
        angle = np.random.uniform(-45, 45)
        rad = np.deg2rad(angle)
        cos, sin = np.cos(rad), np.sin(rad)
        
        # 旋转矩阵 (2x2)
        rot = np.array([[cos, -sin], [sin, cos]])
        
        # 2. 随机缩放 (0.9 ~ 1.1)
        scale = np.random.uniform(0.9, 1.1)
        scale_mat = np.eye(2) * scale
        
        # 3. 随机翻转
        flip_x = np.random.choice([1, -1])
        flip_y = np.random.choice([1, -1])
        flip_mat = np.array([[flip_x, 0], [0, flip_y]])
        
        # 组合变换: Flip -> Scale -> Rotate
        mat = rot @ scale_mat @ flip_mat # (2, 2)
        
        # 构造成 3x3 齐次矩阵 (物理坐标系)
        # PyTorch affine_grid 需要的是逆变换 (Destination -> Source)
        # 所以这里通常需要求逆，或者生成的时候就生成逆矩阵
        # 这里为了演示，我们构造一个标准的 pytorch 2x3 矩阵格式
        # [theta11, theta12, tx]
        # [theta21, theta22, ty]
        
        # 简单构造一个 3x3 用于传入模型
        final_mat = np.eye(3)
        final_mat[:2, :2] = mat
        bda_mats.append(torch.from_numpy(final_mat).float())
        
    return torch.stack(bda_mats) # (B, 3, 3)

# ================= 测试 BDA =================
bev_aug = BEVDataAugmentation()
# 模拟 BEV 特征: Batch=2, C=64, 128x128
fake_bev_feat = torch.randn(2, 64, 128, 128)
# 模拟 BDA 矩阵
bda_mats = get_bda_matrices(2)

# 执行增强
aug_bev_feat = bev_aug(fake_bev_feat, bda_mats)

print("\n=== BDA (BEV Data Augmentation) ===")
print(f"Input BEV Shape: {fake_bev_feat.shape}")
print(f"BDA Matrix Shape: {bda_mats.shape}")
print(f"Augmented BEV Shape: {aug_bev_feat.shape}")