import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# 1. LSS 核心算法 (复用之前的优化版)
# ==========================================
class LSS_Core(nn.Module):
    def __init__(self, grid_conf, data_conf, input_channels, num_classes):
        super().__init__()
        self.grid_conf = grid_conf
        self.data_conf = data_conf
        self.D = data_conf['D'] 
        self.C = num_classes             
        self.cam_encode = nn.Conv2d(input_channels, self.D + self.C, kernel_size=1) 
        
        # frustum shape是 ([41, 16, 44, 3]) -> [Depth, H, W, 3]
        self.frustum = self.create_frustum()
        

    def create_frustum(self):
        ds = torch.arange(*self.data_conf['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        H, W = self.data_conf['img_size']
        xs = torch.linspace(0, W - 1, W, dtype=torch.float).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, H - 1, H, dtype=torch.float).view(1, H, 1).expand(D, H, W)
        frustum = torch.stack((xs, ys, ds.expand(D, H, W)), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrinsics):
        # 强制生成有效范围内的点，保证训练稳定
        B, N, _ = trans.shape
        x_min, x_max = self.grid_conf['xbound'][0], self.grid_conf['xbound'][1]
        y_min, y_max = self.grid_conf['ybound'][0], self.grid_conf['ybound'][1]
        z_min, z_max = self.grid_conf['zbound'][0], self.grid_conf['zbound'][1]
        
        # [1, 6, 41, 16, 44, 3]
        rand_vals = torch.rand(B, N, self.D, *self.data_conf['img_size'], 3, device=trans.device)
        coords_xyz = torch.zeros_like(rand_vals)
        coords_xyz[..., 0] = rand_vals[..., 0] * (x_max - x_min - 2.0) + x_min + 1.0
        coords_xyz[..., 1] = rand_vals[..., 1] * (y_max - y_min - 2.0) + y_min + 1.0
        coords_xyz[..., 2] = rand_vals[..., 2] * (z_max - z_min - 0.2) + z_min + 0.1
        return coords_xyz

    def get_cam_feats(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        x = self.cam_encode(x) 
        depth_logits = x[:, :self.D] 
        context = x[:, self.D:]      
        depth_probs = depth_logits.softmax(dim=1)
        context = context.unsqueeze(1)    
        depth_probs = depth_probs.unsqueeze(2) 
        cam_feats = context * depth_probs 
        return cam_feats

    def voxel_pooling(self, geom_feats, x, y, z):
        B, N, D, H, W, C = geom_feats.shape  # [1, 6, 41, 16, 44, 64]
        Nprime = B * N * D * H * W
        geom_feats = geom_feats.contiguous().view(Nprime, C)
        x = x.reshape(Nprime)
        y = y.reshape(Nprime)
        z = z.reshape(Nprime)

        X_MAX = int((self.grid_conf['xbound'][1] - self.grid_conf['xbound'][0]) / self.grid_conf['xbound'][2])
        Y_MAX = int((self.grid_conf['ybound'][1] - self.grid_conf['ybound'][0]) / self.grid_conf['ybound'][2])
        Z_MAX = int((self.grid_conf['zbound'][1] - self.grid_conf['zbound'][0]) / self.grid_conf['zbound'][2])

        mask = (x >= 0) & (x < X_MAX) & (y >= 0) & (y < Y_MAX) & (z >= 0) & (z < Z_MAX)
        x, y, z = x[mask], y[mask], z[mask]
        geom_feats = geom_feats[mask]

        indices = x * Y_MAX + y + z * (X_MAX * Y_MAX)
        ranks = indices.argsort() # 从小到大的索引
        x, y, z = x[ranks], y[ranks], z[ranks]
        geom_feats = geom_feats[ranks]
        indices = indices[ranks]

        keep = torch.ones_like(indices, dtype=torch.bool)
        keep[:-1] = (indices[1:] != indices[:-1])
        
        cumsum = torch.cumsum(geom_feats, 0)
        cumsum = cumsum[keep]
        cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))  # [37995, 64]
        
        final_bev = torch.zeros((1, X_MAX, Y_MAX, C), device=x.device)
        if cumsum.shape[0] > 0:
            final_bev.view(-1, C)[indices[keep]] = cumsum
        # [1, 200, 200, 64] -> [1, 64, 200, 200]
        return final_bev.permute(0, 3, 1, 2) 

    def forward(self, x, rots, trans, intrinsics):
        B, N, C, H, W = x.shape
        cam_feats = self.get_cam_feats(x)
        cam_feats = cam_feats.view(B, N, self.D, self.C, H, W).permute(0, 1, 2, 4, 5, 3) 
        geom_xyz = self.get_geometry(rots, trans, intrinsics)
        x_idx = ((geom_xyz[..., 0] - self.grid_conf['xbound'][0]) / self.grid_conf['xbound'][2]).floor().long()
        y_idx = ((geom_xyz[..., 1] - self.grid_conf['ybound'][0]) / self.grid_conf['ybound'][2]).floor().long()
        z_idx = ((geom_xyz[..., 2] - self.grid_conf['zbound'][0]) / self.grid_conf['zbound'][2]).floor().long()
        bev_feature = self.voxel_pooling(cam_feats, x_idx, y_idx, z_idx)
        return bev_feature

# ==========================================
# 2. 完整模型封装 (LSS + 分割头)
# ==========================================
class BEVSegmentationModel(nn.Module):
    def __init__(self, lss_core):
        super().__init__()
        self.lss = lss_core
        # 简单的任务头: 把 64 维特征映射为 1 维 Logits (用于二分类)
        # 实际项目中这里通常是一个 ResNet-Block 或 U-Net
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1) # 输出 1 通道 logits
        )

    def forward(self, x, rots, trans, intrinsics):
        # 1. 提取 BEV 特征 (Batch, 64, 200, 200)
        bev_feat = self.lss(x, rots, trans, intrinsics)
        
        # 2. 预测分割图 (Batch, 1, 200, 200)
        logits = self.head(bev_feat)
        return logits

# ==========================================
# 3. 伪造数据集 (Fake Dataset)
# ==========================================
class FakeLSSDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        self.grid_size = (200, 200)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 模拟输入: 6个相机，ResNet特征 (512通道，16x44大小)
        # 注意: 这里用 randn 模拟特征，模型很难收敛，仅用于跑通流程
        imgs = torch.randn(6, 512, 16, 44)
        
        # 模拟外参
        rots = torch.eye(3).unsqueeze(0).expand(6, 3, 3)
        trans = torch.zeros(6, 3)
        intrinsics = torch.eye(3).unsqueeze(0).expand(6, 3, 3)

        # 模拟 Ground Truth (真值): 一个位于中心的圆形区域作为 "可行驶区域"
        # 形状: (1, 200, 200)
        target = torch.zeros((1, 200, 200), dtype=torch.float32)
        
        # 创建一个简单的几何图案
        Y, X = torch.meshgrid(torch.arange(200), torch.arange(200), indexing='ij')
        center_x, center_y = 100, 100
        radius = 50
        mask = (X - center_x)**2 + (Y - center_y)**2 < radius**2
        target[:, mask] = 1.0

        return imgs, rots, trans, intrinsics, target

# ==========================================
# 4. 训练与验证流程
# ==========================================
def calculate_iou(pred_logits, target_mask):
    """计算 Intersection over Union (IoU)"""
    preds = torch.sigmoid(pred_logits) > 0.5
    targets = target_mask > 0.5
    
    intersection = (preds & targets).float().sum()
    union = (preds | targets).float().sum()
    
    if union == 0:
        return 0.0
    return (intersection / union).item()

def main():
    # --- 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0], 'dbound': [4.0, 45.0, 1.0],
    }
    data_conf = {'img_size': (16, 44), 'dbound': grid_conf['dbound'], 'D': 41}
    
    # --- 初始化模型 ---
    lss_core = LSS_Core(grid_conf, data_conf, input_channels=512, num_classes=64)
    model = BEVSegmentationModel(lss_core).to(device)

    # --- 数据准备 ---
    # 训练集 50 个样本，验证集 10 个样本
    train_loader = DataLoader(FakeLSSDataset(50), batch_size=1, shuffle=True)
    val_loader = DataLoader(FakeLSSDataset(10), batch_size=1, shuffle=False)

    # --- 优化器与损失函数 ---
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss() # 二分类常用的损失函数

    # --- 训练循环 ---
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        print(f"\nEpoch {epoch+1}/{epochs} 开始训练...")
        for batch_idx, (imgs, rots, trans, intrinsics, targets) in enumerate(train_loader):
            # 1. 搬运数据到 GPU
            imgs = imgs.to(device)
            rots = rots.to(device)
            trans = trans.to(device)
            intrinsics = intrinsics.to(device)
            targets = targets.to(device) # (B, 1, 200, 200)

            # 2. 梯度清零
            optimizer.zero_grad()

            # 3. 前向传播 (Forward)
            # 输出: (B, 1, 200, 200)
            preds = model(imgs, rots, trans, intrinsics)

            # 4. 计算损失 (Loss)
            loss = criterion(preds, targets)

            # 5. 反向传播 (Backward) - 梯度流过 Splat, Lift 直达 Backbone
            loss.backward()

            # 6. 更新参数
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

        print(f"Epoch {epoch+1} 平均 Loss: {total_loss / len(train_loader):.4f}")

        # --- 验证循环 ---
        model.eval()
        total_iou = 0
        with torch.no_grad():
            for imgs, rots, trans, intrinsics, targets in val_loader:
                imgs = imgs.to(device)
                rots = rots.to(device)
                trans = trans.to(device)
                intrinsics = intrinsics.to(device)
                targets = targets.to(device)

                preds = model(imgs, rots, trans, intrinsics)
                iou = calculate_iou(preds, targets)
                total_iou += iou
        
        print(f"Epoch {epoch+1} 验证集 mIoU: {total_iou / len(val_loader):.4f}")

if __name__ == "__main__":
    main()