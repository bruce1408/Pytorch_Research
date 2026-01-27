import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from bev_data_augmentation import BEVDataAugmentation


# ==========================================
# Part 1: 基础组件 (Backbone, Head, etc.)
# ==========================================

class MockBackbone(nn.Module):
    """模拟 ResNet-50，将图像下采样 16 倍"""
    def __init__(self, out_channels=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=16, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class MockBEVEncoder(nn.Module):
    """模拟 BEV 空间的 ResNet"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class MockHead(nn.Module):
    """模拟 CenterPoint Head"""
    def __init__(self, in_channels):
        super().__init__()
        # 输出18通道: 10个类别热力图 + 8个回归属性
        self.conv = nn.Conv2d(in_channels, 18, kernel_size=1)
    def forward(self, x): return self.conv(x)

# ==========================================
# Part 2: 核心 LSS View Transformer (已修复)
# ==========================================

class LSSViewTransformer(nn.Module):
    def __init__(self, grid_conf, out_channels):
        super(LSSViewTransformer, self).__init__()
        self.grid_conf = grid_conf
        self.dx, self.bx, self.nx = self.gen_dx_bx(self.grid_conf['xbound'], 
                                                   self.grid_conf['ybound'], 
                                                   self.grid_conf['zbound'])
        self.D = int((grid_conf['dbound'][1] - grid_conf['dbound'][0]) / grid_conf['dbound'][2])
        
        # DepthNet: 输入512 -> 输出 D(深度概率) + out_channels(语义特征)
        self.depth_net = nn.Conv2d(512, self.D + out_channels, kernel_size=1)
        self.out_channels = out_channels

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx

    def create_frustum(self):
        # 假设特征图大小为 16x44 (原图 256x704 下采样 16 倍)
        H_feat, W_feat = 16, 44 
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, H_feat, W_feat)
        D, _, _ = ds.shape
        xs = torch.linspace(0, W_feat - 1, W_feat, dtype=torch.float).view(1, 1, W_feat).expand(D, H_feat, W_feat)
        ys = torch.linspace(0, H_feat - 1, H_feat, dtype=torch.float).view(1, H_feat, 1).expand(D, H_feat, W_feat)
        return torch.stack((xs, ys, ds), -1)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        B, N, _ = trans.shape
        
        # 1. 生成视锥
        frustum = self.create_frustum().to(trans.device)
        points = frustum.view(1, 1, *frustum.shape).repeat(B, N, 1, 1, 1, 1)

        # 2. 【IDA 逆变换】: 抵消图像数据增强的影响
        points = points - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        # 3. 图像坐标 -> 相机坐标 (利用透视原理)
        uv = points[..., :2]
        d  = points[..., 2:3]
        points = torch.cat((uv * d, d), dim=-1)
        
        # 4. 相机坐标 -> 自车坐标 (Ego)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        return points

    def voxel_pooling(self, geom_feats, x, y, z):
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W

        # 1. Flatten 所有点
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        geom_feats = geom_feats.reshape(-1, C)

        # 2. 坐标量化
        x = ((x - self.bx[0]) / self.dx[0]).long()
        y = ((y - self.bx[1]) / self.dx[1]).long()
        z = ((z - self.bx[2]) / self.dx[2]).long()
        
        # 3. 边界过滤
        valid = (x >= 0) & (x < self.nx[0]) & \
                (y >= 0) & (y < self.nx[1]) & \
                (z >= 0) & (z < self.nx[2])
        x, y, z, geom_feats = x[valid], y[valid], z[valid], geom_feats[valid]

        # 4. 排序 (Sort) - 为 Cumsum 做准备
        ranks =  x + y * self.nx[0] + z * (self.nx[0] * self.nx[1])
        sort_idx = ranks.argsort()
        x, y, z, ranks, geom_feats = x[sort_idx], y[sort_idx], z[sort_idx], ranks[sort_idx], geom_feats[sort_idx]

        # 5. CumSum (前缀和) - 核心加速
        keep = torch.ones_like(ranks, dtype=torch.bool)
        keep[:-1] = (ranks[1:] != ranks[:-1])
        
        cumsum = torch.cumsum(geom_feats, dim=0)
        cumsum = cumsum[keep]
        cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))
        
        # 6. Scatter 回网格
        final = torch.zeros((1, C, self.nx[2], self.nx[1], self.nx[0]), device=geom_feats.device)
        
        if x.shape[0] > 0:
            # 【核心修复】: 必须使用 x[keep] 对应的唯一坐标进行填充
            final[0, :, z[keep], y[keep], x[keep]] = cumsum.permute(1, 0)

        return final.sum(2) 

    def forward(self, img_feats, rots, trans, intrins, post_rots, post_trans):
        B, N, C, H, W = img_feats.shape
        
        # Lift
        x = self.depth_net(img_feats.view(B*N, C, H, W))
        depth_digit = x[:, :self.D].softmax(dim=1)
        tran_feat = x[:, self.D:]
        
        # Outer Product
        outer = depth_digit.unsqueeze(1) * tran_feat.unsqueeze(2) # (BN, C, D, H, W)
        outer = outer.view(B, N, self.out_channels, self.D, H, W).permute(0, 1, 3, 4, 5, 2)
        
        # Geometry
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        
        # Splat
        x = self.voxel_pooling(outer, geom[..., 0], geom[..., 1], geom[..., 2])
        return x

# ==========================================
# Part 3: BEV Data Augmentation (BDA)
# ==========================================
class BEVDataAugmentation(nn.Module):
    """
    [BDA] BEV 空间数据增强模块
    在模型 Forward 中执行
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, bda_mat):
        # x: (B, C, H, W)
        # bda_mat: (B, 3, 3)
        if bda_mat is None:
            return x
        

        # --- 第一步：正确的变量赋值 ---
        # 必须从 bda_mat 开始处理，不要使用 rot_mat 这个变量名，除非你先定义它
        # 提取前两行 (Batch, 2, 3)
        theta = bda_mat[:, :2, :] 
        
        # --- 第二步：安全检查（可选，但推荐）---
        # 确保矩阵的 Batch Size 和特征图 x 的 Batch Size 一致
        if x.size(0) != theta.size(0):
            print(f"警告: Batch Size 不匹配! x: {x.size(0)}, mat: {theta.size(0)}")
            # 如果这是一个测试，你可能想强制让它们匹配（这就看具体需求了）
            # 这里我们让程序继续，看看上面的 print 输出了什么
        
        # --- 第三步：生成网格 ---
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        # --- 第四步：采样 ---
        x = F.grid_sample(x, grid, align_corners=True, mode='bilinear')
        
        return x

# ==========================================
# Part 4: 完整的 BEVDet 模型
# ==========================================

class BEVDet(nn.Module):
    def __init__(self, grid_conf):
        super(BEVDet, self).__init__()
        self.img_backbone = MockBackbone(out_channels=512)
        self.view_transformer = LSSViewTransformer(grid_conf, out_channels=64)
        self.bda_layer = BEVDataAugmentation() # 新增 BDA 层
        self.bev_encoder = MockBEVEncoder(in_channels=64, out_channels=256)
        self.head = MockHead(in_channels=256)

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, bda_mat=None):
        B, N, C, H, W = imgs.shape
        
        # 1. 图像编码
        imgs = imgs.view(B * N, C, H, W)
        x = self.img_backbone(imgs)
        x = x.view(B, N, x.shape[1], x.shape[2], x.shape[3])
        
        # 2. View Transformer (LSS)
        # 注意: 这里传入 post_rots/post_trans 是为了抵消 IDA
        x = self.view_transformer(x, rots, trans, intrins, post_rots, post_trans)
        
        # 3. BEV Data Augmentation (BDA)
        # 注意: 这里应用 BDA 矩阵，对 BEV 特征做旋转/翻转
        if self.training and bda_mat is not None:
            x = self.bda_layer(x, bda_mat)
        
        # 4. BEV Encoder
        x = self.bev_encoder(x)
        
        # 5. Head
        x = self.head(x)
        return x

# ==========================================
# Part 5: 数据集与数据增强 (IDA & BDA Sim)
# ==========================================

class MockNuScenesDataset(Dataset):
    def __init__(self, length=100, is_train=True):
        self.length = length
        self.is_train = is_train
        self.grid_size = (128, 128) # 与 grid_conf 对应

    def __len__(self):
        return self.length

    def get_ida_matrices(self):
        """模拟 IDA (图像增强) 并返回对应的变换矩阵"""
        # 随机参数
        resize = np.random.uniform(0.8, 1.2) if self.is_train else 1.0
        rotate = np.random.uniform(-5, 5) if self.is_train else 0.0
        flip = (np.random.random() > 0.5) if self.is_train else False
        
        # 图像尺寸
        H, W = 256, 704
        
        # 计算 3x3 变换矩阵 (Post Transformation)
        # 这里简化处理，构造对应的矩阵
        post_rot = np.eye(3)
        post_tran = np.zeros(3)
        
        # 1. Resize
        post_rot[:2, :2] *= resize
        
        # 2. Rotate (简化版)
        theta = np.deg2rad(rotate)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        post_rot[:2, :2] = np.dot(rot, post_rot[:2, :2])
        
        # 3. Flip
        if flip:
            post_rot[0, 0] *= -1
            post_tran[0] += W # 翻转带来的平移

        return torch.from_numpy(post_rot[:3, :3]).float(), torch.from_numpy(post_tran[:3]).float()

    def get_bda_matrix(self):
        """模拟 BDA (BEV 增强) 矩阵"""
        if not self.is_train:
            return torch.eye(3).float()
            
        angle = np.random.uniform(-20, 20)
        scale = np.random.uniform(0.9, 1.1)
        
        rad = np.deg2rad(angle)
        cos, sin = np.cos(rad), np.sin(rad)
        
        mat = np.eye(3)
        mat[:2, :2] = np.array([[cos, -sin], [sin, cos]]) * scale
        
        return torch.from_numpy(mat).float()

    def __getitem__(self, idx):
        N = 6 
        # 模拟 6 张图片
        imgs = torch.randn(N, 3, 256, 704)
        
        # 模拟相机参数
        rots = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        trans = torch.zeros(N, 3)
        intrins = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        
        # 生成 IDA 矩阵 (Post Rots/Trans)
        post_rots_list, post_trans_list = [], []
        for _ in range(N):
            r, t = self.get_ida_matrices()
            post_rots_list.append(r)
            post_trans_list.append(t)
        post_rots = torch.stack(post_rots_list)
        post_trans = torch.stack(post_trans_list)
        
        # 生成 BDA 矩阵
        bda_mat = self.get_bda_matrix()
        
        # 模拟 GT (Heatmap + Regression)
        # 真实训练中，如果 BDA 旋转了 BEV，这里的 targets 也必须跟着旋转！
        # 这里为了简化，我们假设 targets 已经是对齐好的
        targets = torch.zeros((18, 128, 128))
        targets[0, 64, 64] = 1.0 # 放一个物体在中心
        
        return (imgs, rots, trans, intrins, post_rots, post_trans, bda_mat), targets

# ==========================================
# Part 6: 训练与验证循环
# ==========================================

def train_loop():
    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    grid_conf = {
        'xbound': [-51.2, 51.2, 0.8], # 128个格子
        'ybound': [-51.2, 51.2, 0.8], # 128个格子
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    
    # 2. 模型与数据
    model = BEVDet(grid_conf).to(device)
    train_loader = DataLoader(MockNuScenesDataset(100, True), batch_size=1, shuffle=True)
    val_loader = DataLoader(MockNuScenesDataset(20, False), batch_size=1, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss() # 简化版 Loss
    
    # 3. Epoch 循环4
    for epoch in range(2):
        print(f"\n--- Epoch {epoch+1} ---")
        
        # Train
        model.train()
        train_loss = 0
        start = time.time()
        for i, (inputs, targets) in enumerate(train_loader):
            # 搬运数据到 GPU
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            imgs, rots, trans, intrins, post_rots, post_trans, bda_mat = inputs
            
            optimizer.zero_grad()
            
            # Forward (包含 IDA 逆变换 + BDA 正变换)
            preds = model(imgs, rots, trans, intrins, post_rots, post_trans, bda_mat)
            
            # Backward
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 5 == 0:
                print(f"Iter {i}: Loss = {loss.item():.4f}")
        
        print(f"Train Loss Avg: {train_loss/len(train_loader):.4f}, Time: {time.time()-start:.2f}s")
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                imgs, rots, trans, intrins, post_rots, post_trans, bda_mat = inputs
                
                preds = model(imgs, rots, trans, intrins, post_rots, post_trans, bda_mat)
                val_loss += loss_fn(preds, targets).item()
        
        print(f"Val Loss Avg: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    train_loop()