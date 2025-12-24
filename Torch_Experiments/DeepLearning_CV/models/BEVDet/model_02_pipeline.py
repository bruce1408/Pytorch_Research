import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# ==========================================
# 1. 之前已修复好的模型定义 (Model Definition)
# ==========================================

class MockBackbone(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=16, padding=1)
    def forward(self, x): return self.conv(x)

class MockBEVEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x): 
        return self.conv(x)

class MockHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
    
        # 输出18通道: 10个类别热力图 + 8个回归属性(x, y, z, w, l, h, sin, cos)
        self.conv = nn.Conv2d(in_channels, 18, kernel_size=1)
    
    def forward(self, x): 
        return self.conv(x)

class LSSViewTransformer(nn.Module):
    def __init__(self, grid_conf, out_channels):
        super(LSSViewTransformer, self).__init__()
        self.grid_conf = grid_conf
        self.dx, self.bx, self.nx = self.gen_dx_bx(self.grid_conf['xbound'], 
                                                   self.grid_conf['ybound'], 
                                                   self.grid_conf['zbound'])
        
        self.D = int((grid_conf['dbound'][1] - grid_conf['dbound'][0]) / grid_conf['dbound'][2])
        
        self.depth_net = nn.Conv2d(512, self.D + out_channels, kernel_size=1)
        
        self.out_channels = out_channels

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx

    def create_frustum(self):
        H_feat, W_feat = 16, 44 
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, H_feat, W_feat)
        D, _, _ = ds.shape
        xs = torch.linspace(0, W_feat - 1, W_feat, dtype=torch.float).view(1, 1, W_feat).expand(D, H_feat, W_feat)
        ys = torch.linspace(0, H_feat - 1, H_feat, dtype=torch.float).view(1, H_feat, 1).expand(D, H_feat, W_feat)
        return torch.stack((xs, ys, ds), -1)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        B, N, _ = trans.shape
        frustum = self.create_frustum().to(trans.device)
        points = frustum.view(1, 1, *frustum.shape).repeat(B, N, 1, 1, 1, 1)
        points = points - post_trans.view(B, N, 1, 1, 1, 3)
        
        # 修复点：squeeze 维度
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        uv = points[..., :2]
        d  = points[..., 2:3]
        uv_scaled = uv * d
        points = torch.cat((uv_scaled, d), dim=-1)
        
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def voxel_pooling(self, geom_feats, x, y, z):
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
        geom_feats = geom_feats.reshape(-1, C)

        x = ((x - self.bx[0]) / self.dx[0]).long()
        y = ((y - self.bx[1]) / self.dx[1]).long()
        z = ((z - self.bx[2]) / self.dx[2]).long()
        
        valid = (x >= 0) & (x < self.nx[0]) & (y >= 0) & (y < self.nx[1]) & (z >= 0) & (z < self.nx[2])
        x, y, z, geom_feats = x[valid], y[valid], z[valid], geom_feats[valid]

        ranks =  x + y * self.nx[0] + z * (self.nx[0] * self.nx[1])
        sort_idx = ranks.argsort()
        x, y, z, ranks, geom_feats = x[sort_idx], y[sort_idx], z[sort_idx], ranks[sort_idx], geom_feats[sort_idx]

        keep = torch.ones_like(ranks, dtype=torch.bool)
        keep[:-1] = (ranks[1:] != ranks[:-1])
        cumsum = torch.cumsum(geom_feats, dim=0)
        cumsum = cumsum[keep]
        cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))
        
        final_bev = torch.zeros((1, C, self.nx[2], self.nx[1], self.nx[0]), device=geom_feats.device)
        if x.shape[0] > 0:
            final_bev[0, :, z[keep], y[keep], x[keep]] = cumsum.permute(1, 0)
        return final_bev.sum(2) 

    def forward(self, img_feats, rots, trans, intrins, post_rots, post_trans):
        B, N, C, H, W = img_feats.shape
        x = self.depth_net(img_feats.view(B*N, C, H, W))
        depth_digit = x[:, :self.D].softmax(dim=1)
        tran_feat = x[:, self.D:]
        outer = depth_digit.unsqueeze(1) * tran_feat.unsqueeze(2)
        outer = outer.view(B, N, self.out_channels, self.D, H, W).permute(0, 1, 3, 4, 5, 2)
        
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.voxel_pooling(outer, geom[..., 0], geom[..., 1], geom[..., 2])
        return x

class BEVDet(nn.Module):
    def __init__(self, grid_conf):
        super(BEVDet, self).__init__()
        self.img_backbone = MockBackbone(out_channels=512)
        self.view_transformer = LSSViewTransformer(grid_conf, out_channels=64)
        self.bev_encoder = MockBEVEncoder(in_channels=64, out_channels=256)
        self.head = MockHead(in_channels=256)

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans):
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        
        x = self.img_backbone(imgs)
        
        x = x.view(B, N, x.shape[1], x.shape[2], x.shape[3])
        
        x = self.view_transformer(x, rots, trans, intrins, post_rots, post_trans)
        
        x = self.bev_encoder(x)
        
        x = self.head(x)
        return x

# ==========================================
# 2. 模拟数据集 (Mock Dataset)
# ==========================================

class MockNuScenesDataset(Dataset):
    """
    模拟生成数据和标签。
    真实场景下，这里会读取磁盘上的图片和 JSON 标注。
    """
    def __init__(self, length=100, is_train=True):
        self.length = length
        self.is_train = is_train
        self.grid_size = (125, 125) # BEV 网格大小

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. 模拟输入数据
        N = 6 # 6个相机
        imgs = torch.randn(N, 3, 256, 704) # 归一化后的图像
        
        # 模拟相机参数
        rots = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        trans = torch.zeros(N, 3)
        intrins = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        post_rots = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        post_trans = torch.zeros(N, 3)

        # 2. 模拟 Ground Truth (标签)
        # 假设有 10 个类别 + 8 个回归参数 = 18 通道
        # Heatmap: (10, 125, 125) -> 只有在有物体的地方是 1 (高斯分布)，其他是 0
        # Regression: (8, 125, 125) -> 只有在物体中心处有值
        
        # 为了演示，我们随机生成一个“热力图”标签
        # 真实训练中，需要根据 GT Box 绘制高斯热力图
        gt_heatmap = torch.zeros((10, *self.grid_size))
        gt_reg = torch.zeros((8, *self.grid_size))
        
        # 随机放置一个物体在中心
        cx, cy = 60, 60
        gt_heatmap[0, cx, cy] = 1.0 # 第0类，在 (60,60) 处
        gt_reg[:, cx, cy] = 0.5 # 模拟回归值

        # 拼接作为最终 target
        targets = torch.cat([gt_heatmap, gt_reg], dim=0) # (18, 125, 125)

        return (imgs, rots, trans, intrins, post_rots, post_trans), targets

# ==========================================
# 3. 损失函数 (Simplified Loss)
# ==========================================

class SimpleCenterPointLoss(nn.Module):
    """
    简化版 Loss：
    - Heatmap 使用 MSE Loss (真实应用通常用 Gaussian Focal Loss)
    - Regression 使用 L1 Loss
    """
    def __init__(self):
        super().__init__()
        self.loss_hm = nn.MSELoss()
        self.loss_reg = nn.L1Loss()
        self.hm_weight = 1.0
        self.reg_weight = 0.25 # 回归通常权重小一点

    def forward(self, preds, targets):
        # preds: (B, 18, 125, 125)
        # targets: (B, 18, 125, 125)
        
        # 分离热力图和回归头
        pred_hm = preds[:, :10]
        pred_reg = preds[:, 10:]
        
        gt_hm = targets[:, :10]
        gt_reg = targets[:, 10:]
        
        # 计算 Loss
        # 注意：CenterPoint 只计算有物体位置的回归 Loss，这里为了简化计算所有位置
        # 真实代码需使用 mask 过滤
        l_hm = self.loss_hm(pred_hm, gt_hm)
        l_reg = self.loss_reg(pred_reg, gt_reg)
        
        return self.hm_weight * l_hm + self.reg_weight * l_reg, l_hm, l_reg

# ==========================================
# 4. 训练与验证循环 (Main Loop)
# ==========================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(dataloader):
        # 1. 数据搬运到 GPU
        imgs, rots, trans, intrins, post_rots, post_trans = [item.to(device) for item in inputs]
        targets = targets.to(device)
        
        # 2. 梯度清零
        optimizer.zero_grad()
        
        # 3. 前向传播
        preds = model(imgs, rots, trans, intrins, post_rots, post_trans)
        
        # 4. 计算 Loss
        loss, l_hm, l_reg = criterion(preds, targets)
        
        # 5. 反向传播与更新
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}] Step [{i}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} (HM: {l_hm.item():.4f}, Reg: {l_reg.item():.4f})")
            
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}] Train Loss: {avg_loss:.4f} Time: {time.time()-start_time:.2f}s")
    return avg_loss

def validate(model, dataloader, criterion, device):
    model.eval() # 切换到评估模式
    total_loss = 0
    
    with torch.no_grad(): # 不计算梯度，节省显存
        for inputs, targets in dataloader:
            imgs, rots, trans, intrins, post_rots, post_trans = [item.to(device) for item in inputs]
            targets = targets.to(device)
            
            preds = model(imgs, rots, trans, intrins, post_rots, post_trans)
            loss, _, _ = criterion(preds, targets)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# ==========================================
# 5. 主程序入口
# ==========================================

if __name__ == "__main__":
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.8],
        'ybound': [-50.0, 50.0, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }

    # 1. 实例化模型
    model = BEVDet(grid_conf).to(device)
    
    # 2. 准备数据
    train_dataset = MockNuScenesDataset(length=100, is_train=True)
    val_dataset = MockNuScenesDataset(length=20, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 3. 定义优化器和 Loss
    criterion = SimpleCenterPointLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 4. 开始训练
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        val_loss = validate(model, val_loader, criterion, device)
    
    print("\n训练完成！模型已准备好用于推理。")