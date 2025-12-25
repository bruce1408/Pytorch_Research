import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# ==========================================
# 1. 基础模型定义 (沿用上一节的核心逻辑)
# ==========================================
class BEVAligner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev_bev, trans, rot, bev_h, bev_w):
        # trans: [dx, dy] (pixel), rot: [theta] (rad)
        B, C, H, W = prev_bev.shape
        theta = torch.zeros(B, 2, 3, device=prev_bev.device)
        cos_r, sin_r = torch.cos(rot).squeeze(1), torch.sin(rot).squeeze(1)
        
        theta[:, 0, 0], theta[:, 0, 1] = cos_r, -sin_r
        theta[:, 1, 0], theta[:, 1, 1] = sin_r, cos_r
        theta[:, 0, 2] = trans[:, 0] / (W / 2.0) # Pixel to Normalized Grid
        theta[:, 1, 2] = trans[:, 1] / (H / 2.0)
        
        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        return F.grid_sample(prev_bev, grid, align_corners=False, padding_mode='zeros')

class BEVDet4D(nn.Module):
    def __init__(self, bev_c=64, num_classes=1):
        super().__init__()
        # 注意：这里模拟 LSS，先降维再池化到 BEV 大小
        self.lss_conv = nn.Conv2d(256, bev_c, 1)
        self.lss_pool = nn.AdaptiveAvgPool2d((128, 128))
        
        self.aligner = BEVAligner()
        self.encoder = nn.Sequential(
            nn.Conv2d(bev_c * 2, bev_c, 3, padding=1),
            nn.BatchNorm2d(bev_c), nn.ReLU()
        )
        # Head 输出: [heatmap(1), offset_x, offset_y, w, l, sin, cos] = 7 channels
        self.head = nn.Conv2d(bev_c, num_classes + 6, 1) 

    def forward(self, imgs, prev_bev=None, trans=None, rot=None):
        # imgs shape: (B, N, C, H, W) -> [4, 6, 256, 16, 44]
        B, N, C, H, W = imgs.shape
        
        # 1. 维度变换: (B, N, C, H, W) -> (B*N, C, H, W)
        imgs_reshaped = imgs.view(B * N, C, H, W)
        
        # 2. 提取特征 (模拟 LSS 的 Lift 和 Splat)
        # 结果 shape: (B*N, 64, 128, 128)
        feats = self.lss_conv(imgs_reshaped)
        feats = self.lss_pool(feats)
        
        # 3. 维度还原并聚合: (B*N, 64, ...) -> (B, N, 64, ...) -> (B, 64, ...)
        # 这里简单使用 mean() 模拟 Voxel Pooling 将多视角特征融合为一张 BEV
        curr_bev = feats.view(B, N, -1, 128, 128).mean(dim=1)
        
        # 4. 时序融合 (Temporal Fusion)
        if prev_bev is not None:
            # 确保维度匹配，aligner 需要 (B, C, H, W)
            aligned_prev = self.aligner(prev_bev, trans, rot, 128, 128)
            feat = torch.cat([curr_bev, aligned_prev], dim=1)
        else:
            # 第一帧，没有历史，自己拼自己 (填充)
            feat = torch.cat([curr_bev, curr_bev], dim=1)
            
        # 5. BEV Encoder & Head
        feat = self.encoder(feat)
        output = self.head(feat) # (B, 7, 128, 128)
        
        # 返回 output 用于算 Loss, 返回 curr_bev.detach() 用于存入 History
        return output, curr_bev.detach()

# ==========================================
# 2. 数据集模拟 (MockNuScenesSequence)
# ==========================================
class TemporalDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        self.voxel_size = 0.5 # 0.5米/像素
        self.bev_size = 128   # 128x128 像素

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. 模拟当前帧图像
        # Shape: (6, C, H, W) -> 这里简化 C=256 特征图
        curr_imgs = torch.randn(6, 256, 16, 44) 
        
        # 2. 模拟 Ego-Motion (当前帧相对于上一帧的运动)
        # 假设车往前开了 2 米 (dy=4 pixels), 转了 0.1 rad
        dx_meter, dy_meter = 0.5, 2.0 
        d_theta = 0.1
        
        # 转换为像素单位传给模型
        trans_pixel = torch.tensor([dx_meter / self.voxel_size, dy_meter / self.voxel_size]) # [1, 4]
        rot_rad = torch.tensor([d_theta])

        # 3. 模拟上一帧图像 (实际训练中，需要读取上一帧数据)
        # 这里为了演示，我们假设上一帧图像稍有不同
        prev_imgs = curr_imgs + torch.randn_like(curr_imgs) * 0.1

        # 4. 生成 Ground Truth (Target)
        # 假设当前帧有一个物体在 BEV 中心 (64, 64)
        # [class_id, x, y, w, l, yaw]
        gt_box = torch.tensor([0, 64.0, 64.0, 4.0, 2.0, 0.5]) 
        
        return {
            'curr_imgs': curr_imgs,
            'prev_imgs': prev_imgs, # 真实训练中通常通过 Sequential Sampler 获取
            'trans': trans_pixel,
            'rot': rot_rad,
            'gt_box': gt_box
        }

# ==========================================
# 3. Loss 函数 (Gaussian Focal Loss + L1)
# ==========================================
class CenterHeadLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def gaussian_focal_loss(self, pred, target):
        # pred: (B, 1, H, W) Sigmoid后的热力图
        # target: (B, 1, H, W) GT 高斯热力图
        pos_inds = target.eq(1)
        neg_inds = target.lt(1)

        neg_weights = torch.pow(1 - target[neg_inds], 4)
        loss = 0

        pos_loss = torch.log(pred[pos_inds]) * torch.pow(1 - pred[pos_inds], 2)
        neg_loss = torch.log(1 - pred[neg_inds]) * torch.pow(pred[neg_inds], 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos

    def forward(self, preds, gt_boxes):
        """
        preds: (B, 7, 128, 128) -> [heatmap, reg...]
        gt_boxes: (B, 6) -> [cls, x, y, w, l, yaw]
        """
        B, C, H, W = preds.shape
        
        # Split Prediction
        pred_hm = torch.sigmoid(preds[:, 0:1]) # Heatmap
        pred_reg = preds[:, 1:]                # Regression
        
        # Generate Targets (On the fly for demo)
        target_hm = torch.zeros_like(pred_hm)
        target_reg = torch.zeros_like(pred_reg)
        mask = torch.zeros(B, 1, H, W).to(preds.device)

        # 极其简化的 Target 生成 (仅生成一个 GT 点)
        for b in range(B):
            cx, cy = int(gt_boxes[b, 1]), int(gt_boxes[b, 2])
            # 1. 绘制高斯热力图
            if 0 <= cx < W and 0 <= cy < H:
                target_hm[b, 0, cy, cx] = 1.0 # 简化为单点，实际应画高斯圆
                # 简单扩散一下高斯 (3x3)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                         if 0<=cy+dy<H and 0<=cx+dx<W:
                             target_hm[b, 0, cy+dy, cx+dx] = max(target_hm[b, 0, cy+dy, cx+dx], 0.8 if (dx!=0 or dy!=0) else 1.0)
                
                # 2. 回归 Target
                # reg: [dx, dy, w, l, sin, cos]
                target_reg[b, 0, cy, cx] = gt_boxes[b, 1] - cx # offset x
                target_reg[b, 1, cy, cx] = gt_boxes[b, 2] - cy # offset y
                target_reg[b, 2, cy, cx] = torch.log(gt_boxes[b, 3]) # w
                target_reg[b, 3, cy, cx] = torch.log(gt_boxes[b, 4]) # l
                target_reg[b, 4, cy, cx] = torch.sin(gt_boxes[b, 5])
                target_reg[b, 5, cy, cx] = torch.cos(gt_boxes[b, 5])
                
                mask[b, 0, cy, cx] = 1.0

        # Calculate Loss
        loss_hm = self.gaussian_focal_loss(pred_hm, target_hm)
        loss_reg = F.l1_loss(pred_reg * mask, target_reg * mask, reduction='sum') / (mask.sum() + 1e-4)

        return loss_hm + 2.0 * loss_reg, {"loss_hm": loss_hm.item(), "loss_reg": loss_reg.item()}

# ==========================================
# 4. 训练与验证 Engine
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        curr_imgs = batch['curr_imgs'].to(device)
        prev_imgs = batch['prev_imgs'].to(device)
        trans = batch['trans'].to(device).float()
        rot = batch['rot'].to(device).float()
        gt_boxes = batch['gt_box'].to(device)
        
        optimizer.zero_grad()
        
        # --- BEVDet4D 的核心训练逻辑: Teacher Forcing ---
        # 1. 先推理上一帧，获取 prev_bev (Detach!)
        # 在真实训练中，我们通常会随机采样相邻两帧。
        # 为了得到上一帧的特征，我们需要先过一遍模型。
        # 注意: 上一帧不需要梯度，因为我们只训练当前帧如何融合上一帧。
        with torch.no_grad():
            _, prev_bev_feat = model(prev_imgs, prev_bev=None) # 假设上一帧没有历史
        
        # 2. 推理当前帧 (融合上一帧)
        preds, curr_bev_feat = model(curr_imgs, prev_bev=prev_bev_feat, trans=trans, rot=rot)
        
        # 3. 计算 Loss
        loss, loss_dict = criterion(preds, gt_boxes)
        
        # 4. 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Train Batch {batch_idx}: Loss={loss.item():.4f} (HM={loss_dict['loss_hm']:.4f}, Reg={loss_dict['loss_reg']:.4f})")
            
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_detections = 0
    
    with torch.no_grad():
        for batch in loader:
            curr_imgs = batch['curr_imgs'].to(device)
            prev_imgs = batch['prev_imgs'].to(device)
            trans = batch['trans'].to(device).float()
            rot = batch['rot'].to(device).float()
            gt_boxes = batch['gt_box'].to(device)
            
            # 推理上一帧
            _, prev_bev_feat = model(prev_imgs, prev_bev=None)
            # 推理当前帧
            preds, _ = model(curr_imgs, prev_bev=prev_bev_feat, trans=trans, rot=rot)
            
            # 计算验证 Loss
            loss, _ = criterion(preds, gt_boxes)
            total_loss += loss.item()
            
            # --- 简单的后处理 (Post-Processing) ---
            # 找 Heatmap 最大的点
            pred_hm = torch.sigmoid(preds[:, 0:1])
            B, _, H, W = pred_hm.shape
            
            for b in range(B):
                # 展平找最大值索引
                flatten_hm = pred_hm[b, 0].view(-1)
                max_score, max_idx = torch.max(flatten_hm, 0)
                
                # 转换回 (y, x)
                pred_y = max_idx // W
                pred_x = max_idx % W
                
                # 获取 GT 的 (y, x)
                gt_x, gt_y = int(gt_boxes[b, 1]), int(gt_boxes[b, 2])
                
                # 简单计算：如果峰值位置距离 GT 小于 2 个像素，算检测正确
                dist = np.sqrt((pred_x.cpu() - gt_x)**2 + (pred_y.cpu() - gt_y)**2)
                if dist < 2.0:
                    correct_detections += 1
                    
    accuracy = correct_detections / len(loader.dataset)
    return total_loss / len(loader), accuracy

# ==========================================
# 5. 主程序入口
# ==========================================
def main_pipeline():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    LR = 1e-3
    EPOCHS = 5
    
    print(f"Starting BEVDet4D Training Pipeline on {device}...")
    
    # 1. 实例化 Dataset & DataLoader
    train_set = TemporalDataset(length=100)
    val_set = TemporalDataset(length=20)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 实例化模型、Loss、优化器
    model = BEVDet4D().to(device)
    criterion = CenterHeadLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 3. 循环 Epoch
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch Summary: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc (Peak Hit)={val_acc*100:.1f}%")

if __name__ == "__main__":
    main_pipeline()