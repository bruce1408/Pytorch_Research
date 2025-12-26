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

    def forward(self, prev_bev, trans, rot):
        # 注意：这里不需要传入 bev_h, bev_w，直接从 tensor 获取即可
        B, C, H, W = prev_bev.shape
        
        # 仿射变换矩阵
        theta = torch.zeros(B, 2, 3, device=prev_bev.device)
        cos_r, sin_r = torch.cos(rot).squeeze(1), torch.sin(rot).squeeze(1)
        
        theta[:, 0, 0], theta[:, 0, 1] = cos_r, -sin_r
        theta[:, 1, 0], theta[:, 1, 1] = sin_r, cos_r        
        # 将像素位移转换为归一化坐标 (-1 ~ 1)
        theta[:, 0, 2] = trans[:, 0] / (W / 2.0) 
        theta[:, 1, 2] = trans[:, 1] / (H / 2.0)
        
        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        return F.grid_sample(prev_bev, grid, align_corners=False, padding_mode='zeros')

class LSSViewTransformer(nn.Module):
    def __init__(self, grid_conf, data_conf, downsample=16, num_depth_bins=118):
        """
        初始化 LSS 视图转换器
        :param grid_conf: BEV 网格配置 {'xbound': [-51.2, 51.2, 0.8], 'ybound': ..., 'zbound': ...}
        :param data_conf: 数据配置 {'input_size': [256, 704], ...}
        :param downsample: 特征图下采样倍率 (默认16，即输入256x704 -> 特征16x44)
        :param num_depth_bins: 深度桶数量 (D)
        """
        super().__init__()
        self.grid_conf = grid_conf
        self.data_conf = data_conf
        self.downsample = downsample
        self.D = num_depth_bins
        self.C = 64  # 输出 BEV 特征维度

        # 1. 准备 Frustum (视锥)
        # 生成固定的 (D, H_feat, W_feat, 3) 的网格
        self.frustum = self.create_frustum()

        # 2. DepthNet
        # 输入: Image Feature (C_in=256 or 512), 输出: Depth(D) + Context(C)
        self.depth_net = nn.Conv2d(256, self.D + self.C, kernel_size=1, padding=0)

    def create_frustum(self):
        """生成视锥网格 (D, H, W, 3)"""
        # 1. 解析图像尺寸和深度范围
        img_H, img_W = self.data_conf['input_size']
        feat_H, feat_W = img_H // self.downsample, img_W // self.downsample
        
        # 2. 生成深度网格 D
        d_min, d_max, d_step = 4.0, 45.0, 1.0 # 示例参数
        ds = torch.arange(d_min, d_max, d_step).view(-1, 1, 1).expand(-1, feat_H, feat_W)
        self.D = ds.shape[0]

        # 3. 生成像素坐标网格 (u, v)
        # 注意：要映射回原图坐标，所以要乘以 downsample
        xs = torch.linspace(0, img_W - 1, feat_W).view(1, 1, feat_W).expand(self.D, feat_H, -1)
        ys = torch.linspace(0, img_H - 1, feat_H).view(1, feat_H, 1).expand(self.D, -1, feat_W)

        # 4. 堆叠成 (D, H, W, 3) -> (u, v, d)
        # 最后一维是 3: x(u), y(v), z(depth)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        几何投影: 将视锥点从 像素坐标 -> 自车坐标
        """
        B, N, _ = trans.shape
        
        # 1. 抵消数据增强 (Resize/Crop/Rotate 的逆变换)
        # Undo post-transformation
        # formula: x = (x_img - post_trans) @ inv(post_rot)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # 2. 图像平面 -> 相机坐标系
        # formula: x_cam = x_img * depth * inv(intrinsic)
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], 
                            points[:, :, :, :, :, 2:3]), 5)
        
        # 结合内参
        combined_transform = rots.matmul(torch.inverse(intrins))
        points = combined_transform.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        
        # 3. 相机坐标系 -> 自车坐标系 (Ego)
        # formula: x_ego = rot * x_cam + trans
        points += trans.view(B, N, 1, 1, 1, 3)
        
        # Output: (B, N, D, H, W, 3) [x_ego, y_ego, z_ego]
        return points

    def voxel_pooling(self, geom_feats, x):
        """
        [关键] Voxel Pooling 使用 CumSum Trick 实现
        geom_feats: (B, N, D, H, W, 3) 坐标
        x: (B, N, D, H, W, C) 特征
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # 1. 展平数据
        x = x.reshape(Nprime, C)
        geom_feats = geom_feats.reshape(Nprime, 3)

        # 2. 过滤无效点 (超出 BEV 边界的点)
        # 解析 grid 配置
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        
        # 计算 grid index
        # index = (coord - start) / resolution
        geom_feats = ((geom_feats - (bx - dx/2.)) / dx).long()
        
        # 转换成 1D index 用于排序
        geom_feats = geom_feats[:, 0] + geom_feats[:, 1] * nx[0] + geom_feats[:, 2] * nx[0] * nx[1]
        geom_feats = geom_feats.long()

        # 生成 mask: 仅保留在 grid 范围内的点
        ranks = geom_feats
        kept = (ranks >= 0) & (ranks < nx[0] * nx[1] * nx[2])
        x = x[kept]
        ranks = ranks[kept]

        # 3. 排序 (Sorting) - 这是 CumSum 的前提
        # 将落在同一个格子的点排在一起
        ranks, indices = ranks.sort()
        x = x[indices]
        
        # 4. CumSum Trick (核心加速)
        # 通过前缀和快速计算同一个格子内的特征总和
        feat_cumsum = torch.cumsum(x, dim=0)
        
        # 找边界: 只要 ranks[i] != ranks[i+1]，说明换格子了
        mask = torch.ones(ranks.shape, device=x.device, dtype=torch.bool)
        mask[:-1] = (ranks[1:] != ranks[:-1])
        
        # 算出每个格子的 sum
        # sum[i] = cumsum[end_i] - cumsum[start_i - 1]
        feat_cumsum = torch.cat([torch.zeros((1, C), device=x.device), feat_cumsum], dim=0)
        final_feats = feat_cumsum[1:][mask] - feat_cumsum[:-1][mask] # 得到每个 voxel 的特征和
        final_ranks = ranks[mask] # 得到去重后的 voxel index

        # 5. 填回 BEV Grid
        # 初始化全 0 的 BEV
        bev_feat = torch.zeros((B, nx[2], nx[1], nx[0], C), device=x.device)
        
        # 此时 final_ranks 包含 batch 信息，需要拆解
        # 这里简化为 B=1 的情况，多 Batch 需要把 Batch 索引加入 ranks 计算
        # 为了通用性，通常把 Batch 算进 ranks 里
        
        # 简单填入 (Flatten View)
        bev_flat = torch.zeros((nx[0]*nx[1]*nx[2], C), device=x.device)
        bev_flat[final_ranks] = final_feats
        
        # Reshape 回 BEV 形状 (H_bev, W_bev, C)
        # 注意: LSS 通常会 collapse Z 轴 (nx[2]=1)
        bev_feat = bev_flat.view(nx[2], nx[1], nx[0], C)
        
        # 调整为 (B, C, H, W) 格式
        bev_feat = bev_feat.permute(0, 3, 1, 2).contiguous() # (1, C, H, W)
        
        # 如果是多 Batch，需要在上面 ranks 计算时加入 batch_offset，这里做了简化
        if B > 1:
            # 真实代码需处理 Batch 偏移，这里为了演示逻辑略过
            pass
            
        return bev_feat

    def gen_dx_bx(self, xbound, ybound, zbound):
        # 辅助函数: 计算网格分辨率(dx), 起点(bx), 尺寸(nx)
        dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]], dtype=torch.float, device=self.frustum.device)
        bx = torch.tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]], dtype=torch.float, device=self.frustum.device)
        nx = torch.tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=torch.long, device=self.frustum.device)
        return dx, bx, nx

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        :param x: 图像特征 (B, N, C_in, H, W)
        :param rots, trans: 外参
        :param intrins: 内参
        :param post_rots, post_trans: 数据增强参数
        """
        B, N, C_in, H, W = x.shape
        
        # 1. Lift: 推理深度 + 上下文
        x = x.view(B * N, C_in, H, W)
        x = self.depth_net(x) # (B*N, D+C, H, W)
        
        # 拆分 depth (Softmax) 和 context
        depth_digit = x[:, :self.D].softmax(dim=1)
        context = x[:, self.D:]
        
        # 外积: 生成视锥点云特征 (Frustum Features)
        # (B*N, C, 1, H, W) * (B*N, 1, D, H, W) -> (B*N, C, D, H, W)
        outer = context.unsqueeze(2) * depth_digit.unsqueeze(1)
        
        # 调整形状 -> (B, N, D, H, W, C)
        outer = outer.view(B, N, self.C, self.D, H, W).permute(0, 1, 3, 4, 5, 2)
        
        # 2. Splat: 计算几何坐标
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        
        # 3. Shoot: Voxel Pooling
        bev_feat = self.voxel_pooling(geom, outer)
        
        return bev_feat

class BEVDet4D(nn.Module):
    def __init__(self, bev_c=64, num_classes=1):
        super().__init__()
        
        # --- 1. 极简 Backbone ---
        # 目标: 将 (3, 256, 704) -> (256, 16, 44)
        # 计算: 256/16 = 16. 我们需要 16 倍下采样。
        # 实现: 使用一个 stride=16, kernel_size=16 的卷积层直接完成提取+下采样
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=16, stride=16, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # --- 2. LSS ---
        grid_conf = {
            'xbound': [-51.2, 51.2, 0.8], 
            'ybound': [-51.2, 51.2, 0.8], 
            'zbound': [-10.0, 10.0, 20.0]
        }
        
        data_conf = {'input_size': [256, 704]}
        
        self.lss = LSSViewTransformer(grid_conf, data_conf, num_depth_bins=50)        
        
        # --- 3. 对齐与融合 ---
        self.aligner = BEVAligner()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(bev_c * 2, bev_c, 3, padding=1),
            nn.BatchNorm2d(bev_c),
            nn.ReLU()
        )
        
        self.head = nn.Conv2d(bev_c, num_classes + 6, 1) 

    def forward_single_frame(self, imgs, mats_dict):
        # imgs shape: (B, N, 3, 256, 704)
        B, N, C, H, W = imgs.shape
        
        # 1. Flatten Batch & N to pass through Backbone
        imgs = imgs.view(B * N, C, H, W)
        
        # 2. Extract Features
        # Output: (B*N, 256, 16, 44)
        feats = self.backbone(imgs)
        
        # 3. Reshape back for LSS
        feats = feats.view(B, N, 256, feats.shape[2], feats.shape[3])
        
        # 4. LSS Transform
        bev = self.lss(feats, 
                       mats_dict['rots'], mats_dict['trans'], 
                       mats_dict['intrins'], 
                       mats_dict['post_rots'], mats_dict['post_trans'])
        return bev
    
    def forward(self, curr_imgs, curr_mats, prev_bev=None, ego_trans=None, ego_rot=None):
        # Step 1: 当前帧生成 BEV [1, 64, 128, 128]
        curr_bev = self.forward_single_frame(curr_imgs, curr_mats)
        
        # Step 2: 时序融合
        if prev_bev is not None:
            aligned_prev = self.aligner(prev_bev, ego_trans, ego_rot)
            feat = torch.cat([curr_bev, aligned_prev], dim=1)
        else:
            feat = torch.cat([curr_bev, curr_bev], dim=1)
        # [1, 64, 128, 128]
        feat = self.encoder(feat)
        output = self.head(feat)  # [1, 7, 128, 128]
        return output, curr_bev.detach()

# ==========================================
# 4. 增强版 Dataset (生成相机参数)
# ==========================================
class TemporalDataset(Dataset):
    def __init__(self, length=20):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. 图像 (6张, 3通道, 256x704)
        curr_imgs = torch.randn(6, 3, 256, 704)
        # 实际上 LSS 不需要上一帧的 Raw Image，只需要上一帧的 BEV Feature
        # 为了模拟训练中的 Teacher Forcing，我们生成 prev_imgs 用于提取 prev_bev
        prev_imgs = torch.randn(6, 3, 256, 704)
        
        # 2. 生成 LSS 必须的 5 个矩阵 (Mock Data)
        # 真实场景中，这些需要从 nuScenes 的 calib 文件读取
        
        # 相机外参 (Camera -> Ego)
        rots = torch.eye(3).view(1, 3, 3).repeat(6, 1, 1)
        trans = torch.zeros(6, 3)
        
        # 相机内参 (Pixel -> Camera)
        # fx=500, fy=500, cx=352, cy=128
        intrins = torch.eye(3).view(1, 3, 3).repeat(6, 1, 1)
        intrins[:, 0, 0] = 500
        intrins[:, 1, 1] = 500
        intrins[:, 0, 2] = 352
        intrins[:, 1, 2] = 128
        
        # 数据增强矩阵 (Identity for now)
        post_rots = torch.eye(3).view(1, 3, 3).repeat(6, 1, 1)
        post_trans = torch.zeros(6, 3)

        # 3. 自车运动 (Ego Motion)
        dx_pixel, dy_pixel = 1.0, 4.0 
        d_theta = 0.1
        ego_trans = torch.tensor([dx_pixel, dy_pixel]) # 像素单位
        ego_rot = torch.tensor([d_theta])
        
        # GT Box (Mock)
        gt_box = torch.tensor([0, 64.0, 64.0, 4.0, 2.0, 0.5]) 

        return {
            'curr_imgs': curr_imgs,
            'prev_imgs': prev_imgs,
            'mats': {
                'rots': rots, 'trans': trans, 'intrins': intrins,
                'post_rots': post_rots, 'post_trans': post_trans
            },
            'ego_trans': ego_trans, 'ego_rot': ego_rot, 'gt_box': gt_box
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
def main():
    device = torch.device("cpu") # LSS 的 CumSum 可以在 CPU 跑，但建议 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    print(f"Running Real-LSS BEVDet4D on {device}...")
    
    # 1. 只有 batch_size=1 时，简单的 CumSum 代码才不会报错
    # 如果要支持 Batch > 1，需要修改 LSS 中的 ranks 处理逻辑
    loader = DataLoader(TemporalDataset(), batch_size=1, shuffle=True)
    model = BEVDet4D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for i, batch in enumerate(loader):
        # 准备数据 [1, 6, 3, 256, 704]
        curr_imgs = batch['curr_imgs'].to(device)
        prev_imgs = batch['prev_imgs'].to(device)
        
        # 解包相机参数
        mats = batch['mats']
        curr_mats = {k: v.to(device) for k, v in mats.items()}
        
        ego_trans = batch['ego_trans'].to(device).float()
        ego_rot = batch['ego_rot'].to(device).float()
        
        # --- Step 1: 获取上一帧 BEV (Teacher Forcing) ---
        with torch.no_grad():
            # 上一帧也需要过一遍模型来获得 BEV，这里复用 curr_mats 简化
            prev_bev = model.forward_single_frame(prev_imgs, curr_mats)
        
        # --- Step 2: 前向传播 ---
        preds, _ = model(curr_imgs, curr_mats, prev_bev, ego_trans, ego_rot)
        
        # --- Step 3: 简单计算 Loss ---
        # 只要 preds 有输出，且 shape 是 (B, 7, 128, 128)，说明 pipeline 通了
        loss = preds.sum() 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Iter {i}: BEV Shape {preds.shape}, Loss {loss.item():.4f}")
        if i >= 2: break # 跑几步测试即可
        
        
if __name__ == "__main__":
    main()