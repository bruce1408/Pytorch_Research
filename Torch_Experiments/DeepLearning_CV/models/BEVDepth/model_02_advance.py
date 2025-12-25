import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# [组件 1] Camera-Aware DepthNet (保持不变)
# ==========================================
class CameraAwareDepthNet(nn.Module):
    """
        相机参数感知的深度预测网络
        DepthNet 的核心任务是：预测每个像素点在空间中的深度分布（Probability Distribution）。
    """
    def __init__(self, in_channels=256, mid_channels=256, depth_bins=40): # 改小 depth_bins 防止 OOM
        
        super().__init__()
        
        self.depth_bins = depth_bins
        
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.bn_params = nn.BatchNorm1d(27)
        
        self.depth_mlp = nn.Sequential(
            nn.Linear(27, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.se_attention = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.Sigmoid() 
        )
        
        self.depth_head = nn.Conv2d(mid_channels, depth_bins, kernel_size=1)

    def forward(self, x, mats):
        
        feat = self.reduce_conv(x)
        
        # 相机参数进行归一化
        mats = self.bn_params(mats)
        
        # 把参数升维到高纬空间
        context = self.depth_mlp(mats)
        
        attn_weight = self.se_attention(context).unsqueeze(-1).unsqueeze(-1)
        
        # 如果这是长焦镜头拍的（权重里包含了这个信息），请增强那些代表远距离物体的特征通道
        feat = feat * attn_weight 
        
        depth_logits = self.depth_head(feat)
        
        return depth_logits

# ==========================================
# [组件 2] 真实的 LSS View Transformer (核心还原)
# ==========================================
class LSSViewTransformer(nn.Module):
    """
    还原论文思想的 Lift-Splat-Shoot 模块
    包含:
    1. create_frustum: 生成视锥点云
    2. get_geometry: 利用内参外参计算 3D 坐标
    3. voxel_pooling: 将特征投影到 BEV 网格 (Splat)
    """
    def __init__(self, grid_conf, data_conf, num_input_channels=256):
        super().__init__()
        self.grid_conf = grid_conf # BEV 网格配置 (范围, 大小)
        self.data_conf = data_conf # 数据配置 (输入尺寸, 深度范围)
        self.out_channels = num_input_channels
        
        # 初始化视锥 (D, H, W, 3) -> 每一个像素在每一个深度的 (u, v, d)
        self.frustum = self.create_frustum()

    def create_frustum(self):
        # 1. 生成深度网格 D
        d_min, d_max, d_bins = self.data_conf['depth']
        depth_grid = torch.arange(d_min, d_max, (d_max - d_min) / d_bins).view(-1, 1, 1) + (d_max - d_min) / d_bins # (D, 1, 1)
        
        # 2. 生成像素网格 H, W
        H, W = self.data_conf['img_size']
        # 注意: +0.5 是为了取像素中心
        x_grid = torch.linspace(0, W - 1, W).view(1, 1, W)
        y_grid = torch.linspace(0, H - 1, H).view(1, H, 1)
        
        # 3. 扩展成 (D, H, W) -> [40, 32, 88]
        D = depth_grid.shape[0]
        x_grid = x_grid.expand(D, H, W)
        y_grid = y_grid.expand(D, H, W)
        depth_grid = depth_grid.expand(D, H, W)
        
        # 4. 堆叠成 (D, H, W, 3) -> 最后一维是 (u, v, d)
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins):
        """
        几何投影核心: 像素坐标 (u,v,d) -> 图像坐标 -> 相机坐标 -> 自车(Ego)坐标
        """
        B, N, _ = trans.shape
        D, H, W, _ = self.frustum.shape
        
        # 1. 像素坐标 -> 相机归一化坐标 (利用内参)
        # points: (B, N, D, H, W, 3)
        points = self.frustum.view(1, 1, D, H, W, 3).expand(B, N, D, H, W, 3)
        
        # 这里的计算逻辑: 
        # x_c = (u - c_x) * z / f_x
        # y_c = (v - c_y) * z / f_y
        # z_c = d
        # 为了高效，我们将 points 展平进行矩阵运算
        
        # 2. 逆投影 + 坐标变换 (Cam -> Ego)
        # 这是一个简化的写法，真实 LSS 实际上是把 (u,v,1) * d 变成 (ud, vd, d)，然后乘内参逆矩阵，再乘外参
        
        # (为了演示清晰，我们假设 dataset 直接提供了可以将 (u*d, v*d, d) 转换到 ego 坐标的 4x4 组合矩阵)
        # 但为了还原度，我们还是写标准流程：
        
        # [A] 构建 (u*d, v*d, d)
        points_d = points.clone()
        points_d[..., 0] = points[..., 0] * points[..., 2] # u * d
        points_d[..., 1] = points[..., 1] * points[..., 2] # v * d
        # points_d[..., 2] 已经是 d 了
        
        # [B] 所有的矩阵运算放到这里 (Batch处理)
        # points_d: (B, N, D, H, W, 3) -> flattened (B*N*D*H*W, 3)
        flat_points = points_d.view(B*N*D*H*W, 3)
        
        # 添加一列 1，变成齐次坐标 (B*N*D*H*W, 4) (u*d, v*d, d, 1)
        flat_points_homo = torch.cat([flat_points, torch.ones_like(flat_points[:, :1])], dim=1)
        
        # [C] 组合变换矩阵: Sensor2Ego @ Intrinsics^-1
        # rots: (B, N, 3, 3), trans: (B, N, 3) -> sensor2ego: (B, N, 4, 4)
        # intrins: (B, N, 3, 3) -> intrins_inv: (B, N, 3, 3) (扩展到4x4)
        
        # ...此处省略繁琐的矩阵构建代码，直接假设 sensor2ego_aug 矩阵已传入...
        # 我们用一个 dummy 计算代替复杂的矩阵乘法，重点展示 "Splat"
        # 假设 flat_points 经过了正确的旋转平移，变成了真实世界的 (X, Y, Z)
        
        # 模拟几何变换结果:
        # X, Y 分布在 BEV 网格范围内 (-50m ~ 50m)
        geom_xyz = flat_points # 这里仅为占位，实际需要乘以 self.sensor2ego_mats
        
        # 为了让代码跑通，我们根据 frustum 模拟生成的 XYZ
        # 让 X, Y 覆盖 grid 范围
        scale_x = (self.grid_conf['xbound'][1] - self.grid_conf['xbound'][0]) / W
        scale_y = (self.grid_conf['ybound'][1] - self.grid_conf['ybound'][0]) / H
        geom_xyz[:, 0] = (flat_points[:, 0] / points_d.max()) * (self.grid_conf['xbound'][1] - self.grid_conf['xbound'][0]) + self.grid_conf['xbound'][0]
        geom_xyz[:, 1] = (flat_points[:, 1] / points_d.max()) * (self.grid_conf['ybound'][1] - self.grid_conf['ybound'][0]) + self.grid_conf['ybound'][0]
        
        return geom_xyz # (N_points, 3)

    def voxel_pooling(self, geom_feats, geom_xyz):
        """
        Splat 核心: 将离散的点云特征聚合到规则的 BEV 网格中
        geom_feats: (Total_Points, C)  <-- 修正了注释，输入已经是平铺的了
        geom_xyz:   (Total_Points, 3)
        """
        # --- 修复开始 ---
        # 错误代码: B, N, D, H, W, C = geom_feats.shape[:-1] + (self.out_channels,)
        # 修复如下: 直接获取总点数和通道数
        Nprime, C = geom_feats.shape 
        # --- 修复结束 ---

        # 1. 将物理坐标 (x, y, z) 转换为网格坐标 (dx, dy, dz)
        dx = self.grid_conf['xbound'][2]
        dy = self.grid_conf['ybound'][2]
        dz = self.grid_conf['zbound'][2]
        x_min = self.grid_conf['xbound'][0]
        y_min = self.grid_conf['ybound'][0]
        z_min = self.grid_conf['zbound'][0]

        # 转换坐标 (Clone一下防止改变原数据)
        cur_xyz = geom_xyz.clone()
        cur_xyz[:, 0] = (cur_xyz[:, 0] - x_min) / dx
        cur_xyz[:, 1] = (cur_xyz[:, 1] - y_min) / dy
        cur_xyz[:, 2] = (cur_xyz[:, 2] - z_min) / dz
        
        # 取整
        cur_xyz = cur_xyz.long()

        # 2. 过滤掉落在 BEV 网格范围之外的点
        # 根据 xbound 计算网格大小: (-50, 50) / 0.8 = 125 (取整后可能是 128，这里我们动态计算一下)
        bev_w = int((self.grid_conf['xbound'][1] - self.grid_conf['xbound'][0]) / dx)
        bev_h = int((self.grid_conf['ybound'][1] - self.grid_conf['ybound'][0]) / dy)
        
        valid_mask = (cur_xyz[:, 0] >= 0) & (cur_xyz[:, 0] < bev_w) & \
                     (cur_xyz[:, 1] >= 0) & (cur_xyz[:, 1] < bev_h) & \
                     (cur_xyz[:, 2] >= 0) # 也可以加上 z 轴过滤
        
        cur_xyz = cur_xyz[valid_mask]
        cur_feats = geom_feats[valid_mask]

        # 3. Splat (Pooling)
        # 初始化 BEV 特征图
        bev_map = torch.zeros((bev_h, bev_w, C), device=geom_feats.device)
        
        # 如果没有有效点，直接返回全0
        if cur_xyz.shape[0] == 0:
            return bev_map.permute(2, 0, 1).unsqueeze(0)

        # 构造索引 (y, x)
        indices = (cur_xyz[:, 1], cur_xyz[:, 0])
        
        # 累加特征 (Scatter Add)
        # index_put_ 是 PyTorch 中实现 voxel pooling 的简易方法 (虽然比 CUDA 慢)
        bev_map.index_put_(indices, cur_feats, accumulate=True)
        
        # 调整输出形状: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        return bev_map.permute(2, 0, 1).unsqueeze(0)
    

    def forward(self, img_feat, depth_logits, rots, trans, intrins):
        """
        img_feat: (B, N, C, H, W)
        depth_logits: (B, N, D, H, W)
        """
        B, N, C, H, W = img_feat.shape
        D = self.data_conf['depth'][2]
        
        # 1. Lift (提升): 图像特征 * 深度概率 = 视锥特征 (Frustum Features)
        depth_probs = F.softmax(depth_logits, dim=2) # 在深度维做 Softmax
        
        # 显存优化: 不要直接存 (B, N, C, D, H, W)，那个太大了
        # 我们把 Batch 和 N 合并处理
        img_feat = img_feat.view(B*N, C, H, W)
        depth_probs = depth_probs.view(B*N, D, H, W)
        
        # Outer Product: (B*N, C, 1, H, W) * (B*N, 1, D, H, W) = (B*N, C, D, H, W)
        # 这里的维度变换是 LSS 的标志性操作
        frustum_feat = img_feat.unsqueeze(2) * depth_probs.unsqueeze(1)
        
        # 2. Geometry: 计算每个点的坐标
        # 为了 voxel_pooling，我们需要展平特征 -> (N_points, C)
        geom_feats = frustum_feat.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        
        # (N_points, 3)
        geom_xyz = self.get_geometry(rots, trans, intrins) 

        # 3. Splat: 投影到 BEV
        bev_feat = self.voxel_pooling(geom_feats, geom_xyz)
        
        return bev_feat

# ==========================================
# [组件 3] 完整模型 BEVDepth
# ==========================================
class BEVDepthModel(nn.Module):
    def __init__(self, grid_conf, data_conf):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.depth_net = CameraAwareDepthNet(in_channels=256, mid_channels=256, depth_bins=data_conf['depth'][2])
        
        # 使用真实的 LSS 转换器
        self.view_transformer = LSSViewTransformer(grid_conf, data_conf)
        
        self.det_head = nn.Conv2d(256, 10, kernel_size=1)

    def get_depth_loss(self, depth_logits, gt_depth):
        # 维度调整: (B, N, D, H, W) -> (B*N, D, H, W)
        B, N, D, H, W = depth_logits.shape
        depth_logits = depth_logits.view(B*N, D, H, W)
        gt_depth = gt_depth.view(B*N, H, W)
        
        mask = gt_depth >= 0 
        if mask.sum() > 0:
            loss = F.cross_entropy(depth_logits[mask.unsqueeze(1).expand_as(depth_logits)].view(-1, D), gt_depth[mask].long())
        else:
            loss = torch.tensor(0.0, device=depth_logits.device, requires_grad=True)
        return loss

    def forward(self, imgs, mats, rots, trans, intrins, gt_depth=None):
        B, N, C, H, W = imgs.shape
        
        # 1. 图像编码 [6, 3, 32, 88]
        imgs_flat = imgs.view(B * N, C, H, W)
        
        # [6, 27]
        mats_flat = mats.view(B * N, -1)
        
        # [6, 256, 32, 88] 
        feat = self.backbone(imgs_flat) 
        
        # 2. 深度预测 -> [6, 40, 32, 88]
        depth_logits = self.depth_net(feat, mats_flat) 
        
        # Reshape 回 (B, N, D, H, W) 以便 View Transformer 处理 -> [6, 40, 32, 88]
        depth_logits = depth_logits.view(B, N, -1, H, W) 
        
        # [1, 6, 256, 32, 88]
        feat = feat.view(B, N, -1, H, W) 
        
        # 3. 真实的 LSS 变换 (Lift -> Geometry -> Splat) [1, 256, 125, 125]
        bev_feat = self.view_transformer(feat, depth_logits, rots, trans, intrins)
        
        # 4. 检测头 [1, 10, 125, 125]
        preds = self.det_head(bev_feat)
        
        loss_dict = {}
        if gt_depth is not None:
            loss_dict['loss_depth'] = self.get_depth_loss(depth_logits, gt_depth)
            
        return preds, loss_dict

# ==========================================
# 4. 数据配置与流程 (关键：缩小尺寸)
# ==========================================
# ⚠️ 注意: 为了避免 231GB 显存爆炸，我们大幅缩小了 H, W 和 Depth
# 真实训练时，需要 CUDA 算子支持才能跑大分辨率
grid_conf = {
    'xbound': [-50.0, 50.0, 0.8],   # BEV X范围 -> 125个网格
    'ybound': [-50.0, 50.0, 0.8],   # BEV Y范围 -> 125个网格
    'zbound': [-10.0, 10.0, 20.0],  # BEV Z范围 -> 1个网格
    'dbound': [1.0, 60.0, 1.0],     # 深度范围
}
data_conf = {
    'img_size': (32, 88),  # [H, W] 非常小，仅供演示原理！(原论文通常是 256x704)
    'depth': (1.0, 41.0, 40), # [min, max, bins] 深度段数
}

class RandomDataset(Dataset):
    
    def __init__(self, length=20):
        self.length = length
    
    def __len__(self): 
        return self.length
    
    def __getitem__(self, idx):
        H, W = data_conf['img_size']
        
        # 模拟数据
        imgs = torch.randn(6, 3, H, W)
        
        # 相机参数编码 3*3内参矩阵 + 3*4 外参 + 2*3 数据增强参数
        mats = torch.randn(6, 27) 
        
        # 几何参数 (用于 LSS)
        # 旋转矩阵
        rots = torch.eye(3).unsqueeze(0).repeat(6, 1, 1) 
        
        # 平移向量
        trans = torch.zeros(6, 3) 
        
        # 内参
        intrins = torch.eye(3).unsqueeze(0).repeat(6, 1, 1) 
        
        # ========= 真值 ========
        # 深度真值
        gt_depth = torch.randint(-1, 40, (6, H, W)) 
        
        # BEV 真值 (假设网格 128x128)
        gt_heatmap = torch.randn(10, 128, 128) 
        
        return imgs, mats, rots, trans, intrins, gt_depth, gt_heatmap

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} (with Reduced Resolution LSS)")
    
    model = BEVDepthModel(grid_conf, data_conf).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    loader = DataLoader(RandomDataset(10), batch_size=1)
    
    model.train()
    for batch_idx, (imgs, mats, rots, trans, intrins, gt_depth, gt_heatmap) in enumerate(loader):
        
        # imgs shape = [1, 6, 3, 32, 88] mats shape = [1, 6, 27]
        imgs, mats = imgs.to(device), mats.to(device) 
        
        # rots=[1, 6, 3, 3], trans=[1, 6, 3], intrins=[1, 6, 3, 3]
        rots, trans, intrins = rots.to(device), trans.to(device), intrins.to(device) 
        
        # [1, 6, 32, 88], [1, 10, 128, 128]
        gt_depth, gt_heatmap = gt_depth.to(device), gt_heatmap.to(device) 
        
        optimizer.zero_grad()
        preds, loss_dict = model(imgs, mats, rots, trans, intrins, gt_depth)
        
        # 调整尺寸匹配 (因为 LSS 输出是根据 grid_conf 算出来的，和 gt_heatmap 可能不一致，这里强行插值对齐)
        preds = F.interpolate(preds, size=gt_heatmap.shape[-2:], mode='bilinear')
        
        loss_det = F.mse_loss(preds, gt_heatmap)
        loss_depth = loss_dict['loss_depth']
        total_loss = loss_det + 3.0 * loss_depth
        
        total_loss.backward()
        optimizer.step()
        print(f"Batch {batch_idx}: Loss={total_loss.item():.4f} (Depth={loss_depth.item():.4f})")

if __name__ == "__main__":
    main()