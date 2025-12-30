import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
class Config:
    # --- 统一的 BEV 坐标系定义 ---
    # 这是 Fusion 的基石：Camera 和 LiDAR 必须映射到同一个物理网格上
    xbound = [-51.2, 51.2, 0.8]  # [min, max, resolution] -> 128 格
    ybound = [-51.2, 51.2, 0.8]  # [min, max, resolution] -> 128 格
    zbound = [-10.0, 10.0, 20.0] # Z轴通常在 BEV 中被压缩
    dbound = [1.0, 60.0, 1.0]    # 相机深度的估计范围
    
    # 得到的 BEV 尺寸
    bev_h = int((ybound[1] - ybound[0]) / ybound[2]) # 128
    bev_w = int((xbound[1] - xbound[0]) / xbound[2]) # 128
    
    # 通道配置
    cam_c = 64          # 图像 Backbone 输出通道
    cam_bev_c = 80      # Camera BEV 特征通道
    lidar_in_c = 4      # 点云输入通道 (x,y,z,intensity)
    lidar_bev_c = 32    # LiDAR BEV 特征通道 (简化版 PointPillars)
    fusion_out_c = 128  # 融合后通道
    
    # 输入尺寸
    input_size = [256, 704] # [H, W]

# ==========================================
# 2. Camera Branch (LSS 实现)
# ==========================================
class LSSViewTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.downsample = 16 # Backbone 下采样倍率
        self.out_C = cfg.cam_bev_c

        # 解析 Grid 参数
        self.dx, self.bx, self.nx = self.gen_dx_bx(
            cfg.xbound, cfg.ybound, cfg.zbound
        )
        
        # 解析 Depth 参数
        d_min, d_max, d_step = cfg.dbound
        self.D = int((d_max - d_min) / d_step) 
        
        # 1. 预计算视锥 (Frustum)
        self.frustum = self.create_frustum()

        # 2. DepthNet (Lift 核心)
        # 输出: 深度分布(D) + 语义特征(out_C)
        self.depth_net = nn.Conv2d(cfg.cam_c, self.D + self.out_C, kernel_size=1, padding=0)

    def gen_dx_bx(self, xbound, ybound, zbound):
        # 计算网格分辨率(dx), 起点(bx), 尺寸(nx)
        dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx

    def create_frustum(self):
        # 生成静态的 (u, v, d) 视锥网格
        img_H, img_W = self.cfg.input_size
        feat_H, feat_W = img_H // self.downsample, img_W // self.downsample
        
        d_min, d_max, d_step = self.cfg.dbound
        ds = torch.arange(d_min, d_max, d_step).view(-1, 1, 1).expand(-1, feat_H, feat_W)
        
        xs = torch.linspace(0, img_W - 1, feat_W).view(1, 1, feat_W).expand(self.D, feat_H, -1)
        ys = torch.linspace(0, img_H - 1, feat_H).view(1, feat_H, 1).expand(self.D, -1, feat_W)

        # (D, H, W, 3)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins):
        # 几何投影: Pixel (u,v,d) -> World (x,y,z)
        B, N, _ = trans.shape
        D, H, W, _ = self.frustum.shape
        
        # (1, 1, D, H, W, 3) -> (B, N, D, H, W, 3)
        points = self.frustum.view(1, 1, D, H, W, 3).repeat(B, N, 1, 1, 1, 1)
        
        # Pixel -> Camera
        # 利用相似三角形原理 x = u * z / f
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], 
                            points[:, :, :, :, :, 2:3]), 5)
        
        combined_transform = rots.matmul(torch.inverse(intrins))
        points = combined_transform.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        # Camera -> Ego (World)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        return points

    def voxel_pooling(self, geom_feats, x):
        # LSS 的核心: 将离散点聚合到 BEV 网格
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W 

        # Flatten
        x = x.reshape(Nprime, C)
        geom_feats = geom_feats.reshape(Nprime, 3)

        # 转换到网格坐标
        dx, bx, nx = self.dx.to(x.device), self.bx.to(x.device), self.nx.to(x.device)
        
        # 提取整数维度 (修复之前的报错)
        nx_int, ny_int, nz_int = nx[0].long().item(), nx[1].long().item(), nx[2].long().item()

        # 量化坐标 (Physical -> Grid Index)
        geom_feats = ((geom_feats - (bx - dx/2.)) / dx).long()
        
        # 过滤出界点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) & \
               (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) & \
               (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # 计算 Rank 用于排序
        # Rank = Batch_ID * Size + Z * Size + Y * Size + X
        batch_ix = torch.cat([torch.full([N*D*H*W], i, device=x.device, dtype=torch.long) for i in range(B)])
        batch_ix = batch_ix[kept]
        
        ranks = (batch_ix * (nx[0] * nx[1] * nx[2]) + 
                 geom_feats[:, 0] + 
                 geom_feats[:, 1] * nx[0] + 
                 geom_feats[:, 2] * nx[0] * nx[1]).long()
        
        # 排序
        sorts = ranks.argsort()
        x, ranks = x[sorts], ranks[sorts]
        
        # CumSum Trick (并行求和)
        if len(ranks) == 0:
            return torch.zeros((B, self.out_C, ny_int, nx_int), device=x.device)

        kept_mask = torch.ones(ranks.shape[0], device=x.device, dtype=torch.bool)
        kept_mask[:-1] = (ranks[1:] != ranks[:-1])
        
        cumsum = torch.cumsum(x, dim=0)
        cumsum = torch.cat([torch.zeros((1, C), device=x.device), cumsum], dim=0)
        
        voxel_feats = cumsum[1:][kept_mask] - cumsum[:-1][kept_mask]
        voxel_ranks = ranks[kept_mask]
        
        # 填回 BEV
        flat_size = B * nz_int * ny_int * nx_int
        final_bev = torch.zeros((flat_size, C), device=x.device)
        final_bev[voxel_ranks] = voxel_feats
        
        # Reshape & Collapse Z
        final_bev = final_bev.view(B, nz_int, ny_int, nx_int, C)
        final_bev = final_bev.permute(0, 4, 1, 2, 3).contiguous() 
        final_bev = final_bev.sum(2) # Sum pooling along Z
        
        return final_bev

    def forward(self, x, rots, trans, intrins):
        B, N, C_in, H, W = x.shape
        x = x.view(B * N, C_in, H, W)
        x = self.depth_net(x) 
        
        # Lift
        depth_digit = x[:, :self.D].softmax(dim=1)
        context = x[:, self.D:] 
        outer = context.unsqueeze(2) * depth_digit.unsqueeze(1) 
        outer = outer.view(B, N, self.out_C, self.D, H, W).permute(0, 1, 3, 4, 5, 2)
        
        # Splat
        geom = self.get_geometry(rots, trans, intrins)
        bev_feat = self.voxel_pooling(geom, outer)
        return bev_feat

class CameraBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Mock Backbone (stride=16)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, cfg.cam_c, kernel_size=3, stride=16, padding=1), # 粗暴下采样模拟
            nn.BatchNorm2d(cfg.cam_c),
            nn.ReLU()
        )
        self.vtransform = LSSViewTransformer(cfg)

    def forward(self, imgs, rots, trans, intrins):
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        feat = self.backbone(imgs) 
        feat = feat.view(B, N, -1, feat.shape[2], feat.shape[3])
        return self.vtransform(feat, rots, trans, intrins)

# ==========================================
# 3. LiDAR Branch (还原算法细节: Point-to-Grid)
# ==========================================
class SimplePointPillars(nn.Module):
    """
    一个极其简化的 PointPillars 实现，用于展示 LiDAR 如何转换到和 Camera 同样的 BEV 空间。
    算法步骤：
    1. 计算每个点的网格索引 (Grid Index)
    2. 过滤出界点
    3. Scatter Mean (将落入同一个格子的点取平均) - 类似 Voxel Pooling
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 解析网格参数 (复用 Camera 的配置以保证对齐)
        self.dx = torch.tensor([row[2] for row in [cfg.xbound, cfg.ybound, cfg.zbound]])
        self.bx = torch.tensor([row[0] + row[2]/2.0 for row in [cfg.xbound, cfg.ybound, cfg.zbound]])
        self.nx = torch.tensor([(row[1] - row[0]) / row[2] for row in [cfg.xbound, cfg.ybound, cfg.zbound]])
        
        # 一个简单的 MLP 处理每个点 (模拟 Voxel Feature Encoding)
        self.pfn = nn.Sequential(
            nn.Linear(4, cfg.lidar_bev_c),
            nn.BatchNorm1d(cfg.lidar_bev_c),
            nn.ReLU()
        )

    def forward(self, points):
        """
        points: (B, N_points, 4)
        """
        B, N, C = points.shape
        device = points.device
        
        # 1. 扁平化处理
        points_flat = points.reshape(-1, C) # (B*N, 4)
        
        # 2. VFE (简单的特征提取)
        feat = self.pfn(points_flat) # (B*N, C_lidar)
        
        # 3. 计算坐标索引 (Quantization)
        # 这一步的逻辑和 Camera 的 voxel_pooling 是一模一样的！
        # 都是为了对齐到同一个 BEV Grid。
        dx, bx, nx = self.dx.to(device), self.bx.to(device), self.nx.to(device)
        
        coords = ((points_flat[:, :3] - (bx - dx/2.)) / dx).long()
        
        # 4. 过滤出界点
        kept = (coords[:, 0] >= 0) & (coords[:, 0] < nx[0]) & \
               (coords[:, 1] >= 0) & (coords[:, 1] < nx[1]) & \
               (coords[:, 2] >= 0) & (coords[:, 2] < nx[2])
        
        feat = feat[kept]
        coords = coords[kept]
        
        # 5. Scatter to BEV (类似于 Camera 的 Pooling)
        # 计算 Rank
        batch_ix = torch.cat([torch.full([N], i, device=device, dtype=torch.long) for i in range(B)])
        batch_ix = batch_ix.reshape(-1)[kept] # 对应 kept 后的点
        
        # Rank 计算公式必须和 Camera 完全一致
        ranks = (batch_ix * (nx[0] * nx[1] * nx[2]) + 
                 coords[:, 0] + 
                 coords[:, 1] * nx[0] + 
                 coords[:, 2] * nx[0] * nx[1]).long()
        
        # 排序
        sorts = ranks.argsort()
        feat, ranks = feat[sorts], ranks[sorts]
        
        # CumSum 聚合 (Scatter Mean)
        # 这里为了简单，直接用 CumSum 模拟 Mean/Sum Pooling
        if len(ranks) == 0:
            return torch.zeros((B, self.cfg.lidar_bev_c, int(nx[1]), int(nx[0])), device=device)

        kept_mask = torch.ones(ranks.shape[0], device=device, dtype=torch.bool)
        kept_mask[:-1] = (ranks[1:] != ranks[:-1])
        
        cumsum = torch.cumsum(feat, dim=0)
        cumsum = torch.cat([torch.zeros((1, self.cfg.lidar_bev_c), device=device), cumsum], dim=0)
        
        voxel_feats = cumsum[1:][kept_mask] - cumsum[:-1][kept_mask]
        voxel_ranks = ranks[kept_mask]
        
        # 填入 BEV Canvas
        flat_size = B * int(nx[2]) * int(nx[1]) * int(nx[0])
        final_bev = torch.zeros((flat_size, self.cfg.lidar_bev_c), device=device)
        final_bev[voxel_ranks] = voxel_feats
        
        final_bev = final_bev.view(B, int(nx[2]), int(nx[1]), int(nx[0]), -1)
        final_bev = final_bev.permute(0, 4, 1, 2, 3).contiguous()
        
        # 压缩 Z 轴 (Sum)
        final_bev = final_bev.sum(2) # (B, C, H, W)
        
        return final_bev

class LiDARBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 1. Point -> Grid (PointPillars Style)
        self.pfn_scatter = SimplePointPillars(cfg)
        
        # 2. BEV Backbone (处理稀疏特征，模拟 SECOND/CenterPoint 的 Backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(cfg.lidar_bev_c, cfg.lidar_bev_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.lidar_bev_c),
            nn.ReLU()
        )

    def forward(self, points):
        # points -> bev_map
        lidar_bev = self.pfn_scatter(points)
        # bev_map -> refined_bev
        lidar_bev = self.backbone(lidar_bev)
        return lidar_bev

# ==========================================
# 4. Fusion Module (详细融合逻辑)
# ==========================================
class ConvFuser(nn.Module):
    """
    BEVFusion 的融合非常直接：在通道维度拼接，然后用卷积混匀。
    关键在于：前置的 Camera 和 LiDAR 分支必须输出完全一致的空间尺寸 (H, W)。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 输入通道 = Cam + Lidar
        in_channels = cfg.cam_bev_c + cfg.lidar_bev_c
        out_channels = cfg.fusion_out_c
        
        # 融合网络
        self.fuser = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # 这是一个典型的 ResNet Block 风格的融合层
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, cam_bev, lidar_bev):
        """
        cam_bev: (B, 80, 128, 128)
        lidar_bev: (B, 32, 128, 128)
        """
        # 1. 拼接 (Concatenation)
        # 这是多模态融合最核心的一步
        # 由于我们严格控制了 LSS 和 PointPillars 的 Grid 参数 (dx, bx, nx)
        # 这里的 H 和 W 是严格对齐的，不需要 Resize 或 Crop
        fusion_feat = torch.cat([cam_bev, lidar_bev], dim=1) # (B, 112, 128, 128)
        
        # 2. 卷积融合
        output = self.fuser(fusion_feat) # (B, 128, 128, 128)
        return output

# ==========================================
# 5. BEVFusion 主模型
# ==========================================
class BEVFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.camera_branch = CameraBranch(cfg)
        self.lidar_branch = LiDARBranch(cfg)
        self.fuser = ConvFuser(cfg)
        self.head = nn.Conv2d(cfg.fusion_out_c, 10, kernel_size=1) 

    def forward(self, imgs, points, rots, trans, intrins):
        # 1. 视觉分支: Image -> Feature -> Lift -> Splat -> BEV
        # 输出: (B, 80, 128, 128)
        cam_bev = self.camera_branch(imgs, rots, trans, intrins)
        
        # 2. 雷达分支: Points -> Quantize -> Scatter -> BEV
        # 输出: (B, 32, 128, 128)
        lidar_bev = self.lidar_branch(points)
        
        # 3. 融合: Concat + Conv
        # 输出: (B, 128, 128, 128)
        fused_bev = self.fuser(cam_bev, lidar_bev)
        
        # 4. 检测头
        out = self.head(fused_bev)
        return out, fused_bev

# ==========================================
# 6. 运行验证
# ==========================================
def main():
    cfg = Config()
    model = BEVFusion(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"=== BEVFusion Config ===")
    print(f"Grid Size: {cfg.bev_h} x {cfg.bev_w}")
    print(f"Camera BEV Channels: {cfg.cam_bev_c}")
    print(f"LiDAR BEV Channels: {cfg.lidar_bev_c}")
    
    # --- Mock Data ---
    B = 1
    # 模拟 6 张图片 [B, 6, 3, 256, 704]
    imgs = torch.randn(B, 6, 3, 256, 704).to(device)
    # 模拟相机参数
    rots = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    trans = torch.zeros(B, 6, 3).to(device)
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    intrins[:, :, 0, 0] = 500.0 # 简单的焦距模拟
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0
    
    # 模拟点云: 随机生成一些在有效范围内的点
    # 生成 10000 个点，范围在 -50~50 之间
    points = (torch.rand(B, 10000, 4) * 100 - 50).to(device)
    points[:, :, 2] = torch.rand(B, 10000).to(device) * 4 - 2 # Z轴 -2~2
    
    print("\n--- Starting Inference ---")
    output, fused_bev = model(imgs, points, rots, trans, intrins)
    
    print(f"✅ Successful!")
    print(f"Fused BEV Shape:  {fused_bev.shape} (Expected: [1, 128, 128, 128])") 
    print(f"Detection Output: {output.shape} (Expected: [1, 10, 128, 128])")

if __name__ == "__main__":
    main()