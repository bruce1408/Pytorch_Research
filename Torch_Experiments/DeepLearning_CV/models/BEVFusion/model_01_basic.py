import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
class Config:
    # BEV 空间设置
    bev_h = 128         # BEV 高度 (Y轴格子数)
    bev_w = 128         # BEV 宽度 (X轴格子数)
    
    # 相机分支设置
    num_cams = 6        # 环视相机数量
    cam_c = 64          # 图像特征维度
    cam_d = 40          # LSS 深度估计的 Bin 数量 (Depth Bins)
    cam_bev_c = 80      # Camera BEV 特征的通道数
    
    # 雷达分支设置
    lidar_bev_c = 128   # LiDAR BEV 特征的通道数 (VoxelNet输出)
    
    # 融合设置
    fusion_out_c = 256  # 融合后的特征通道数
    
    # 配置参数
    grid_conf = {
        'xbound': [-51.2, 51.2, 0.8],   # X 范围和分辨率
        'ybound': [-51.2, 51.2, 0.8],   # Y 范围和分辨率
        'zbound': [-10.0, 10.0, 20.0],  # Z 范围 (这里只有一层)
        'dbound': [4.0, 45.0, 1.0],     # 深度范围 4m~45m
    }
    
    input_size = [128, 352] # [H, W]

# ==========================================
# 2. Camera Branch (相机分支: LSS 逻辑)
# ==========================================
class LSSViewTransformer(nn.Module):
    def __init__(self, cfg, downsample=16, cam_c=64, bev_c=64):
        super().__init__()
        
        self.grid_conf = cfg.grid_conf
        self.downsample = downsample
        self.input_size = cfg.input_size
        self.out_C = bev_c

        # 解析 Grid 配置
        self.dx, self.bx, self.nx = self.gen_dx_bx(
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']
        )
        
        # 解析深度配置 (Depth Bins)
        d_min, d_max, d_step = self.grid_conf['dbound']
        self.D = int((d_max - d_min) / d_step) 
        
        # 1. 准备 Frustum (视锥网格)
        self.frustum = self.create_frustum()

        # 2. DepthNet
        # 输入: Image Feature (C), 输出: Depth Distribution (D) + Context (C_out)
        self.depth_net = nn.Conv2d(cam_c, self.D + self.out_C, kernel_size=1, padding=0)

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx

    def create_frustum(self):
        img_H, img_W = self.input_size
        feat_H, feat_W = img_H // self.downsample, img_W // self.downsample
        
        d_min, d_max, d_step = self.grid_conf['dbound']
        ds = torch.arange(d_min, d_max, d_step).view(-1, 1, 1).expand(-1, feat_H, feat_W)
        
        xs = torch.linspace(0, img_W - 1, feat_W).view(1, 1, feat_W).expand(self.D, feat_H, -1)
        ys = torch.linspace(0, img_H - 1, feat_H).view(1, feat_H, 1).expand(self.D, -1, feat_W)

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        B, N, _ = trans.shape
        D, H, W, _ = self.frustum.shape
        
        # (D, H, W, 3) -> (B, N, D, H, W, 3)
        points = self.frustum.view(1, 1, D, H, W, 3).repeat(B, N, 1, 1, 1, 1)
        
        # Undo Post-transformation (Data Augmentation)
        points = points - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # Image -> Camera
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], 
                            points[:, :, :, :, :, 2:3]), 5)
        combined_transform = rots.matmul(torch.inverse(intrins))
        points = combined_transform.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        
        # Camera -> Ego
        points += trans.view(B, N, 1, 1, 1, 3)
        
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W 

        x = x.reshape(Nprime, C)
        geom_feats = geom_feats.reshape(Nprime, 3)

        dx, bx, nx = self.dx.to(x.device), self.bx.to(x.device), self.nx.to(x.device)
        
        # 【关键修改 1】：提取整数维度的 BEV 尺寸，供后续 zeros 初始化使用
        nx_int = int(nx[0].item())
        ny_int = int(nx[1].item())
        nz_int = int(nx[2].item())

        geom_feats = ((geom_feats - (bx - dx/2.)) / dx).long()        
        # 使用 tensor 的 nx 进行掩码计算是没问题的
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) & \
               (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) & \
               (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        batch_ix = torch.cat([torch.full([N*D*H*W], i, device=x.device, dtype=torch.long) for i in range(B)])
        batch_ix = batch_ix[kept]
        
        # 计算索引
        ranks = (batch_ix * (nx[0] * nx[1] * nx[2]) + 
                 geom_feats[:, 0] + 
                 geom_feats[:, 1] * nx[0] + 
                 geom_feats[:, 2] * nx[0] * nx[1]).long()
        
        sorts = ranks.argsort()
        x, ranks = x[sorts], ranks[sorts]
        
        # 【关键修改 2】：这里也要用整数
        if len(ranks) == 0:
            return torch.zeros((B, self.out_C, ny_int, nx_int), device=x.device)

        kept_mask = torch.ones(ranks.shape[0], device=x.device, dtype=torch.bool)
        kept_mask[:-1] = (ranks[1:] != ranks[:-1])
        
        cumsum = torch.cumsum(x, dim=0)
        cumsum = torch.cat([torch.zeros((1, C), device=x.device), cumsum], dim=0)
        
        voxel_feats = cumsum[1:][kept_mask] - cumsum[:-1][kept_mask]
        voxel_ranks = ranks[kept_mask]
        
        # 【关键修改 3】：torch.zeros 的形状参数必须是整数
        # 之前报错就是因为 nx[i] 是 Tensor，导致 B * nx[...] 也是 Tensor
        flat_size = B * nz_int * ny_int * nx_int
        final_bev = torch.zeros((flat_size, C), device=x.device)
        
        final_bev[voxel_ranks] = voxel_feats
        
        # 【关键修改 4】：view 里面也建议用整数
        final_bev = final_bev.view(B, nz_int, ny_int, nx_int, C)
        final_bev = final_bev.permute(0, 4, 1, 2, 3).contiguous() 
        final_bev = final_bev.sum(2) # Collapse Z
        
        return final_bev

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        B, N, C_in, H, W = x.shape
        
        x = x.view(B * N, C_in, H, W)
        x = self.depth_net(x) 
        
        depth_digit = x[:, :self.D].softmax(dim=1)
        context = x[:, self.D:] 
        
        outer = context.unsqueeze(2) * depth_digit.unsqueeze(1) 
        outer = outer.view(B, N, self.out_C, self.D, H, W).permute(0, 1, 3, 4, 5, 2)
        
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        bev_feat = self.voxel_pooling(geom, outer)
        return bev_feat

class CameraBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # LSS 默认 downsample=16。输入 128x352 -> 输出应该为 8x22
        self.backbone = nn.Sequential(
            # 使用 kernel=16, stride=16 模拟 ResNet 的下采样效果
            nn.Conv2d(3, cfg.cam_c, kernel_size=16, stride=16, padding=0),
            nn.BatchNorm2d(cfg.cam_c),
            nn.ReLU()
        )
        
        self.vtransform = LSSViewTransformer(cfg, cam_c=cfg.cam_c, bev_c=cfg.cam_bev_c)

    def forward(self, imgs, rots, trans, intrins):
        # imgs: (B, N, 3, H, W)
        B, N, C, H, W = imgs.shape
        
        # 1. Backbone Extraction
        imgs = imgs.view(B * N, C, H, W)
        feat = self.backbone(imgs) # (B*N, C, H/16, W/16) -> (B*N, 64, 8, 22)
        
        # Reshape back
        feat = feat.view(B, N, self.cfg.cam_c, feat.shape[2], feat.shape[3])
        
        # 2. View Transform (2D -> 3D BEV)
        # 【修复点 2】: 确保 Post-Rot/Trans 与输入在同一个 device
        device = imgs.device
        post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
        post_trans = torch.zeros(B, N, 3, device=device)
        
        cam_bev = self.vtransform(feat, rots, trans, intrins, post_rots, post_trans)
        return cam_bev

# ==========================================
# 3. LiDAR Branch (雷达分支)
# ==========================================
class LiDARBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = nn.Sequential(
            nn.Conv2d(32, cfg.lidar_bev_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.lidar_bev_c),
            nn.ReLU()
        )

    def forward(self, points):
        B = points.shape[0]
        
        # Mock Voxel Features (B, 32, 128, 128)
        mock_voxel_feat = torch.randn(B, 
                                      32, 
                                      self.cfg.bev_h, 
                                      self.cfg.bev_w, device=points.device)
        
        lidar_bev = self.backbone(mock_voxel_feat)
        return lidar_bev

# ==========================================
# 4. Fusion Module (核心融合模块)
# ==========================================
class ConvFuser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.cam_bev_c + cfg.lidar_bev_c
        out_channels = cfg.fusion_out_c
        
        self.fuser = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, cam_bev, lidar_bev):
        fusion_feat = torch.cat([cam_bev, lidar_bev], dim=1) # [1, 208, 128, 128]
        output = self.fuser(fusion_feat)
        return output

# ==========================================
# 5. BEVFusion Main Model (主模型)
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
        # 1. Camera BEV
        cam_bev = self.camera_branch(imgs, rots, trans, intrins)
        
        # 2. LiDAR BEV
        lidar_bev = self.lidar_branch(points)
        
        # 3. Fusion
        fused_bev = self.fuser(cam_bev, lidar_bev)
        
        # 4. Head
        out = self.head(fused_bev)
        return out, fused_bev

# ==========================================
# 6. Pipeline Simulation
# ==========================================
def main():
    cfg = Config()
    model = BEVFusion(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("=== BEVFusion Architecture Initialized ===")
    print(f"Lidar BEV Channel: {cfg.lidar_bev_c}")
    print(f"Camera BEV Channel: {cfg.cam_bev_c}")
    print(f"Fusion Output Channel: {cfg.fusion_out_c}")
    
    # --- Mock Data ---
    B = 1
    # 输入尺寸 [B, N, C, 128, 352]
    imgs = torch.randn(B, 6, 3, 128, 352).to(device)
    rots = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    trans = torch.zeros(B, 6, 3).to(device)
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    
    # 模拟激光雷达点云数据，10000个点，x,y,z 和 i 反射强度
    points = torch.randn(B, 10000, 4).to(device)
    
    # --- Forward ---
    print("\n--- Starting Inference ---")
    output, fused_bev = model(imgs, points, rots, trans, intrins)
    
    print(f"Camera BEV Shape: (Hidden inside) [1, 80, 128, 128]")
    print(f"LiDAR BEV Shape:  (Hidden inside) [1, 128, 128, 128]")
    print(f"Fused BEV Shape:  {fused_bev.shape}") 
    print(f"Detection Output: {output.shape}")    
    
    print("\n✅ BEVFusion Pipeline Test Passed!")

if __name__ == "__main__":
    main()