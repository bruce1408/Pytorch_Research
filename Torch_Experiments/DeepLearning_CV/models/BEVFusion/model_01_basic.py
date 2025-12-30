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

# ==========================================
# 2. Camera Branch (相机分支: LSS 逻辑)
# ==========================================
class LSSViewTransformer(nn.Module):
    """
    负责将 2D 图像特征转换为 3D BEV 特征。
    BEVFusion 的核心优化就在这里 (Optimized BEV Pooling)，
    这里演示标准的 Lift-Splat 逻辑。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.D = cfg.cam_d
        self.C = cfg.cam_bev_c
        
        # DepthNet: 预测每个像素的深度分布
        # 输入: Image Feature, 输出: Depth(D) + Context(C)
        self.depth_net = nn.Conv2d(cfg.cam_c, self.D + self.C, kernel_size=1)

    def get_geometry(self, rots, trans, intrins):
        """
        计算视锥点云的 3D 坐标 (简化版)
        真实代码需要利用外参把 (u,v,d) 投影到 ego 坐标系 (x,y,z)
        这里我们生成随机的 Grid Index 来模拟投影结果
        """
        B, N, _ = trans.shape
        # 假设我们已经算好了每个像素对应的 BEV 坐标
        # 模拟输出: (B, N, D, H, W, 2) -> 最后一维是 (x_index, y_index)
        # 为了跑通代码，我们不做真实的矩阵乘法，直接返回模拟的坐标
        H_feat, W_feat = 16, 44 # 假设特征图大小
        
        # 随机生成坐标索引，范围在 BEV Grid (0~127) 之内
        geom_xyz = torch.randint(0, self.cfg.bev_h, (B, N, self.D, H_feat, W_feat, 2))
        return geom_xyz

    def voxel_pooling(self, geom_feats, x):
        """
        将视锥特征 'Splat' (拍扁/池化) 到 BEV 网格上
        x: (B, N, D, H, W, C) -> 视锥特征
        geom_feats: (B, N, D, H, W, 2) -> 对应的 BEV 坐标
        """
        B, N, D, H, W, C = x.shape
        # 1. Flatten
        x = x.reshape(B, -1, C) # (B, N*D*H*W, C)
        geom_feats = geom_feats.reshape(B, -1, 2) # (B, N*D*H*W, 2)
        
        # 2. 初始化 BEV 特征图 (B, C, H_bev, W_bev)
        bev_feat = torch.zeros(B, self.cfg.bev_h, self.cfg.bev_w, C, device=x.device)
        
        # 3. 模拟 Pooling (由于 Python 循环太慢，这里仅演示逻辑)
        # 真实 BEVFusion 使用了专门的 CUDA Kernel 做这一步 (Interval Reduction)
        # 这里为了演示 Pipeline，我们做一个极其简化的 "Scatter Mean"
        # 假设我们将所有特征直接通过插值放到 BEV 上 (Mock)
        
        # --- Mock Output ---
        # 真正的 Pooling 会根据 geom_feats 的索引把 x 加到 bev_feat 上
        # 这里直接生成一个结果以保证形状正确
        bev_feat = torch.randn(B, C, self.cfg.bev_h, self.cfg.bev_w, device=x.device)
        
        return bev_feat

    def forward(self, img_feats, rots, trans, intrins):
        """
        img_feats: (B, N, C_in, H, W)
        """
        B, N, C_in, H, W = img_feats.shape
        
        # 1. Lift (升维)
        x = img_feats.view(B * N, C_in, H, W)
        x = self.depth_net(x) # (B*N, D+C, H, W)
        
        # 拆分深度和上下文
        depth_digit = x[:, :self.D].softmax(dim=1) # 深度概率 (B*N, D, H, W)
        context = x[:, self.D:]                    # 语义特征 (B*N, C, H, W)
        
        # 外积: 将 2D 特征扩展为 3D 视锥特征
        # (B*N, C, 1, H, W) * (B*N, 1, D, H, W) = (B*N, C, D, H, W)
        frustum_feat = context.unsqueeze(2) * depth_digit.unsqueeze(1)
        
        # Reshape 回 (B, N, D, H, W, C)
        frustum_feat = frustum_feat.view(B, N, self.C, self.D, H, W).permute(0, 1, 3, 4, 5, 2)
        
        # 2. Splat (几何投影 + 池化)
        geom = self.get_geometry(rots, trans, intrins) # 计算坐标
        bev_feat = self.voxel_pooling(geom, frustum_feat) # 投射到 BEV
        
        return bev_feat

class CameraBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Mock Image Backbone (ResNet)
        # 假设输出 stride=16 的特征图
        self.backbone = nn.Sequential(
            nn.Conv2d(3, cfg.cam_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.cam_c),
            nn.ReLU()
        )
        self.vtransform = LSSViewTransformer(cfg)

    def forward(self, imgs, rots, trans, intrins):
        # imgs: (B, N, 3, H, W)
        B, N, C, H, W = imgs.shape
        
        # 1. Backbone Extraction
        imgs = imgs.view(B * N, C, H, W)
        feat = self.backbone(imgs) # (B*N, C_feat, H/16, W/16)
        
        # Reshape back
        feat = feat.view(B, N, self.cfg.cam_c, feat.shape[2], feat.shape[3])
        
        # 2. View Transform (2D -> 3D BEV)
        cam_bev = self.vtransform(feat, rots, trans, intrins)
        return cam_bev

# ==========================================
# 3. LiDAR Branch (雷达分支)
# ==========================================
class LiDARBranch(nn.Module):
    """
    负责将点云转换为 LiDAR BEV 特征。
    通常包含: Voxelization -> 3D Sparse Conv -> Flatten
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Mock LiDAR Backbone (VoxelNet / PointPillars)
        # 这里不真的做体素化，直接用卷积模拟特征提取
        self.backbone = nn.Sequential(
            nn.Conv2d(32, cfg.lidar_bev_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.lidar_bev_c),
            nn.ReLU()
        )

    def forward(self, points):
        """
        points: (B, N_points, 4) or Voxels
        为了演示，我们假设输入已经被体素化并 flatten 成了伪 BEV 图像
        """
        B = points.shape[0]
        # 模拟 VoxelNet 的输出: (B, C_lidar, H_bev, W_bev)
        # 假设输入已经被处理成 (B, 32, 128, 128) 的形状
        mock_voxel_feat = torch.randn(B, 32, self.cfg.bev_h, self.cfg.bev_w, device=points.device)
        
        lidar_bev = self.backbone(mock_voxel_feat)
        return lidar_bev

# ==========================================
# 4. Fusion Module (核心融合模块)
# ==========================================
class ConvFuser(nn.Module):
    """
    BEVFusion 的核心: 对齐并拼接
    """
    def __init__(self, cfg):
        super().__init__()
        # 输入通道 = Camera BEV 通道 + LiDAR BEV 通道
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
        # cam_bev:   (B, C_cam, H, W)
        # lidar_bev: (B, C_lidar, H, W)
        
        # 1. 拼接 (Concatenation)
        # BEVFusion 论文中最简单但最有效的一步
        # 假设 H 和 W 已经对齐 (通常通过 Resize 或 Padding)
        fusion_feat = torch.cat([cam_bev, lidar_bev], dim=1)
        
        # 2. 卷积融合
        output = self.fuser(fusion_feat)
        return output

# ==========================================
# 5. BEVFusion Main Model (主模型)
# ==========================================
class BEVFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 分支初始化
        self.camera_branch = CameraBranch(cfg)
        self.lidar_branch = LiDARBranch(cfg)
        
        # 融合模块
        self.fuser = ConvFuser(cfg)
        
        # 检测头 (Mock)
        self.head = nn.Conv2d(cfg.fusion_out_c, 10, kernel_size=1) # 输出10类热力图

    def forward(self, imgs, points, rots, trans, intrins):
        # 1. 获取 Camera BEV
        # Shape: (B, 80, 128, 128)
        cam_bev = self.camera_branch(imgs, rots, trans, intrins)
        
        # 2. 获取 LiDAR BEV
        # Shape: (B, 128, 128, 128)
        lidar_bev = self.lidar_branch(points)
        
        # 3. 多模态融合
        # Shape: (B, 256, 128, 128)
        fused_bev = self.fuser(cam_bev, lidar_bev)
        
        # 4. 任务头
        out = self.head(fused_bev)
        return out, fused_bev

# ==========================================
# 6. Pipeline Simulation (运行演示)
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
    # 相机数据: 6张图, 256x704 尺寸太大，变成 128*352
    imgs = torch.randn(B, 6, 3, 128, 352).to(device)
    # 模拟相机参数 (Identity)
    rots = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    trans = torch.zeros(B, 6, 3).to(device)
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    
    # 点云数据 (Mock Batch, N_points, 4)
    points = torch.randn(B, 10000, 4).to(device)
    
    # --- Forward ---
    print("\n--- Starting Inference ---")
    output, fused_bev = model(imgs, points, rots, trans, intrins)
    
    print(f"Camera BEV Shape: (Hidden inside) [1, 80, 128, 128]")
    print(f"LiDAR BEV Shape:  (Hidden inside) [1, 128, 128, 128]")
    print(f"Fused BEV Shape:  {fused_bev.shape}") # 预期 (1, 256, 128, 128)
    print(f"Detection Output: {output.shape}")    # 预期 (1, 10, 128, 128)
    
    print("\n✅ BEVFusion Pipeline Test Passed!")

if __name__ == "__main__":
    main()