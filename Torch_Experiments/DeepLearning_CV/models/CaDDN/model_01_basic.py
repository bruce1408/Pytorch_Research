import torch
import torch.nn as nn
import torch.nn.functional as F

class CaDDN_Core(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 feat_channels=64, 
                 depth_bins=80, 
                 image_shape=(384, 1280),
                 voxel_grid_size=(200, 200, 16), # X, Y, Z 的网格数量
                 pc_range=(0, -40, -3, 70.4, 40, 1), # (x_min, y_min, z_min, x_max, y_max, z_max)
                 depth_range=(2.0, 42.0)): # 视锥的深度范围
        super().__init__()
        
        self.feat_channels = feat_channels
        self.depth_bins = depth_bins
        self.image_shape = image_shape
        self.depth_range = depth_range
        self.pc_range = pc_range
        
        # -----------------------------------------------------------
        # 1. Backbone & Heads (模拟)
        # -----------------------------------------------------------
        # 语义分支: 提取图像纹理特征 F
        self.image_head = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU()
        )
        
        # 深度分支: 预测深度分布 logits D
        # 注意输出通道是 depth_bins (分类问题)
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, depth_bins, kernel_size=3, padding=1),
        )
        
        # BEV 压缩层: 把 Z 轴压扁
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(feat_channels * voxel_grid_size[2], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 预先生成体素网格坐标 (X, Y, Z) -> [1, 3, 16, 200, 200]
        self.voxel_grid = self.create_voxel_grid(voxel_grid_size, pc_range)

    def create_voxel_grid(self, grid_size, pc_range):
        """
        这是一个规则的长方体（比如自车前方 80米，左右 40米）。 这个 grid 存储的不是特征，而是“地址”
        因为网格是固定的，不需要 Batch 维度（这只是坐标值）。
        生成目标 3D Voxel 的物理坐标。
        输出 shape: (1, 3, Z, Y, X) -> 3 代表 (x, y, z)
        我理解这个体素网格就是，xyz 表示 体素网格的坐标，在 这个坐标里面，
        存储的是 当前格子的中心，距离我车的中心的距离，比如车头前方 10.5米，车左侧-3.2米，车上方的0.5米
        grid[0, 0, 0, 0] 可能存着 [0.0, -10.0, -1.0]
        车头原点，左边10米，地下1米
        """
        X_n, Y_n, Z_n = grid_size
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        
        # 生成网格中心点坐标
        xs = torch.linspace(x_min, x_max, X_n)
        ys = torch.linspace(y_min, y_max, Y_n)
        zs = torch.linspace(z_min, z_max, Z_n)
        
        # Meshgrid 生成 3D 坐标体
        # 注意: indexing='ij' 对应 Z, Y, X 顺序 -> [16, 200, 200]
        zs, ys, xs = torch.meshgrid(zs, ys, xs, indexing='ij')
        
        # 堆叠: (Z, Y, X, 3) -> (3, Z, Y, X)
        grid = torch.stack([xs, ys, zs], dim=-1).permute(3, 0, 1, 2)
        return grid.unsqueeze(0) # (1, 3, Z, Y, X)

    def frustum_to_voxel_sampling(self, frustum_feat, grid, calib_mats):
        """
        frustum_feat 上下文*深度 = 特征
        grid 体素空间的网格
        calib_mats = 相机参数矩阵
        [核心步骤 2] 视锥 -> 体素变换
        原理: 逆向采样。
        1. 拿到 Voxel 的 (x,y,z) 物理坐标。
        2. 投影回相机平面 -> 得到 (u, v) 和 深度 d。
        3. 用 (u, v, d) 去 Frustum Feature 里采样。
        """
        B = frustum_feat.shape[0] # [2, 64, 80, 96, 320]
        device = frustum_feat.device
        
        # 1. 准备 Voxel 坐标 (B, 3, Z, Y, X) -> Flatten -> (B, 3, N_voxels)
        voxels = grid.to(device).repeat(B, 1, 1, 1, 1)
        B, _, Z, Y, X = voxels.shape
        voxels_flat = voxels.view(B, 3, -1)
        
        # 2. 投影到图像平面 (Project to Image)
        # 这里简化假设 calib_mats 是 lidar2img 矩阵 (4x4)
        # 变为齐次坐标 (4, N) -> [2, 1, 640000]
        ones = torch.ones((B, 1, voxels_flat.shape[-1]), device=device)
        
        # x,y,z 变成 x,y,z,1
        voxels_homo = torch.cat([voxels_flat, ones], dim=1) # (B, 4, N)
        
        # 矩阵乘法: P_img = K * T * P_world，把世界坐标系 转换为 像素坐标系，这个结果是 [u*z, v*z, z, 1]
        img_points = torch.bmm(calib_mats, voxels_homo) # (B, 4, N) -> [2, 4, 640000]
        
        # 3. 归一化得到 (u, v) 和 深度 depth
        eps = 1e-5
        depth = img_points[:, 2:3, :] # z 也就是深度
        u = img_points[:, 0:1, :] / (depth + eps)
        v = img_points[:, 1:2, :] / (depth + eps)
        
        # 4. 坐标归一化 (Normalize to [-1, 1] for grid_sample)
        # u: [0, W] -> [-1, 1]
        # v: [0, H] -> [-1, 1]
        # d: [min_d, max_d] -> [-1, 1] (这是关键！我们要去深度桶里采样)
        
        img_H, img_W = self.image_shape
        d_min, d_max = self.depth_range
        
        u_norm = (u / (img_W - 1) * 2) - 1
        v_norm = (v / (img_H - 1) * 2) - 1
        
        # 深度归一化稍微复杂一点，因为深度是分桶的 (Log space or Linear space)
        # 这里假设是线性分桶 (Linear Input Depth -> Normalized Grid Index)
        d_norm = (depth - d_min) / (d_max - d_min) # 0~1
        d_norm = d_norm * 2 - 1 # -1~1
        
        # 拼接采样网格: (B, N, 3) -> (u, v, d)
        # grid_sample 需要的顺序是 (x, y, z) -> 对应这里的 (u, v, d)
        sample_grid = torch.cat([u_norm, v_norm, d_norm], dim=1).permute(0, 2, 1) # (B, N, 3)
        
        # Reshape 回 3D 网格形状以便采样
        sample_grid = sample_grid.view(B, Z, Y, X, 3)
        
        # 5. 三线性插值 (Trilinear Interpolation)
        # input: (B, C, D_bins, H_feat, W_feat) -> [2, 64, 80, 96, 320]
        # grid:  (B, Z_out, Y_out, X_out, 3)    -> [2, 16, 200, 200, 3]
        voxel_features = F.grid_sample(
            frustum_feat, 
            sample_grid, 
            mode='bilinear', # 3D采样通常叫 trilinear，但在PyTorch里5D输入+3D网格用mode='bilinear'指代线性插值
            padding_mode='zeros', # 投影到图片外面的点填 0
            align_corners=False
        )
        
        return voxel_features # (B, C, Z, Y, X) -> [2, 64, 16, 200, 200]

    def forward(self, images, calib_mats):
        """
        images: (B, 3, H, W)
        calib_mats: (B, 4, 4) Lidar/World 到 Image 的投影矩阵
        """
        B, _, H, W = images.shape
        
        # ------------------------------------------------
        # Step 1: 视锥特征生成 (Frustum Feature Generation)
        # ------------------------------------------------
        
        # 1.1 语义特征 (Image Features)
        # Shape: (B, C, H, W) -> [2, 64, 96, 320]
        img_feats = self.image_head(images) 
        
        # 1.2 深度分布 (Depth Distribution)
        # Shape: (B, D, H, W) -> [2, 80, 96, 320]
        depth_logits = self.depth_head(images)
        # 关键：做 Softmax，变成概率分布
        depth_probs = F.softmax(depth_logits, dim=1) 
        
        # 1.3 外积 (Outer Product) -> 混合语义和深度
        # Feature: (B, C, 1, H, W)
        # Depth:   (B, 1, D, H, W)
        # Result:  (B, C, D, H, W) -> [2, 64, 80, 96, 320]
        # 这是一个视锥体 (Frustum Volume) 它长得像一个从相机出发的放射状棱锥。
        # 在这个棱锥里，每一个坐标点 (u, v, d)都存储了一个 64 维的特征向量
        # 它是源头数据，是我们一会要去“抓取”的地方。
        frustum_feat = img_feats.unsqueeze(2) * depth_probs.unsqueeze(1)
        
        
        # ------------------------------------------------
        # Step 2: 视锥 -> 体素变换 (Frustum to Voxel)
        # ------------------------------------------------
        # 这是一个 5D -> 5D 的插值过程
        # 输入是视锥形状的 (C, D, H, W)，输出是长方体形状的 (C, Z, Y, X) self.voxel_grid -> [1, 3, 16, 200, 200]
        voxel_feat_3d = self.frustum_to_voxel_sampling(frustum_feat, self.voxel_grid, calib_mats)
        
        # ------------------------------------------------
        # Step 3: 体素塌缩 (Voxel Collapse)
        # ------------------------------------------------
        # Input: (B, C, Z, Y, X) -> [2, 64, 16, 200, 200]
        B, C, Z, Y, X = voxel_feat_3d.shape
        
        # 把 Z 轴和 C 轴合并，准备变 BEV
        # Shape: (B, C*Z, Y, X)
        voxel_feat_collapsed = voxel_feat_3d.view(B, C * Z, Y, X)
        
        # 用卷积降维并融合 Z 轴信息
        # Shape: (B, 128, Y, X) -> 最终的 BEV Feature -> [2, 128, 200, 200]
        bev_feat = self.bev_compressor(voxel_feat_collapsed)
        
        return bev_feat, depth_logits # 返回 depth_logits 用于计算 Loss

# ==========================================
# 额外部分: 深度监督 Loss 怎么算？ (模拟代码)
# ==========================================
def compute_caddn_loss(depth_logits, gt_depth_map, depth_bins=80, depth_range=(2, 42)):
    """
    修复后的 Loss 计算函数
    """
    min_d, max_d = depth_range
    interval = (max_d - min_d) / depth_bins
    
    # 1. 计算 Bin Index
    gt_bins = ((gt_depth_map - min_d) / interval).long()
    
    # 2. 生成掩码 (Masking)
    # 找出所有 有效深度范围 (2m ~ 42m) 内的点
    valid_mask = (gt_depth_map > min_d) & (gt_depth_map < max_d)
    
    # -------------------------------------------------------------------------
    # [关键修复步骤]
    # 在传入 Loss 函数之前，必须把越界的索引改成一个"无害"的值。
    # 我们通常将其设为 -1，并在 cross_entropy 中设置 ignore_index=-1
    # -------------------------------------------------------------------------
    
    # 将所有无效位置 (深度太近或太远) 的索引设为 -1
    gt_bins[~valid_mask] = -1 
    
    # 3. 计算 CrossEntropy Loss
    loss_depth = F.cross_entropy(
        depth_logits,       # (B, D, H, W)
        gt_bins,            # (B, H, W) -> 里面只有 0~79 和 -1
        ignore_index=-1,    # [重点] 告诉 PyTorch 忽略所有值为 -1 的位置，不计算 Loss 也不报错
        reduction='mean'    # 直接求平均即可，因为 ignore_index 会自动处理分母
    )
    
    return loss_depth

def generate_realistic_projection_matrix(B, img_H=96, img_W=320):
    """
    模拟生成真实的 Lidar2Img 投影矩阵 (4x4)
    数学原理: P_img = K_homo * T_lidar2cam
    """
    # ==========================================
    # 1. 定义相机内参 (Intrinsics K) - 3x3
    # ==========================================
    # 假设这是缩小了4倍后的特征图尺寸 (96x320)
    # 原始图像可能是 (384x1280)
    # 焦距 f_x, f_y 通常跟图像宽度差不多量级
    f_x = 300.0 
    f_y = 300.0
    # 光心 (Principal Point) 通常在图像中心
    c_u = img_W / 2.0
    c_v = img_H / 2.0
    
    # 构造 K 矩阵 (B, 3, 3)
    K = torch.zeros(B, 3, 3)
    K[:, 0, 0] = f_x
    K[:, 1, 1] = f_y
    K[:, 0, 2] = c_u
    K[:, 1, 2] = c_v
    K[:, 2, 2] = 1.0
    
    # ==========================================
    # 2. 定义相机外参 (Extrinsics T) - 4x4
    # Lidar 到 Camera 的刚体变换
    # ==========================================
    # 初始化单位矩阵
    T_lidar2cam = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    
    # --- 模拟旋转 (Rotation) ---
    # 自动驾驶中，Lidar 坐标系通常是: X前, Y左, Z上
    # 相机坐标系通常是: X右, Y下, Z前
    # 我们需要一个旋转矩阵来对齐这两个坐标系
    # R_rect: 这是一个标准的坐标系转换矩阵
    # [ 0 -1  0 ]  (Lidar Y -> Camera -X) -> (左 -> 左)
    # [ 0  0 -1 ]  (Lidar Z -> Camera -Y) -> (上 -> 上)
    # [ 1  0  0 ]  (Lidar X -> Camera  Z) -> (前 -> 前)
    
    # 这里是把Lidar坐标系转换为camera坐标系，lidar坐标系是 前,左,上; 相加是右,下,前;
    R_rect = torch.tensor([
        [0.0, -1.0, 0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0,  0.0]
    ])
    
    T_lidar2cam[:, :3, :3] = R_rect
    
    # --- 模拟平移 (Translation) ---
    # 假设相机安装在 Lidar 的位置 (相对偏移很小)
    # 在这个旋转下，平移也需要对应旋转，为了简单，我们假设中心重合
    # 仅做微小调整
    T_lidar2cam[:, 0, 3] = 0.0
    T_lidar2cam[:, 1, 3] = 0.0
    T_lidar2cam[:, 2, 3] = 0.0 # 保持在原点，方便理解

    # ==========================================
    # 3. 将内参 K 扩充为 4x4 (K_homo)
    # ==========================================
    # 关键步骤！为了能和 4x4 外参相乘，K 必须扩充
    K_homo = torch.zeros(B, 4, 4)
    K_homo[:, :3, :3] = K
    # 这一步是为了保留深度信息 (Z) 和平移项
    K_homo[:, 3, 3] = 1.0 # 右下角填1
    # 注意：标准的投影矩阵第3行通常是 [0, 0, 1, 0]，这样投影后 w=z
    # 我们的代码里 img_points[:, 2] 取的就是 z，所以这样构造没问题
    
    # ==========================================
    # 4. 计算最终投影矩阵 (Lidar -> Image)
    # P = K_homo * T_lidar2cam
    # ==========================================
    # (B, 4, 4) x (B, 4, 4) -> (B, 4, 4)
    calib_mats = torch.bmm(K_homo, T_lidar2cam)
    
    return calib_mats

# ==========================================
# 测试运行
# ==========================================
# if __name__ == "__main__":
#     # 模拟数据
#     B = 2
#     H, W = 96, 320 # 假设这是 Backbone 输出的特征图大小
    
#     # 模型初始化
#     model = CaDDN_Core(image_shape=(H, W))
    
#     # 随机输入
#     dummy_img = torch.randn(B, 3, H, W)
#     # 模拟投影矩阵 (Lidar -> Image)
#     dummy_calib = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    
#     print(">>> 正在运行 CaDDN 核心流程...")
#     bev_out, depth_out = model(dummy_img, dummy_calib)
    
#     print(f"1. 输入图像: {dummy_img.shape}")
#     print(f"2. 深度预测 (Logits): {depth_out.shape} -> 用于计算 Depth Loss")
#     print(f"3. 最终 BEV 特征: {bev_out.shape} -> 用于 3D 检测")
    
#     # 模拟 Loss 计算
#     dummy_gt_depth = torch.rand(B, H, W) * 50 # 0~50米的随机深度
#     loss = compute_caddn_loss(depth_out, dummy_gt_depth)
#     print(f"4. 深度监督 Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # ==========================================
    # 1. 基础配置
    # ==========================================
    B = 2
    # 注意：这里的 H, W 对应的是特征图大小，不是原图大小
    H, W = 96, 320 
    
    # ==========================================
    # 2. 生成真实的投影矩阵 (代替原来的 eye(4))
    # ==========================================
    # 这就是你想要的详细计算过程产物
    real_calib_mats = generate_realistic_projection_matrix(B, img_H=H, img_W=W)
    
    print(f"生成的真实投影矩阵 Shape: {real_calib_mats.shape}")
    print(f"矩阵示例 (Batch 0):\n{real_calib_mats[0]}")

    # ==========================================
    # 3. 验证一下矩阵是否靠谱 (验证环节)
    # ==========================================
    # 我们找一个 Lidar 坐标系下的点: 车头正前方 20米
    # P_lidar = (20, 0, 0)
    test_point = torch.tensor([20.0, 0.0, 0.0, 1.0]).view(1, 4, 1).repeat(B, 1, 1)
    
    # 投影: P_img = Matrix * P_lidar
    proj_point = torch.bmm(real_calib_mats, test_point) # (B, 4, 1)
    
    # 归一化 (透视除法)
    z = proj_point[:, 2, 0]        # 深度
    u = proj_point[:, 0, 0] / z    # 像素 u
    v = proj_point[:, 1, 0] / z    # 像素 v
    
    print("\n--------------------------------")
    print("【投影验证】车头正前方 20米 的点 (20, 0, 0)")
    print(f" -> 投影深度 Z: {z[0].item():.2f} 米 (应该接近 20.0)")
    print(f" -> 像素坐标 U: {u[0].item():.2f} (应该接近图像中心 W/2 = 160)")
    print(f" -> 像素坐标 V: {v[0].item():.2f} (应该接近图像中心 H/2 = 48)")
    print("--------------------------------\n")

    # ==========================================
    # 4. 运行 CaDDN 模型
    # ==========================================
    model = CaDDN_Core(image_shape=(H, W))
    
    # 随机输入图像
    dummy_img = torch.randn(B, 3, H, W)
    
    print(">>> 正在运行 CaDDN 核心流程...")
    # 把真实的矩阵传进去
    bev_out, depth_out = model(dummy_img, real_calib_mats)
    
    print(f"1. 输入图像: {dummy_img.shape}")
    print(f"2. 深度预测: {depth_out.shape}")
    print(f"3. 最终 BEV 特征: {bev_out.shape}")
    
    # Loss 计算
    dummy_gt_depth = torch.rand(B, H, W) * 50 
    loss = compute_caddn_loss(depth_out, dummy_gt_depth)
    print(f"4. 深度监督 Loss: {loss.item():.4f}")