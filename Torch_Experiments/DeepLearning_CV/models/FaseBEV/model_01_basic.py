import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================
# 1. 全局配置 (Configuration)
# =========================================================
class FastBEVConfig:
    # --- 输入参数 ---
    num_cams = 6
    img_h = 256
    img_w = 704
    
    # --- Backbone & FPN ---
    # Fast-BEV 使用多尺度投影，通常是 1/4, 1/8, 1/16
    scales = [4, 8, 16] 
    feat_dims = [64, 128, 256] # FPN 各层通道数
    
    # --- Voxel (3D 空间定义) ---
    # [x_min, y_min, z_min, x_max, y_max, z_max]
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.4, 0.4, 1.333] # 论文中 Z 轴只有 6 格: 8m / 6 ≈ 1.33
    
    # 计算出的体素分辨率: 102.4 / 0.4 = 256
    # Grid Size: [256, 256, 6] (X, Y, Z)
    
    # --- BEV Encoder ---
    bev_dim = 256
    
    # --- Temporal ---
    num_frames = 4  # 论文使用 4 帧融合

# =========================================================
# 2. 多尺度图像编码器 (Mock FPN)
# =========================================================
class MultiScaleBackbone(nn.Module):
    """
    模拟 ResNet + FPN，输出 3 层多尺度特征
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 用简单的卷积模拟 FPN 的 3 个 Level
        # Level 0: 1/4 stride
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), # /2
            nn.ReLU(),
            nn.Conv2d(32, cfg.feat_dims[0], 3, 2, 1), # /4
            nn.BatchNorm2d(cfg.feat_dims[0]),
            nn.ReLU()
        )
        # Level 1: 1/8 stride
        self.stage2 = nn.Sequential(
            nn.Conv2d(cfg.feat_dims[0], cfg.feat_dims[1], 3, 2, 1), # /8
            nn.BatchNorm2d(cfg.feat_dims[1]),
            nn.ReLU()
        )
        # Level 2: 1/16 stride
        self.stage3 = nn.Sequential(
            nn.Conv2d(cfg.feat_dims[1], cfg.feat_dims[2], 3, 2, 1), # /16
            nn.BatchNorm2d(cfg.feat_dims[2]),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B*N, 3, H, W)
        f1 = self.stage1(x) # 1/4
        f2 = self.stage2(f1) # 1/8
        f3 = self.stage3(f2) # 1/16
        return [f1, f2, f3]

# =========================================================
# 3. 核心算法: Fast-Ray Transformation (LUT 查表法)
# =========================================================
class FastRayTransformation(nn.Module):
    def __init__(self, cfg, grid_size):
        super().__init__()
        self.cfg = cfg
        self.grid_size = grid_size # [X, Y, Z]
        self.nx, self.ny, self.nz = grid_size
        
        # 预先构建体素中心坐标 (Voxel Centers)
        self.register_buffer("voxel_centers", self._create_voxel_grid(), persistent=False)
        
        # 存储 LUT (Look-Up-Table)
        # LUT 的内容是: 每一个体素对应哪个相机的哪个像素坐标
        # Shape: (N_scales, N_voxels_X, N_voxels_Y, N_voxels_Z, 3) 
        # 最后一维 3 代表: [camera_id, u, v]
        self.lut_dict = nn.ParameterDict() 
        self.has_lut_computed = False

    def _create_voxel_grid(self):
        # 生成体素中心的世界坐标
        pc_range = self.cfg.pc_range
        voxel_size = self.cfg.voxel_size
        
        xs = torch.arange(self.nx, dtype=torch.float32) * voxel_size[0] + pc_range[0] + voxel_size[0]/2
        ys = torch.arange(self.ny, dtype=torch.float32) * voxel_size[1] + pc_range[1] + voxel_size[1]/2
        zs = torch.arange(self.nz, dtype=torch.float32) * voxel_size[2] + pc_range[2] + voxel_size[2]/2
        
        # Meshgrid: (X, Y, Z)
        # 注意: indexing='ij' 表示第一个维度是 x，第二个是 y
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
        
        # Flatten: (N_voxels, 3)
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    @torch.no_grad()
    def compute_lut(self, intrins, extrinsics, img_shapes):
        """
        【算法核心】: 预计算投影索引 (Build Look-Up-Table)
        这个函数在初始化或标定改变时调用一次，推理时不再调用。
        
        intrins: (N_cam, 3, 3)
        extrinsics: (N_cam, 4, 4)  World -> Camera
        img_shapes: List of [(H_1, W_1), (H_2, W_2)...] 对应不同 FPN 尺度
        """
        device = intrins.device
        num_voxels = self.voxel_centers.shape[0]
        num_cams = intrins.shape[0]
        
        # 1. 将所有体素点转到相机坐标系
        # Homogeneous: (N_voxels, 4)
        voxels_homo = torch.cat([self.voxel_centers, torch.ones(num_voxels, 1, device=device)], dim=-1)
        
        # World -> Cam (对于所有相机)
        # (N_cams, 4, 4) @ (N_voxels, 4)^T -> (N_cams, 4, N_voxels) -> (N_cams, N_voxels, 4)
        # 这里实际上就是计算: P_cam = T_ext * P_world
        points_cam = torch.matmul(extrinsics, voxels_homo.t()).permute(0, 2, 1)
        
        # 深度mask: 只保留相机前方的点 (z > 0)
        depth = points_cam[..., 2] # (N_cams, N_voxels)
        valid_mask_z = depth > 0.1
        
        # 2. 投影到图像平面
        # P_img = K * P_cam
        # (N_cams, 3, 3) @ (N_cams, N_voxels, 3)^T 
        # 为了方便矩阵乘法，只取前3维
        points_cam_xyz = points_cam[..., :3]
        points_img = torch.bmm(intrins, points_cam_xyz.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 归一化 (u, v, 1)
        u = points_img[..., 0] / (points_img[..., 2] + 1e-6)
        v = points_img[..., 1] / (points_img[..., 2] + 1e-6)
        
        # 3. 为每个尺度生成 LUT
        for scale_idx, (H, W) in enumerate(img_shapes):
            # 缩放内参对应的 uv 坐标 (假设 intrins 是基于原图的，如果 feature map 缩小了，uv也要缩)
            # FPN 1/4 -> uv / 4
            stride = self.cfg.scales[scale_idx]
            u_feat = u / stride
            v_feat = v / stride
            
            # 检查边界
            valid_mask_u = (u_feat >= 0) & (u_feat < W)
            valid_mask_v = (v_feat >= 0) & (v_feat < H)
            valid_mask = valid_mask_z & valid_mask_u & valid_mask_v # (N_cams, N_voxels)
            
            # --- Multi-View to One-Voxel ---
            # 论文策略: "directly adopt the first encountered view"
            # 我们需要为每个 voxel 找到第一个有效的 camera
            
            # 初始化 LUT: (N_voxels, 3) -> [cam_idx, u, v]
            # 填 -1 表示无效
            lut = torch.full((num_voxels, 3), -1, dtype=torch.long, device=device)
            
            # 倒序遍历相机，这样前面的相机会覆盖后面的 (或者正序均可，取决于"first encountered"定义)
            # 实际上用 torch.max 或 mask 填充更快
            for cam_i in range(num_cams):
                mask = valid_mask[cam_i]
                # 只有当前 voxel 还没被填过(-1) 且 当前相机视角有效 时才填
                # 但为了并行，我们可以利用 scatter 或者 mask fill
                # 简单写法:
                # 找到该相机所有有效的 voxel 索引
                valid_indices = torch.where(mask)[0]
                
                # 更新这些 voxel 的 LUT
                # 注意：为了实现"First Encountered"，我们其实应该先填后面的，再填前面的？
                # 不，简单的策略是：只要有效就填。如果有多个，后面的会覆盖前面的。
                # 论文说"First"，我们可以只填那些还是 -1 的位置。
                
                not_filled = (lut[valid_indices, 0] == -1)
                indices_to_fill = valid_indices[not_filled]
                
                if len(indices_to_fill) > 0:
                    u_vals = u_feat[cam_i, indices_to_fill].long()
                    v_vals = v_feat[cam_i, indices_to_fill].long()
                    
                    # 填入 [cam_i, u, v]
                    # PyTorch 不支持一行赋值多列，需分开
                    lut[indices_to_fill, 0] = cam_i
                    lut[indices_to_fill, 1] = u_vals
                    lut[indices_to_fill, 2] = v_vals
            
            # 存入 ParameterDict (不可训练)
            self.lut_dict[f"scale_{scale_idx}"] = nn.Parameter(lut, requires_grad=False)
            
        self.has_lut_computed = True

    def forward(self, features_list):
        """
        【算法核心】: 推理阶段 (Inference)
        直接查表，极速投影。
        
        features_list: List of (B, N, C, H, W) 不同尺度的特征
        Returns: 
            voxel_feats: (B, C_sum, X, Y, Z) 融合后的体素特征
        """
        if not self.has_lut_computed:
            raise RuntimeError("LUT not computed! Call compute_lut() first.")
            
        B = features_list[0].shape[0]
        voxel_feats_list = []
        
        # 对每个尺度进行 Fast-Ray 变换
        for i, feat in enumerate(features_list):
            # feat: (B, N, C, H, W)
            N, C, H, W = feat.shape[1:]
            
            # 取出对应的 LUT: (N_voxels, 3) -> [cam_idx, u, v]
            lut = self.lut_dict[f"scale_{i}"] # type: torch.Tensor
            
            # 1. 扁平化特征以便索引
            # (B, N, C, H, W) -> (B, N, H, W, C) -> (B, N*H*W, C)
            feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(B, -1, C)
            
            # 2. 计算 Gather 索引
            # LUT 里的 cam_idx, u, v 需要转换成 N*H*W 里的线性索引
            # linear_idx = cam_idx * (H*W) + v * W + u
            
            valid_mask = (lut[:, 0] != -1) # 哪些体素是有效的
            valid_lut = lut[valid_mask]    # (M, 3)
            
            cam_idx = valid_lut[:, 0]
            u_idx = valid_lut[:, 1]
            v_idx = valid_lut[:, 2]
            
            flat_indices = cam_idx * (H * W) + v_idx * W + u_idx # (M,)
            
            # 3. 查表 (Gather)
            # 输出容器: (B, N_voxels, C)
            out_voxels = torch.zeros((B, self.nx * self.ny * self.nz, C), 
                                     dtype=feat.dtype, device=feat.device)
            
            # 对每个 Batch 执行 gather
            # expand indices to (B, M, C) is expensive, use loop or advanced indexing
            # fast way:
            for b in range(B):
                # (M, C) = (N*H*W, C)[(M,)]
                selected_feats = feat_flat[b, flat_indices, :] 
                out_voxels[b, valid_mask, :] = selected_feats
            
            # Reshape to (B, C, X, Y, Z)
            # 注意 meshgrid 顺序, 我们的 lut 是按 voxel_centers (X,Y,Z) 顺序生成的
            # reshape 为 (B, X, Y, Z, C) -> permute -> (B, C, X, Y, Z)
            out_voxels = out_voxels.reshape(B, self.nx, self.ny, self.nz, C).permute(0, 4, 1, 2, 3)
            
            voxel_feats_list.append(out_voxels)
            
        return voxel_feats_list

# =========================================================
# 4. 高效 BEV 编码器 & 时序融合
# =========================================================
class EfficientBEVEncoder(nn.Module):
    def __init__(self, cfg, in_channels_list):
        super().__init__()
        self.cfg = cfg
        
        # 1. MSCF (Multi-Scale Concatenation Fusion)
        # Fast-Ray 输出了3个尺度的 Voxel，我们需要把它们对齐并拼起来
        # 但其实 Fast-Ray 投影时用的都是同一个 Voxel Grid (200x200x6)，
        # 所以它们的空间维度 X,Y,Z 是一样的！不需要上采样，直接 Concat Channel 即可。
        # 论文中 "upsample ... to the same size" 指的是如果 Fast-Ray 设置了不同的 voxel grid，
        # 但为了极速，通常只用一个统一的 Grid。这里我们假设 Grid 统一。
        
        self.total_in_channels = sum(in_channels_list)
        
        # 2. S2C (Space to Channel)
        # 将 Z 轴压扁: (C, X, Y, Z) -> (C*Z, X, Y)
        self.nz = 6 # 假设 Z=6
        self.s2c_dim = self.total_in_channels * self.nz
        
        # 3. MFCF (Multi-Frame Concatenation Fusion)
        # 输入维度会变成 s2c_dim * num_frames
        self.fusion_dim = self.s2c_dim * cfg.num_frames
        
        # 4. 2D BEV Encoder (降维 + 融合)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.fusion_dim, cfg.bev_dim, 3, 1, 1),
            nn.BatchNorm2d(cfg.bev_dim),
            nn.ReLU(),
            nn.Conv2d(cfg.bev_dim, cfg.bev_dim, 3, 1, 1),
            nn.BatchNorm2d(cfg.bev_dim),
            nn.ReLU()
        )

    def forward(self, voxel_feats_list, prev_bev_feats_queue=None):
        """
        voxel_feats_list: 当前帧 3 个尺度的 Voxel 特征 [(B, C1, X, Y, Z), (B, C2...), ...]
        prev_bev_feats_queue: 历史帧的 BEV 特征列表 (list of 2D tensors)
        """
        # --- MSCF: 多尺度融合 ---
        # 已经在同一个 Grid 上了，直接 concat channel
        # cat([(B, C1, X, Y, Z), ...]) -> (B, C_total, X, Y, Z)
        voxel_feat = torch.cat(voxel_feats_list, dim=1)
        
        # --- S2C: Space to Channel ---
        B, C, X, Y, Z = voxel_feat.shape
        # Permute to (B, C, Z, X, Y) -> Reshape (B, C*Z, X, Y)
        bev_feat = voxel_feat.permute(0, 1, 4, 2, 3).reshape(B, C*Z, X, Y)
        
        # --- MFCF: 时序融合 ---
        # 论文使用 BEVDet4D 的对齐方式，这里简化为直接 Concat (假设已经对齐或作为队列输入)
        # 实际上应该有 feature alignment (warp)，为了代码简洁，这里展示 Concat 逻辑
        current_feat = bev_feat
        
        if prev_bev_feats_queue is not None and len(prev_bev_feats_queue) > 0:
            # 队列: [T-3, T-2, T-1]
            # Concat: [T-3, T-2, T-1, Curr]
            history_feats = torch.cat(prev_bev_feats_queue, dim=1) # Channel 维度拼接
            fused_feat = torch.cat([history_feats, current_feat], dim=1)
        else:
            # 如果没有历史帧 (或第一帧)，简单复制填充或补零
            # 为了保持 Channel 维度一致以便送入 Conv
            fused_feat = torch.cat([current_feat] * self.cfg.num_frames, dim=1)
            
        # --- 2D Encoder ---
        out = self.encoder(fused_feat)
        
        return out, current_feat # 返回当前特征以便存入队列

# =========================================================
# 5. Fast-BEV 整体模型
# =========================================================
class FastBEV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. Backbone
        self.backbone = MultiScaleBackbone(cfg)
        
        # 2. Fast-Ray Projector
        # 计算 Grid Size
        grid_size = [
            int((cfg.pc_range[3]-cfg.pc_range[0])/cfg.voxel_size[0]),
            int((cfg.pc_range[4]-cfg.pc_range[1])/cfg.voxel_size[1]),
            int((cfg.pc_range[5]-cfg.pc_range[2])/cfg.voxel_size[2])
        ] # [256, 256, 6]
        self.fast_ray = FastRayTransformation(cfg, grid_size)
        
        # 3. BEV Encoder
        self.bev_encoder = EfficientBEVEncoder(cfg, cfg.feat_dims)
        
        # 4. Head (简单示例)
        self.head = nn.Conv2d(cfg.bev_dim, 10, 1) # 10类检测

    def forward(self, imgs, prev_bev_queue=None, **kwargs):
        """
        imgs: (B, N, 3, H, W)
        kwargs: 包含 intrins, extrinsics (仅在第一次计算 LUT 时需要)
        """
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B*N, C, H, W)
        
        # 1. 提取多尺度特征
        feats_list = self.backbone(imgs) 
        # feats_list: [(B*N, 64, H/4, W/4), ...]
        
        # Reshape back to (B, N, C, H, W)
        feats_list = [f.view(B, N, f.shape[1], f.shape[2], f.shape[3]) for f in feats_list]
        
        # 2. 检查 LUT 是否就绪 (One-time Setup)
        if not self.fast_ray.has_lut_computed:
            if 'intrins' not in kwargs or 'extrinsics' not in kwargs:
                raise ValueError("First run requires intrins/extrinsics to build LUT!")
            
            # 获取特征图尺寸
            feat_shapes = [(f.shape[3], f.shape[4]) for f in feats_list]
            
            # 假设 Batch 中所有相机的参数一致 (Fast-BEV 的静态假设)
            # 取第一个样本的参数构建 LUT
            self.fast_ray.compute_lut(kwargs['intrins'][0], kwargs['extrinsics'][0], feat_shapes)
        
        # 3. Fast-Ray 投影
        # 返回 3 个尺度的 Voxel: List of (B, C, X, Y, Z)
        voxel_feats = self.fast_ray(feats_list)
        
        # 4. BEV Encoder & Fusion
        # bev_out: (B, 256, X, Y)
        # curr_bev: (B, C*Z, X, Y) 原始 BEV 特征，用于更新历史队列
        bev_out, curr_bev = self.bev_encoder(voxel_feats, prev_bev_queue)
        
        # 5. Detection Head
        preds = self.head(bev_out)
        
        return preds, curr_bev

# =========================================================
# 6. 运行 Demo
# =========================================================
if __name__ == "__main__":
    cfg = FastBEVConfig()
    model = FastBEV(cfg)
    
    # Mock Data
    B, N = 1, 6
    imgs = torch.randn(B, N, 3, 256, 704)
    
    # Mock Calibration (第一次运行时需要)
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    intrins[:, :, 0, 0] = 500 # fx
    intrins[:, :, 1, 1] = 500 # fy
    intrins[:, :, 0, 2] = 352 # cx
    intrins[:, :, 1, 2] = 128 # cy
    
    # 简单的相机外参 (World -> Cam)
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    
    print("--- 1. First Pass (Build LUT & Inference) ---")
    preds, curr_bev = model(imgs, intrins=intrins, extrinsics=extrinsics)
    print(f"Output Shape: {preds.shape}") # (1, 10, 256, 256)
    
    print("\n--- 2. Second Pass (Fast Inference using LUT) ---")
    # 模拟时序: 把上一帧的 bev 放入队列
    queue = [curr_bev, curr_bev, curr_bev] # 模拟 T-3, T-2, T-1
    preds_2, curr_bev_2 = model(imgs, prev_bev_queue=queue)
    print(f"Output Shape (Temporal): {preds_2.shape}")
    print("✅ Fast-BEV implemented successfully!")