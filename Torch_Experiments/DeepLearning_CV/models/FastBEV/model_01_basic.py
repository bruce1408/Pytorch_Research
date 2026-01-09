import torch, os
import torch.nn as nn
import numpy as np
from collections import deque
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from spectrautils.common_utils import enter_workspace

enter_workspace()

# =========================================================
# 1. 全局配置 (Configuration)
# =========================================================
@dataclass
class FastBEVConfig:
    """
    Fast-BEV 模型的全局配置类
    
    这个类定义了模型所需的所有超参数，包括：
    - 输入图像尺寸和相机数量
    - 特征提取器的配置（多尺度FPN）
    - 3D体素空间的划分方式
    - BEV编码器的通道数
    - 时序融合的帧数
    """
    
    # # --- 输入参数 ---
    # num_cams = 6          # 相机数量（通常为6个，覆盖360度）
    # img_h = 256           # 输入图像高度（像素）
    # img_w = 704           # 输入图像宽度（像素）
    
    # --- Backbone & FPN ---
    # Fast-BEV 的核心思想是使用多尺度特征进行投影
    # 不同尺度的特征图包含不同层次的信息：
    # - 1/4 尺度：细节丰富，适合近距离目标
    # - 1/8 尺度：中等分辨率，平衡细节和感受野
    # - 1/16 尺度：感受野大，适合远距离目标
    scales = [4, 8, 16]   # FPN特征图的下采样倍数（相对于原图）
    feat_dims = [64, 128, 256]  # 对应每个尺度的特征通道数
    
    # --- Voxel (3D 空间定义) ---
    # 定义3D空间的体素网格，用于将图像特征投影到3D空间
    # [x_min, y_min, z_min, x_max, y_max, z_max]（单位：米）
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # 每个体素的尺寸 [dx, dy, dz]（单位：米）
    # 注意：Z轴通常只有几层（如6层），因为BEV主要关注水平面的信息
    # 8米高度范围 / 6层 ≈ 1.333米每层
    voxel_size = [0.4, 0.4, 1.333]
    
    # 计算出的体素网格分辨率：
    # X方向：(51.2 - (-51.2)) / 0.4 = 102.4 / 0.4 = 256
    # Y方向：同样为 256
    # Z方向：(3.0 - (-5.0)) / 1.333 ≈ 6
    # Grid Size: [256, 256, 6] (X, Y, Z)
    
    # --- BEV Encoder ---
    bev_dim = 256  # BEV特征图的通道数（经过编码后的最终特征维度）
    
    # --- Temporal (时序融合) ---
    num_frames = 4  # 融合的历史帧数（当前帧 + 前3帧）
    # 时序信息有助于处理动态物体和遮挡问题
    
    # Input
    num_cams: int = 6
    img_h: int = 256
    img_w: int = 704

    # Backbone scales
    scales: List[int] = None
    feat_dims: List[int] = None

    # Voxel space
    pc_range: List[float] = None  # [x_min, y_min, z_min, x_max, y_max, z_max]
    voxel_size: List[float] = None  # [dx, dy, dz]

    # BEV encoder
    bev_dim: int = 256

    # Temporal
    num_frames: int = 4  # current + (num_frames-1) history
    feat_stride: int = 4  # not used here; kept for future compatibility

    # Task
    num_classes: int = 10  # your head output channels

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 3
    batch_size: int = 2
    num_workers: int = 2
    amp: bool = True
    grad_clip_norm: float = 5.0

    def __post_init__(self):
        if self.scales is None:
            self.scales = [4, 8, 16]
        if self.feat_dims is None:
            self.feat_dims = [64, 128, 256]
        if self.pc_range is None:
            self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if self.voxel_size is None:
            self.voxel_size = [0.4, 0.4, 1.333]

# =========================================================
# 2. 多尺度图像编码器 (Mock FPN)
# =========================================================
class MultiScaleBackbone(nn.Module):
    """
    多尺度特征提取器（模拟 ResNet + FPN）
    
    功能：
    - 从输入图像提取多尺度特征
    - 输出3个不同分辨率的特征图（1/4, 1/8, 1/16）
    - 每个尺度包含不同层次的信息，用于后续的3D投影
    
    输入：原始图像 (B*N, 3, H, W)
    输出：3个尺度的特征图列表
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Stage 1: 提取 1/4 尺度特征
        # 通过两次 stride=2 的卷积实现下采样
        # 输入：(B*N, 3, H, W) -> 输出：(B*N, 64, H/4, W/4)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 第一次下采样：/2
            nn.ReLU(),
            nn.Conv2d(32, cfg.feat_dims[0], 3, stride=2, padding=1),  # 第二次下采样：/4
            nn.BatchNorm2d(cfg.feat_dims[0]),  # 批归一化，稳定训练
            nn.ReLU()
        )
        
        # Stage 2: 提取 1/8 尺度特征
        # 在 stage1 的基础上再下采样一次
        # 输入：(B*N, 64, H/4, W/4) -> 输出：(B*N, 128, H/8, W/8)
        self.stage2 = nn.Sequential(
            nn.Conv2d(cfg.feat_dims[0], cfg.feat_dims[1], 3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(cfg.feat_dims[1]),
            nn.ReLU()
        )
        
        # Stage 3: 提取 1/16 尺度特征
        # 在 stage2 的基础上再下采样一次
        # 输入：(B*N, 128, H/8, W/8) -> 输出：(B*N, 256, H/16, W/16)
        self.stage3 = nn.Sequential(
            nn.Conv2d(cfg.feat_dims[1], cfg.feat_dims[2], 3, stride=2, padding=1),  # /16
            nn.BatchNorm2d(cfg.feat_dims[2]),
            nn.ReLU()
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B*N, 3, H, W) - 输入图像，B=batch_size, N=相机数
        
        Returns:
            List of 3 tensors:
            - f1: (B*N, 64, H/4, W/4)  - 1/4尺度特征
            - f2: (B*N, 128, H/8, W/8) - 1/8尺度特征
            - f3: (B*N, 256, H/16, W/16) - 1/16尺度特征
        """
        f1 = self.stage1(x)  # 1/4 尺度
        f2 = self.stage2(f1)  # 1/8 尺度
        f3 = self.stage3(f2)  # 1/16 尺度
        return [f1, f2, f3]

# =========================================================
# 3. 核心算法: Fast-Ray Transformation (LUT 查表法)
# =========================================================
class FastRayTransformation(nn.Module):
    """
        Fast-BEV 的核心创新：Fast-Ray 投影变换
        
        核心思想：
        - 传统方法：每次推理都要计算3D体素到2D图像的投影关系（耗时）
        - Fast-BEV：预先计算投影关系，存储为查找表（LUT），推理时直接查表（极快）
        
        工作流程：
        1. 初始化阶段（compute_lut）：预计算每个体素对应哪个相机的哪个像素
        2. 推理阶段（forward）：直接查表，快速完成投影
        
        优势：
        - 推理速度极快（避免了重复的矩阵运算）
        - 适合固定标定的场景（如自动驾驶）
    """
    
    def __init__(self, cfg, grid_size):
        super().__init__()
        self.cfg = cfg
        self.grid_size = grid_size  # [X, Y, Z]
        self.nx, self.ny, self.nz = grid_size

        self.register_buffer("voxel_centers", self._create_voxel_grid(), persistent=False)

        # 用 ParameterDict 存 LUT（requires_grad=False），保持你原先风格
        self.lut_dict = nn.ParameterDict()
        self.has_lut_computed = False

    def _create_voxel_grid(self):
        pc_range = self.cfg.pc_range
        voxel_size = self.cfg.voxel_size

        xs = torch.arange(self.nx, dtype=torch.float32) * voxel_size[0] + pc_range[0] + voxel_size[0] / 2
        ys = torch.arange(self.ny, dtype=torch.float32) * voxel_size[1] + pc_range[1] + voxel_size[1] / 2
        zs = torch.arange(self.nz, dtype=torch.float32) * voxel_size[2] + pc_range[2] + voxel_size[2] / 2

        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (N_vox, 3)

    @torch.no_grad()
    def compute_lut(self, intrins, extrinsics, img_shapes):
        """
        【算法核心】预计算投影索引（构建查找表）
        
        这个函数只在初始化时或相机标定改变时调用一次。
        之后推理时不再调用，直接使用预计算的LUT。
        
        算法步骤：
        1. 将所有体素中心点从世界坐标系转换到相机坐标系
        2. 投影到图像平面，得到像素坐标
        3. 检查有效性（深度>0，在图像边界内）
        4. 为每个体素选择第一个有效的相机视角
        5. 存储为LUT：体素索引 -> [相机ID, u, v]
        
        Args:
            intrins: (N_cam, 3, 3) - 相机内参矩阵（每个相机一个）
            extrinsics: (N_cam, 4, 4) - 相机外参矩阵（世界坐标系到相机坐标系）
            img_shapes: List of [(H_1, W_1), (H_2, W_2), ...] - 不同尺度特征图的尺寸
        """
        device = intrins.device
        dtype = intrins.dtype

        voxel_centers = self.voxel_centers.to(device=device, dtype=dtype)
        num_voxels = voxel_centers.shape[0]
        num_cams = intrins.shape[0]

        # (N_vox,4)
        voxels_homo = torch.cat(
            [voxel_centers, torch.ones((num_voxels, 1), device=device, dtype=dtype)],
            dim=-1
        )

        # world -> cam : (N_cam, N_vox, 4)
        points_cam = torch.matmul(extrinsics.to(dtype), voxels_homo.t()).permute(0, 2, 1)
        depth = points_cam[..., 2]  # (N_cam, N_vox)
        valid_z = depth > 0.1

        # cam -> image (pixel)
        points_cam_xyz = points_cam[..., :3]  # (N_cam, N_vox, 3)
        points_img = torch.bmm(intrins.to(dtype), points_cam_xyz.permute(0, 2, 1)).permute(0, 2, 1)
        u = points_img[..., 0] / (points_img[..., 2].clamp(min=1e-6))
        v = points_img[..., 1] / (points_img[..., 2].clamp(min=1e-6))

        cam_ids = torch.arange(num_cams, device=device).view(num_cams, 1)  # (N_cam,1)
        vox_ids = torch.arange(num_voxels, device=device)                  # (N_vox,)

        for scale_idx, (H, W) in enumerate(img_shapes):
            stride = float(self.cfg.scales[scale_idx])

            u_feat = u / stride
            v_feat = v / stride

            valid_u = (u_feat >= 0) & (u_feat < W)
            valid_v = (v_feat >= 0) & (v_feat < H)
            valid = valid_z & valid_u & valid_v  # (N_cam, N_vox)

            # has[v] = voxel v 是否被任意相机看到
            has = valid.any(dim=0)  # (N_vox,)

            # 选取最小 cam index：把无效 cam 置为 +inf，然后取 min
            # min_cam[v] = 最小可见相机id（无效会是很大）
            big = torch.full_like(cam_ids.expand_as(valid), fill_value=10**9)
            cam_mat = torch.where(valid, cam_ids.expand_as(valid), big)  # (N_cam,N_vox)
            min_cam, _ = cam_mat.min(dim=0)  # (N_vox,)

            # 组 LUT
            lut = torch.full((num_voxels, 3), -1, dtype=torch.long, device=device)

            # 有效 voxel 的 cam idx
            cam_sel = min_cam[has].long()  # (M,)
            vox_sel = vox_ids[has]         # (M,)

            # 从 u_feat/v_feat 里按 (cam_sel, vox_sel) 取对应坐标
            u_sel = u_feat[cam_sel, vox_sel].long()
            v_sel = v_feat[cam_sel, vox_sel].long()

            lut[vox_sel, 0] = cam_sel
            lut[vox_sel, 1] = u_sel
            lut[vox_sel, 2] = v_sel

            self.lut_dict[f"scale_{scale_idx}"] = nn.Parameter(lut, requires_grad=False)

        self.has_lut_computed = True

    def forward(self, features_list):
        """
        推理阶段：向量化查表投影
        features_list: List of (B, N, C, H, W)
        return: List of (B, C, X, Y, Z)
        """
        if not self.has_lut_computed:
            raise RuntimeError("LUT not computed! Call compute_lut() first.")

        B = features_list[0].shape[0]
        voxel_feats_list = []

        num_voxels = self.nx * self.ny * self.nz

        for i, feat in enumerate(features_list):
            # feat: (B, N, C, H, W)
            N, C, H, W = feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4]
            lut = self.lut_dict[f"scale_{i}"]  # (N_vox,3)

            valid_mask = (lut[:, 0] != -1)     # (N_vox,)
            if valid_mask.any():
                valid_lut = lut[valid_mask]   # (M,3)
                cam_idx = valid_lut[:, 0]
                u_idx = valid_lut[:, 1]
                v_idx = valid_lut[:, 2]

                flat_indices = cam_idx * (H * W) + v_idx * W + u_idx  # (M,)

                # (B, N*H*W, C)
                feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(B, N * H * W, C)

                # 一次性 gather：(B, M, C)
                selected = feat_flat[:, flat_indices, :]

                out_voxels = torch.zeros((B, num_voxels, C), dtype=feat.dtype, device=feat.device)
                
                out_voxels[:, valid_mask, :] = selected
            else:
                out_voxels = torch.zeros((B, num_voxels, C), dtype=feat.dtype, device=feat.device)

            # reshape to (B, C, X, Y, Z)
            out_voxels = out_voxels.reshape(B, self.nx, self.ny, self.nz, C).permute(0, 4, 1, 2, 3).contiguous()
            voxel_feats_list.append(out_voxels)

        return voxel_feats_list



# =========================================================
# 4. 高效 BEV 编码器 & 时序融合
# =========================================================
class EfficientBEVEncoder(nn.Module):
    """
    高效的BEV编码器，包含三个关键操作：
    
    1. MSCF (Multi-Scale Concatenation Fusion)
       - 融合多个尺度的体素特征
    
    2. S2C (Space to Channel)
       - 将Z维度压缩到通道维度，避免使用昂贵的3D卷积
    
    3. MFCF (Multi-Frame Concatenation Fusion)
       - 融合时序信息（当前帧 + 历史帧）
    """
    def __init__(self, cfg, in_channels_list, nz):
        super().__init__()
        self.cfg = cfg
        self.nz = nz

        # ========== MSCF: 多尺度特征融合 ==========
        # Fast-Ray输出了3个尺度的体素特征，它们都在同一个体素网格上
        # 所以可以直接在通道维度拼接，不需要上采样
        # 例如：[(B,64,X,Y,Z), (B,128,X,Y,Z), (B,256,X,Y,Z)] 
        #   -> (B, 64+128+256, X, Y, Z) = (B, 448, X, Y, Z)
        self.total_in_channels = sum(in_channels_list)
        
        # ========== S2C: Space to Channel ==========
        # 将Z维度压缩到通道维度：(C, X, Y, Z) -> (C*Z, X, Y)
        # 这样可以用2D卷积处理，比3D卷积快得多
        # 例如：(B, 448, X, Y, 6) -> (B, 448*6, X, Y) = (B, 2688, X, Y)
        self.s2c_dim = self.total_in_channels * self.nz
        self.temporal_c = 256
        
        self.reduce = nn.Sequential(
            nn.Conv2d(self.s2c_dim, self.temporal_c, 1, bias=False),
            nn.BatchNorm2d(self.temporal_c),
            nn.ReLU(inplace=True),
        )

        # ========== MFCF: 时序融合 ==========
        # 融合多帧信息：当前帧 + 前3帧
        # 输入维度：s2c_dim * num_frames
        # 例如：2688 * 4 = 10752
        self.fusion_dim = self.temporal_c * cfg.num_frames

        # self.fusion_dim = self.s2c_dim * cfg.num_frames
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.fusion_dim, cfg.bev_dim, 3, padding=1),
            nn.BatchNorm2d(cfg.bev_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.bev_dim, cfg.bev_dim, 3, padding=1),
            nn.BatchNorm2d(cfg.bev_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, voxel_feats_list, prev_bev_feats_queue=None):
        """
            前向传播：多尺度融合 + S2C + 时序融合
            
            Args:
                voxel_feats_list: List of (B, C_i, X, Y, Z)
                    当前帧的3个尺度体素特征
                prev_bev_feats_queue: List of (B, C*Z, X, Y) 或 None
                    历史帧的BEV特征队列（用于时序融合）
            
            Returns:
                out: (B, bev_dim, X, Y) - 编码后的BEV特征
                current_feat: (B, C*Z, X, Y) - 当前帧的BEV特征（用于更新队列）
        """
        
        # 1) MSCF: 多尺度融合
        voxel_feat = torch.cat(voxel_feats_list, dim=1)  # (B, 448, X, Y, Z)

        # 2) S2C: (B, 448, X, Y, Z) -> (B, 2688, X, Y)
        B, C, X, Y, Z = voxel_feat.shape
        bev_feat = voxel_feat.permute(0, 1, 4, 2, 3).reshape(B, C * Z, X, Y)  # (B, s2c_dim, X, Y)

        # 3) per-frame reduce: (B, 2688, X, Y) -> (B, 256, X, Y)
        current_feat = self.reduce(bev_feat)

        # 4) Temporal fusion: padding 到固定长度
        hist_len = self.cfg.num_frames - 1
        if prev_bev_feats_queue is None:
            prev_bev_feats_queue = []

        # 注意：prev_bev_feats_queue 里存的也必须是 reduce 之后的 (B,256,X,Y)
        # 如果你之前队列里存的是 2688 通道，这里会再次不匹配
        # 所以建议你训练时队列也存 current_feat（下面会讲）

        if len(prev_bev_feats_queue) < hist_len:
            # padding 到固定长度
            pad = [torch.zeros_like(current_feat) for _ in range(hist_len - len(prev_bev_feats_queue))]
            prev_bev_feats_queue = pad + list(prev_bev_feats_queue)

        history_feats = torch.cat(prev_bev_feats_queue, dim=1)  # (B, 256*(T-1), X, Y)
        fused_feat = torch.cat([history_feats, current_feat], dim=1)  # (B, 256*T, X, Y) = (B,1024,X,Y)

        # ========== 2D BEV编码器 ==========
        # 使用2D卷积对融合后的特征进行编码和降维
        # 输入：(B, fusion_dim, X, Y)
        # 输出：(B, bev_dim, X, Y)
        out = self.encoder(fused_feat)

        # ✅ 返回 reduce 后的 current_feat（队列应存这个）
        return out, current_feat


# =========================================================
# 5. Fast-BEV 整体模型
# =========================================================
class FastBEV(nn.Module):
    """
    Fast-BEV 完整模型
    
    模型流程：
    1. 多尺度特征提取（Backbone）
    2. Fast-Ray投影（2D特征 -> 3D体素）
    3. BEV编码和时序融合
    4. 任务头（检测/分割等）
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. 多尺度特征提取器（Backbone + FPN）
        self.backbone = MultiScaleBackbone(cfg)
        
        # 2. Fast-Ray投影器
        # 计算体素网格尺寸
        grid_size = [
            int((cfg.pc_range[3] - cfg.pc_range[0]) / cfg.voxel_size[0]),  # X: 256
            int((cfg.pc_range[4] - cfg.pc_range[1]) / cfg.voxel_size[1]),  # Y: 256
            int((cfg.pc_range[5] - cfg.pc_range[2]) / cfg.voxel_size[2])   # Z: 6
        ]
        
        self.fast_ray = FastRayTransformation(cfg, grid_size)
        
        # 3. BEV编码器（包含多尺度融合、S2C、时序融合）
        self.bev_encoder = EfficientBEVEncoder(cfg, cfg.feat_dims, grid_size[2])
        
        # 4. 任务头（这里简化为检测头）
        # 实际应用中可能有多个头：检测头、分割头等
        self.head = nn.Conv2d(cfg.bev_dim, 10, 1)  # 10类检测

    def forward(self, imgs, prev_bev_queue=None, **kwargs):
        """
        前向传播
        
        Args:
            imgs: (B, N, 3, H, W) - 输入图像，B=batch_size, N=相机数
            prev_bev_queue: List of (B, C*Z, X, Y) 或 None - 历史BEV特征队列
            **kwargs: 可能包含相机参数（仅在第一次运行时需要）
                - intrins: 相机内参
                - extrinsics: 相机外参
        
        Returns:
            preds: (B, num_classes, X, Y) - 预测结果
            curr_bev: (B, C*Z, X, Y) - 当前帧BEV特征（用于更新队列）
        """
        B, N, C, H, W = imgs.shape
        
        # ========== 步骤1：提取多尺度特征 ==========
        # 将图像展平：(B, N, 3, H, W) -> (B*N, 3, H, W)
        imgs = imgs.view(B*N, C, H, W)
        
        # 提取特征：输出3个尺度的特征图
        feats_list = self.backbone(imgs)
        # feats_list: [(B*N, 64, H/4, W/4), (B*N, 128, H/8, W/8), (B*N, 256, H/16, W/16)]
        
        # 恢复batch和相机维度：(B*N, C, H, W) -> (B, N, C, H, W)
        feats_list = [f.view(B, N, f.shape[1], f.shape[2], f.shape[3]) for f in feats_list]
        
        # ========== 步骤2：检查并构建LUT（一次性操作） ==========
        if not self.fast_ray.has_lut_computed:
            # 第一次运行时需要相机参数来构建LUT
            if 'intrins' not in kwargs or 'extrinsics' not in kwargs:
                raise ValueError("First run requires intrins/extrinsics to build LUT!")
            
            # 获取特征图尺寸（用于LUT计算）
            feat_shapes = [(f.shape[3], f.shape[4]) for f in feats_list]
            # 例如：[(H/4, W/4), (H/8, W/8), (H/16, W/16)]
            
            # 假设batch中所有相机的参数一致（Fast-BEV的静态假设）
            # 取第一个样本的参数构建LUT
            self.fast_ray.compute_lut(kwargs['intrins'][0], kwargs['extrinsics'][0], feat_shapes)
        
        # ========== 步骤3：Fast-Ray投影 ==========
        # 将2D特征投影到3D体素空间
        # 返回3个尺度的体素特征：List of (B, C, X, Y, Z)
        voxel_feats = self.fast_ray(feats_list)
        
        # ========== 步骤4：BEV编码和时序融合 ==========
        # bev_out: (B, bev_dim, X, Y) - 编码后的BEV特征
        # curr_bev: (B, C*Z, X, Y) - 当前帧原始BEV特征（用于更新历史队列）
        bev_out, curr_bev = self.bev_encoder(voxel_feats, prev_bev_queue)
        
        # ========== 步骤5：任务头预测 ==========
        # 使用BEV特征进行检测/分割等任务
        preds = self.head(bev_out)  # (B, num_classes, X, Y)
        
        return preds, curr_bev


# =========================================================
# 6) Dataset (demo) — replace with your real dataset
# =========================================================
class SyntheticFastBEVDataset(Dataset):
    """
    生成一个“可训练”的假数据集，用于跑通 pipeline：
    - imgs: (Ncam,3,H,W)
    - gt: (nx,ny) 0..num_classes-1
    - intrins/extrinsics: 固定标定
    - scene_id/frame_idx: 用于时序队列
    """
    def __init__(self, cfg: FastBEVConfig, num_scenes=10, frames_per_scene=20, seed=0):
        self.cfg = cfg
        self.num_scenes = num_scenes
        self.frames_per_scene = frames_per_scene
        self.total = num_scenes * frames_per_scene
        g = torch.Generator().manual_seed(seed)
        self._rand = g

        # fixed calibration (B,N,3,3)/(B,N,4,4) produced in __getitem__ per sample
        self.fx = 500.0
        self.fy = 500.0
        self.cx = cfg.img_w / 2.0
        self.cy = cfg.img_h / 2.0

        # BEV grid (nx, ny)
        self.nx = int((cfg.pc_range[3] - cfg.pc_range[0]) / cfg.voxel_size[0])
        self.ny = int((cfg.pc_range[4] - cfg.pc_range[1]) / cfg.voxel_size[1])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        scene_id = idx // self.frames_per_scene
        frame_idx = idx % self.frames_per_scene

        # images
        imgs = torch.randn(self.cfg.num_cams, 3, self.cfg.img_h, self.cfg.img_w, generator=self._rand)

        # BEV segmentation GT
        gt = torch.randint(
            low=0, high=self.cfg.num_classes, size=(self.nx, self.ny), generator=self._rand, dtype=torch.long
        )

        # intrinsics/extrinsics
        intrins = torch.eye(3).unsqueeze(0).repeat(self.cfg.num_cams, 1, 1)
        intrins[:, 0, 0] = self.fx
        intrins[:, 1, 1] = self.fy
        intrins[:, 0, 2] = self.cx
        intrins[:, 1, 2] = self.cy

        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.cfg.num_cams, 1, 1)  # identity for demo

        sample = {
            "imgs": imgs,                 # (N,3,H,W)
            "gt_bev": gt,                 # (nx,ny)
            "intrins": intrins,           # (N,3,3)
            "extrinsics": extrinsics,     # (N,4,4)
            "scene_id": scene_id,
            "frame_idx": frame_idx,
        }
        return sample


def collate_fn(batch):
    # stack
    imgs = torch.stack([b["imgs"] for b in batch], dim=0)         # (B,N,3,H,W)
    gt = torch.stack([b["gt_bev"] for b in batch], dim=0)         # (B,nx,ny)
    intrins = torch.stack([b["intrins"] for b in batch], dim=0)   # (B,N,3,3)
    extrinsics = torch.stack([b["extrinsics"] for b in batch], dim=0)  # (B,N,4,4)
    scene_id = torch.tensor([b["scene_id"] for b in batch], dtype=torch.long)
    frame_idx = torch.tensor([b["frame_idx"] for b in batch], dtype=torch.long)
    return {"imgs": imgs, "gt_bev": gt, "intrins": intrins, "extrinsics": extrinsics,
            "scene_id": scene_id, "frame_idx": frame_idx}


# =========================================================
# 7) Metrics
# =========================================================
@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    # logits: (B,C,H,W), target: (B,H,W)
    pred = logits.argmax(dim=1)
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / max(total, 1)).item()


# =========================================================
# 8) Train / Validate
# =========================================================
def train_one_epoch(
    model: FastBEV,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: FastBEVConfig,
    epoch: int,
):
    model.train()

    # scene_id -> deque of (B_i, CZ, X, Y) for THAT sample's batch element
    # 为了简单起见，这里假设 batch 内 scene 不混或你不强依赖时序。
    # 工程上更严谨的做法：对 batch 内每个样本单独维护队列（见备注）。
    queue_dict: Dict[int, deque] = {}

    running_loss = 0.0
    running_acc = 0.0

    for it, batch in enumerate(loader):
        imgs = batch["imgs"].to(device)
        gt = batch["gt_bev"].to(device)  # (B,nx,ny)
        intrins = batch["intrins"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        scene_ids = batch["scene_id"].tolist()
        frame_idxs = batch["frame_idx"].tolist()

        # ---- build per-batch prev_queue (simple version) ----
        # 这里做一个“简化但可跑”的时序队列：
        # - 若 batch 中 scene 混杂，会变得复杂（要 per-sample queues）。
        # - demo 默认 loader 不 shuffle，且 batch_size 小，scene 不乱。
        prev_queue = None
        # 取 batch 里第一个样本的队列（简化）
        sid0, fid0 = scene_ids[0], frame_idxs[0]
        if fid0 == 0 or sid0 not in queue_dict:
            queue_dict[sid0] = deque(maxlen=cfg.num_frames - 1)
        prev_queue = list(queue_dict[sid0])

        optimizer.zero_grad(set_to_none=True)

        use_amp = (scaler is not None) and cfg.amp and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            
            # first iteration will build LUT inside model
            preds, curr_bev = model(imgs, prev_bev_queue=prev_queue, intrins=intrins, extrinsics=extrinsics)
            
            # CrossEntropy expects (B,C,H,W) and target (B,H,W)
            loss = F.cross_entropy(preds, gt)

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        # 更新队列（stop-grad，避免跨帧反传）
        queue_dict[sid0].append(curr_bev.detach())

        running_loss += loss.item()
        running_acc += pixel_accuracy(preds.detach(), gt)

        if (it + 1) % 10 == 0:
            avg_loss = running_loss / (it + 1)
            avg_acc = running_acc / (it + 1)
            print(f"[Train][Epoch:{epoch}] iter={it+1}/{len(loader)} loss={avg_loss:.4f} acc={avg_acc:.4f}")

    return running_loss / max(len(loader), 1), running_acc / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(
    model: FastBEV,
    loader: DataLoader,
    device: torch.device,
    cfg: FastBEVConfig,
    epoch: int,
):
    model.eval()

    queue_dict: Dict[int, deque] = {}
    running_loss = 0.0
    running_acc = 0.0

    for it, batch in enumerate(loader):
        imgs = batch["imgs"].to(device)
        gt = batch["gt_bev"].to(device)
        intrins = batch["intrins"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        scene_ids = batch["scene_id"].tolist()
        frame_idxs = batch["frame_idx"].tolist()

        sid0, fid0 = scene_ids[0], frame_idxs[0]
        if fid0 == 0 or sid0 not in queue_dict:
            queue_dict[sid0] = deque(maxlen=cfg.num_frames - 1)
        prev_queue = list(queue_dict[sid0])

        preds, curr_bev = model(imgs, prev_bev_queue=prev_queue, intrins=intrins, extrinsics=extrinsics)
        loss = F.cross_entropy(preds, gt)

        queue_dict[sid0].append(curr_bev.detach())

        running_loss += loss.item()
        running_acc += pixel_accuracy(preds, gt)

    avg_loss = running_loss / max(len(loader), 1)
    avg_acc = running_acc / max(len(loader), 1)
    print(f"[Val][E{epoch}] loss={avg_loss:.4f} acc={avg_acc:.4f}")
    return avg_loss, avg_acc


# =========================================================
# 1) 一个可跑通的 Dataset 模板（你换成真实数据读取）
# =========================================================
class DummyFastBEVDataset(Dataset):
    """
    用于跑通训练/验证 pipeline 的 dummy 数据集：
    - imgs: (N,3,H,W)
    - intrins: (N,3,3), extrinsics:(N,4,4)
    - gt_bev: (X,Y) 取值 0..num_classes-1  (用于CE loss)
    - scene_id / frame_idx: 用于时序队列
    """
    def __init__(self, cfg, num_scenes=8, frames_per_scene=20, seed=0):
        super().__init__()
        self.cfg = cfg
        self.num_scenes = num_scenes
        self.frames_per_scene = frames_per_scene
        self.total = num_scenes * frames_per_scene

        g = torch.Generator().manual_seed(seed)
        self.g = g

        # 计算 BEV 网格大小（与你 FastBEV 里一致）
        self.nx = int((cfg.pc_range[3] - cfg.pc_range[0]) / cfg.voxel_size[0])
        self.ny = int((cfg.pc_range[4] - cfg.pc_range[1]) / cfg.voxel_size[1])

        # 固定标定（符合你 LUT “静态标定” 假设）
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = cfg.img_w / 2.0, cfg.img_h / 2.0

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        scene_id = idx // self.frames_per_scene
        frame_idx = idx % self.frames_per_scene

        # imgs: (N,3,H,W)
        imgs = torch.randn(self.cfg.num_cams, 3, self.cfg.img_h, self.cfg.img_w, generator=self.g)

        # intrinsics: (N,3,3)
        intrins = torch.eye(3).unsqueeze(0).repeat(self.cfg.num_cams, 1, 1)
        intrins[:, 0, 0] = self.fx
        intrins[:, 1, 1] = self.fy
        intrins[:, 0, 2] = self.cx
        intrins[:, 1, 2] = self.cy

        # extrinsics: (N,4,4) 这里简化为单位阵；真实数据用真实外参
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.cfg.num_cams, 1, 1)

        # gt_bev: (X,Y)，10类（对应你的 head 输出10通道）
        gt_bev = torch.randint(low=0, high=10, size=(self.nx, self.ny), dtype=torch.long, generator=self.g)

        return {
            "imgs": imgs,
            "intrins": intrins,
            "extrinsics": extrinsics,
            "gt_bev": gt_bev,
            "scene_id": scene_id,
            "frame_idx": frame_idx,
        }


def collate_fn(batch):
    imgs = torch.stack([b["imgs"] for b in batch], dim=0)               # (B,N,3,H,W)
    intrins = torch.stack([b["intrins"] for b in batch], dim=0)         # (B,N,3,3)
    extrinsics = torch.stack([b["extrinsics"] for b in batch], dim=0)   # (B,N,4,4)
    gt_bev = torch.stack([b["gt_bev"] for b in batch], dim=0)           # (B,X,Y)
    scene_id = torch.tensor([b["scene_id"] for b in batch], dtype=torch.long)
    frame_idx = torch.tensor([b["frame_idx"] for b in batch], dtype=torch.long)
    return {
        "imgs": imgs,
        "intrins": intrins,
        "extrinsics": extrinsics,
        "gt_bev": gt_bev,
        "scene_id": scene_id,
        "frame_idx": frame_idx,
    }


# =========================================================
# 2) 时序队列：支持 batch 内不同 scene 混杂
# =========================================================
class TemporalQueueManager:
    """
    维护每个 scene 的历史 BEV 特征队列（长度 = num_frames-1）
    - 每个 entry 存单样本特征 (1, C, X, Y)
    - batch 输入时把不同 scene 的队列按 “lag=1..T-1” 组装成 list[(B,C,X,Y), ...]
    """
    def __init__(self, num_frames: int):
        assert num_frames >= 1
        self.hist_len = num_frames - 1
        self.queues: Dict[int, deque] = {}  # scene_id -> deque of tensors (1,C,X,Y)

    def reset_scene(self, scene_id: int):
        self.queues[scene_id] = deque(maxlen=self.hist_len)

    def get_prev_queue(self, scene_ids: torch.Tensor, frame_idxs: torch.Tensor,
                       feat_shape: Optional[Tuple[int, int, int]] = None,
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None) -> Optional[List[torch.Tensor]]:
        """
        返回 prev_bev_queue: List[ (B,C,X,Y), ... ] 长度 <= hist_len
        如果某个 sample 的历史不足，则用 0 填充（更合理 than 复制当前帧，因为当前帧还没算出来）
        """
        if self.hist_len <= 0:
            return None

        B = scene_ids.shape[0]

        # 对每个 batch 样本检查是否需要 reset
        for i in range(B):
            sid = int(scene_ids[i].item())
            fid = int(frame_idxs[i].item())
            if sid not in self.queues or fid == 0:
                self.reset_scene(sid)

        # 如果还不知道 feature shape（第一次迭代可能），先不返回 history
        if feat_shape is None:
            # 没法构造 padding
            # 这里选择：如果所有队列都为空，就返回 None
            all_empty = True
            for i in range(B):
                sid = int(scene_ids[i].item())
                if len(self.queues[sid]) > 0:
                    all_empty = False
                    break
            if all_empty:
                return None
            else:
                raise RuntimeError("Need feat_shape to build padded temporal queue.")

        C, X, Y = feat_shape
        zeros = torch.zeros((1, C, X, Y), device=device, dtype=dtype)

        # 组装每个 lag 的 batch tensor
        # lag=1 表示上一帧（队列末尾），lag=hist_len 表示最老的
        prev_list: List[torch.Tensor] = []
        for lag in range(1, self.hist_len + 1):
            per_sample = []
            for i in range(B):
                sid = int(scene_ids[i].item())
                q = self.queues[sid]
                if len(q) >= lag:
                    # q[-1] 是最近一帧，q[-lag] 是第 lag 帧前
                    per_sample.append(q[-lag])
                else:
                    per_sample.append(zeros)
            prev_list.append(torch.cat(per_sample, dim=0))  # (B,C,X,Y)
        # prev_list 是 [lag1, lag2, ...]，你的模型里是 cat(history_feats, dim=1)，顺序影响不大
        # 这里给的顺序是 [t-1, t-2, ...]
        return prev_list

    def push(self, scene_ids: torch.Tensor, frame_idxs: torch.Tensor, curr_bev: torch.Tensor):
        """
        curr_bev: (B,C,X,Y)，逐样本 push 到对应 scene 的队列
        """
        B = scene_ids.shape[0]
        for i in range(B):
            sid = int(scene_ids[i].item())
            fid = int(frame_idxs[i].item())
            if sid not in self.queues or fid == 0:
                self.reset_scene(sid)
            self.queues[sid].append(curr_bev[i:i+1].detach())


# =========================================================
# 3) 指标（简单像素准确率）
# =========================================================
@torch.no_grad()
def pixel_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    logits: (B,C,X,Y)
    target: (B,X,Y)
    """
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


# =========================================================
# 4) 训练 / 验证循环
# =========================================================
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    use_amp: bool,
                    num_frames: int,
                    epoch: int,
                    grad_clip: float = 0.0):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    tqm = TemporalQueueManager(num_frames=num_frames)

    running_loss = 0.0
    running_acc = 0.0

    for it, batch in enumerate(loader):
        imgs = batch["imgs"].to(device)             # (B,N,3,H,W)
        intrins = batch["intrins"].to(device)       # (B,N,3,3)
        extrinsics = batch["extrinsics"].to(device) # (B,N,4,4)
        gt = batch["gt_bev"].to(device)             # (B,X,Y)
        scene_id = batch["scene_id"].to(device)
        frame_idx = batch["frame_idx"].to(device)

        # ---- 先拿 history queue（第一次迭代可以先不给 history）
        prev_queue = None

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            # 第一次 forward 会触发 LUT 构建，所以必须传 intrins/extrinsics
            preds, curr_bev = model(imgs, prev_bev_queue=None, intrins=intrins, extrinsics=extrinsics)

            # 拿到 curr_bev shape 后，重新组装 prev_queue，再跑一遍（可选）
            # 为了“严格符合时序融合”，我们在第一个 batch 后开始使用 history：
            C, X, Y = curr_bev.shape[1], curr_bev.shape[2], curr_bev.shape[3]
            prev_queue = tqm.get_prev_queue(scene_id, frame_idx, feat_shape=(C, X, Y),
                                            device=device, dtype=curr_bev.dtype)

            if prev_queue is not None:
                preds, curr_bev = model(imgs, prev_bev_queue=prev_queue, intrins=intrins, extrinsics=extrinsics)

            loss = F.cross_entropy(preds, gt)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 更新时序队列（stop-grad）
        tqm.push(scene_id, frame_idx, curr_bev)

        running_loss += float(loss.item())
        running_acc += pixel_acc(preds.detach(), gt)

        if (it + 1) % 10 == 0:
            print(f"[Train][E{epoch}] it {it+1}/{len(loader)} "
                  f"loss {running_loss/(it+1):.4f} acc {running_acc/(it+1):.4f}")

    return running_loss / max(len(loader), 1), running_acc / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model: nn.Module,
                       loader: DataLoader,
                       device: torch.device,
                       num_frames: int,
                       epoch: int):
    model.eval()

    tqm = TemporalQueueManager(num_frames=num_frames)

    running_loss = 0.0
    running_acc = 0.0

    for it, batch in enumerate(loader):
        imgs = batch["imgs"].to(device)
        intrins = batch["intrins"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        gt = batch["gt_bev"].to(device)
        scene_id = batch["scene_id"].to(device)
        frame_idx = batch["frame_idx"].to(device)

        # 先跑一次得到 curr_bev shape（如果已有队列，会用队列再跑一遍）
        preds, curr_bev = model(imgs, prev_bev_queue=None, intrins=intrins, extrinsics=extrinsics)
        C, X, Y = curr_bev.shape[1], curr_bev.shape[2], curr_bev.shape[3]
        prev_queue = tqm.get_prev_queue(scene_id, frame_idx, feat_shape=(C, X, Y),
                                        device=device, dtype=curr_bev.dtype)

        if prev_queue is not None:
            preds, curr_bev = model(imgs, prev_bev_queue=prev_queue, intrins=intrins, extrinsics=extrinsics)

        loss = F.cross_entropy(preds, gt)

        tqm.push(scene_id, frame_idx, curr_bev)

        running_loss += float(loss.item())
        running_acc += pixel_acc(preds, gt)

    avg_loss = running_loss / max(len(loader), 1)
    avg_acc = running_acc / max(len(loader), 1)
    print(f"[Val][E{epoch}] loss {avg_loss:.4f} acc {avg_acc:.4f}")
    return avg_loss, avg_acc


# =========================================================
# 5) Main：组装所有东西
# =========================================================
def main():
    # ---- 你原始配置 ----
    cfg = FastBEVConfig()
    
    # 训练超参（你可以放进 cfg）
    max_epochs = 3
    batch_size = 2
    lr = 1e-4
    weight_decay = 1e-2
    use_amp = True
    grad_clip = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 模型 ----
    model = FastBEV(cfg).to(device)

    # ---- 数据 ----
    # 时序训练建议 shuffle=False（保证同scene连续帧顺序）；真实工程可以用 sampler 做按scene分组
    train_ds = DummyFastBEVDataset(cfg, num_scenes=12, frames_per_scene=20, seed=0)
    val_ds = DummyFastBEVDataset(cfg, num_scenes=4, frames_per_scene=20, seed=123)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, drop_last=False
    )

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- 训练 / 验证 ----
    os.makedirs("checkpoints", exist_ok=True)
    best_val = 1e9

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            use_amp=use_amp, num_frames=cfg.num_frames, epoch=epoch, grad_clip=grad_clip
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, device, num_frames=cfg.num_frames, epoch=epoch
        )

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "best_val": best_val,
            }, "checkpoints/fastbev_best.pt")
            print(f"✅ Saved best checkpoint: checkpoints/fastbev_best.pt (val_loss={val_loss:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()


