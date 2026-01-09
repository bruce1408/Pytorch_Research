import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================
# 1. 全局配置 (Configuration)
# =========================================================
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
    
    # --- 输入参数 ---
    num_cams = 6          # 相机数量（通常为6个，覆盖360度）
    img_h = 256           # 输入图像高度（像素）
    img_w = 704           # 输入图像宽度（像素）
    
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
        self.grid_size = grid_size  # [X, Y, Z] 体素网格尺寸，如 [256, 256, 6]
        self.nx, self.ny, self.nz = grid_size
        
        # 预先构建体素中心坐标（世界坐标系）
        # 这些坐标在初始化时计算一次，之后保持不变
        # Shape: (N_voxels, 3)，其中 N_voxels = nx * ny * nz
        self.register_buffer("voxel_centers", self._create_voxel_grid(), persistent=False)
        
        # 存储查找表（Look-Up-Table）的字典
        # 每个尺度对应一个LUT
        # LUT内容：每个体素对应 [camera_id, u, v]
        #   - camera_id: 哪个相机能看到这个体素
        #   - u, v: 该体素在该相机特征图中的像素坐标
        self.lut_dict = nn.ParameterDict()
        self.has_lut_computed = False  # 标记LUT是否已计算

    def _create_voxel_grid(self):
        """
        创建体素网格的中心坐标
        
        功能：
        - 根据 pc_range 和 voxel_size 生成所有体素的中心点坐标
        - 这些坐标是世界坐标系下的3D点
        
        Returns:
            voxel_centers: (N_voxels, 3) - 所有体素中心的世界坐标 [x, y, z]
        """
        pc_range = self.cfg.pc_range
        voxel_size = self.cfg.voxel_size
        
        # 生成每个维度的坐标序列
        # 例如：x方向从 -51.2+0.2 到 51.2-0.2，步长0.4
        xs = torch.arange(self.nx, dtype=torch.float32) * voxel_size[0] + pc_range[0] + voxel_size[0]/2
        ys = torch.arange(self.ny, dtype=torch.float32) * voxel_size[1] + pc_range[1] + voxel_size[1]/2
        zs = torch.arange(self.nz, dtype=torch.float32) * voxel_size[2] + pc_range[2] + voxel_size[2]/2
        
        # 使用 meshgrid 生成3D网格
        # indexing='ij' 表示第一个维度是x，第二个是y，第三个是z
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
        
        # 展平为 (N_voxels, 3) 的形状
        # 每一行是一个体素中心的 [x, y, z] 坐标
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

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
        num_voxels = self.voxel_centers.shape[0]  # 总体素数，如 256*256*6 = 393216
        num_cams = intrins.shape[0]  # 相机数量，如 6
        
        # ========== 步骤1：世界坐标系 -> 相机坐标系 ==========
        # 将体素中心点转换为齐次坐标 (N_voxels, 4)
        voxels_homo = torch.cat([self.voxel_centers, torch.ones(num_voxels, 1, device=device)], dim=-1)
        
        # 矩阵乘法：P_cam = T_extrinsic * P_world
        # (N_cams, 4, 4) @ (N_voxels, 4)^T -> (N_cams, 4, N_voxels)
        # 然后转置为 (N_cams, N_voxels, 4)
        points_cam = torch.matmul(extrinsics, voxels_homo.t()).permute(0, 2, 1)
        
        # 提取深度信息（Z坐标）
        # 只保留相机前方的点（深度 > 0.1，避免数值不稳定）
        depth = points_cam[..., 2]  # (N_cams, N_voxels)
        valid_mask_z = depth > 0.1
        
        # ========== 步骤2：相机坐标系 -> 图像平面 ==========
        # 投影公式：P_img = K * P_cam
        # K是内参矩阵，将3D点投影到2D图像平面
        points_cam_xyz = points_cam[..., :3]  # 只取前3维（x, y, z）
        
        # 批量矩阵乘法：对每个相机分别计算
        # (N_cams, 3, 3) @ (N_cams, 3, N_voxels) -> (N_cams, 3, N_voxels)
        points_img = torch.bmm(intrins, points_cam_xyz.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 归一化得到像素坐标 (u, v)
        # u = X/Z, v = Y/Z（透视投影）
        u = points_img[..., 0] / (points_img[..., 2] + 1e-6)  # (N_cams, N_voxels)
        v = points_img[..., 1] / (points_img[..., 2] + 1e-6)  # (N_cams, N_voxels)
        
        # ========== 步骤3：为每个尺度生成LUT ==========
        for scale_idx, (H, W) in enumerate(img_shapes):
            # 将像素坐标缩放到特征图尺度
            # 例如：原图坐标 (u, v) -> 特征图坐标 (u/4, v/4) for 1/4 scale
            stride = self.cfg.scales[scale_idx]
            u_feat = u / stride  # 特征图上的u坐标
            v_feat = v / stride  # 特征图上的v坐标
            
            # 检查边界：确保投影点在特征图范围内
            valid_mask_u = (u_feat >= 0) & (u_feat < W)
            valid_mask_v = (v_feat >= 0) & (v_feat < H)
            # 综合有效性：深度有效 + u有效 + v有效
            valid_mask = valid_mask_z & valid_mask_u & valid_mask_v  # (N_cams, N_voxels)
            
            # ========== 步骤4：多视角融合策略 ==========
            # Fast-BEV策略："directly adopt the first encountered view"
            # 为每个体素选择第一个能看到的相机
            
            # 初始化LUT：每个体素对应 [cam_idx, u, v]
            # -1 表示无效（该体素没有被任何相机看到）
            lut = torch.full((num_voxels, 3), -1, dtype=torch.long, device=device)
            
            # 遍历每个相机，填充LUT
            for cam_i in range(num_cams):
                mask = valid_mask[cam_i]  # 当前相机能看到哪些体素
                
                # 找到当前相机能看到的所有体素索引
                valid_indices = torch.where(mask)[0]
                
                # 只填充那些还没有被填充的体素（实现"first encountered"策略）
                not_filled = (lut[valid_indices, 0] == -1)
                indices_to_fill = valid_indices[not_filled]
                
                if len(indices_to_fill) > 0:
                    # 获取这些体素在特征图中的整数坐标
                    u_vals = u_feat[cam_i, indices_to_fill].long()
                    v_vals = v_feat[cam_i, indices_to_fill].long()
                    
                    # 填入LUT：[相机ID, u坐标, v坐标]
                    lut[indices_to_fill, 0] = cam_i
                    lut[indices_to_fill, 1] = u_vals
                    lut[indices_to_fill, 2] = v_vals
            
            # 将LUT存储为不可训练的参数（不会在反向传播中更新）
            self.lut_dict[f"scale_{scale_idx}"] = nn.Parameter(lut, requires_grad=False)
            
        self.has_lut_computed = True  # 标记LUT已计算完成

    def forward(self, features_list):
        """
        【算法核心】推理阶段：快速查表投影
        
        这是Fast-BEV速度快的核心：不需要重复计算投影关系，直接查表！
        
        算法步骤：
        1. 将特征图展平为一维数组
        2. 使用LUT中的索引直接gather特征
        3. 重塑为3D体素特征
        
        Args:
            features_list: List of (B, N, C, H, W)
                - B: batch size
                - N: 相机数量
                - C: 特征通道数（不同尺度可能不同）
                - H, W: 特征图高度和宽度
        
        Returns:
            voxel_feats_list: List of (B, C, X, Y, Z)
                每个尺度对应的3D体素特征
        """
        if not self.has_lut_computed:
            raise RuntimeError("LUT not computed! Call compute_lut() first.")
            
        B = features_list[0].shape[0]
        voxel_feats_list = []
        
        # 对每个尺度分别进行投影
        for i, feat in enumerate(features_list):
            # feat: (B, N, C, H, W)
            N, C, H, W = feat.shape[1:]
            
            # 取出对应尺度的LUT
            lut = self.lut_dict[f"scale_{i}"]  # (N_voxels, 3) -> [cam_idx, u, v]
            
            # ========== 步骤1：展平特征图 ==========
            # 将 (B, N, C, H, W) 转换为 (B, N*H*W, C)
            # 这样每个像素的特征可以用一个线性索引访问
            feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(B, -1, C)
            # 解释：permute(0,1,3,4,2) -> (B, N, H, W, C)
            #       reshape -> (B, N*H*W, C)
            
            # ========== 步骤2：计算gather索引 ==========
            # LUT中存储的是 [cam_idx, u, v]，需要转换为线性索引
            # 线性索引公式：linear_idx = cam_idx * (H*W) + v * W + u
            
            # 找出有效的体素（LUT中cam_idx != -1的）
            valid_mask = (lut[:, 0] != -1)  # (N_voxels,) bool
            valid_lut = lut[valid_mask]  # (M, 3)，M是有效体素数
            
            cam_idx = valid_lut[:, 0]  # (M,) 相机索引
            u_idx = valid_lut[:, 1]    # (M,) u坐标
            v_idx = valid_lut[:, 2]    # (M,) v坐标
            
            # 计算线性索引
            flat_indices = cam_idx * (H * W) + v_idx * W + u_idx  # (M,)
            
            # ========== 步骤3：查表gather特征 ==========
            # 输出容器：(B, N_voxels, C)
            out_voxels = torch.zeros((B, self.nx * self.ny * self.nz, C), 
                                     dtype=feat.dtype, device=feat.device)
            
            # 对每个batch分别gather（因为索引是2D的，需要逐batch处理）
            for b in range(B):
                # 从展平的特征图中gather对应的特征
                # feat_flat[b]: (N*H*W, C)
                # flat_indices: (M,) - 要gather的索引
                # selected_feats: (M, C) - gather到的特征
                selected_feats = feat_flat[b, flat_indices, :]
                
                # 将gather到的特征填入对应体素位置
                out_voxels[b, valid_mask, :] = selected_feats
            
            # ========== 步骤4：重塑为3D体素特征 ==========
            # 将 (B, N_voxels, C) 重塑为 (B, C, X, Y, Z)
            # 注意：voxel_centers是按 (X, Y, Z) 顺序生成的，所以reshape也要对应
            out_voxels = out_voxels.reshape(B, self.nx, self.ny, self.nz, C).permute(0, 4, 1, 2, 3)
            # reshape -> (B, X, Y, Z, C)
            # permute -> (B, C, X, Y, Z)
            
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
    def __init__(self, cfg, in_channels_list):
        super().__init__()
        self.cfg = cfg
        
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
        self.nz = 6  # Z维度大小
        self.s2c_dim = self.total_in_channels * self.nz
        
        # ========== MFCF: 时序融合 ==========
        # 融合多帧信息：当前帧 + 前3帧
        # 输入维度：s2c_dim * num_frames
        # 例如：2688 * 4 = 10752
        self.fusion_dim = self.s2c_dim * cfg.num_frames
        
        # ========== 2D BEV编码器 ==========
        # 使用2D卷积对融合后的特征进行编码和降维
        # 输入：(B, fusion_dim, X, Y)
        # 输出：(B, bev_dim, X, Y)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.fusion_dim, cfg.bev_dim, 3, padding=1),
            nn.BatchNorm2d(cfg.bev_dim),
            nn.ReLU(),
            nn.Conv2d(cfg.bev_dim, cfg.bev_dim, 3, padding=1),
            nn.BatchNorm2d(cfg.bev_dim),
            nn.ReLU()
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
        # ========== MSCF: 多尺度特征融合 ==========
        # 在通道维度拼接所有尺度的特征
        # cat([(B, C1, X, Y, Z), (B, C2, X, Y, Z), (B, C3, X, Y, Z)], dim=1)
        # -> (B, C1+C2+C3, X, Y, Z)
        voxel_feat = torch.cat(voxel_feats_list, dim=1)
        
        # ========== S2C: Space to Channel ==========
        # 将Z维度压缩到通道维度
        B, C, X, Y, Z = voxel_feat.shape
        # permute(0,1,4,2,3): (B, C, X, Y, Z) -> (B, C, Z, X, Y)
        # reshape: (B, C, Z, X, Y) -> (B, C*Z, X, Y)
        bev_feat = voxel_feat.permute(0, 1, 4, 2, 3).reshape(B, C*Z, X, Y)
        
        # ========== MFCF: 时序融合 ==========
        # 融合当前帧和历史帧的特征
        current_feat = bev_feat  # (B, C*Z, X, Y)
        
        if prev_bev_feats_queue is not None and len(prev_bev_feats_queue) > 0:
            # 如果有历史帧，在通道维度拼接
            # 队列：[T-3帧, T-2帧, T-1帧]
            # cat([T-3, T-2, T-1], dim=1) -> (B, 3*C*Z, X, Y)
            history_feats = torch.cat(prev_bev_feats_queue, dim=1)
            # 拼接当前帧：cat([历史, 当前], dim=1) -> (B, 4*C*Z, X, Y)
            fused_feat = torch.cat([history_feats, current_feat], dim=1)
        else:
            # 如果没有历史帧（第一帧），复制当前帧填充
            # 保持通道维度一致，便于送入卷积层
            fused_feat = torch.cat([current_feat] * self.cfg.num_frames, dim=1)
            
        # ========== 2D编码器 ==========
        # 使用2D卷积对融合特征进行编码和降维
        # (B, fusion_dim, X, Y) -> (B, bev_dim, X, Y)
        out = self.encoder(fused_feat)
        
        # 返回编码后的特征和当前帧特征（用于更新队列）
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
        self.bev_encoder = EfficientBEVEncoder(cfg, cfg.feat_dims)
        
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
# 6. 运行 Demo
# =========================================================
if __name__ == "__main__":
    # 创建配置和模型
    cfg = FastBEVConfig()
    model = FastBEV(cfg)
    
    # ========== 模拟输入数据 ==========
    B, N = 1, 6  # batch_size=1, 6个相机
    imgs = torch.randn(B, N, 3, 256, 704)  # 随机图像数据
    
    # ========== 模拟相机标定参数 ==========
    # 内参矩阵 K (3x3)
    # [fx  0  cx]
    # [0  fy  cy]
    # [0   0   1]
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    intrins[:, :, 0, 0] = 500  # fx: x方向焦距
    intrins[:, :, 1, 1] = 500  # fy: y方向焦距
    intrins[:, :, 0, 2] = 352  # cx: 主点x坐标（图像中心）
    intrins[:, :, 1, 2] = 128  # cy: 主点y坐标（图像中心）
    
    # 外参矩阵（世界坐标系到相机坐标系）
    # 这里简化为单位矩阵（实际应用中需要真实的标定数据）
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    
    # ========== 第一次前向传播（构建LUT + 推理） ==========
    print("--- 1. First Pass (Build LUT & Inference) ---")
    preds, curr_bev = model(imgs, intrins=intrins, extrinsics=extrinsics)
    print(f"Output Shape: {preds.shape}")  # 期望：(1, 10, 256, 256)
    
    # ========== 第二次前向传播（使用LUT快速推理） ==========
    print("\n--- 2. Second Pass (Fast Inference using LUT) ---")
    # 模拟时序融合：将前3帧的BEV特征放入队列
    queue = [curr_bev, curr_bev, curr_bev]  # 模拟 T-3, T-2, T-1 帧
    preds_2, curr_bev_2 = model(imgs, prev_bev_queue=queue)
    print(f"Output Shape (Temporal): {preds_2.shape}")
    print("✅ Fast-BEV implemented successfully!")