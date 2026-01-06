import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. 配置参数 (Configuration)
# 对应论文 Experiment 部分的设置 (nuScenes)
# ==============================================================================
class M2BEVConfig:
    # 3D 空间范围 (X_min, Y_min, Z_min, X_max, Y_max, Z_max)
    # 论文 Sec 3.1: Range ±50m, Z: -1~5m
    pc_range = [-50.0, -50.0, -1.0, 50.0, 50.0, 5.0]
    
    # Voxel 大小
    # 论文 Sec 3.1: (0.25m, 0.25m, 0.5m)
    voxel_size = [0.25, 0.25, 0.5]
    
    # 计算 Grid 尺寸: 100m / 0.25m = 400
    # X, Y, Z = (400, 400, 12)
    grid_size = [
        int((pc_range[3] - pc_range[0]) / voxel_size[0]),
        int((pc_range[4] - pc_range[1]) / voxel_size[1]),
        int((pc_range[5] - pc_range[2]) / voxel_size[2])
    ]
    
    # 输入图像参数 (假设)
    num_cams = 6
    img_size = (900, 1600) # H, W
    
    # 特征维度
    feat_dim = 64        # 2D 特征维度 (C)
    bev_dim = 128        # BEV 特征维度
    num_classes_det = 10 # 检测类别数
    num_classes_seg = 2  # 分割类别数 (Drivable area, Lane)

cfg = M2BEVConfig()

# ==============================================================================
# 2. 简化的 2D 图像编码器 (Image Encoder)
# ==============================================================================
class SimpleImageEncoder(nn.Module):
    """
    论文中使用 ResNet-50/101 + FPN。
    这里用简化的 3 层卷积模拟提取特征，输出 stride=4 的特征图。
    论文 Sec 3.1: "FPN... fuse them to form a tensor F... shape H/4 x W/4 x C"
    """
    def __init__(self, in_channels=3, out_channels=64):
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), # /2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),  # 特征融合
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B*N, 3, H, W)
        return self.net(x)

# ==============================================================================
# 3. 核心：Uniform 2D->3D 投影 (View Transformation)
# ==============================================================================
class UniformViewTransformer(nn.Module):
    """
    论文核心贡献 Sec 3.2: Efficient 2D->3D Projection
    与 LSS 不同，M2BEV 不预测深度分布，而是假设射线上的深度是均匀的 (Uniform)。
    这意味着：投影到同一像素的所有 Voxels 共享相同的图像特征。
    """
    def __init__(self, config):
        super().__init__()
        self.grid_size = config.grid_size
        self.pc_range = config.pc_range
        self.voxel_size = config.voxel_size
        
        # 预先生成 Voxel 的物理坐标 (X, Y, Z)
        # Shape: (1, 3, Z, Y, X)
        self.register_buffer('voxel_coords', self._create_voxel_grid())

    def _create_voxel_grid(self):
        # 生成网格中心坐标
        coords_x = torch.linspace(
            self.pc_range[0] + self.voxel_size[0]/2, 
            self.pc_range[3] - self.voxel_size[0]/2, 
            self.grid_size[0]
        )
        coords_y = torch.linspace(
            self.pc_range[1] + self.voxel_size[1]/2, 
            self.pc_range[4] - self.voxel_size[1]/2, 
            self.grid_size[1]
        )
        coords_z = torch.linspace(
            self.pc_range[2] + self.voxel_size[2]/2, 
            self.pc_range[5] - self.voxel_size[2]/2, 
            self.grid_size[2]
        )
        
        # Note: indexing 'ij' gives (Z, Y, X) order for meshgrid
        coords_z, coords_y, coords_x = torch.meshgrid(coords_z, coords_y, coords_x, indexing='ij')
        
        # Stack to (3, Z, Y, X) -> (x, y, z)
        grid = torch.stack([coords_x, coords_y, coords_z], dim=0)
        return grid.unsqueeze(0) # (1, 3, Z, Y, X)

    def forward(self, img_feats, mats_dict):
        """
        Args:
            img_feats: (B*N, C, H_f, W_f) 图像特征
            mats_dict: 包含内参 intrinsic 和外参 extrinsic
                - ego2cam: (B*N, 4, 4) 世界/自车坐标系到相机坐标系
                - cam2img: (B*N, 4, 4) 相机到像素坐标系 (内参)
        Returns:
            voxel_feat: (B, C, Z, Y, X) 填充后的 3D 体素特征
        """
        B_N, C, Hf, Wf = img_feats.shape
        # 假设 B=1, N=6 -> B_N=6
        
        # 1. 准备 Voxel 坐标
        # (1, 3, Z, Y, X) -> (B*N, 3, Z*Y*X)
        voxel_coords = self.voxel_coords.repeat(B_N, 1, 1, 1, 1)
        _, _, Z, Y, X = voxel_coords.shape
        voxel_points = voxel_coords.view(B_N, 3, -1)
        
        # 转为齐次坐标 (B*N, 4, N_points)
        ones = torch.ones((B_N, 1, voxel_points.shape[-1]), device=img_feats.device)
        voxel_points = torch.cat([voxel_points, ones], dim=1)
        
        # 2. 逆向投影 (Voxel -> Image)
        # P_img = K * T * P_world
        # 组合变换矩阵: (B*N, 4, 4)
        # 注意：这里简化处理，假设 mats_dict 已经包含了从 ego 到 image 的完整变换
        # 实际代码可能需要 ego2lidar @ lidar2cam @ intrinsic
        proj_mat = torch.bmm(mats_dict['cam2img'], mats_dict['ego2cam']) 
        
        img_points = torch.bmm(proj_mat, voxel_points) # (B*N, 4, N_points)
        
        # 3. 归一化坐标 & 深度过滤
        eps = 1e-5
        depth = img_points[:, 2:3, :]
        u = img_points[:, 0:1, :] / (depth + eps)
        v = img_points[:, 1:2, :] / (depth + eps)
        
        # 归一化到 [-1, 1] 用于 grid_sample
        # 注意：这里的 Hf, Wf 是特征图尺寸，需要根据下采样倍率调整
        # M2BEV 论文中 feature map 是 input 的 1/4
        u_norm = (u / (Wf - 1) * 2) - 1
        v_norm = (v / (Hf - 1) * 2) - 1
        
        # 4. 生成采样 Grid
        # (B*N, N_points, 2)
        sample_grid = torch.cat([u_norm, v_norm], dim=1).permute(0, 2, 1)
        sample_grid = sample_grid.view(B_N, Z * Y * X, 1, 2) # grid_sample 需要 4D/5D 输入
        
        # 5. 采样图像特征 (Grid Sample)
        # Input: (B*N, C, Hf, Wf)
        # Grid:  (B*N, Z*Y*X, 1, 2)
        sampled_feat = F.grid_sample(
            img_feats, 
            sample_grid, 
            align_corners=True, 
            padding_mode='zeros' # 投影到图像外的 Voxel 填充 0
        ) 
        # Output: (B*N, C, Z*Y*X, 1) -> (B*N, C, Z*Y*X)
        sampled_feat = sampled_feat.view(B_N, C, -1)
        
        # 6. 处理深度有效性
        # 只保留相机前方的点 (depth > 0)
        valid_mask = (depth > 0.1).view(B_N, 1, -1)
        sampled_feat = sampled_feat * valid_mask
        
        # 7. 多视角融合 (Multi-View Aggregation)
        # 论文 Sec 3.1: "The voxel feature contains image features with all the views"
        # 这里使用简单的 Mean Pooling 或 Max Pooling 将 6 个视角的特征融合到一个 Voxel
        # Reshape to (B, N, C, N_voxels) -> Sum/Mean over N
        B = B_N // 6 # 假设 N=6
        sampled_feat = sampled_feat.view(B, 6, C, -1)
        valid_mask = valid_mask.view(B, 6, 1, -1)
        
        # 加权平均 (避免除以0)
        fused_feat = torch.sum(sampled_feat, dim=1) / (torch.sum(valid_mask, dim=1) + 1e-6)
        
        # 8. 恢复 Voxel 形状
        # (B, C, Z, Y, X)
        voxel_feat = fused_feat.view(B, C, Z, Y, X)
        
        return voxel_feat

# ==============================================================================
# 4. Efficient BEV Encoder (S2C + Conv)
# ==============================================================================
class EfficientBEVEncoder(nn.Module):
    """
    论文 Sec 3.3: Efficient BEV Encoder
    核心操作: "Spatial to Channel (S2C)"
    把 4D Tensor (C, Z, Y, X) 变换为 3D Tensor (Z*C, Y, X)
    然后用 2D 卷积降维，而不是用昂贵的 3D 卷积。
    """
    def __init__(self, in_channels, out_channels, z_dim):
        super().__init__()
        self.z_dim = z_dim
        # S2C 后的通道数 = C * Z
        s2c_channels = in_channels * z_dim
        
        self.net = nn.Sequential(
            # 论文中提到使用 2D 卷积来处理压缩后的特征
            nn.Conv2d(s2c_channels, s2c_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(s2c_channels // 2),
            nn.ReLU(),
            nn.Conv2d(s2c_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, voxel_feat):
        """
        Args:
            voxel_feat: (B, C, Z, Y, X)
        Returns:
            bev_feat: (B, C_out, Y, X)
        """
        B, C, Z, Y, X = voxel_feat.shape
        
        # 1. Spatial to Channel (S2C)
        # View: (B, C*Z, Y, X)
        x = voxel_feat.view(B, C * Z, Y, X)
        
        # 2. 2D Convolutions
        bev_feat = self.net(x)
        
        return bev_feat

# ==============================================================================
# 5. Detection Head & Dynamic Box Assignment Logic
# ==============================================================================
class DetectionHead(nn.Module):
    """
    论文 Sec 3.1 Part 4: 3D Detection Head
    直接采用 PointPillars 的 Head 结构 (三个并行 1x1 卷积)
    预测: Class, Box, Direction
    """
    def __init__(self, in_channels, n_anchors, n_classes):
        super().__init__()
        # 3 parallel 1x1 convs
        self.conv_cls = nn.Conv2d(in_channels, n_anchors * n_classes, 1)
        self.conv_box = nn.Conv2d(in_channels, n_anchors * 7, 1) # (x,y,z,w,l,h,theta)
        self.conv_dir = nn.Conv2d(in_channels, n_anchors * 2, 1) # Direction bins

    def forward(self, x):
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        dir_preds = self.conv_dir(x)
        return cls_preds, box_preds, dir_preds

class DynamicBoxAssignment:
    """
    论文 Sec 3.3: Dynamic Box Assignment
    这通常是一个 Loss 计算过程中的逻辑。
    灵感来自 FreeAnchor。不使用固定的 IoU 阈值分配 Anchor。
    逻辑描述:
    1. 为每个 GT box 选择 Top-K 个 IoU 最高的 Anchors (Candidate Bag).
    2. 计算每个 Candidate 的 Classification Score 和 Localization Score.
    3. 综合得分 Mean-Max 策略来决定正样本。
    """
    def assign(self, anchors, gt_boxes, cls_preds, box_preds):
        # 这是一个简化的逻辑示意，完整的 FreeAnchor 实现非常复杂
        # 此处展示论文核心思想
        
        # 1. 计算 Anchors 和 GT 的 IoU
        # ious = compute_iou(anchors, gt_boxes)
        
        # 2. 预选 (Pre-selection): 每个 GT 选 Top-50 anchors
        # candidate_mask = topk_mask(ious, k=50)
        
        # 3. 计算得分
        # cls_score = sigmoid(cls_preds)
        # loc_score = exp(- smooth_l1(box_preds, gt_boxes))
        
        # 4. 联合得分 Q = cls_score * loc_score
        
        # 5. 更新 Loss 权重 (根据 Q 值)
        # return assigned_targets, weights
        pass

# ==============================================================================
# 6. Segmentation Head & BEV Centerness
# ==============================================================================
class SegmentationHead(nn.Module):
    """
    论文 Sec 3.1 Part 5: BEV Segmentation Head
    结构: 4 个 3x3 卷积 + 1 个 1x1 卷积
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # Final projection
            nn.Conv2d(in_channels, out_channels, 1) 
        )

    def forward(self, x):
        return self.net(x)

def compute_bev_centerness(bev_dims):
    """
    论文 Sec 3.3 Formula (2): BEV Centerness
    用于 Segmentation Loss 的加权。
    思想：远处的物体像素少，需要更大的权重。
    Centerness = 1 + sqrt( (dist_to_center) / (max_dist) )
    Range: [1, 2]
    """
    H, W = bev_dims
    
    # 生成坐标网格
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # 中心点
    yc, xc = H / 2.0, W / 2.0
    
    # 计算距离平方
    dist_sq = (xs - xc)**2 + (ys - yc)**2
    max_dist_sq = (max(xs.flatten()) - xc)**2 + (max(ys.flatten()) - yc)**2
    
    # 公式实现
    centerness = 1.0 + torch.sqrt(dist_sq / max_dist_sq)
    
    return centerness # (H, W)

# ==============================================================================
# 7. M2BEV 整体网络 (Main Model)
# ==============================================================================
class M2BEV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # 1. 2D Image Encoder (Backbone + FPN simplified)
        self.img_encoder = SimpleImageEncoder(out_channels=config.feat_dim)
        
        # 2. View Transformer (Uniform Projection)
        self.view_transformer = UniformViewTransformer(config)
        
        # 3. Efficient BEV Encoder (S2C)
        self.bev_encoder = EfficientBEVEncoder(
            in_channels=config.feat_dim, 
            out_channels=config.bev_dim,
            z_dim=config.grid_size[2]
        )
        
        # 4. Heads
        self.det_head = DetectionHead(
            in_channels=config.bev_dim, 
            n_anchors=2, # 假设每个位置2个anchor
            n_classes=config.num_classes_det
        )
        
        self.seg_head = SegmentationHead(
            in_channels=config.bev_dim,
            out_channels=config.num_classes_seg
        )
        
        # Pre-compute Centerness map for loss (buffer)
        self.register_buffer(
            'centerness_map', 
            compute_bev_centerness((config.grid_size[1], config.grid_size[0]))
        )

    def forward(self, imgs, mats_dict):
        """
        imgs: (B, N, 3, H, W)
        mats_dict: camera matrices
        """
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        
        # 1. Extract 2D Features
        # x: (B*N, C_feat, H/4, W/4)
        img_feats = self.img_encoder(imgs)
        
        # 2. 2D -> 3D Projection (Uniform)
        # voxel_feat: (B, C_feat, Z, Y, X)
        voxel_feat = self.view_transformer(img_feats, mats_dict)
        
        # 3. BEV Encoding (S2C + Conv)
        # bev_feat: (B, C_bev, Y, X)
        bev_feat = self.bev_encoder(voxel_feat)
        
        # 4. Task Heads
        det_output = self.det_head(bev_feat)
        seg_output = self.seg_head(bev_feat)
        
        return {
            'det_preds': det_output, # (cls, box, dir)
            'seg_preds': seg_output,  # (B, N_seg, Y, X)
            'bev_feat': bev_feat
        }

    def compute_seg_loss(self, seg_preds, seg_targets):
        """
        论文 Sec 3.4: L_seg = Dice + BCE
        并应用 BEV Centerness Re-weighting
        """
        # seg_preds: (B, C, H, W)
        # seg_targets: (B, C, H, W)
        
        # 加权 BCE Loss
        # weight shape (1, 1, H, W)
        weight = self.centerness_map.view(1, 1, *self.centerness_map.shape)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            seg_preds, seg_targets, weight=weight, reduction='mean'
        )
        
        # Dice Loss (Simplified)
        probs = torch.sigmoid(seg_preds)
        intersection = (probs * seg_targets).sum()
        union = probs.sum() + seg_targets.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        
        return bce_loss + dice_loss

# ==============================================================================
# 8. 测试运行 (Demo)
# ==============================================================================
if __name__ == "__main__":
    # 初始化模型
    model = M2BEV(cfg)
    print("Model initialized.")
    
    # 模拟输入数据
    B, N = 1, 6
    dummy_imgs = torch.randn(B, N, 3, 900, 1600)
    
    # 模拟矩阵 (单位阵作为占位符)
    dummy_mats = {
        'ego2cam': torch.eye(4).unsqueeze(0).repeat(B*N, 1, 1),
        'cam2img': torch.eye(4).unsqueeze(0).repeat(B*N, 1, 1)
    }
    
    # 前向推理
    outputs = model(dummy_imgs, dummy_mats)
    
    print("\nOutput Shapes:")
    print(f"BEV Feature: {outputs['bev_feat'].shape}") # Expected: (1, 128, 400, 400)
    print(f"Seg Prediction: {outputs['seg_preds'].shape}") # Expected: (1, 2, 400, 400)
    print(f"Det Cls Prediction: {outputs['det_preds'][0].shape}") # Expected: (1, 2*10, 400, 400)
    
    print("\nBEV Centerness Map:")
    print(model.centerness_map.shape)
    print(f"Range: [{model.centerness_map.min():.2f}, {model.centerness_map.max():.2f}]") # Expected ~ [1.0, 2.0]