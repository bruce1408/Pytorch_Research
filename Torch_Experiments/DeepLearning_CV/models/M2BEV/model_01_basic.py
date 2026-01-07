import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. 配置参数 (Configuration)
# 对应论文 Experiment 部分的设置 (nuScenes)
# ==============================================================================
class M2BEVConfig:
    pc_range = [-50.0, -50.0, -1.0, 50.0, 50.0, 5.0]
    voxel_size = [0.25, 0.25, 0.5]
    grid_size = [
        int((pc_range[3] - pc_range[0]) / voxel_size[0]),
        int((pc_range[4] - pc_range[1]) / voxel_size[1]),
        int((pc_range[5] - pc_range[2]) / voxel_size[2])
    ]

    num_cams = 6
    img_size = (900, 1600)  # H, W

    feat_dim = 64
    bev_dim = 128
    num_classes_det = 10
    num_classes_seg = 2

    # 新增：特征图 stride（你的 encoder 是 /4）
    feat_stride = 4

    # 新增：多视角融合方式：'mean'/'sum'/'max'
    view_fusion = "mean"


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
    M2BEV: Uniform 2D->3D Projection (voxel-centric pull)
    关键修复：
    - u,v 是像素坐标，要先 /stride -> 特征图坐标，再用 Hf/Wf 归一化
    - 加图像边界 mask
    - chunk grid_sample 防爆显存
    - 多视角融合 mean/sum/max
    """
    def __init__(self, config, chunk_size=200_000):
        super().__init__()
        
        self.grid_size = config.grid_size
        self.pc_range = config.pc_range
        self.voxel_size = config.voxel_size
        self.num_cams = config.num_cams
        self.stride = config.feat_stride
        self.view_fusion = getattr(config, "view_fusion", "mean")
        self.chunk_size = int(chunk_size)

        self.register_buffer('voxel_coords', self._create_voxel_grid())

        # 原图尺寸，用于像素边界判断
        self.img_H, self.img_W = config.img_size

    def _create_voxel_grid(self):
        coords_x = torch.linspace(
            self.pc_range[0] + self.voxel_size[0] / 2,
            self.pc_range[3] - self.voxel_size[0] / 2,
            self.grid_size[0]
        )
        
        coords_y = torch.linspace(
            self.pc_range[1] + self.voxel_size[1] / 2,
            self.pc_range[4] - self.voxel_size[1] / 2,
            self.grid_size[1]
        )
        
        coords_z = torch.linspace(
            self.pc_range[2] + self.voxel_size[2] / 2,
            self.pc_range[5] - self.voxel_size[2] / 2,
            self.grid_size[2]
        )

        coords_z, coords_y, coords_x = torch.meshgrid(coords_z, coords_y, coords_x, indexing='ij')
        grid = torch.stack([coords_x, coords_y, coords_z], dim=0)  # (3, Z, Y, X)
        return grid.unsqueeze(0)  # (1, 3, Z, Y, X)

    def forward(self, img_feats, mats_dict):
        """
        img_feats: (B*N, C, Hf, Wf)
        mats_dict:
          - ego2cam: (B*N,4,4)
          - cam2img: (B*N,4,4)  (像素坐标系投影)
        return:
          voxel_feat: (B, C, Z, Y, X)
        """
        B_N, C, Hf, Wf = img_feats.shape
        device = img_feats.device

        # -------- 1) voxel coords flatten --------
        voxel_coords = self.voxel_coords.to(device).repeat(B_N, 1, 1, 1, 1)  # (B*N,3,Z,Y,X)
        _, _, Z, Y, X = voxel_coords.shape
        N_pts = Z * Y * X

        pts = voxel_coords.view(B_N, 3, -1)  # (B*N,3,N)
        ones = torch.ones((B_N, 1, N_pts), device=device)
        
        pts_h = torch.cat([pts, ones], dim=1)  # (B*N,4,N)

        # -------- 2) projection ego->img (pixel) --------
        proj_mat = torch.bmm(mats_dict['cam2img'], mats_dict['ego2cam'])  # (B*N,4,4)
        
        # 自车坐标系投影到像素坐标系下 u,v,d
        img_pts = torch.bmm(proj_mat, pts_h)  # (B*N,4,N)

        depth = img_pts[:, 2:3, :]  # (B*N,1,N)
        eps = 1e-6
        u = img_pts[:, 0:1, :] / (depth + eps)  # pixel x
        v = img_pts[:, 1:2, :] / (depth + eps)  # pixel y

        # -------- 3) valid mask：前方 + 图像边界 --------
        valid = (depth > 0.1) & (u >= 0) & (u <= (self.img_W - 1)) & (v >= 0) & (v <= (self.img_H - 1))
        valid = valid.view(B_N, 1, N_pts)

        # -------- 4) pixel -> feature coords (IMPORTANT FIX) --------
        uf = u / float(self.stride)
        vf = v / float(self.stride)

        # 防止除零（Wf/Hf 可能为1的极端情况）
        Wf_denom = max(Wf - 1, 1)
        Hf_denom = max(Hf - 1, 1)

        # -------- 5) normalize to [-1,1] for grid_sample (feature coords!) --------
        u_norm_all = 2.0 * (uf / Wf_denom) - 1.0
        v_norm_all = 2.0 * (vf / Hf_denom) - 1.0

        # 无效点推到范围外（grid_sample 会输出 0）
        oob = torch.full_like(u_norm_all, 2.0)
        u_norm_all = torch.where(valid, u_norm_all, oob)
        v_norm_all = torch.where(valid, v_norm_all, oob)

        # -------- 6) chunked grid_sample --------
        sampled_feat = torch.zeros((B_N, C, N_pts), device=device, dtype=img_feats.dtype)
        
        # 这里没有关于深度的部分，完全是通过u和v来采样
        for s in range(0, N_pts, self.chunk_size):
            e = min(s + self.chunk_size, N_pts)

            u_norm = u_norm_all[:, :, s:e]  # (B*N,1,chunk)
            v_norm = v_norm_all[:, :, s:e]  # (B*N,1,chunk)

            # grid: (B*N, chunk, 1, 2)  last dim is (x,y)
            grid = torch.cat([u_norm, v_norm], dim=1).permute(0, 2, 1)  # (B*N,chunk,2)
            grid = grid.view(B_N, e - s, 1, 2)

            # output: (B*N, C, chunk, 1) -> (B*N, C, chunk)
            out = F.grid_sample(
                img_feats,
                grid,
                align_corners=True,
                padding_mode='zeros',
                mode='bilinear'
            ).squeeze(-1)

            sampled_feat[:, :, s:e] = out

        # -------- 7) multi-view fusion to B --------
        B = B_N // self.num_cams
        sampled_feat = sampled_feat.view(B, self.num_cams, C, N_pts)   # (B,N,C,Npts)
        valid_mask = valid.view(B, self.num_cams, 1, N_pts).to(sampled_feat.dtype)

        if self.view_fusion == "sum":
            fused = sampled_feat.sum(dim=1)  # (B,C,Npts)

        elif self.view_fusion == "max":
            
            # 无效位置置为很小值，避免 max 被 0 干扰
            neg_inf = torch.finfo(sampled_feat.dtype).min
            masked = torch.where(valid_mask > 0, sampled_feat, torch.tensor(neg_inf, device=device, dtype=sampled_feat.dtype))
            fused = masked.max(dim=1).values
            
            # 如果所有视角都无效，max 会是 -inf，改回 0
            all_invalid = (valid_mask.sum(dim=1) == 0)
            fused = torch.where(all_invalid.expand_as(fused), torch.zeros_like(fused), fused)

        else:  # mean (default)
            denom = valid_mask.sum(dim=1).clamp(min=1.0)  # (B,1,Npts)
            fused = sampled_feat.sum(dim=1) / denom

        # -------- 8) reshape to voxel volume --------
        voxel_feat = fused.view(B, C, Z, Y, X)
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

def compute_bev_centerness(bev_dims, device=None, dtype=torch.float32):
    """
    centerness = 1 + sqrt(dist^2 / max_dist^2)  ∈ [1,2]
    """
    H, W = bev_dims
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    yc, xc = (H - 1) / 2.0, (W - 1) / 2.0
    dist_sq = (xs - xc) ** 2 + (ys - yc) ** 2

    #  角点最大距离（更明确）
    corners = torch.tensor(
        [[0.0, 0.0], [0.0, W - 1.0], [H - 1.0, 0.0], [H - 1.0, W - 1.0]],
        device=device, dtype=dtype
    )
    max_dist_sq = ((corners[:, 1] - xc) ** 2 + (corners[:, 0] - yc) ** 2).max()

    centerness = 1.0 + torch.sqrt(dist_sq / (max_dist_sq + 1e-6))
    return centerness  # (H, W)

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
        H_bev, W_bev = config.grid_size[1], config.grid_size[0]
        cent = compute_bev_centerness((H_bev, W_bev), device=torch.device("cpu"))
        self.register_buffer('centerness_map', cent)
    

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