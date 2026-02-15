# =============================================================================
# MV3D 模型进阶实现 (model_02_advance.py)
# =============================================================================
# 在保持 Mock Backbone 的前提下，本版重点做到：
# - 坐标/尺度严谨：BEV ROI 使用特征图坐标（非网格坐标），与 roi_align 一致
# - 可复现：MockBackbone 支持 deterministic 模式
# - 结构清晰：Projection 抽离、Proposal3D 封装、Config 集中 stride/roi 等
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from dataclasses import dataclass
import faulthandler

# 崩溃时打印 Python 调用栈，便于调试 C 扩展/段错误
faulthandler.enable()


# -----------------------------
# 1) Config：论文参数与网络超参集中管理
# -----------------------------
class MV3DConfig:
    """
    论文中的关键参数配置类（常量集中管理）。
    来源: Section 3.4 Implementation Details；backbone_stride / roi_out_size 为本实现扩展。
    """
    # ----- BEV 范围（单位：米） -----
    x_range = (0.0, 70.4)
    y_range = (-40.0, 40.0)
    z_range = (-3.0, 1.0)
    voxel_size = 0.1  # 每个网格格子的边长（米），即 BEV 分辨率

    # BEV 网格尺寸：用于将连续 (x,y) 映射到离散网格索引
    bev_h = int((x_range[1] - x_range[0]) / voxel_size)   # 704
    bev_w = int((y_range[1] - y_range[0]) / voxel_size)   # 800

    # ----- 前视图 (FV) 分辨率 -----
    # 论文 64 线雷达，水平 512，对应 FV 图像高×宽
    fv_h = 64
    fv_w = 512

    # ----- RGB 图像分辨率 -----
    rgb_h = 500
    rgb_w = 1280

    # ----- Anchor 配置 (Section 3.2) -----
    # 2 种尺寸 × 2 种朝向 = 4 个 anchor/位置
    anchor_sizes = [[3.9, 1.6, 1.56], [1.0, 0.6, 1.56]]  # (l, w, h)
    anchor_rotations = [0.0, 90.0]  # 度

    # ----- 各视图输入通道数 -----
    bev_channels = 6   # 多高度切片 + 密度 + 强度等
    fv_channels = 3    # 高度、距离、强度等

    # ----- Backbone 下采样倍数（Mock 与论文一致） -----
    # 特征图相对输入为 H/stride, W/stride；用于 BEV 网格坐标 -> 特征图坐标
    backbone_stride = 8  # MockBackbone 输出 H/8, W/8

    # ROI Align 输出空间尺寸（7×7 与论文一致）
    roi_out_size = (7, 7)


# -----------------------------
# 2) Mock Backbone：可复现版 + 可选随机版
# -----------------------------
class MockBackbone(nn.Module):
    """
    模拟 VGG-16 的占位 Backbone，保留下采样倍率。
    - deterministic=True：用 1×1 卷积 + 插值下采样，同一输入可复现
    - deterministic=False：用 randn 生成特征，与原 basic 版行为类似
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 8, 
                 deterministic: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.deterministic = deterministic
        # 小卷积使输出可复现，且更接近“真实网络输出”
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先按 stride 做双线性插值下采样（与论文 8× 一致）
        feat = F.interpolate(x, scale_factor=1.0 / self.stride, mode="bilinear", align_corners=False)
        if self.deterministic:
            return self.proj(feat)
        # 非确定模式：随机特征，便于与旧版对比
        b, _, h, w = feat.shape
        return torch.randn(b, self.out_channels, h, w, device=x.device, dtype=x.dtype)


# -----------------------------
# 3) Proposal 数据结构：便于类型与语义清晰
# -----------------------------
@dataclass
class Proposal3D:
    """
    RPN 输出的 3D 提案封装。
    data: [N, 8]，每行为 (batch_idx, x, y, z, l, w, h, yaw), LiDAR 坐标系。
    """
    data: torch.Tensor  # [N, 8]


# -----------------------------
# 4) RPN：仅用 BEV 特征生成 3D Proposals（decode/NMS 此处用固定值代替）
# -----------------------------
class ProposalNetwork(nn.Module):
    """
    Section 3.2: 仅使用 BEV 特征图生成 3D 候选框，避免 3D 全局搜索。
    """
    def __init__(self, in_channels: int, cfg: MV3DConfig):
        super().__init__()
        self.cfg = cfg
        
        # 论文：最后一层 conv 后 2× 上采样，相对输入为 4× 下采样
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        num_anchors = len(cfg.anchor_sizes) * len(cfg.anchor_rotations)
        
        # 分类头：目标 vs 背景，每 anchor 2 类
        self.score_conv = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        
        # 回归头：每 anchor 6 维 (tx, ty, tz, tl, tw, th)
        self.reg_conv = nn.Conv2d(in_channels, num_anchors * 6, kernel_size=1)

    def forward(self, bev_feat: torch.Tensor) -> Proposal3D:
        """
        输入: bev_feat [B, C, H/8, W/8]
        论文：上采样到 H/4 再做 proposal heads；此处 decode/NMS 用固定 proposals 代替。
        """
        x = self.upsample(bev_feat)   # [B, C, H/4, W/4]
        _scores = self.score_conv(x)  # [B, A*2, H/4, W/4]
        _deltas = self.reg_conv(x)    # [B, A*6, H/4, W/4]

        # 省略真实 decode + NMS，仅返回固定 proposals 以验证 pipeline
        device = bev_feat.device
        proposals = torch.tensor([
            [0, 10.0,  5.0, -1.5, 3.9, 1.6, 1.56, 0.0],
            [0, 20.0, -5.0, -1.5, 3.9, 1.6, 1.56, 1.57],
        ], device=device, dtype=bev_feat.dtype)
        return Proposal3D(proposals)


# -----------------------------
# 5) 投影与坐标转换：LiDAR/网格/特征图坐标
# -----------------------------
class Projection:
    """静态方法：LiDAR 米制 -> BEV 网格 -> 特征图坐标，以及 ROI 边界裁剪。"""

    @staticmethod
    def lidar_xy_to_bev_grid(x: torch.Tensor, y: torch.Tensor, cfg: MV3DConfig):
        """
        LiDAR 平面坐标 (米) -> BEV 网格坐标（像素索引，连续值）。
        roi_align 若直接吃 BEV 图像则用此坐标；若吃特征图则需再除以 stride。
        """
        x_img = (x - cfg.x_range[0]) / cfg.voxel_size
        y_img = (y - cfg.y_range[0]) / cfg.voxel_size
        return x_img, y_img

    @staticmethod
    def bev_grid_to_feat(x_img: torch.Tensor, y_img: torch.Tensor, cfg: MV3DConfig, feat_stride: int):
        """
        BEV 网格像素 -> BEV 特征图像素（特征图是输入的 1/stride）。
        feat_stride 通常为 backbone_stride（如 8）。
        """
        return x_img / feat_stride, y_img / feat_stride

    @staticmethod
    def clamp_boxes(x1, y1, x2, y2, w, h):
        """将 ROI 边界裁剪到 [0, w-1] x [0, h-1]，避免 roi_align 越界。"""
        x1 = x1.clamp(0, w - 1)
        x2 = x2.clamp(0, w - 1)
        y1 = y1.clamp(0, h - 1)
        y2 = y2.clamp(0, h - 1)
        return x1, y1, x2, y2


# -----------------------------
# 6) Region Fusion Network：多视图 ROI + Deep Fusion + 分类/回归头
# -----------------------------
class RegionFusionNetwork(nn.Module):
    """
    Section 3.3: 将 3D proposal 投影到 BEV/FV/RGB 三个视图，
    做 ROI Align -> 展平 -> FC 到统一维度 -> Deep Fusion（mean + MLP）-> 分类与 24 维回归。
    """
    def __init__(self, backbone_dims, cfg: MV3DConfig):
        super().__init__()
        self.cfg = cfg
        self.shared_dim = 512

        # 各视图 ROI 池化后 7×7，展平后维度 = backbone_dim * 7 * 7
        self.fc_bev = nn.Linear(backbone_dims[0] * cfg.roi_out_size[0] * cfg.roi_out_size[1], self.shared_dim)
        self.fc_fv  = nn.Linear(backbone_dims[1] * cfg.roi_out_size[0] * cfg.roi_out_size[1], self.shared_dim)
        self.fc_rgb = nn.Linear(backbone_dims[2] * cfg.roi_out_size[0] * cfg.roi_out_size[1], self.shared_dim)

        # Deep Fusion：ReLU -> Linear -> ReLU，与论文中“中间层交互”一致
        self.fusion_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Linear(self.shared_dim, 2)   # 二分类（如 Car vs BG）
        self.reg_head = nn.Linear(self.shared_dim, 24) # 8 角点 × 3 = 24 维

    def project_to_rois(self, proposals: Proposal3D, view: str, feat_map: torch.Tensor) -> torch.Tensor:
        """
        将 3D 提案投影到指定视图的 2D ROI，且坐标必须在**特征图**空间（与 roi_align 输入一致）。
        返回: [N, 5]，每行为 (batch_idx, x1, y1, x2, y2)，单位为特征图像素。
        """
        p = proposals.data
        batch_idx = p[:, 0].float()

        if view == "BEV":
            # BEV：LiDAR (x,y,l,w) -> BEV 网格 -> 再除以 stride 得到特征图坐标
            x, y, l, w = p[:, 1], p[:, 2], p[:, 4], p[:, 5]
            x_img, y_img = Projection.lidar_xy_to_bev_grid(x, y, self.cfg)
            l_img = l / self.cfg.voxel_size
            w_img = w / self.cfg.voxel_size

            stride = self.cfg.backbone_stride
            x_f, y_f = Projection.bev_grid_to_feat(x_img, y_img, self.cfg, stride)
            l_f = l_img / stride
            w_f = w_img / stride

            x1 = x_f - l_f / 2
            y1 = y_f - w_f / 2
            x2 = x_f + l_f / 2
            y2 = y_f + w_f / 2

            _, _, Hf, Wf = feat_map.shape
            x1, y1, x2, y2 = Projection.clamp_boxes(x1, y1, x2, y2, Wf, Hf)
            return torch.stack([batch_idx, x1, y1, x2, y2], dim=1)

        elif view == "FV":
            # FV：可解释假投影——用 (y,z) 线性映射到 FV 特征图 (u,v)，再给固定 ROI 大小
            x, y, z = p[:, 1], p[:, 2], p[:, 3]
            _, _, Hf, Wf = feat_map.shape
            u = (y - self.cfg.y_range[0]) / (self.cfg.y_range[1] - self.cfg.y_range[0]) * (Wf - 1)
            v = (z - self.cfg.z_range[0]) / (self.cfg.z_range[1] - self.cfg.z_range[0]) * (Hf - 1)
            box_w, box_h = 10.0, 6.0
            x1, y1 = u - box_w/2, v - box_h/2
            x2, y2 = u + box_w/2, v + box_h/2
            x1, y1, x2, y2 = Projection.clamp_boxes(x1, y1, x2, y2, Wf, Hf)
            return torch.stack([batch_idx, x1, y1, x2, y2], dim=1)

        elif view == "RGB":
            # RGB：可解释假投影——(x,y) 线性映射到特征图，再固定 ROI 大小
            x, y = p[:, 1], p[:, 2]
            _, _, Hf, Wf = feat_map.shape
            u = (x - self.cfg.x_range[0]) / (self.cfg.x_range[1] - self.cfg.x_range[0]) * (Wf - 1)
            v = (y - self.cfg.y_range[0]) / (self.cfg.y_range[1] - self.cfg.y_range[0]) * (Hf - 1)
            box_w, box_h = 12.0, 12.0
            x1, y1 = u - box_w/2, v - box_h/2
            x2, y2 = u + box_w/2, v + box_h/2
            x1, y1, x2, y2 = Projection.clamp_boxes(x1, y1, x2, y2, Wf, Hf)
            return torch.stack([batch_idx, x1, y1, x2, y2], dim=1)

        else:
            raise ValueError(f"Unknown view: {view}")

    def forward(self, feats: dict, proposals: Proposal3D):
        """多视图 ROI Align -> FC -> mean 融合 -> fusion_layer -> 分类与 24 维回归。"""
        rois_bev = self.project_to_rois(proposals, "BEV", feats["BEV"])
        rois_fv  = self.project_to_rois(proposals, "FV",  feats["FV"])
        rois_rgb = self.project_to_rois(proposals, "RGB", feats["RGB"])

        # ROI Align：aligned=True 表示 half-pixel 对齐，与常见实现一致
        pool_bev = roi_align(feats["BEV"], rois_bev, output_size=self.cfg.roi_out_size, aligned=True)
        pool_fv  = roi_align(feats["FV"],  rois_fv,  output_size=self.cfg.roi_out_size, aligned=True)
        pool_rgb = roi_align(feats["RGB"], rois_rgb, output_size=self.cfg.roi_out_size, aligned=True)

        def flatten(x): return x.flatten(1)

        f_bev = self.fc_bev(flatten(pool_bev))
        f_fv  = self.fc_fv(flatten(pool_fv))
        f_rgb = self.fc_rgb(flatten(pool_rgb))

        # Deep Fusion：三视图逐元素平均后过 MLP（论文中 mean + 后续层）
        fused = (f_bev + f_fv + f_rgb) / 3.0
        fused = self.fusion_layer(fused)

        cls = self.cls_head(fused)
        reg = self.reg_head(fused)
        return cls, reg


# -----------------------------
# 7) MV3D 完整模型：三视图 Backbone -> BEV RPN -> Region Fusion
# -----------------------------
class MV3D(nn.Module):
    """
    完整 MV3D：BEV/FV/RGB 三个 Backbone -> 仅 BEV 上 RPN -> 多视图 ROI + Deep Fusion -> 分类 + 24 维回归。
    deterministic_backbone=True 时 Mock 输出可复现，便于单元测试与对比实验。
    """
    def __init__(self, deterministic_backbone: bool = True):
        super().__init__()
        self.cfg = MV3DConfig()

        self.backbone_bev = MockBackbone(self.cfg.bev_channels, 256,
                                         stride=self.cfg.backbone_stride,
                                         deterministic=deterministic_backbone)
        self.backbone_fv  = MockBackbone(self.cfg.fv_channels, 256,
                                         stride=self.cfg.backbone_stride,
                                         deterministic=deterministic_backbone)
        self.backbone_rgb = MockBackbone(3, 256, stride=self.cfg.backbone_stride,
                                         deterministic=deterministic_backbone)

        self.rpn = ProposalNetwork(256, self.cfg)
        self.fusion = RegionFusionNetwork([256, 256, 256], self.cfg)

    def forward(self, bev, fv, rgb):
        """
        输入: bev [B,6,704,800], fv [B,3,64,512], rgb [B,3,500,1280]
        输出: cls [N,2], reg [N,24]
        """
        feat_bev = self.backbone_bev(bev)   # [B, 256, H/8, W/8]
        feat_fv  = self.backbone_fv(fv)
        feat_rgb = self.backbone_rgb(rgb)

        proposals = self.rpn(feat_bev)
        feats = {"BEV": feat_bev, "FV": feat_fv, "RGB": feat_rgb}

        cls, reg = self.fusion(feats, proposals)
        return cls, reg


# ==========================================
# 脚本入口：用假数据验证维度与 pipeline
# ==========================================
if __name__ == "__main__":
    dummy_bev = torch.randn(1, 6, 704, 800)
    dummy_fv  = torch.randn(1, 3, 64, 512)
    dummy_rgb = torch.randn(1, 3, 500, 1280)

    model = MV3D(deterministic_backbone=True)
    cls, reg = model(dummy_bev, dummy_fv, dummy_rgb)

    print("MV3D Pipeline Verification (Optimized Mock):")
    print("BEV feat:", model.backbone_bev(dummy_bev).shape)
    print("Cls:", cls.shape, "Reg:", reg.shape)