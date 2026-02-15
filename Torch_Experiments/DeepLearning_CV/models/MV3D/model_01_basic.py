# =============================================================================
# MV3D 模型基础实现 (model_01_basic.py)
# =============================================================================
# 本文件实现 MV3D (Multi-View 3D) 论文中的核心算法流程：
# - 多视图输入：BEV(鸟瞰)、FV(前视)、RGB 图像
# - 3D Proposal 仅在 BEV 上生成，再投影到各视图做 Region-based Fusion
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import roi_align
import faulthandler
# 开启 faulthandler：程序崩溃（如段错误）时会自动打印 Python 调用栈，便于调试 C 扩展/底层错误
faulthandler.enable()


class MV3DConfig:
    """
    论文中的关键参数配置类（所有常量集中管理）。
    来源: Section 3.4 Implementation Details
    """
    # ----- 1. 鸟瞰图 (BEV) 配置 -----
    # LiDAR 点云在 X-Y 平面上的范围（单位：米），用于生成 BEV 网格
    x_range = (0, 70.4)
    y_range = (-40, 40)
    z_range = (-3, 1)       # 高度范围，用于体素化时切层
    voxel_size = 0.1        # 体素边长 0.1m，即 BEV 分辨率

    # BEV 网格尺寸：每个方向上的格子数 = 范围长度 / 体素大小
    # 用于将连续坐标 (x,y) 映射到离散网格索引
    bev_h = int((x_range[1] - x_range[0]) / voxel_size)   # 704
    bev_w = int((y_range[1] - y_range[0]) / voxel_size)   # 800

    # ----- 2. 前视图 (FV) 配置 -----
    # 论文使用 64 线激光雷达，水平 512 分辨率，对应 FV 图像高×宽
    fv_h = 64
    fv_w = 512

    # ----- 3. RGB 图像配置 -----
    # 论文将 RGB 最短边缩放到 500，这里宽为假设值，实际随比例变化
    rgb_h = 500
    rgb_w = 1280

    # ----- 4. Anchor 配置 (Section 3.2) -----
    # 每个网格位置有多个 anchor：2 种尺寸 × 2 种朝向 = 4 个
    # 尺寸 (l, w, h)：Car 大尺寸 (3.9, 1.6, 1.56)、小尺寸 (1.0, 0.6, 1.56)
    anchor_sizes = [[3.9, 1.6, 1.56], [1.0, 0.6, 1.56]]
    anchor_rotations = [0, 90]   # 单位：度，0° 与 90° 两种朝向

    # ----- 5. 网络输入通道数 -----
    # BEV：多高度切片 + 密度 + 强度等，共 6 通道
    # FV：高度、距离、强度等，共 3 通道
    bev_channels = 6
    fv_channels = 3


class MockBackbone(nn.Module):
    """
    模拟 VGG-16 Backbone 的占位模块（用随机特征代替真实卷积）。
    目的：保持论文中的下采样倍率，便于验证整体数据流和维度。
    论文：卷积部分 8× 下采样；Proposal 前有 2× 上采样 → 最终 Proposal 特征图为 4× 下采样。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 保存输出通道数，forward 时用于构造随机特征
        self.out_channels = out_channels

    def forward(self, x):
        
        # 模拟 8× 下采样：特征图高宽各除以 8
        feat_h, feat_w = x.shape[2] // 8, x.shape[3] // 8
        
        # 输出形状 [B, out_channels, H/8, W/8]，设备与输入一致
        return torch.randn(x.shape[0], self.out_channels, feat_h, feat_w).to(x.device)


class ProposalNetwork(nn.Module):
    """
    Section 3.2: 3D Proposal Network。
    仅使用 BEV 特征图生成 3D 候选框，避免在 3D 空间全局搜索，降低计算量。
    """
    def __init__(self, in_channels, config):
        super().__init__()
        self.cfg = config

        # 论文：最后一层卷积后进行 2× 双线性上采样
        # 因此相对原始 BEV 输入：8× 下采样再 2× 上采样 → 4× 下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 每个位置 anchor 数量 = 尺寸数 × 朝向数
        num_anchors = len(config.anchor_sizes) * len(config.anchor_rotations)

        # 分类头：每个 anchor 二分类（目标 / 背景），输出通道数 = num_anchors * 2
        self.score_conv = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)

        # 回归头：每个 anchor 预测 6 个值 (tx, ty, tz, tl, tw, th)，即中心与尺寸的偏移
        # 论文在 Proposal 阶段不做朝向连续回归，只做 0°/90° 分类
        self.reg_conv = nn.Conv2d(in_channels, num_anchors * 6, kernel_size=1)

    def forward(self, bev_features):
        # 1. 上采样到 1/4 尺度
        x = self.upsample(bev_features)   # [B, C, H/4, W/4]

        # 2. 预测每个位置的 objectness 和 3D 回归量
        scores = self.score_conv(x)       # [B, A*2, H/4, W/4]
        reg_deltas = self.reg_conv(x)     # [B, A*6, H/4, W/4]

        # 3. 解码 + NMS 得到最终 Proposals（此处用固定 dummy 代替真实解码/NMS）
        # 真实实现需要：生成网格坐标、用 reg_deltas 解码成 3D 框、按分数做 NMS
        batch_size = bev_features.shape[0]

        # 模拟的 Proposal 格式：[batch_idx, x, y, z, l, w, h, rot]（LiDAR 坐标系）
        dummy_proposals = torch.tensor([
            [0, 10.0, 5.0, -1.5, 3.9, 1.6, 1.56, 0.0],
            [0, 20.0, -5.0, -1.5, 3.9, 1.6, 1.56, 1.57],
        ]).to(bev_features.device)

        return dummy_proposals


class InputEncoder:
    """
    Section 3.1: 3D 点云表示。
    将原始点云编码成 BEV/FV 等视图（通常放在数据预处理里，这里仅展示投影逻辑）。
    """
    @staticmethod
    def project_to_fv_coords(points, config):
        """
        论文公式 (1)：将 3D 点投影到前视图 (FV) 的像素坐标（柱面投影）。
        输入: points (N, 3)，每行为 (x, y, z)。
        """
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # 水平距离，用于计算仰角
        dist = torch.sqrt(x**2 + y**2)

        # 垂直方向：r = floor(atan2(z, dist) / delta_phi)  （仰角离散化）
        # 水平方向：c = floor(atan2(y, x) / delta_theta)   （方位角离散化）
        # delta_phi、delta_theta 由雷达线数、水平分辨率决定，此处仅保留逻辑框架
        pass


class RegionFusionNetwork(nn.Module):
    """
    Section 3.3: Region-based Fusion Network。
    MV3D 核心：将 3D Proposal 投影到 BEV/FV/RGB 三个视图，在各视图上做 ROI 池化，
    再将多视图特征融合（Deep Fusion）后做分类与 3D 框回归。
    """
    def __init__(self, backbone_dims, config):
        super().__init__()
        self.cfg = config

        # 每个视图 ROI 池化后为 7×7 特征图，展平后通过全连接映射到统一维度
        # backbone_dims[i] 为各 backbone 输出通道数，7*7 为 ROI 空间尺寸
        self.shared_dim = 512
        self.fc_bev = nn.Sequential(nn.Linear(backbone_dims[0] * 7 * 7, self.shared_dim))
        self.fc_fv = nn.Sequential(nn.Linear(backbone_dims[1] * 7 * 7, self.shared_dim))
        self.fc_rgb = nn.Sequential(nn.Linear(backbone_dims[2] * 7 * 7, self.shared_dim))

        # Deep Fusion：对多视图特征做进一步交互（论文中用 element-wise mean 等）
        self.fusion_layer1 = nn.Linear(self.shared_dim, self.shared_dim)

        # 分类头：二分类（如 Car vs 背景）
        self.cls_head = nn.Linear(self.shared_dim, 2)

        # 回归头：论文采用 8 个角点表示 3D 框，共 8*3=24 维
        self.reg_head = nn.Linear(self.shared_dim, 24)

    def project_proposals(self, proposals3d, view_type):
        """
        将 3D 提案投影到指定视图的 2D 坐标，供 ROI Align 使用。
        对应论文 Eq 2。
        proposals3d: [N, 8]，每行为 (batch_idx, x, y, z, l, w, h, rot)。
        返回: [N, 5]，每行为 (batch_idx, x1, y1, x2, y2) 的归一化/像素坐标格式（依 roi_align 要求）。
        """
        batch_idx = proposals3d[:, 0]

        if view_type == 'BEV':
            # BEV 下只需 x, y, l, w（忽略 z, h, rot），在 BEV 网格上画 2D 框
            x, y, l, w = proposals3d[:, 1], proposals3d[:, 2], proposals3d[:, 4], proposals3d[:, 5]
            # 从 LiDAR 坐标转换为 BEV 网格坐标（连续值，供 roi_align 用）
            x_img = (x - self.cfg.x_range[0]) / self.cfg.voxel_size
            y_img = (y - self.cfg.y_range[0]) / self.cfg.voxel_size
            w_img = w / self.cfg.voxel_size
            l_img = l / self.cfg.voxel_size
            # 左上 (x1,y1) 与右下 (x2,y2)
            return torch.stack([batch_idx, x_img - l_img/2, y_img - w_img/2,
                               x_img + l_img/2, y_img + w_img/2], dim=1)

        elif view_type == 'FV':
            # FV 投影：此处用随机坐标模拟，真实实现需按雷达几何做 3D→2D 投影
            random_coords = torch.rand(proposals3d.shape[0], 4).to(proposals3d.device) * 50
            return torch.cat([batch_idx.unsqueeze(1), random_coords], dim=1)

        elif view_type == 'RGB':
            # RGB 投影：同样用随机坐标模拟，真实实现需 3D→相机投影
            random_coords = torch.rand(proposals3d.shape[0], 4).to(proposals3d.device) * 100
            return torch.cat([batch_idx.unsqueeze(1), random_coords], dim=1)

    def forward(self, features_dict, proposals_3d):
        """
        多视图 Region-based Fusion 前向。
        features_dict: {'BEV', 'FV', 'RGB'} 各视图的 backbone 特征图。
        proposals_3d: [N, 8]，来自 RPN 的 3D 提案。
        返回: 分类 logits [N, 2]，3D 框回归 [N, 24]（8 角点）。
        """
        # 1. 多视图 ROI：将每个 3D 框投影到三个视图并做 ROI Align (Eq 3)
        rois_bev = self.project_proposals(proposals_3d, 'BEV')
        rois_fv = self.project_proposals(proposals_3d, 'FV')
        rois_rgb = self.project_proposals(proposals_3d, 'RGB')

        pool_bev = roi_align(features_dict['BEV'], rois_bev, output_size=(7, 7))
        pool_fv  = roi_align(features_dict['FV'],  rois_fv,  output_size=(7, 7))
        pool_rgb = roi_align(features_dict['RGB'], rois_rgb, output_size=(7, 7))

        # 展平为向量
        feat_bev = pool_bev.view(pool_bev.size(0), -1)
        feat_fv  = pool_fv.view(pool_fv.size(0), -1)
        feat_rgb = pool_rgb.view(pool_rgb.size(0), -1)

        # 2. Deep Fusion (Eq 6)：先映射到同一维度，再逐元素平均做初步融合
        f0_bev = self.fc_bev(feat_bev)
        f0_fv  = self.fc_fv(feat_fv)
        f0_rgb = self.fc_rgb(feat_rgb)
        f_fused = (f0_bev + f0_fv + f0_rgb) / 3.0

        # 再经过融合层
        f_layer1 = self.fusion_layer1(f_fused)

        # 3. 输出分类与 24 维角点回归
        cls_scores = self.cls_head(f_layer1)
        bbox_pred = self.reg_head(f_layer1)
        return cls_scores, bbox_pred


class MV3D(nn.Module):
    """
    完整 MV3D 模型：三视图 Backbone → BEV RPN → Region-based Fusion → 分类 + 3D 框回归。
    """
    def __init__(self):
        super().__init__()
        self.cfg = MV3DConfig()

        # 三个视图的 Backbone（此处为 Mock，论文为 VGG-16，通道减半 512→256）
        self.backbone_bev = MockBackbone(self.cfg.bev_channels, 256)
        self.backbone_fv  = MockBackbone(self.cfg.fv_channels, 256)
        self.backbone_rgb = MockBackbone(3, 256)

        # 3D Proposal 仅依赖 BEV 特征
        self.rpn = ProposalNetwork(256, self.cfg)

        # 多视图融合网络，三个 backbone 输出通道均为 256
        self.fusion_net = RegionFusionNetwork([256, 256, 256], self.cfg)

    def forward(self, input_bev, input_fv, input_rgb):
        """
        完整前向。
        input_bev:  [B, 6, 704, 800]
        input_fv:   [B, 3, 64, 512]
        input_rgb:  [B, 3, 500, 1280]
        返回: cls_scores [N, 2], bbox_pred [N, 24]
        """
        # Step 1: 三视图特征提取（约为输入的 1/8 空间尺寸）
        feat_bev = self.backbone_bev(input_bev)
        feat_fv  = self.backbone_fv(input_fv)
        feat_rgb = self.backbone_rgb(input_rgb)

        # Step 2: 仅在 BEV 上生成 3D Proposals，避免 3D 全局搜索
        proposals_3d = self.rpn(feat_bev)

        # Step 3: 将 Proposals 投影到三视图，做 ROI 池化并融合，得到分类与 3D 框
        features_dict = {'BEV': feat_bev, 'FV': feat_fv, 'RGB': feat_rgb}
        cls_scores, bbox_pred = self.fusion_net(features_dict, proposals_3d)
        return cls_scores, bbox_pred


# ==========================================
# 脚本入口：用随机输入验证数据流与维度
# ==========================================
if __name__ == "__main__":
    # 构造与 config 一致的假输入
    dummy_bev = torch.randn(1, 6, 704, 800)
    dummy_fv = torch.randn(1, 3, 64, 512)
    dummy_rgb = torch.randn(1, 3, 500, 1280)

    model = MV3D()
    scores, boxes = model(dummy_bev, dummy_fv, dummy_rgb)

    print("MV3D Algorithm Pipeline Verification:")
    print(f"1. Input BEV Shape: {dummy_bev.shape}")
    print(f"2. Generated Proposals: {model.rpn(model.backbone_bev(dummy_bev)).shape} (Mocked)")
    print(f"3. Final Classification Score Shape: {scores.shape} (N_Proposals, 2)")
    print(f"4. Final Box Regression Shape: {boxes.shape} (N_Proposals, 24)")
    print("   Note: Output is 24-dim because MV3D regresses 8 corners (8*3=24)")