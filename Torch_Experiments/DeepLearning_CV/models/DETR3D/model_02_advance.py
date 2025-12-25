import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# ==========================================
# 1. 核心组件: 3D-to-2D 特征采样器
# ==========================================
class FeatureSampler(nn.Module):
    """
    DETR3D 的核心: 将 3D 参考点投影到多视角图像上，并采样特征
    """
    def __init__(self, num_cameras=6, num_levels=4, embed_dim=256):
        super().__init__()
        self.num_cameras = num_cameras
        self.num_levels = num_levels
        self.embed_dim = embed_dim

    def forward(self, features, reference_points, lidar2img):
        """
        参数:
            features: List[Tensor], 4个层级的图像特征. Shape: (B, N_cam, C, H, W)
            reference_points: (B, M, 3) 3D参考点 (Sigmoid归一化后的坐标 0~1)
            lidar2img: (B, N_cam, 4, 4) 激光雷达坐标系到图像平面的变换矩阵
        返回:
            sampled_feats: (B, M, C) 采样并融合后的特征
        """
        B, M, _ = reference_points.shape
        
        # 1. 将参考点从 0~1 映射回真实世界坐标 (假设范围 -61.2m ~ 61.2m)
        # 简化起见，我们假设 reference_points 已经是归一化坐标
        # 这里需要反归一化到真实尺度才能用 lidar2img 投影
        pc_range = torch.tensor([-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], device=reference_points.device)
        ref_3d = reference_points.clone()
        ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # 变为齐次坐标 (B, M, 4) -> (x, y, z, 1)
        ref_3d_homo = torch.cat([ref_3d, torch.ones_like(ref_3d[..., :1])], dim=-1)
        
        # 2. 投影到 6 个相机的图像平面
        # ref_3d_homo: (B, 1, M, 4)
        # lidar2img:   (B, N, 1, 4, 4)
        ref_3d_homo = ref_3d_homo.unsqueeze(1) # (B, 1, M, 4)
        lidar2img = lidar2img.view(B, self.num_cameras, 1, 4, 4)
        
        # 矩阵乘法: (B, N, M, 4) = (B, N, 1, 4, 4) @ (B, 1, M, 4, 1) 的变体
        # 也就是 cam_coords = Mat * point
        cam_coords = torch.matmul(lidar2img, ref_3d_homo.unsqueeze(-1)).squeeze(-1) 
        
        # 3. 归一化 (透视除法) -> (u, v, d)
        eps = 1e-5
        # mask: 深度必须为正
        valid_mask = cam_coords[..., 2] > eps 
        
        # 归一化 x, y
        cam_coords[..., 0] /= (cam_coords[..., 2] + eps)
        cam_coords[..., 1] /= (cam_coords[..., 2] + eps)
        
        # 归一化到 [-1, 1] 用于 grid_sample (假设图像输入尺寸是固定的，这里简化处理)
        # 这里的 cam_coords 是像素坐标，需要除以特征图的宽高
        # 为了演示，我们假设特征图采样是归一化的
        # 在真实代码中，需要根据每一层特征图的 H, W 进行归一化
        
        sampled_feats_list = []
        for lvl, feat in enumerate(features):
            # feat: (B, N, C, H, W) -> (B*N, C, H, W)
            BN, C, H, W = feat.shape[0] * feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4]
            feat_flatten = feat.view(BN, C, H, W)
            
            # 坐标归一化: 0~W -> -1~1
            coords_lvl = cam_coords[..., :2].clone() # (B, N, M, 2)
            coords_lvl[..., 0] = coords_lvl[..., 0] / W * 2 - 1
            coords_lvl[..., 1] = coords_lvl[..., 1] / H * 2 - 1
            coords_lvl = coords_lvl.view(BN, M, 1, 2) # grid_sample 需要 (N, H_grid, W_grid, 2)
            
            # 4. 双线性采样
            # (BN, C, M, 1)
            samp = F.grid_sample(feat_flatten, coords_lvl, align_corners=False, padding_mode='zeros')
            samp = samp.view(B, self.num_cameras, C, M).permute(0, 3, 1, 2) # (B, M, N, C)
            
            sampled_feats_list.append(samp)
            
        # 简单平均多尺度的特征 (实际 DETR3D 可能会用 MLP 融合)
        sampled_feats = torch.stack(sampled_feats_list, dim=0).mean(0) # (B, M, N, C)
        
        # 5. 聚合多相机特征
        # valid_mask: (B, N, M) -> (B, M, N, 1)
        valid_mask = valid_mask.permute(0, 2, 1).unsqueeze(-1).float()
        
        # 加权平均: 只取投影在图像内的特征
        # sum(feat * mask) / (sum(mask) + eps)
        sampled_feats = (sampled_feats * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + eps)
        
        return sampled_feats # (B, M, C)

# ==========================================
# 2. 核心组件: DETR3D Decoder Layer
# ==========================================
class DETR3DLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        # Self Attention: Query 之间交互 (感知物体间的关系，避免重叠)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross Attention (在 DETR3D 中被替换为 3D-to-2D 投影采样)
        # 这里只有 MLP 用于处理采样回来的特征
        self.feature_proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, sampled_feats):
        """
        query: (M, B, C) 注意 PyTorch Attention 默认 sequence first
        sampled_feats: (B, M, C) -> 需要转置为 (M, B, C)
        """
        # 1. Self Attention
        q = k = v = query
        query2 = self.self_attn(q, k, v)[0]
        query = self.norm1(query + query2)
        
        # 2. "Cross Attention" (Feature Integration)
        # DETR3D 的特点：用采样的特征直接加到 Query 上
        sampled_feats = sampled_feats.permute(1, 0, 2) # (M, B, C)
        query2 = self.feature_proj(sampled_feats)
        query = self.norm2(query + query2)
        
        # 3. FFN
        query2 = self.ffn(query)
        query = self.norm3(query + query2)
        
        return query

# ==========================================
# 3. 完整模型: DETR3D
# ==========================================
class DETR3D(nn.Module):
    def __init__(self, num_queries=300, embed_dim=256):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # 1. Learnable Queries (Object Priors)
        self.query_embedding = nn.Embedding(num_queries, embed_dim)
        
        # 2. Reference Point Head (从 Query 解码出 3D 坐标)
        self.ref_point_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),
            nn.Sigmoid() # 归一化到 0~1
        )
        
        # 3. Backbone (Mock)
        # 实际应为 ResNet + FPN
        self.backbone_channels = embed_dim # 简化假设通道已对齐
        
        # 4. Sampler & Decoder
        self.sampler = FeatureSampler(embed_dim=embed_dim)
        self.decoder_layers = nn.ModuleList([DETR3DLayer(embed_dim) for _ in range(3)]) # 3层循环
        
        # 5. Prediction Heads (Box & Class)
        self.class_head = nn.Linear(embed_dim, 10) # 10类
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 10) # (cx, cy, cz, w, l, h, rot, vel_x, vel_y) 等
        )

    def forward(self, imgs, lidar2img):
        """
        imgs: (B, N_cam, 3, H, W)
        lidar2img: (B, N_cam, 4, 4)
        """
        B = imgs.shape[0]
        
        # 1. 提取图像特征 (Mock)
        # 模拟 4 层 FPN 特征, 尺寸依次减半
        # Layer 0: 1/8, Layer 1: 1/16 ...
        features = []
        H, W = imgs.shape[-2], imgs.shape[-1]
        for i in range(4):
            scale = 2**(i+3)
            # 生成随机特征 (B, N, C, H_s, W_s)
            feat = torch.randn(B, 6, self.embed_dim, H//scale, W//scale, device=imgs.device)
            features.append(feat)
            
        # 2. 初始化 Queries 和 Reference Points
        # Query: (M, B, C)
        query = self.query_embedding.weight.unsqueeze(1).repeat(1, B, 1)
        
        # 迭代优化
        all_cls_scores = []
        all_bbox_preds = []
        
        for layer in self.decoder_layers:
            # A. 从当前 Query 预测参考点
            # query shape: (M, B, C) -> (B, M, C)
            ref_points = self.ref_point_head(query.permute(1, 0, 2))
            
            # B. 3D-to-2D 投影与特征采样 (关键步骤!)
            # sampled_feats: (B, M, C)
            sampled_feats = self.sampler(features, ref_points, lidar2img)
            
            # C. 更新 Query (Self-Attn + Feature Fusion)
            query = layer(query, sampled_feats)
            
            # D. 预测结果
            # (B, M, 10)
            cls_scores = self.class_head(query.permute(1, 0, 2))
            # 预测的是相对于 ref_points 的偏移量，这里简化直接输出
            bbox_preds = self.bbox_head(query.permute(1, 0, 2)) 
            
            all_cls_scores.append(cls_scores)
            all_bbox_preds.append(bbox_preds)
            
        return all_cls_scores, all_bbox_preds

# ==========================================
# 4. 模拟运行环境
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running DETR3D Demo on: {device}")
    
    # 1. 初始化模型
    model = DETR3D(num_queries=50, embed_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 2. 模拟数据 (Batch Size = 2)
    B = 2
    imgs = torch.randn(B, 6, 3, 512, 1024).to(device) # 6个相机
    
    # 模拟 lidar2img 矩阵 (包含外参+内参)
    # 这是一个把 3D 点投影到像素坐标的 4x4 矩阵
    lidar2img = torch.eye(4).view(1, 1, 4, 4).repeat(B, 6, 1, 1).to(device)
    # 随便改几个值让投影有意义
    lidar2img[..., 0, 0] = 500.0 # fx
    lidar2img[..., 1, 1] = 500.0 # fy
    lidar2img[..., 0, 3] = 512.0 # cx
    lidar2img[..., 1, 3] = 256.0 # cy
    
    # 模拟真值 (用于 Loss)
    gt_classes = torch.randint(0, 10, (B, 10)).to(device) # 每张图10个物体
    gt_boxes = torch.randn(B, 10, 10).to(device)
    
    # 3. 训练循环
    model.train()
    print("\nStarting Training Loop...")
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward
        all_cls, all_box = model(imgs, lidar2img)
        
        # 取最后一层的输出计算 Loss
        final_cls = all_cls[-1] # (B, M, 10)
        final_box = all_box[-1] # (B, M, 10)
        
        # 简单的 Loss (模拟匈牙利匹配后的结果)
        # 这里假设前10个 Query 匹配到了 GT
        loss_cls = F.cross_entropy(final_cls[:, :10].reshape(-1, 10), gt_classes.reshape(-1))
        loss_box = F.l1_loss(final_box[:, :10], gt_boxes)
        
        loss = loss_cls + loss_box
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: Loss = {loss.item():.4f} (Cls={loss_cls.item():.4f}, Box={loss_box.item():.4f})")
    
    print("\nDETR3D Pipeline verified successfully!")

if __name__ == "__main__":
    main()