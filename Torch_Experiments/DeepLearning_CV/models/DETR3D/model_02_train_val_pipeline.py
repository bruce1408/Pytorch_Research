import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
import numpy as np

# ==========================================
# Part 1: DETR3D 专用 Cross-Attention 模块
# ==========================================
class Detr3dCrossAtten(nn.Module):
    """
    DETR3D 的核心创新点：
    用 '3D-to-2D 投影 + Grid Sample' 替代了传统的 'Query-Key 点积注意力'。
    """
    def __init__(self, embed_dim=256, num_cameras=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        
        # 这里的 output_proj 对应标准 Attention 最后的 Wo 矩阵
        # 作用是将采样到的视觉特征融合回 Query 的维度
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, reference_points, image_features, lidar2img):
        """
        query: (B, Q, C) - Object Queries
        reference_points: (B, Q, 3) - 3D 参考点 (0-1归一化)
        image_features: (B, N, C, H, W) - 环视图像特征
        lidar2img: (B, N, 4, 4) - 投影矩阵
        """
        # 1. 投影: 3D 参考点 -> 2D 像素坐标
        # points_2d: (B, N, Q, 2), mask: (B, N, Q)
        points_2d, mask = self.project_3d_to_2d(reference_points, lidar2img)
        
        # 2. 采样: 利用 grid_sample 提取特征 (相当于 Attention 中的 Value 提取)
        # sampled_feats: (B, Q, N, C)
        sampled_feats = self.feature_sampling(image_features, points_2d, mask)
        
        # 3. 聚合: 把 N 个相机的特征平均一下 (相当于 Attention 中不同 Head 的融合)
        # (B, Q, N, C) -> (B, Q, C)
        # 这里简化处理：直接求和或平均。官方代码会用 Attention Weight 来加权不同相机。
        aggregated_feats = sampled_feats.mean(dim=2) 
        
        # 4. 输出投影: 线性映射 + Dropout
        return self.dropout(self.output_proj(aggregated_feats))

    def project_3d_to_2d(self, reference_points, lidar2img):
        B, Q, _ = reference_points.shape
        N = self.num_cameras # 6
        
        # (B, 1, Q, 3) -> (B, N, Q, 3) 复制多份
        points_3d = reference_points.unsqueeze(1).expand(-1, N, -1, -1)
        
        # 构造齐次坐标 (x, y, z, 1)
        ones = torch.ones_like(points_3d[..., :1])
        points_3d_homo = torch.cat([points_3d, ones], dim=-1) # (B, N, Q, 4)
        
        # 矩阵乘法: (B, N, 4, 4) @ (B, N, Q, 4, 1) -> (B, N, Q, 4)
        # lidar2img 需要 unsqueeze(2) 来广播 Q 维度
        points_2d_homo = torch.matmul(lidar2img.unsqueeze(2), points_3d_homo.unsqueeze(-1)).squeeze(-1)
        
        # 透视除法: u = x/z, v = y/z
        eps = 1e-5
        z_depth = points_2d_homo[..., 2:3]
        points_2d = points_2d_homo[..., 0:2] / (torch.abs(z_depth) + eps)
        
        # 生成 Mask: 只保留在相机前方的点 (z > 0)
        mask = (z_depth > eps).squeeze(-1) # (B, N, Q)
        return points_2d, mask

    def feature_sampling(self, image_features, points_2d, mask):
        B, N, C, H, W = image_features.shape
        Q = points_2d.shape[2]
        
        # 归一化坐标到 [-1, 1] 以供 grid_sample 使用
        points_2d_norm = points_2d.clone()
        points_2d_norm[..., 0] = points_2d[..., 0] / (W - 1) * 2 - 1
        points_2d_norm[..., 1] = points_2d[..., 1] / (H - 1) * 2 - 1
        
        # 准备 Grid Sample
        # 将 Batch 和 Camera 维度合并: (B*N, C, H, W)
        feats_flat = image_features.view(B*N, C, H, W)
        
        # Grid: (B*N, Q, 1, 2)
        grid = points_2d_norm.view(B*N, Q, 1, 2)
        
        # 双线性插值采样
        # output: (B*N, C, Q, 1)
        sampled_feats = F.grid_sample(feats_flat, grid, align_corners=False)
        
        # 恢复形状: (B, N, C, Q) -> (B, Q, N, C)
        sampled_feats = sampled_feats.view(B, N, C, Q).permute(0, 3, 1, 2)
        
        # 应用 Mask: 把投影到图片范围外或相机背后的特征置为 0
        # 判断坐标是否在 [-1, 1] 范围内
        in_image_mask = (points_2d_norm.abs().max(dim=-1)[0] <= 1) # (B, N, Q)
        valid_mask = mask & in_image_mask # (B, N, Q)
        
        # (B, Q, N, 1)
        valid_mask = valid_mask.permute(0, 2, 1).unsqueeze(-1)
        
        return sampled_feats * valid_mask

# ==========================================
# Part 2: 完整的 Transformer Decoder Layer
# ==========================================
class DETR3D_DecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        
        # 1. Self Attention (Query 之间的交互)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        # 2. Cross Attention (Query 与 Image 的交互) -> 这里使用了我们自定义的模块
        self.cross_attn = Detr3dCrossAtten(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        # 3. FFN (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, query, reference_points, image_features, lidar2img):
        """
        标准的 Transformer Decoder 流程:
        Query -> SelfAttn -> Norm -> CrossAttn (DETR3D版) -> Norm -> FFN -> Norm
        """
        
        # A. Self Attention
        # query2 = self.self_attn(query, query, query)[0]
        # query = self.norm1(query + self.dropout1(query2))
        # (为了简化，这里先做 Cross 再做 Self，或者直接使用)
        
        q2 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + self.dropout1(q2))
        
        # B. Cross Attention (Detr3d Special)
        # 输入除了 Query，还需要 Reference Points 和 图像特征
        q2 = self.cross_attn(query, reference_points, image_features, lidar2img)
        query = self.norm2(query + self.dropout2(q2))
        
        # C. FFN
        q2 = self.ffn(query)
        query = self.norm3(query + self.dropout3(q2))
        
        return query

# ==========================================
# Part 3: 主模型结构更新
# ==========================================
class DETR3D_Model(nn.Module):
    def __init__(self, num_queries=300, embed_dim=256, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Object Queries (可学习)
        self.query_embedding = nn.Embedding(num_queries, embed_dim)
        
        # 生成参考点的 Head (Query -> 3D Point)
        self.ref_point_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3)
        )
        
        # 堆叠多层 Decoder
        self.layers = nn.ModuleList([
            DETR3D_DecoderLayer(embed_dim) for _ in range(num_layers)
        ])
        
        # 输出头
        self.class_head = nn.Linear(embed_dim, 2) # 0:BG, 1:FG
        
        # 预测相对参考点的偏移量
        self.box_head = nn.Linear(embed_dim, 6) 

    def forward(self, image_features, lidar2img):
        B = image_features.shape[0]
        
        # 初始化 Query: (B, Q, C)
        query = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        
        # 从 Query 预测初始参考点 (Sigmoid归一化到0-1)
        reference_points = self.ref_point_head(query).sigmoid()
        
        # 逐层 Refine (Iterative Refinement)
        # DETR3D 的特点：每一层都会更新 Query，进而可以更新参考点(可选)
        for layer in self.layers:
            # 1. 过 Decoder Layer
            query = layer(query, reference_points, image_features, lidar2img)
            
            # 2. (可选) 可以在这里用 box_head 预测偏移量，更新 reference_points
            # 为了代码简单，这里只在最后预测
            
        # 最终预测
        cls_logits = self.class_head(query)
        box_offsets = self.box_head(query)
        
        # 最终 Box = 参考点 + 预测的偏移量 (简化逻辑)
        # 实际代码中需要用 InverseSigmoid 做坐标转换
        pred_boxes = torch.cat([
            (reference_points + box_offsets[..., :3]).clamp(0, 1), # xyz
            box_offsets[..., 3:].sigmoid() # wlh
        ], dim=-1)
        
        return {"pred_logits": cls_logits, "pred_boxes": pred_boxes}

# -------------------------------------------------------------
# 下面是保持不变的训练代码部分 (Matcher, Loss, Data loop)...
# 为了运行，你可以直接把之前的 HungarianMatcher, SetCriterion,
# FakeDETR3DDataset, main 函数复制到这里。
# -------------------------------------------------------------
# (这里仅展示如何修改 main 函数中的 model 初始化)


    
# ==========================================
# Part 1: DETR3D 模型定义 (精简版)
# ==========================================
# class DETR3D_Core(nn.Module):
#     def __init__(self, num_queries=300, embed_dim=256, num_cameras=6):
#         super().__init__()
#         self.num_queries = num_queries
#         self.embed_dim = embed_dim
#         self.num_cameras = num_cameras
        
#         # Object Queries
#         self.query_embedding = nn.Embedding(num_queries, embed_dim)
        
#         # Heads
#         self.reference_point_head = nn.Linear(embed_dim, 3)
#         self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
        
#         # 输出头: 
#         # class_head: 输出 2 类 (0: 背景, 1: 物体)。实际中通常是 num_classes + 1
#         self.class_head = nn.Linear(embed_dim, 2) 
#         # box_head: 输出 [x, y, z, w, l, h] (简化版, 不含旋转速度)
#         self.box_head = nn.Linear(embed_dim, 6) 

#     def get_reference_points(self, query_embed):
#         return self.reference_point_head(query_embed).sigmoid()

#     def project_3d_to_2d(self, reference_points, lidar2img):
#         B, Q, _ = reference_points.shape
#         N = self.num_cameras
        
#         # 简化坐标变换逻辑用于演示 (假设场景归一化在 0-1 之间) -> (B, N, Q, 3)
#         points_3d = reference_points.unsqueeze(1).expand(-1, N, -1, -1) 
#         ones = torch.ones_like(points_3d[..., :1])
#         points_3d_homo = torch.cat([points_3d, ones], dim=-1)
        
#         points_2d_homo = torch.matmul(lidar2img.unsqueeze(2), points_3d_homo.unsqueeze(-1)).squeeze(-1)
        
#         eps = 1e-5
#         z_depth = points_2d_homo[..., 2:3]
#         points_2d = points_2d_homo[..., 0:2] / (torch.abs(z_depth) + eps)
#         mask = (z_depth > eps).squeeze(-1)
#         return points_2d, mask

#     def feature_sampling(self, image_features, points_2d, mask):
#         B, N, C, H, W = image_features.shape
#         Q = points_2d.shape[2]
        
#         # 归一化到 grid_sample 所需的 [-1, 1]
#         points_2d_norm = points_2d.clone()
#         points_2d_norm[..., 0] = points_2d[..., 0] / (W - 1) * 2 - 1
#         points_2d_norm[..., 1] = points_2d[..., 1] / (H - 1) * 2 - 1
        
#         feats_flat = image_features.view(B*N, C, H, W)
#         grid = points_2d_norm.view(B*N, Q, 1, 2)
        
#         sampled_feats = F.grid_sample(feats_flat, grid, align_corners=False) # (B*N, C, Q, 1)
#         sampled_feats = sampled_feats.view(B, N, C, Q).permute(0, 3, 1, 2)   # (B, Q, N, C)
        
#         # 简单平均融合
#         mask = mask.permute(0, 2, 1).unsqueeze(-1) # (B, Q, N, 1)
#         valid_mask = mask & (points_2d_norm.view(B, Q, N, 2).abs().max(dim=-1, keepdim=True)[0] < 1)
        
#         sampled_feats = sampled_feats * valid_mask
#         sum_feats = sampled_feats.sum(dim=2)
#         count = valid_mask.sum(dim=2).clamp(min=1)
#         return sum_feats / count

#     def forward(self, image_features, lidar2img):
#         B = image_features.shape[0]
#         query_embed = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        
#         # 1. Top-down: 猜位置
#         ref_points = self.get_reference_points(query_embed)
        
#         # 2. Geometry: 投影采样
#         points_2d, mask = self.project_3d_to_2d(ref_points, lidar2img)
#         tgt = self.feature_sampling(image_features, points_2d, mask)
        
#         # 3. Refine: 交互与预测
#         query_embed = query_embed + tgt
#         query_embed, _ = self.self_attn(query_embed, query_embed, query_embed)
#         query_embed = self.norm1(query_embed)
        
#         # 输出绝对坐标 (ref_points + 偏移量)
#         # 这是一个简化的回归逻辑，实际 DETR3D 会更复杂
#         box_preds = self.box_head(query_embed)
#         # 强制前3维是坐标，加上参考点 (Inverse Sigmoid 略过，直接加)
#         box_preds[..., :3] = box_preds[..., :3].sigmoid() # 归一化坐标
        
#         cls_logits = self.class_head(query_embed)
        
#         return {"pred_logits": cls_logits, "pred_boxes": box_preds}


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Numerically stable inverse sigmoid (logit) function.

    Args:
        x (Tensor): input in range [0, 1]
        eps (float): clamp value to avoid numerical overflow

    Returns:
        Tensor: logit(x)
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


# class DETR3D_Core(nn.Module):
#     def __init__(self, num_queries=300, embed_dim=256, num_cameras=6):
#         super().__init__()
#         self.num_queries = num_queries
#         self.embed_dim = embed_dim
#         self.num_cameras = num_cameras

#         # Object Queries
#         self.query_embedding = nn.Embedding(num_queries, embed_dim)

#         # Predict 3D reference points (normalized [0,1])
#         self.reference_point_head = nn.Linear(embed_dim, 3)

#         # Cross-attention: query attends to sampled multi-view features
#         self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
#         self.norm_cross = nn.LayerNorm(embed_dim)

#         # Self-attention among queries (dedup / interaction)
#         self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
#         self.norm_self = nn.LayerNorm(embed_dim)

#         # Heads
#         self.class_head = nn.Linear(embed_dim, 2)
#         self.box_head = nn.Linear(embed_dim, 6)  # [cx,cy,cz,w,l,h] in normalized space for cx,cy,cz

#     def get_reference_points(self, query_embed):
        
#         # (B,Q,C) -> (B,Q,3) in [0,1]
#         return self.reference_point_head(query_embed).sigmoid()

#     def project_3d_to_2d(self, reference_points, lidar2img):
#         """
#         reference_points: (B,Q,3) normalized 3D
#         lidar2img: (B,N,4,4)  (here used as generic projection matrix)
#         returns:
#             points_2d: (B,N,Q,2) in pixel space (not normalized to [-1,1] yet)
#             mask:      (B,N,Q) valid if depth>0
#         """
#         B, Q, _ = reference_points.shape
#         N = self.num_cameras

#         points_3d = reference_points.unsqueeze(1).expand(-1, N, -1, -1)  # (B,N,Q,3)
#         ones = torch.ones_like(points_3d[..., :1])
#         points_3d_homo = torch.cat([points_3d, ones], dim=-1)  # (B,N,Q,4)

#         # (B,N,4,4) @ (B,N,Q,4,1) -> (B,N,Q,4)
#         points_2d_homo = torch.matmul(lidar2img.unsqueeze(2), points_3d_homo.unsqueeze(-1)).squeeze(-1)

#         eps = 1e-5
#         z = points_2d_homo[..., 2:3]  # (B,N,Q,1)
#         points_2d = points_2d_homo[..., 0:2] / (z.abs() + eps)  # (B,N,Q,2)

#         mask = (z > eps).squeeze(-1)  # (B,N,Q)
#         return points_2d, mask

#     def feature_sampling_per_cam(self, image_features, points_2d, mask):
#         """
#         image_features: (B,N,C,H,W)
#         points_2d:      (B,N,Q,2) in pixel coords
#         mask:           (B,N,Q) depth-valid

#         returns:
#           sampled_feats: (B,Q,N,C)  per-camera token for each query
#           valid_mask:    (B,Q,N)    valid camera mask for each query
#         """
#         B, N, C, H, W = image_features.shape
#         _, _, Q, _ = points_2d.shape

#         # normalize to [-1,1] for grid_sample
#         points_norm = points_2d.clone()
#         points_norm[..., 0] = points_2d[..., 0] / (W - 1) * 2 - 1
#         points_norm[..., 1] = points_2d[..., 1] / (H - 1) * 2 - 1  # (B,N,Q,2)

#         # in-bounds check
#         in_bounds = (points_norm.abs().amax(dim=-1) < 1.0)  # (B,N,Q)
#         valid = mask & in_bounds  # (B,N,Q)

#         # grid_sample expects (BN,C,H,W) and grid (BN, Q, 1, 2)
#         feats_flat = image_features.view(B * N, C, H, W)
#         grid = points_norm.view(B * N, Q, 1, 2)

#         sampled = F.grid_sample(feats_flat, grid, align_corners=False)  # (BN,C,Q,1)
#         sampled = sampled.squeeze(-1).permute(0, 2, 1)                  # (BN,Q,C)
#         sampled = sampled.view(B, N, Q, C).permute(0, 2, 1, 3).contiguous()  # (B,Q,N,C)

#         valid_mask = valid.permute(0, 2, 1).contiguous()  # (B,Q,N)

#         # mask out invalid cams (so K/V tokens become 0; attention also uses key_padding_mask)
#         sampled = sampled * valid_mask.unsqueeze(-1)
#         return sampled, valid_mask

#     def forward(self, image_features, lidar2img):
#         """
#         image_features: (B,N,C,H,W)
#         lidar2img:      (B,N,4,4)
#         returns dict:
#           pred_logits: (B,Q,2)
#           pred_boxes:  (B,Q,6)
#         """
#         B = image_features.shape[0]
#         Q = self.num_queries

#         query_embed = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)  # (B,Q,C)

#         # 1) predict 3D reference points
#         ref_points = self.get_reference_points(query_embed)  # (B,Q,3)

#         # 2) project to each camera and sample features (per camera tokens)
#         points_2d, depth_mask = self.project_3d_to_2d(ref_points, lidar2img)  # (B,N,Q,2),(B,N,Q)
#         sampled_feats, valid_mask = self.feature_sampling_per_cam(image_features, points_2d, depth_mask)
#         # sampled_feats: (B,Q,N,C), valid_mask: (B,Q,N)

#         # -----------------------------
#         # 3) CROSS-ATTENTION (核心修正)
#         #   Q: query_embed         (B,Q,C)
#         #   K,V: per-cam sampled   (B,Q,N,C) -> reshape to (B*Q, N, C)
#         # -----------------------------
#         q = query_embed.reshape(B * Q, 1, self.embed_dim)          # (BQ,1,C)
#         kv = sampled_feats.reshape(B * Q, self.num_cameras, self.embed_dim)  # (BQ,N,C)

#         # key_padding_mask: True means "ignore"
#         kpm = (~valid_mask).reshape(B * Q, self.num_cameras)       # (BQ,N)

#         cross_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=kpm)  # (BQ,1,C)
#         cross_out = cross_out.reshape(B, Q, self.embed_dim)              # (B,Q,C)

#         query_embed = self.norm_cross(query_embed + cross_out)           # residual + norm

#         # 4) SELF-ATTENTION among queries (DETR-style)
#         self_out, _ = self.self_attn(query_embed, query_embed, query_embed)  # (B,Q,C)
#         query_embed = self.norm_self(query_embed + self_out)

#         # 5) heads
#         cls_logits = self.class_head(query_embed)   # (B,Q,2)

#         delta = self.box_head(query_embed)          # (B,Q,6)

#         # more stable center regression: center = sigmoid(inv_sigmoid(ref) + delta)
#         center_logit = inverse_sigmoid(ref_points) + delta[..., :3]
#         center = center_logit.sigmoid()

#         # keep sizes positive (optional, but helpful)
#         size = F.relu(delta[..., 3:])  # w,l,h >= 0

#         pred_boxes = torch.cat([center, size], dim=-1)  # (B,Q,6)

#         return {"pred_logits": cls_logits, "pred_boxes": pred_boxes}


# ==========================================
# Part 2: 匈牙利匹配器 (核心难点)
# ==========================================
class HungarianMatcher(nn.Module):
    """
    DETR 的灵魂：解决预测框和真值框 "谁负责谁" 的问题。
    这是一个二分图匹配问题。
    """
    def __init__(self, cost_class=1, cost_bbox=5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad() # 匹配过程不需要梯度
    def forward(self, outputs, targets):
        """
        outputs: 字典，包含 pred_logits (B, Q, 2) 和 pred_boxes (B, Q, 6)
        targets: 列表，每个元素是一个 dict，包含 'labels' (M,) 和 'boxes' (M, 6)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 展平 Batch，把大家放在一起匹配 (为了并行计算)
        # out_prob: (B*Q, 2)
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  
        # out_bbox: (B*Q, 6)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # 2. 准备真值
        # 拼接所有 batch 的真值
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 3. 计算成本矩阵 (Cost Matrix)
        # Cost 1: 分类成本 (我们希望预测成正确类别的概率越大越好 -> Cost越小)
        # 取出对应真值类别的概率。这里假设 targets label 全是 1 (物体)
        # out_prob[:, tgt_ids] 提取出对应真值类别的概率列
        cost_class = -out_prob[:, tgt_ids]

        # Cost 2: 回归成本 (L1 距离)
        # cdist 计算成对距离 (B*Q, Total_GT_Num)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 总成本 C
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        
        # 重塑回 (B, Q, GT_Num_in_Batch) 进行单独匹配
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        
        # 对 Batch 里每一张图单独做匈牙利匹配
        # linear_sum_assignment 是 scipy 提供的匈牙利算法实现
        for i, c in enumerate(C.split(sizes, -1)):
            # c[i] shape: (Q, num_gt_i)
            # result: (row_ind, col_ind) -> (预测索引, 真值索引)
            ind = linear_sum_assignment(c[i])
            indices.append(ind)

        # 返回匹配结果: [(pred_idx, gt_idx), ...]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# ==========================================
# Part 3: 损失函数 (SetCriterion)
# ==========================================
class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        # CrossEntropyLoss: 0是背景，1是物体
        # 如果匹配上了，就学物体；没匹配上(indices之外的)，就学背景
        self.loss_ce = nn.CrossEntropyLoss(reduction='mean') 

    def forward(self, outputs, targets):
        # 1. 进行匹配
        indices = self.matcher(outputs, targets)
        
        # 2. 提取匹配好的索引
        # idx 是一个 tuple (batch_idx, src_idx)，代表哪些预测框匹配上了真值
        idx = self._get_src_permutation_idx(indices)
        
        # --- Loss 1: 分类损失 ---
        # 目标构建: 初始化全为 0 (背景)
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"] for t in targets]) # 真值类别(全为1)
        
        # 构建一个全 0 的 target_classes (B, Q)
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        # 将匹配上的位置设为 1 (物体)
        target_classes[idx] = target_classes_o

        # 计算分类损失 (permute 为了适配 CrossEntropy: B, C, Q)
        loss_ce = self.loss_ce(src_logits.transpose(1, 2), target_classes)

        # --- Loss 2: 回归损失 (只计算匹配上的框) ---
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='mean')

        return {'loss_ce': loss_ce, 'loss_bbox': loss_bbox}

    def _get_src_permutation_idx(self, indices):
        # 辅助函数：把 list of tuples 变成 tensor索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

# ==========================================
# Part 4: 模拟数据与训练循环
# ==========================================
class FakeDETR3DDataset(Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 模拟输入
        features = torch.randn(6, 256, 16, 44) # (N, C, H, W)
        lidar2img = torch.eye(4).unsqueeze(0).repeat(6, 1, 1) # 简化矩阵
        
        # 模拟真值 (Target)
        # 每张图随机生成 3-5 个物体
        num_objs = np.random.randint(3, 6)
        # Box: [x, y, z, w, l, h] 都在 0-1 之间
        boxes = torch.rand(num_objs, 6)
        # Label: 全是 1 (代表 Car)
        labels = torch.ones(num_objs, dtype=torch.long)
        
        target = {"boxes": boxes, "labels": labels}
        return features, lidar2img, target

def collate_fn(batch):
    # 自定义 collate，因为 target 是不定长的 list
    features = torch.stack([item[0] for item in batch])
    lidar2imgs = torch.stack([item[1] for item in batch])
    targets = [item[2] for item in batch]
    return features, lidar2imgs, targets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化
    # model = DETR3D_Core().to(device)
    model = DETR3D_Model(num_layers=2).to(device) # 使用新模型

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5)
    criterion = SetCriterion(matcher, weight_dict={'loss_ce': 1, 'loss_bbox': 5}).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 2. 数据
    dataset = FakeDETR3DDataset(length=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 3. 训练循环
    print("开始训练...")
    for epoch in range(3): # 跑 3 个 epoch
        model.train()
        total_loss = 0
        
        for batch_idx, (feats, l2i, targets) in enumerate(dataloader):
            feats, l2i = feats.to(device), l2i.to(device)
            # targets 需要把 tensor 移到 GPU
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward
            optimizer.zero_grad()
            outputs = model(feats, l2i)

            # Loss (内部自动做 Hungarian Matching)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = loss_dict['loss_ce'] * weight_dict['loss_ce'] + \
                     loss_dict['loss_bbox'] * weight_dict['loss_bbox']

            # Backward
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            
            if batch_idx % 15 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | "
                      f"CE: {loss_dict['loss_ce'].item():.4f} | "
                      f"Box: {loss_dict['loss_bbox'].item():.4f}")

    # 4. 简单的验证步骤
    print("\n简单验证 (看是否预测出物体)...")
    model.eval()
    with torch.no_grad():
        feats, l2i, _ = next(iter(dataloader))
        feats, l2i = feats.to(device), l2i.to(device)
        outputs = model(feats, l2i)
        
        # 取第一个样本
        probs = outputs['pred_logits'][0].softmax(-1) # (Q, 2)
        boxes = outputs['pred_boxes'][0] # (Q, 6)
        
        # 只有当 '物体类(1)' 概率 > 0.7 才算检测到
        scores, labels = probs.max(-1) # 0:背景, 1:物体
        keep = (labels == 1) & (scores > 0.7)
        
        print(f"检测到的物体数量: {keep.sum().item()}")
        if keep.sum() > 0:
            print(f"Top 1 Box Score: {scores[keep][0]:.4f}")
            print(f"Top 1 Box Coord: {boxes[keep][0].cpu().numpy()}")


if __name__ == "__main__":
    main()
    
    