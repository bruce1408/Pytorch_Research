import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DETR3D_Core(nn.Module):
    def __init__(self, num_queries=300, embed_dim=256, num_cameras=6):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras

        # -----------------------------------------------------------
        # 1. Object Queries (侦探)
        # -----------------------------------------------------------
        # 这是 DETR3D 的起点。这些是可学习的参数，一开始随机初始化。
        # 它们最终会学会"我想找什么位置的物体"。
        self.query_embedding = nn.Embedding(num_queries, embed_dim)

        # -----------------------------------------------------------
        # 2. Reference Point Head (侦探猜位置)
        # -----------------------------------------------------------
        # 一个小的 MLP，负责把 Query 翻译成 3D 坐标 (x, y, z)
        # 输出是 3 维，经过 sigmoid 归一化到 [0, 1] 区间
        self.reference_point_head = nn.Linear(embed_dim, 3)

        # -----------------------------------------------------------
        # 3. Attention & Refinement (侦探修正)
        # -----------------------------------------------------------
        # 这是一个简化版的 Transformer Decoder Layer
        # 用于融合从图片里抓回来的特征，并更新 Query
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 最终预测头 (3D 框回归 + 类别分类)
        self.box_head = nn.Linear(embed_dim, 10) # x, y, z, w, h, l, rot, vel_x, vel_y, score
        
    def get_reference_points(self, query_embed):
        """
        [Step 1] 自顶向下：从 Query 猜出 3D 坐标
        """
        # query_embed: (B, Q, C)
        ref_points = self.reference_point_head(query_embed)
        
        # 使用 sigmoid 归一化到 [0, 1]，代表在场景空间中的相对位置
        ref_points = ref_points.sigmoid() 
        return ref_points

    def project_3d_to_2d(self, reference_points, lidar2img):
        """
        [Step 2] 几何投影：3D 点 -> 2D 像素坐标
        这是 DETR3D 与 LSS 最大的反差：LSS 是 Lift (2D->3D)，这是 Project (3D->2D)
        
        参数:
            reference_points: (B, Q, 3) 归一化的 3D 坐标
            lidar2img: (B, N, 4, 4) 变换矩阵 (Lidar -> Camera -> Image)
        """
        B, Q, _ = reference_points.shape
        N = self.num_cameras

        # 1. 反归一化：将 [0, 1] 映射回真实的物理场景范围 (假设场景 -50m ~ 50m)
        # 这里简化处理，假设 reference_points 已经是真实坐标
        # 真实代码中会有: real_pt = ref_pt * (pc_range_max - pc_range_min) + pc_range_min
        
        # 扩展维度以适配多相机: (B, 1, Q, 3) -> (B, N, Q, 3)
        points_3d = reference_points.unsqueeze(1).expand(-1, N, -1, -1)
        
        # 变成齐次坐标 (x, y, z, 1): (B, N, Q, 4)
        ones = torch.ones_like(points_3d[..., :1])
        points_3d_homo = torch.cat([points_3d, ones], dim=-1)
        
        # 2. 矩阵乘法投影
        # lidar2img: (B, N, 4, 4)
        # points_3d: (B, N, Q, 4)
        # result:    (B, N, Q, 4) -> (u*w, v*w, w, 1)
        points_2d_homo = torch.matmul(lidar2img.unsqueeze(2), points_3d_homo.unsqueeze(-1)).squeeze(-1)
        
        # 3. 深度归一化 (除以 w) 得到像素坐标 (u, v)
        eps = 1e-5
        z_depth = points_2d_homo[..., 2:3]
        points_2d = points_2d_homo[..., 0:2] / (torch.abs(z_depth) + eps)
        
        # 4. 判断点是否在相机前方 (深度 > 0)
        mask = (z_depth > eps).squeeze(-1) # (B, N, Q)
        
        return points_2d, mask

    def feature_sampling(self, image_features, points_2d, mask, img_h, img_w):
        """
        [Step 3] 找证据：双线性插值采样 (Bilinear Sampling)
        去图片上的 (u, v) 位置把特征抓出来。
        """
        B, N, C, H, W = image_features.shape
        Q = points_2d.shape[2]
        
        # 1. 坐标归一化
        # grid_sample 要求坐标在 [-1, 1] 之间
        # points_2d 是像素坐标 (u, v)，u在[0, W], v在[0, H]
        points_2d_norm = torch.zeros_like(points_2d)
        points_2d_norm[..., 0] = points_2d[..., 0] / (img_w - 1) * 2 - 1 # x (u)
        points_2d_norm[..., 1] = points_2d[..., 1] / (img_h - 1) * 2 - 1 # y (v)
        
        # 2. 准备 grid_sample 的输入
        # 输入特征: (B*N, C, H, W)
        # 采样网格: (B*N, Q, 1, 2)
        feats_flat = image_features.view(B*N, C, H, W)
        grid = points_2d_norm.view(B*N, Q, 1, 2)
        
        # 3. 执行采样! (Top-down 的精髓)
        # 这里的 sampled_feats 就是我们找到的"证据"
        sampled_feats = F.grid_sample(feats_flat, grid, align_corners=False) # (B*N, C, Q, 1)
        
        # 还原维度: (B, N, C, Q)
        sampled_feats = sampled_feats.view(B, N, C, Q).permute(0, 3, 1, 2) # (B, Q, N, C)
        
        # 4. 掩码过滤 (Masking)
        # 如果投影点在图片外面，或者在相机背后，这个特征就是无效的(0)
        mask = mask.unsqueeze(-1) # (B, N, Q, 1) -> (B, Q, N, 1)
        mask = mask.permute(0, 2, 1, 3)
        
        # 还要过滤掉 u,v 超出 [-1, 1] 的点
        valid_coords = (points_2d_norm[..., 0] > -1) & (points_2d_norm[..., 0] < 1) & \
                       (points_2d_norm[..., 1] > -1) & (points_2d_norm[..., 1] < 1)
                       
        valid_coords = valid_coords.view(B, Q, N, 1)
        
        final_mask = mask & valid_coords
        
        # 5. 特征聚合 (Aggregation)
        # 一个 3D 点可能被多个相机同时看到 (重叠区域)
        # DETR3D 的做法很简单：对有效相机的特征求平均 (Mean) 或 求和 (Sum)
        sampled_feats = sampled_feats * final_mask # 无效位置置零
        sum_feats = sampled_feats.sum(dim=2)       # 对 N 维度求和 -> (B, Q, C)
        count = final_mask.sum(dim=2).clamp(min=1) # 计数
        
        avg_feats = sum_feats / count # (B, Q, C)
        
        return avg_feats

    def forward(self, image_features, lidar2img):
        """
        image_features: (B, N, C, H, W) 来自 ResNet 的 2D 特征
        lidar2img: (B, N, 4, 4) 投影矩阵
        """
        B, N, C, H, W = image_features.shape
        
        # 1. 拿到所有的侦探 (Query)
        query_embed = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1) # (B, Q, C)
        
        # 2. 侦探先猜一个 3D 位置 (Reference Points)
        reference_points_3d = self.get_reference_points(query_embed) # (B, Q, 3)
        
        # 3. 把 3D 位置投影回 2D 图片
        points_2d, mask = self.project_3d_to_2d(reference_points_3d, lidar2img)
        
        # 4. 去图片里抓特征 (Sampling)
        # 这就是 Cross-Attention 的核心：Value 是图片特征，Key 是位置
        tgt = self.feature_sampling(image_features, points_2d, mask, H, W) # (B, Q, C)
        
        # 5. Self-Attention (侦探们互相交流)
        # "哎，我找到车头了，你找到车尾了吗？"
        query_embed = query_embed + tgt # 残差连接
        query_embed = query_embed.transpose(0, 1) # (Q, B, C) required by nn.MultiheadAttention
        query_embed, _ = self.self_attn(query_embed, query_embed, query_embed)
        query_embed = query_embed.transpose(0, 1) # (B, Q, C)
        
        # 6. 最终预测 (Refinement)
        outputs = self.box_head(self.norm1(query_embed))
        
        return outputs, reference_points_3d

# ==========================================
# 模拟运行
# ==========================================
if __name__ == "__main__":
    print("DETR3D 核心流程演示...")
    
    # 参数设置
    B, N, Q, C = 1, 6, 300, 256
    H, W = 16, 44 # 特征图大小
    
    model = DETR3D_Core(num_queries=Q, embed_dim=C, num_cameras=N)
    
    # 模拟输入数据
    # 1. 2D 图像特征 (ResNet 提取出来的)
    fake_img_feats = torch.randn(B, N, C, H, W)
    
    # 2. 投影矩阵 (模拟)
    # 正常情况下的变换结果是
    # lidar2img = (相机内参矩阵) @ (相机外参矩阵) @ (Lidar到车身的变换矩阵)

    # [ R_11  R_12  R_13 | t_x ]
    # [ R_21  R_22  R_23 | t_y ]
    # [ R_31  R_32  R_33 | t_z ]
    # [ -----------------|---- ]
    # [   0     0     0  |  1  ]
    fake_lidar2img = torch.eye(4).view(1, 1, 4, 4).expand(B, N, 4, 4)
    
    print("Step 1: 随机初始化 Query")
    print("Step 2: 预测 Reference Points (3D)")
    print("Step 3: 投影到 2D 并采样特征")
    print("Step 4: 更新 Query 并输出结果")
    
    outputs, ref_pts = model(fake_img_feats, fake_lidar2img)
    
    print("-" * 30)
    print(f"输入特征: {fake_img_feats.shape}")
    print(f"输出结果: {outputs.shape} (B, Q, 10)")
    print(f"参考点位置: {ref_pts.shape} (B, Q, 3)")
    print("DETR3D 运行成功！")