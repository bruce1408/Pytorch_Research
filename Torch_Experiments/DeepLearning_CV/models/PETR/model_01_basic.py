import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
class PETRConfig:
    # 图像参数
    num_cams = 6
    img_h = 256
    img_w = 704
    
    # 特征参数
    stride = 16           # Backbone 下采样倍率
    feat_h = img_h // stride
    feat_w = img_w // stride
    embed_dim = 256       # Transformer 特征维度 (C)
    
    # 深度离散化 (PETR 用来生成 3D 坐标网格)
    depth_step = 4        # 简单起见，PETR通常只取几个深度点或者归一化方向向量
                          # 这里我们还原论文，生成视锥网格
    
    # 3D 空间范围 (用于归一化 3D 坐标)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # Transformer
    num_queries = 900     # Object Query 数量
    num_decoder_layers = 6
    num_heads = 8

# ==========================================
# 2. 简单的 Backbone (Mock)
# ==========================================
class SimpleBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 假设输入是 (B, N, 3, H, W)
        # 用几个卷积层模拟 ResNet-50 的输出
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), # /2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),     # /4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# /8
            nn.ReLU(),
            nn.Conv2d(128, cfg.embed_dim, kernel_size=3, stride=2, padding=1), # /16
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B*N, 3, H, W)
        return self.net(x) # -> (B*N, 256, H/16, W/16)

# ==========================================
# 3. 核心算法: PETR 3D 位置编码生成器
# ==========================================
class PETRImpl(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. Backbone
        self.backbone = SimpleBackbone(cfg)
        
        # 2. 3D Coordinates Generator (生成 3D 坐标)
        # 这里需要创建一个固定的网格用于反投影
        self.coords_mesh = self.create_meshgrid(cfg.feat_h, cfg.feat_w)
        
        # 3. 3D Position Encoder (3D PE MLP)
        # 论文结构: Linear -> ReLU -> Linear
        # 输入是 3D 坐标 (x,y,z)，一般会先编码成更高维或者直接映射
        # 这里严格还原论文：3D coords -> PE
        self.position_encoder = nn.Sequential(
            nn.Linear(3, cfg.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
        )
        
        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=False # PyTorch Transformer 默认 (Seq, Batch, Dim)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)
        
        # 5. Object Queries (可学习的锚点)
        self.object_queries = nn.Embedding(cfg.num_queries, cfg.embed_dim)
        
        # 6. Detection Head (简单版)
        self.cls_head = nn.Linear(cfg.embed_dim, 10) # 10类
        self.reg_head = nn.Linear(cfg.embed_dim, 10) # x,y,z,w,l,h,sin,cos,vel_x,vel_y

    def create_meshgrid(self, H, W):
        """
        生成像素坐标网格 (y, x)
        """
        ys = torch.linspace(0, 1, H)
        xs = torch.linspace(0, 1, W)
        # 注意：PETR 这里生成的通常是归一化的像素坐标，或者真实像素坐标
        # 这里我们生成真实像素坐标，后续配合内参使用
        ys = ys * (self.cfg.img_h - 1)
        xs = xs * (self.cfg.img_w - 1)
        
        coords_y, coords_x = torch.meshgrid(ys, xs, indexing='ij')
        
        # Stack 得到 (H, W, 2) -> (u, v)
        coords = torch.stack([coords_x, coords_y], dim=-1)
        return nn.Parameter(coords, requires_grad=False)

    def get_3d_coordinates(self, rots, trans, intrins, inv_intrins):
        """
        [算法核心]: 将 2D 像素反投影回 3D 世界坐标
        
        rots, trans: 相机外参 (Cam -> Ego/World)
        intrins: 相机内参
        """
        B, N, _ = trans.shape # (B, 6, 3)
        H, W = self.cfg.feat_h, self.cfg.feat_w
        
        # 1. 准备像素坐标 (B, N, H, W, 2)
        # 扩展维度以匹配 Batch 和 Cam
        coords = self.coords_mesh.view(1, 1, H, W, 2).repeat(B, N, 1, 1, 1)
        
        # 2. 像素 -> 归一化相机坐标 (u, v, 1) * d
        # 我们先把 (u, v) 变成 (u, v, 1)
        coords_d = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1) # (B, N, H, W, 3)
        
        # 为了做矩阵乘法，把 H,W 展平 -> (B, N, H*W, 3)
        coords_flat = coords_d.view(B, N, -1, 3)
        
        # 乘以内参逆矩阵: P_cam = K_inv * P_pixel
        # inv_intrins: (B, N, 3, 3)
        # 维度对齐: (B, N, 1, 3, 3) @ (B, N, HW, 3, 1) -> (B, N, HW, 3)
        # 为了方便，转置一下坐标进行乘法
        img_points = torch.matmul(inv_intrins.unsqueeze(2), coords_flat.unsqueeze(-1)).squeeze(-1)
        
        # 3. 引入深度信息 (Discretize Depth)
        # PETR 的做法是生成一条射线，或者取几个深度点。
        # 最经典的做法是：直接使用单位深度 (Depth=1) 的方向向量，
        # 或者在深度维度上复制几次 (D=4)。
        # 论文中提到: "We discretize the camera frustum..."
        # 实际上，PETR 经常使用 "Depth-adaptive" 的方式，
        # 简单实现：我们假设 Depth=1，让网络自己学习尺度。
        # 或者我们模拟 4 个深度的点 (PETR paper settings)
        
        D = 4 # 深度采样数
        depth_vals = torch.tensor([10.0, 20.0, 30.0, 40.0], device=coords.device).view(1,1,1,D)
        
        # (B, N, HW, 3) -> (B, N, HW, D, 3)
        img_points = img_points.unsqueeze(3) * depth_vals.unsqueeze(-1)
        
        # 4. 相机坐标 -> 世界坐标 (Ego)
        # P_world = R * P_cam + T
        # rots: (B, N, 3, 3), trans: (B, N, 3)
        
        # 旋转: (B, N, 1, 1, 3, 3) @ (B, N, HW, D, 3, 1)
        rots_expand = rots.view(B, N, 1, 1, 3, 3)
        img_points_rotated = torch.matmul(rots_expand, img_points.unsqueeze(-1)).squeeze(-1)
        
        # 平移: + T (B, N, 1, 1, 3)
        coords_3d = img_points_rotated + trans.view(B, N, 1, 1, 3)
        
        # Output: (B, N, HW, D, 3) -> 包含 x, y, z
        return coords_3d

    def position_embed(self, coords_3d):
        """
        [算法核心]: 3D 坐标归一化 + MLP 编码
        """
        # 1. 归一化 (Normalize 3D coordinates)
        # 将坐标映射到 [0, 1] 之间 (使用 sigmoid 或 min-max)
        # PETR 论文使用 sigmoid 对坐标进行处理
        # 或者按 pc_range 进行归一化
        
        # 这里演示 Sigmoid 归一化 (PETR 官方代码做法)
        # 因为 3D 坐标范围很大，直接送 MLP 不好学
        # 这里我们把 D 维度和 HW 维度融合，或者做 Global Avg Pooling
        # PETR 简化版通常直接对 D=1 或 D=4 的坐标 embedding 求和或拼接
        
        # 形状变化: (B, N, HW, D, 3) -> (B, N, HW, 3*D)
        B, N, HW, D, C = coords_3d.shape
        coords_feat = coords_3d.permute(0, 1, 2, 4, 3).contiguous().view(B, N, HW, -1)
        
        # 简单的归一化技巧
        coords_feat = torch.sigmoid(coords_feat) 
        
        # 注意: 此时 coords_feat 维度是 3*D (例如 12)
        # 我们需要先把它映射到 embed_dim
        # 为了对齐上面的 position_encoder 定义 (Linear(3 -> ..))
        # 我们这里做一个简化的适配：假设只取一个深度，或者取平均
        coords_mean = coords_3d.mean(dim=3) # (B, N, HW, 3)
        
        # 归一化
        coords_norm = torch.sigmoid(coords_mean)
        
        # 2. MLP 编码
        # (B, N, HW, 3) -> (B, N, HW, Embed_Dim)
        pe = self.position_encoder(coords_norm)
        
        return pe

    def forward(self, imgs, rots, trans, intrins):
        """
        imgs: (B, N, 3, H, W)
        rots, trans, intrins: 相机参数
        """
        B, N, C, H, W = imgs.shape
        
        # 1. Image Feature Extraction
        # -> (B*N, 256, H/16, W/16)
        x = imgs.view(B*N, C, H, W)
        img_feats = self.backbone(x)
        
        # Reshape back to (B, N, C, Hf, Wf)
        _, C_feat, Hf, Wf = img_feats.shape
        img_feats = img_feats.view(B, N, C_feat, Hf, Wf)
        
        # 2. Generate 3D Coordinates (Core of PETR)
        # 计算内参逆矩阵
        inv_intrins = torch.inverse(intrins)
        # 获取 3D 坐标 (B, N, HW, D, 3)
        coords_3d = self.get_3d_coordinates(rots, trans, intrins, inv_intrins)
        
        # 3. Generate 3D Position Embeddings
        # -> (B, N, HW, 256)
        pos_embeds = self.position_embed(coords_3d)
        
        # Reshape Image features to (B, N, HW, 256)
        # 【修复点1】：确保内存连续
        img_feats_flat = img_feats.flatten(3).permute(0, 1, 3, 2).contiguous()
        
        # 4. Add PE to Features (Fusion)
        # 这是 PETR 最关键的一步：将 3D PE 直接加到 2D 特征上
        # 这样 Transformer 就能感知到每个像素的 3D 位置
        transformer_input = img_feats_flat + pos_embeds
        
        # 5. Transformer Decoder
        # (1) Prepare Key/Value
        # Flatten all cameras: (B, N*HW, C)
        # Permute for Transformer: (Seq_Len, Batch, Dim) -> (N*HW, B, C)
        # 【修复点2】：使用 reshape 替代 view，或者 flatten
        memory = transformer_input.flatten(1, 2).permute(1, 0, 2)
        
        # (2) Prepare Query
        # (Num_Queries, 1, C) -> (Num_Queries, B, C)
        query_embed = self.object_queries.weight.unsqueeze(1).repeat(1, B, 1)
        # 初始 Query 通常设为 0，位置编码由 query_embed 提供
        target = torch.zeros_like(query_embed)
        
        # 因为 nn.TransformerDecoder 不支持 query_pos 参数，
        # 我们直接把 位置编码(query_embed) 当作输入的一部分传进去。
        # 在 DETR 中，通常是 Content(0) + Position(Learned)
        tgt = target + query_embed
        
        # (3) Decoding
        # Out: (Num_Queries, B, C)
        hs = self.decoder(tgt, memory, tgt_key_padding_mask=None)
        # (4) Prediction Head
        hs = hs.permute(1, 0, 2) # (B, Num_Queries, C)
        
        all_cls_scores = self.cls_head(hs)
        all_bbox_preds = self.reg_head(hs)
        
        return all_cls_scores, all_bbox_preds

# ==========================================
# 4. 运行验证 (Smoke Test)
# ==========================================
def main():
    cfg = PETRConfig()
    model = PETRImpl(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Mock Input Data
    B = 1
    imgs = torch.randn(B, 6, 3, 256, 704).to(device)
    
    # 简单的相机参数模拟
    rots = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    trans = torch.zeros(B, 6, 3).to(device)
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    intrins[:, :, 0, 0] = 500.0 # fx
    intrins[:, :, 1, 1] = 500.0 # fy
    intrins[:, :, 0, 2] = 352.0 # cx
    intrins[:, :, 1, 2] = 128.0 # cy
    
    print("=== Start PETR Inference ===")
    cls_scores, bbox_preds = model(imgs, rots, trans, intrins)
    
    print(f"Input Shape: {imgs.shape}")
    print(f"Output Cls Shape: {cls_scores.shape} (Expected: [1, 900, 10])")
    print(f"Output Box Shape: {bbox_preds.shape} (Expected: [1, 900, 10])")
    print("✅ PETR Forward Pass Successful")

if __name__ == "__main__":
    main()