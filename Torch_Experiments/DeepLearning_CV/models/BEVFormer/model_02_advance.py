import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础配置 (Configuration)
# ==========================================
class Config:
    bev_h = 50          # BEV 高度 (格子数)
    bev_w = 50          # BEV 宽度
    bev_z = 4           # Pillar z轴采样点数 (论文中 N_ref)
    embed_dims = 256    # 特征维度
    num_heads = 8       # 注意力头数
    num_cams = 6        # 相机数量
    
    # [x_min, y_min, z_min, x_max, y_max, z_max]
    pc_range = [-50, -50, -5, 50, 50, 3] 
    

# ==========================================
# 2. 核心组件: 空间交叉注意力 (Spatial Cross-Attention)
# 对应论文 Section 3.3
# ==========================================

class SpatialCrossAttention(nn.Module):
    """
    SCA 模块（简化版，但更接近 BEVFormer 思想）：
    - 生成 pillar reference points: (x,y) -> (x,y,z_k)
    - lidar2img 投影到每个相机 2D
    - grid_sample 采样图像特征
    - 用 query 预测 (cam,z) 权重做加权聚合（替代简单 mean）
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 让每个 query 预测对 (cam,z) 的权重
        self.attn_weights = nn.Linear(cfg.embed_dims, cfg.num_cams * cfg.bev_z)
        self.output_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)

    def get_reference_points(self, H, W, Z, bs, device):
        """
        返回归一化 3D reference points: (B, N_query, Z, 3)
        x,y in [0,1], z in [0,1]（后面会映射到 pc_range）
        """
        
        # (H,W) meshgrid in "ij" order
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )

        # flatten -> (1, Nq)
        ref_y = (ref_y.reshape(-1)[None] / H)  # (1,Nq)
        ref_x = (ref_x.reshape(-1)[None] / W)  # (1,Nq)

        # expand to (1, Nq, Z)
        ref_x = ref_x.unsqueeze(-1).repeat(1, 1, Z)
        ref_y = ref_y.unsqueeze(-1).repeat(1, 1, Z)

        # z in [0,1] with Z bins
        ref_z = torch.linspace(0.5, Z - 0.5, Z, dtype=torch.float32, device=device)
        ref_z = (ref_z / Z).view(1, 1, Z).repeat(1, H * W, 1)  # (1,Nq,Z)

        # stack -> (1, Nq, Z, 3)
        ref_3d = torch.stack((ref_x, ref_y, ref_z), dim=-1)

        # repeat batch -> (B, Nq, Z, 3)
        return ref_3d.repeat(bs, 1, 1, 1)

    
    def point_sampling(self, reference_points, img_feats, lidar2img):
        """
        reference_points: (B, Nq, Z, 3) 归一化
        img_feats:        (B, Ncam, C, Hf, Wf)
        lidar2img:        (B, Ncam, 4, 4)

        return sampled_feats: (B, Ncam, C, Nq, Z)
        """
        B, Nq, Z, _ = reference_points.shape
        Ncam = img_feats.shape[1]
        C = img_feats.shape[2]
        Hf, Wf = img_feats.shape[-2], img_feats.shape[-1]

        # ---- A) 归一化(0~1) -> 世界坐标(pc_range) ----
        pc = self.cfg.pc_range
        
        # clone 防止 in-place 污染外部 [1, 2500, 4, 3]
        ref = reference_points.clone()
        ref[..., 0] = ref[..., 0] * (pc[3] - pc[0]) + pc[0]
        ref[..., 1] = ref[..., 1] * (pc[4] - pc[1]) + pc[1]
        ref[..., 2] = ref[..., 2] * (pc[5] - pc[2]) + pc[2]

        # homogeneous: (x,y,z,1)
        ref = torch.cat([ref, torch.ones_like(ref[..., :1])], dim=-1)  # (B,Nq,Z,4)

        # ---- B) 投影到每个相机 ----
        # (B, 1, Nq*Z, 4) -> (B, Ncam, Nq*Z, 4)
        ref = ref.view(B, 1, Nq * Z, 4).repeat(1, Ncam, 1, 1)
        # (B,Ncam,1,4,4)
        l2i = lidar2img.view(B, Ncam, 1, 4, 4)

        # (B, Ncam, Nq*Z, 4)
        cam = torch.matmul(l2i, ref.unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        depth = cam[..., 2:3]
        valid = depth > eps

        # perspective divide -> pixel coords
        xy = cam[..., 0:2] / torch.clamp(depth, min=eps)  # (B,Ncam,Nq*Z,2)

        # ---- C) 像素坐标 -> [-1,1] 给 grid_sample ----
        # 这里假设 xy 已经是以特征图坐标为基准的像素坐标
        # 如果你的 lidar2img 是原图像像素坐标，需要再除以 stride 映射到 feature map
        x = xy[..., 0] / (Wf - 1)
        y = xy[..., 1] / (Hf - 1)
        grid = torch.stack([(x - 0.5) * 2.0, (y - 0.5) * 2.0], dim=-1)  # (B,Ncam,Nq*Z,2)

        # 视野内 mask
        valid = valid & (grid[..., 0:1] > -1.0) & (grid[..., 0:1] < 1.0) & \
                      (grid[..., 1:2] > -1.0) & (grid[..., 1:2] < 1.0)

        # ---- D) grid_sample ----
        img = img_feats.view(B * Ncam, C, Hf, Wf)
        grid_rs = grid.view(B * Ncam, Nq * Z, 1, 2)

        # (B*Ncam, C, Nq*Z, 1)
        sampled = F.grid_sample(img, grid_rs, align_corners=False)  
        sampled = sampled.view(B, Ncam, C, Nq, Z)

        valid = valid.view(B, Ncam, 1, Nq, Z)
        sampled = sampled * valid
        return torch.nan_to_num(sampled)

    def forward(self, query, img_feats, lidar2img):
        """
        query:    (B, Nq, C) 时间自注意力机制的输出结果
        img_feats:(B, Ncam, C, Hf, Wf)
        lidar2img:(B, Ncam, 4, 4)
        return:   (B, Nq, C)
        """
        B, Nq, C = query.shape
        Ncam = img_feats.shape[1]
        Z = self.cfg.bev_z

        # 1) reference points
        ref = self.get_reference_points(self.cfg.bev_h, self.cfg.bev_w, Z, B, query.device)  # (B,Nq,Z,3)

        # 2) sample image features
        sampled = self.point_sampling(ref, img_feats, lidar2img)  # (B,Ncam,C,Nq,Z)

        # 3) query-conditioned weights over cam*Z
        w = self.attn_weights(query).view(B, Nq, Ncam, Z)  # (B,Nq,Ncam,Z)
        w = F.softmax(w.flatten(2), dim=-1).view(B, Nq, Ncam, Z)

        # 4) weighted sum
        # sampled: (B,Ncam,C,Nq,Z) -> (B,Nq,Ncam,Z,C)
        sampled = sampled.permute(0, 3, 1, 4, 2)
        
        # [B, Nq, C]
        out = (sampled * w.unsqueeze(-1)).sum(dim=2).sum(dim=2)  

        out = self.output_proj(out)
        
        return out + query


# ==========================================
# 3. 核心组件: 时序自注意力 (Temporal Self-Attention)
# 对应论文 Section 3.4
# ==========================================
class TemporalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.self_attn = nn.MultiheadAttention(cfg.embed_dims, cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.embed_dims)

    def _warp_prev_bev(self, prev_bev, ego_motion, H, W):
        """
        prev_bev: (B, N, C)
        ego_motion: (B, 4, 4), T_{t-1 -> t}
        return aligned_prev_bev: (B, N, C)
        """
        B, N, C = prev_bev.shape
        prev_bev_spatial = prev_bev.permute(0, 2, 1).reshape(B, C, H, W)  # (B,C,H,W)

        # ---- 1) 从 ego_motion 取平移(tx,ty) 和 yaw ----
        # ego_motion 是 4x4: [R t; 0 1]
        tx = ego_motion[:, 0, 3]  # (B,)
        ty = ego_motion[:, 1, 3]  # (B,)
        # yaw from rotation matrix around z
        yaw = torch.atan2(ego_motion[:, 1, 0], ego_motion[:, 0, 0])  # (B,)

        # ---- 2) 米 -> 像素（根据 pc_range 和 bev 尺度）----
        pc = self.cfg.pc_range  # [x_min,y_min,z_min,x_max,y_max,z_max]
        meter_per_px_x = (pc[3] - pc[0]) / W
        meter_per_px_y = (pc[4] - pc[1]) / H
        dx_px = tx / meter_per_px_x  # (B,)
        dy_px = ty / meter_per_px_y  # (B,)

        # ---- 3) 构造 2x3 affine（把 prev warp 到 curr）----
        # 关键：affine_grid 的平移是归一化到 [-1,1]
        cos_r = torch.cos(yaw)
        sin_r = torch.sin(yaw)

        theta = torch.zeros(B, 2, 3, device=prev_bev.device, dtype=prev_bev.dtype)
        theta[:, 0, 0] = cos_r
        theta[:, 0, 1] = -sin_r
        theta[:, 1, 0] = sin_r
        theta[:, 1, 1] = cos_r

        # 像素 -> 归一化（align_corners=False 时用 W/2, H/2 是常见近似）
        theta[:, 0, 2] = dx_px * 2.0 / W
        theta[:, 1, 2] = dy_px * 2.0 / H

        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        aligned = F.grid_sample(prev_bev_spatial, grid, align_corners=False, padding_mode='zeros')
        aligned_prev_bev = aligned.flatten(2).permute(0, 2, 1)  # (B,N,C)
        return aligned_prev_bev

    def forward(self, query, prev_bev, ego_motion):
        """
        query: (B, N, C)
        prev_bev: (B, N, C) or None
        ego_motion: (B, 4, 4)
        """
        if prev_bev is None:
            attn_out, _ = self.self_attn(query, query, query) # [1, 2500, 256]
            return self.norm(query + attn_out)

        H, W = self.cfg.bev_h, self.cfg.bev_w
        aligned_prev = self._warp_prev_bev(prev_bev, ego_motion, H, W)  # (B, N, C)

        # 论文语义：query 去查 (当前 query + 历史 memory)
        key_value = torch.cat([query, aligned_prev], dim=1)  # (B,2N,C)
        attn_out, _ = self.self_attn(query, key_value, key_value)       # (B,N,C)
        return self.norm(query + attn_out)


# ==========================================
# 4. 完整的 BEVFormer 编码器层
# ==========================================
class BEVFormerEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.tsa = TemporalSelfAttention(cfg)
        
        self.sca = SpatialCrossAttention(cfg)
        
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dims, cfg.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(cfg.embed_dims * 2, cfg.embed_dims),
            nn.LayerNorm(cfg.embed_dims)
        )
        
    def forward(self, query, prev_bev, img_feats, ego_motion, lidar2img):
        
        # 1. Temporal Self-Attention 时间自注意力机制
        query = self.tsa(query, prev_bev, ego_motion)
        
        # 2. Spatial Cross-Attention 空间交叉注意力机制
        query = self.sca(query, img_feats, lidar2img)
        
        # 3. Feed Forward
        query = query + self.ffn(query)
        
        return query

# ==========================================
# 5. 主模型: BEVFormer
# ==========================================
class BEVFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Mock Backbone (1x1 conv 代替 ResNet)
        self.backbone = nn.Conv2d(3, cfg.embed_dims, kernel_size=1)
        
        # BEV Queries (可学习参数)
        # 形状: (H*W, C) [2500, 256]
        self.bev_queries = nn.Parameter(torch.randn(cfg.bev_h * cfg.bev_w, cfg.embed_dims))
        
        # Encoder Layers (论文通常用6层)
        self.layers = nn.ModuleList([BEVFormerEncoderLayer(cfg) for _ in range(3)])
        
    def forward(self, imgs, prev_bev=None, ego_motion=None, lidar2img=None):
        """
        imgs: (B, N_cam, 3, H_img, W_img)
        prev_bev: (B, N_query, C) 上一帧的输出
        """
        B, N_cam, C_img, H_img, W_img = imgs.shape
        
        # 1. 提取图像特征 (Mock Backbone)
        # flatten cams to batch: (B*N_cam, 3, H, W) [6, 3, 128, 128]
        imgs_reshaped = imgs.view(-1, C_img, H_img, W_img)
        img_feats = self.backbone(imgs_reshaped) # [6, 256, 128, 128] 
        # reshape back: (B, N_cam, C, H, W) [1, 6, 256, 128, 128]
        img_feats = img_feats.view(B, N_cam, self.cfg.embed_dims, H_img, W_img)
        
        # 2. 准备 BEV Queries
        # 复制 batch 份: (B, N_query, C) [1, 2500, 256]
        queries = self.bev_queries.unsqueeze(0).repeat(B, 1, 1)
        
        # 3. 进入 Transformer Encoder
        for layer in self.layers:
            queries = layer(queries, prev_bev, img_feats, ego_motion, lidar2img)
            
        return queries

# ==========================================
# 6. 模拟运行 Pipeline (Pipeline Simulation)
# ==========================================
def main():
    cfg = Config()
    model = BEVFormer(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model initialized. Device: {device}")
    
    # --- 模拟输入数据 ---
    B = 1
    H_img, W_img = 128, 128
    
    # 1. 当前帧图像 [1, 6, 3, 128, 128]
    imgs = torch.randn(B, cfg.num_cams, 3, H_img, W_img).to(device)
    
    # 2. 投影矩阵 (Lidar -> Image)
    # 随机生成投影矩阵 (仅为跑通形状，数值无物理意义) [1, 6, 4, 4]
    lidar2img = torch.randn(B, cfg.num_cams, 4, 4).to(device)
    
    # 3. 自车运动 (Ego Motion)
    # T_{t-1 -> t}
    ego_motion = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
    
    # --- 推理过程 (Inference Loop) ---
    # 模拟两帧: t-1 和 t
    
    # Frame 1 (t-1): 历史上没有 BEV，prev_bev=None
    print("\nProcessing Frame t-1...")
    bev_t_minus_1 = model(imgs, prev_bev=None, ego_motion=ego_motion, lidar2img=lidar2img)
    
    # (B, 2500, 256)
    print(f"Output BEV Shape: {bev_t_minus_1.shape}") 
    
    # Frame 2 (t): 传入上一帧的 BEV
    # 关键：必须 detach，切断梯度流，模拟 inference 时的 recurrence
    prev_bev = bev_t_minus_1.detach()
    
    print("\nProcessing Frame t (with history)...")
    bev_t = model(imgs, prev_bev=prev_bev, ego_motion=ego_motion, lidar2img=lidar2img)
    print(f"Output BEV Shape: {bev_t.shape}")
    
    print("\nPipeline Test Passed! ✅")

if __name__ == "__main__":
    main()