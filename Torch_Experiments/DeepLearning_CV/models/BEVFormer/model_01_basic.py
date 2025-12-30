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
    
    # 定义了感知的物理范围，单位是米
    # [x_min, y_min, z_min, x_max, y_max, z_max]
    pc_range = [-50, -50, -5, 50, 50, 3] 
    

# ==========================================
# 2. 核心组件: 空间交叉注意力 (Spatial Cross-Attention)
# 对应论文 Section 3.3
# ==========================================
class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力 (Spatial Cross-Attention, SCA) 模块

    这是 BEVFormer 的核心组件。它的作用是让每一个 BEV Query (可以理解为鸟瞰图上的一个格子)，
    去多个相机的 2D 图像特征中，主动地查找和自己相关的 3D 空间信息，然后把这些信息聚合起来，更新自己。
    整个过程可以分解为四个核心步骤：
    
    1. Lift (提升):  
        为每个 2D 的 BEV Query 在垂直方向(Z轴)生成一系列 3D 参考点，形成一个 "Pillar" (柱子)。
        这一步由 `get_reference_points` 完成。
        
    2. Project (投影): 
        利用相机内外参矩阵，将这些 3D 参考点从自动驾驶车的坐标系投影到各个相机的 2D 图像平面上。
        这一步是 `point_sampling` 的核心逻辑。
    
    3. Sample (采样):  
        根据投影到 2D 图像上的坐标，使用 `grid_sample` 从图像特征图中提取出对应的特征。
        这一步同样在 `point_sampling` 中完成。
    
    4. Aggregate (聚合): 
        将从多个相机、多个高度点采样到的特征进行加权聚合，形成最终的输出。
        在论文中，这一步由 Deformable Attention 完成，但在此简化代码中，只做了简单的平均。
        
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # deformable_attention_weights 用于预测采样点的权重和偏移量
        self.deformable_attention_weights = nn.Linear(cfg.embed_dims, cfg.num_heads * cfg.bev_z)
        
        # output_proj 用于在特征聚合后进行一次线性变换
        self.output_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)

    def get_reference_points(self, H, W, Z, bs, device):
        """
        生成 3D 参考点 (Reference Points)
        这些参考点定义了 BEV Query 在 3D 空间中需要去哪里查找信息。
        对于 BEV 平面上的每一个点 (x, y)，我们都会在 Z 轴上生成 Z 个点，形成一个垂直的柱子。

        Args:
            H (int): BEV 网格的高度
            W (int): BEV 网格的宽度
            Z (int): Z 轴方向的采样点数量 (Pillar 的高度)
            bs (int): Batch Size
            device: torch.device

        Returns:
            torch.Tensor: 归一化的 3D 参考点坐标，形状为 (bs, H*W, Z, 3)，3 代表 (x, y, z)
        """
        
        # 1. 生成 BEV 平面的 2D 网格坐标
        # torch.meshgrid 会创建两个张量，`ref_y` 的每一行都是 0.5, 1.5, ..., H-0.5
        # `ref_x` 的每一列都是 0.5, 1.5, ..., W-0.5
        # `indexing='ij'` 保证了输出的 ref_y, ref_x 的形状是 (H, W)，符合直觉
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # 归一化坐标到 [0, 1] 范围
        # Flatten: (H, W) -> (H*W,) -> (N_query,) -> (1, N_query)
        ref_y = ref_y.reshape(-1)[None] / H 
        ref_x = ref_x.reshape(-1)[None] / W 

        # 2. 扩展 2D 坐标以匹配 Z 轴
        # 目标形状: (1, N_query, Z)
        # ref_x / ref_y 当前是 (1, N_query)，需要扩展最后一维变成 (1, N_query, 1)，然后复制 Z 次
        # (1, H*W, Z)
        ref_x = ref_x.unsqueeze(-1).repeat(1, 1, Z) 
        
        # (1, H*W, Z)
        ref_y = ref_y.unsqueeze(-1).repeat(1, 1, Z) 
        
        # 3. 生成 Z 轴坐标
        # ref_z 初始是 (Z,) -> (1, 1, Z) -> 复制 H*W 次变成 (1, H*W, Z)
        ref_z = torch.linspace(0.5, Z - 0.5, Z, dtype=torch.float32, device=device)
        
        # (1, H*W, Z)
        ref_z = ref_z.view(1, 1, Z).repeat(1, H * W, 1) 
        
        # 4. 堆叠成 3D 坐标
        # (1, H*W, Z) + (1, H*W, Z) + (1, H*W, Z) -> (1, H*W, Z, 3)
        # stack 之后变成 (1, H*W, Z, 3)
        ref_3d = torch.stack((ref_x, ref_y, ref_z), -1)
        
        # 5. 复制 Batch 维度
        # (1, H*W, Z, 3) -> (bs, H*W, Z, 3)
        return ref_3d.repeat(bs, 1, 1, 1)

    def point_sampling(self, reference_points, img_feats, lidar2img):
        """
        核心的投影与采样函数 (对应论文 Eq.4)
        将 3D 参考点投影到 2D 图像，并采样该处的图像特征。

        Args:
            reference_points (torch.Tensor): 归一化的 3D 坐标      (bs, N_query, Z, 3)
            img_feats (torch.Tensor): 从 backbone 提取的图像特征    (bs, N_cam, C, H_img, W_img)
            lidar2img (torch.Tensor): 投影矩阵                     (bs, N_cam, 4, 4)，用于将激光雷达/车辆坐标系下的点投影到图像坐标系

        Returns:
            torch.Tensor: 从图像上采样到的特征 (bs, N_cam, C, N_query, Z)
        """
        
        B, N_query, Z, _ = reference_points.shape
        N_cam = img_feats.shape[1]
        
        # --- A. 坐标反归一化 (0~1 -> 真实世界坐标) ---
        # 将 [0, 1] 范围的参考点坐标，映射回 `pc_range` 定义的真实世界物理坐标
        pc_range = self.cfg.pc_range
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        
        # 变成齐次坐标 (x, y, z, 1)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        
        # --- B. 3D -> 2D 投影 ---
        # 准备进行批量矩阵乘法，需要精心调整形状
        # 目标: 将每个 3D 点都用每个相机的投影矩阵进行转换，调整形状以进行矩阵乘法
        # (bs, N_query, Z, 4) -> (bs, 1, N_query*Z, 4) -> (bs, N_cam, N_query*Z, 4)
        reference_points = reference_points.view(B, 1, N_query * Z, 4).repeat(1, N_cam, 1, 1)
        lidar2img = lidar2img.view(B, N_cam, 1, 4, 4)
        
        # 矩阵乘法: img_pt = Matrix @ world_pt
        # 矩阵乘法: (bs, N_cam, N_q*Z, 4, 4) @ (bs, N_cam, N_q*Z, 4, 1) -> (bs, N_cam, N_q*Z, 4, 1)
        # (B, N_cam, N_q*Z, 4, 1)
        # matmul 后 squeeze(-1) 去掉最后的 1，形状变为 (bs, N_cam, N_q*Z, 4)
        reference_points_cam = torch.matmul(lidar2img, reference_points.unsqueeze(-1)).squeeze(-1)
        
        # 齐次坐标除以 Z (深度): x_img = x_cam / z_cam, y_img = y_cam / z_cam
        # 齐次坐标 -> 像素坐标: (x_cam, y_cam, z_cam, w_cam) -> (x_img, y_img)
        # x_img = x_cam / z_cam,  y_img = y_cam / z_cam
        eps = 1e-5
        
        # 过滤掉深度值 z_cam <= 0 的点 (这些点在相机背后)
        mask = (reference_points_cam[..., 2:3] > eps) # 过滤掉相机背后的点
        
        # 除以深度，得到归一化的图像平面坐标
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        # --- C. 归一化到 [-1, 1] 用于 grid_sample ---
        # 假设输入特征图大小 (这里简单模拟，真实情况需根据 img_feats shape 计算)
        H_img, W_img = img_feats.shape[-2], img_feats.shape[-1]
        
        # (-1, -1) 是左上角, (1, 1) 是右下角
        reference_points_cam[..., 0] /= W_img
        reference_points_cam[..., 1] /= H_img
        
        # 将 [0, 1] 范围映射到 [-1, 1]
        reference_points_cam = (reference_points_cam - 0.5) * 2 
        
        # --- D. 采样 (Sampling) ---
        # 这里的 mask 非常重要，如果点投到了图片外面，要置零
        mask = mask & (reference_points_cam[..., 0:1] > -1.0) \
                    & (reference_points_cam[..., 0:1] < 1.0) \
                    & (reference_points_cam[..., 1:2] > -1.0) \
                    & (reference_points_cam[..., 1:2] < 1.0)
        
        # 使用 grid_sample 进行双线性插值
        # Input: (B*N_cam, C, H_img, W_img)
        # Grid:  (B*N_cam, N_q*Z, 1, 2)
        img_feats_reshaped = img_feats.view(B * N_cam, self.cfg.embed_dims, H_img, W_img)
        
        # 准备格网：grid 需要是 (N, H_out, W_out, 2)
        # 这里我们只想对一串点采样，所以 H_out=N_q*Z, W_out=1
        # (B, N_cam, N_q*Z, 2) -> (B*N_cam, N_q*Z, 1, 2)
        grid = reference_points_cam.view(B * N_cam, N_query * Z, 1, 2)
        
        # grid_sample!
        # 输出形状: (B*N_cam, C, N_q*Z, 1)
        sampled_feats = F.grid_sample(img_feats_reshaped, grid, align_corners=False) # (B*N_cam, C, N_q*Z, 1)
        sampled_feats = sampled_feats.view(B, N_cam, self.cfg.embed_dims, N_query, Z)
        
        # 应用 mask (把投到图片外的点特征清零)
        # 将无效点 (在相机背后或图像外) 的特征置零
        # (B, N_cam, C, N_q, Z) * (B, N_cam, 1, N_q, Z) (自动广播)
        mask = mask.view(B, N_cam, 1, N_query, Z)
        
        # grid_sample 在边界外可能会产生 nan，用 0 替换
        return torch.nan_to_num(sampled_feats * mask)

    def forward(self, query, img_feats, lidar2img):
        """
        SCA 的前向传播函数

        Args:
            query (torch.Tensor): BEV Queries 时序融合结果 (bs, N_query, C)
            img_feats (torch.Tensor): 图像特征 (bs, N_cam, C, H_img, W_img)
            lidar2img (torch.Tensor): 投影矩阵 (bs, N_cam, 4, 4)
        """
        bs, num_query, _ = query.shape
        
        # 1. 为 BEV Query 创建 3D 空间中的参考点
        # 输出形状: (bs, N_query, Z, 3)
        ref_points = self.get_reference_points(self.cfg.bev_h, self.cfg.bev_w, self.cfg.bev_z, bs, query.device)
        
        # 2. 从多视角图像中采样特征，像素的特征点
        # 输出形状: (bs, N_cam, C, N_query, Z)
        # sampled_feats shape: (B, N_cam, C, N_query, Z)
        sampled_feats = self.point_sampling(ref_points, img_feats, lidar2img)
        
        # 3. 聚合采样到的特征 (Simplified Aggregation)
        # 这是一个极简化的 Attention 聚合过程，用于教学目的
        # 论文中的 Deformable Attention 会用 query 去和 sampled_feats 计算注意力权重，然后加权求和
        # 简化步骤 3.1: 简单地在相机维度上取平均，聚合来自不同视角的信息
        # (bs, N_cam, C, N_query, Z) -> (bs, C, N_query, Z)
        # 这是一个简化的 Attention：只用 Linear 层预测权重，而不是 Query @ Key
        # (B, N_query, heads, Z) -> 这里为了简化代码，不模拟多头，直接对 Z 轴和 Cam 轴加权
        weights = torch.mean(sampled_feats, dim=1) # 简单平均 6 个相机 (B, C, N_query, Z)
        weights = torch.mean(weights, dim=3)       # 简单平均 Z 轴 (B, C, N_query)
        
        output = weights.permute(0, 2, 1) # (B, N_query, C)
        return self.output_proj(output) + query # Residual connection

# ==========================================
# 3. 核心组件: 时序自注意力 (Temporal Self-Attention)
# 对应论文 Section 3.4
# ==========================================
class TemporalSelfAttention(nn.Module):
    """
    TSA 模块：利用历史 BEV 特征增强当前 Query。
    关键：Alignment (对齐)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.self_attn = nn.MultiheadAttention(cfg.embed_dims, cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.embed_dims)

    def forward(self, query, prev_bev, ego_motion):
        """
        query: (B, N_query, C) 当前时刻的 Query
        prev_bev: (B, N_query, C) 上一帧算好的 BEV 特征
        ego_motion: (B, 4, 4) 两帧之间的位姿变换矩阵
        """
        if prev_bev is None:
            # 第一帧没有历史，做普通的 Self-Attention
            attn_out, _ = self.self_attn(query, query, query)
            return self.norm(query + attn_out)
        
        # --- 1. Alignment (对齐) ---
        # 真正的对齐需要 grid_sample，这里用简化版：假设车只平移了
        # 将 prev_bev reshape 回 (B, C, H, W) 进行 grid_sample
        B, N, C = prev_bev.shape
        H, W = self.cfg.bev_h, self.cfg.bev_w
        prev_bev_spatial = prev_bev.permute(0, 2, 1).view(B, C, H, W)
        
        # 模拟对齐：生成一个 grid (实际应该用 ego_motion 计算 affine grid)
        # 这里仅为演示 Pipeline，不做真实的 grid 计算
        # 做仿射变换
        grid = F.affine_grid(
            torch.eye(2, 3, device=query.device).unsqueeze(0).repeat(B,1,1), 
            [B, C, H, W], 
            align_corners=False
        )
        
        aligned_prev_bev = F.grid_sample(prev_bev_spatial, grid, align_corners=False)
        aligned_prev_bev = aligned_prev_bev.flatten(2).permute(0, 2, 1) # (B, N, C)
        
        # --- 2. Temporal Fusion (Concat Query + History) ---
        # 论文中是 Query 去查 (Query + History)
        key_value = torch.cat([query, aligned_prev_bev], dim=1) # (B, 2*N, C)
        
        # --- 3. Attention ---
        # Query 查 Key_Value
        attn_out, _ = self.self_attn(query, key_value, key_value)
        
        # 典型的 Transformer 残差连接结构: x = norm(x + dropout(sub_layer(x)))
        # 也就是： 原有 Query + 历史信息 -> 归一化
        return self.norm(query + attn_out)

# ==========================================
# 4. 完整的 BEVFormer 编码器层
# ==========================================
import torch
import torch.nn as nn

class BEVFormerEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # ==================================================
        # 1. 时序自注意力 (Temporal Self-Attention, TSA)
        # ==================================================
        # 作用：让当前的 BEV Query "看一眼" 历史，继承上一帧的特征
        # 注意：这里不仅是 Query 自己查自己，更是查 "对齐后的历史 BEV"
        self.tsa = TemporalSelfAttention(cfg)
        
        # 这里的 Norm 层用于 TSA 之后的归一化 (Pre-Norm 或 Post-Norm 结构)
        self.norm1 = nn.LayerNorm(cfg.embed_dims)
        self.dropout1 = nn.Dropout(0.1) # 防止过拟合

        # ==================================================
        # 2. 空间交叉注意力 (Spatial Cross-Attention, SCA)
        # ==================================================
        # 作用：让 BEV Query "看一眼" 当前的 6 张环视图片，提取视觉特征
        # 这是 BEVFormer 最核心的创新点 (3D -> 2D 投影采样)
        self.sca = SpatialCrossAttention(cfg)
        
        # SCA 之后的归一化
        self.norm2 = nn.LayerNorm(cfg.embed_dims)
        self.dropout2 = nn.Dropout(0.1)

        # ==================================================
        # 3. 前馈网络 (Feed Forward Network, FFN)
        # ==================================================
        # 作用：标准的 MLP，用于特征的非线性变换和整合
        # 结构通常是: Linear -> Activation -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dims, cfg.embed_dims * 2), # 升维 (通常是 2倍或4倍)
            nn.ReLU(),                                     # 激活函数
            nn.Dropout(0.1),
            nn.Linear(cfg.embed_dims * 2, cfg.embed_dims), # 降维回原尺寸
            nn.Dropout(0.1)
        )
        
        # FFN 之后的归一化
        self.norm3 = nn.LayerNorm(cfg.embed_dims)

    def forward(self, query, prev_bev, img_feats, ego_motion, lidar2img):
        """
        参数说明:
        query:      (B, N_query, C)      -> 当前时刻初始化的 BEV 网格查询向量
        prev_bev:   (B, N_query, C)      -> 上一时刻输出的 BEV 特征 (历史记忆)
        img_feats:  (B, N_cam, C, H, W)  -> 当前时刻 6 个相机的图像特征
        ego_motion: (B, 4, 4)            -> 两帧之间的自车运动矩阵 (用于对齐历史)
        lidar2img:  (B, N_cam, 4, 4)     -> 3D 到 2D 的投影矩阵 (用于 SCA)
        """
        
        # --------------------------------------------------
        # Step 1: 时序融合 (Temporal Self-Attention)
        # --------------------------------------------------
        # 这里的 query 是残差连接的基准 (Identity)
        # 这里的 src 是 TSA 算出来的增量信息 (包含了历史信息)
        src = self.tsa(query, prev_bev, ego_motion)
        
        # 典型的 Transformer 残差连接结构: x = norm(x + dropout(sub_layer(x)))
        # 也就是： 原有 Query + 历史信息 -> 归一化
        query = self.norm1(query + self.dropout1(src))

        # --------------------------------------------------
        # Step 2: 空间感知 (Spatial Cross-Attention)
        # --------------------------------------------------
        # 拿着融合了历史信息的 Query，去当前的图片里找特征
        src = self.sca(query, img_feats, lidar2img)
        
        # 再次残差连接 + 归一化
        # 也就是：(Query+历史) + 当前图片特征 -> 归一化
        query = self.norm2(query + self.dropout2(src))

        # --------------------------------------------------
        # Step 3: 特征精修 (Feed Forward Network)
        # --------------------------------------------------
        # 通过 MLP 进行特征的非线性变换
        src = self.ffn(query)
        
        # 最后一次残差连接 + 归一化
        query = self.norm3(query + src)

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
        # 形状: (H*W, C)
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
        # flatten cams to batch: (B*N_cam, 3, H, W)
        imgs_reshaped = imgs.view(-1, C_img, H_img, W_img)
        img_feats = self.backbone(imgs_reshaped) 
        # reshape back: (B, N_cam, C, H, W)
        img_feats = img_feats.view(B, N_cam, self.cfg.embed_dims, H_img, W_img)
        
        # 2. 准备 BEV Queries
        # 复制 batch 份: (B, N_query, C)
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
    
    # 1. 当前帧图像
    imgs = torch.randn(B, cfg.num_cams, 3, H_img, W_img).to(device)
    
    # 2. 投影矩阵 (Lidar -> Image)
    # 随机生成投影矩阵 (仅为跑通形状，数值无物理意义)
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